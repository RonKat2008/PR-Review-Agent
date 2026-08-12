"""Unwired-export detector (P13): a new public symbol nobody outside this PR calls.

Their best finding class, in one line: "PR adds public function `X`, exported and
tested, but grep shows ZERO production call sites — only tests. The PR body calls
this channel essential. Wire it or defer explicitly." High-value because it is
deterministic — either callers exist or they do not — and low-noise because it only
ever fires on symbols this very diff introduced.

This is the mirror image of P9 (`blast.py`), and that relationship is the whole
design:

    P9  (blast)    "what does this change BREAK outside the diff?"
                    -> cares about symbols that EXISTED before: old_signature != ""
    P13 (exports)  "what does this change ADD that nothing outside the diff uses?"
                    -> cares about symbols that did NOT exist before: old_signature == ""

`blast.extract_changed_symbols` deliberately computes only `old_sigs - new_sigs`
(removed) and `old_sigs & new_sigs` (modified) — a symbol with no prior existence
cannot break an existing caller, so blast has no reason to ever emit one, and it
does not: a purely-added Python function or class never appears in its output at
all (see its own test `test_a_newly_added_function_with_no_prior_existence_is_not_reported`).
The one exception is blast's non-Python regex fallback, whose `removed`/`added`
dicts are unioned rather than intersected, so a brand-new non-Python declaration
*does* come through with `old_signature == ""` already.

So step 1 here reuses blast's own signature-collection primitives
(`_reconstructed_lines`, `_collect_signatures`, `SIGNATURE_START`) — never
re-parsing the diff — but combines them with the complement set blast itself never
takes: `new_sigs.keys() - old_sigs.keys()`. For non-Python files, blast's public
`extract_changed_symbols` already carries that complement, so it is reused as-is
and simply filtered down to entries with an empty `old_signature`.

Step 2 is `git grep -n`, exactly blast's convention (same over-match tolerance: a
false candidate costs a glance, reusing blast's own grep-line parser so malformed
and binary-match lines degrade the same way in both places). Hits are partitioned
three ways: inside the diff's own files (the definition itself, or same-PR usage —
ignored for the caller count, but the one place `git grep` still finds the
definition line itself, so it doubles as this module's line-number source); files
matching `test_globs` (counted, but not disqualifying); everything else (a real
production caller — any single one found excludes the symbol entirely).

There is no step 3 here — no model call. This module hands the model evidence and
an instruction (`render`); the judgment of whether an unwired export actually
deserves a finding, given the PR's stated intent, is left entirely to the review
prompt this feeds, the same way `analysis.py`'s lint findings are grounding rather
than a verdict.

One deliberate difference from blast's per-symbol degrade: here, a `git` failure
aborts the whole sweep step and returns `()` rather than treating just that one
symbol as zero call sites. Blast can afford per-symbol degrade because each symbol
is judged independently by the model anyway; this module's entire value is the
deterministic claim "zero production callers", and if `git grep` could not actually
run, that claim is not known to be true for anything it touched this pass — better
to say nothing than to assert an unverified zero.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .blast import (
    SIGNATURE_START,
    CallSite,
    ChangedSymbol,
    _collect_signatures,
    _parse_grep_line,
    _reconstructed_lines,
    extract_changed_symbols,
)
from .context import GitError, GitRunner
from .diffs import FileDiff, split_by_file

DEFAULT_TEST_GLOBS: tuple[str, ...] = (
    "test_*",
    "*_test.*",
    "tests/**",
    "**/tests/**",
    "conftest*",
)


@dataclass(frozen=True)
class UnwiredExport:
    """A public symbol this diff added that nothing outside the diff calls yet."""

    symbol: str
    file: str
    line: int | None
    test_references: int
    definition_kind: str  # "function" | "class"


def find_unwired_exports(
    diff: str,
    git_runner: GitRunner,
    repo_root: Path | str,
    test_globs: Sequence[str] = DEFAULT_TEST_GLOBS,
) -> tuple[UnwiredExport, ...]:
    """Every newly added public symbol with zero production callers, sorted.

    Deterministic and model-free. A `git` failure or zero added symbols both
    degrade to `()` — see the module docstring for why a git failure aborts here
    rather than degrading symbol by symbol the way blast does.
    """
    symbols = _added_public_symbols(diff)
    if not symbols:
        return ()

    diff_paths = frozenset(f.path for f in split_by_file(diff))
    root = Path(repo_root)

    try:
        evaluated = [
            _evaluate(symbol, diff_paths, root, git_runner, test_globs) for symbol in symbols
        ]
    except GitError:
        return ()

    unwired = tuple(u for u in evaluated if u is not None)
    return tuple(sorted(unwired, key=lambda u: (u.file, u.symbol)))


def render(unwired: tuple[UnwiredExport, ...]) -> str:
    """Markdown block for the review prompt. Empty input renders to `""`."""
    if not unwired:
        return ""

    lines = ["## Newly added public symbols with no production callers", ""]
    for item in unwired:
        location = f"{item.file}:{item.line}" if item.line is not None else item.file
        lines.append(
            f"- `{location}` — `{item.symbol}` ({item.definition_kind}), "
            f"{item.test_references} test reference(s), 0 production callers"
        )
    lines.append("")
    lines.append(
        "_Cross-check each of these against the PR's stated intent. Raise a finding "
        "ONLY if the PR body or title implies this symbol should already be wired "
        "into production; otherwise suggest the PR note explicitly that wiring it "
        "up is deferred._"
    )
    return "\n".join(lines) + "\n"


def activity_note(unwired: tuple[UnwiredExport, ...]) -> str:
    """One-line sweep observability note, or `""` when there is nothing to say."""
    if not unwired:
        return ""
    return f"exports: {len(unwired)} added public symbol(s) with zero production callers"


# --- step 1: added-and-public symbol extraction, reusing blast's machinery ----


def _added_public_symbols(diff: str) -> tuple[ChangedSymbol, ...]:
    """Symbols this diff introduces for the first time: public, name is new.

    Python files: `new_sigs.keys() - old_sigs.keys()`, the complement blast's own
    extraction never takes (see module docstring). Everything else: blast's own
    fallback output already represents "added" as an empty `old_signature`, so it
    is reused verbatim rather than re-parsed.
    """
    added: list[ChangedSymbol] = []
    for file_diff in split_by_file(diff):
        if file_diff.path.endswith(".py"):
            added.extend(_python_additions(file_diff))
    added.extend(
        symbol
        for symbol in extract_changed_symbols(diff)
        if symbol.language != "python" and not symbol.old_signature and symbol.new_signature
    )
    return tuple(s for s in added if not s.name.startswith("_"))


def _python_additions(file_diff: FileDiff) -> list[ChangedSymbol]:
    old_lines = _reconstructed_lines(file_diff.body, keep="-", header="---")
    new_lines = _reconstructed_lines(file_diff.body, keep="+", header="+++")
    old_sigs = _collect_signatures(old_lines)
    new_sigs = _collect_signatures(new_lines)

    return [
        ChangedSymbol(
            name=name,
            file=file_diff.path,
            language="python",
            old_signature="",
            new_signature=new_sigs[name],
            change=f"'{name}' added",
            kind_hint="added",
        )
        for name in sorted(new_sigs.keys() - old_sigs.keys())
    ]


def _definition_kind(symbol: ChangedSymbol) -> str:
    if symbol.language == "python":
        match = SIGNATURE_START.match(symbol.new_signature)
        if match is not None and match.group(1) == "class":
            return "class"
    return "function"


# --- step 2: git grep, partitioned into diff / test / production --------------


def _evaluate(
    symbol: ChangedSymbol,
    diff_paths: frozenset[str],
    repo_root: Path,
    git_runner: GitRunner,
    test_globs: Sequence[str],
) -> UnwiredExport | None:
    """`None` when a real production caller exists; otherwise the finding."""
    hits = _grep(symbol.name, repo_root, git_runner)
    in_diff = tuple(h for h in hits if h.file in diff_paths)
    outside = tuple(h for h in hits if h.file not in diff_paths)
    production = tuple(h for h in outside if not _matches_any_glob(h.file, test_globs))
    if production:
        return None

    return UnwiredExport(
        symbol=symbol.name,
        file=symbol.file,
        line=_definition_line(symbol, in_diff),
        test_references=len(outside) - len(production),
        definition_kind=_definition_kind(symbol),
    )


def _grep(name: str, repo_root: Path, git_runner: GitRunner) -> tuple[CallSite, ...]:
    """Every `git grep -n` hit for `name`. A `GitError` here is not caught: the
    caller decides how a search failure should degrade the whole sweep step."""
    raw = git_runner(["-C", str(repo_root), "grep", "-n", "--", name])
    hits = (_parse_grep_line(line) for line in raw.splitlines() if line.strip())
    return tuple(h for h in hits if h is not None)


def _definition_line(symbol: ChangedSymbol, in_diff_hits: tuple[CallSite, ...]) -> int | None:
    """The definition's own line, recovered from the grep hits already found inside
    the diff's own files rather than a second, independent hunk-parsing pass.

    Matches by exact text against the first line of the signature blast already
    extracted — the same source text on both sides, since grep reports the raw
    line untouched and blast's collector only strips the def/class line's own
    leading indentation.
    """
    if not symbol.new_signature:
        return None
    first_line = symbol.new_signature.splitlines()[0].strip()
    matches = sorted(
        h.line for h in in_diff_hits if h.file == symbol.file and h.text == first_line
    )
    return matches[0] if matches else None


def _matches_any_glob(path: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)
