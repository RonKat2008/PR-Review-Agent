"""Blast radius (P9): what does this change break outside the diff?

A diff cannot show that six other files still call a function the old way — those
files have no lines in the diff at all. This module answers "what else does this
change break?" by looking outside the diff for the first time in the pipeline.

DESIGN CONSTRAINT — deterministic first, model last. Do not collapse this:

    Step 1  (pure Python, no model)  extract changed symbols from the diff itself
                                      -> functions, classes, constants
    Step 2  (git grep, no model)     find every real reference to those symbols
                                      outside the files this diff already touches
    Step 3  (model)                  judge each (symbol, caller) pair: does this
                                      change actually break this call site?

Steps 1 and 2 must never involve the model, for one reason: if a model were asked
"who calls this function", it does not know — it has no way to see the rest of
the repository from a diff — and it will answer anyway, with a plausible file
path and line number stated with exactly the same confidence a real one would
carry. A fabricated "this breaks shop/invoice.py:44" is the single most
dangerous output this system could produce: it looks indistinguishable from a
real finding and sends someone chasing a caller that was never there. So the
list of callers handed to the model in step 3 is always the real one, found by
`git grep` against files git actually tracks. The model's only job is judgment —
given a real call site and a real signature diff, does this call still work? —
never discovery. If you are tempted to let the model "just find the callers
itself" for a language ast can't parse, don't; the regex-plus-git-grep fallback
below over-matches on purpose so the model still only ever judges real lines.

`BlastRadius.unbroken_callers` (defined in `review.py`) exists so a "0 broken"
result is distinguishable from "we never checked" — every call site step 2
finds must land in either `breaks` or the unbroken count.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .context import GitError, GitRunner
from .diffs import FileDiff, split_by_file
from .github import PullRequest
from .review import BlastRadius, _extract_json, _parse_blast

ModelFn = Callable[[str], str]

DEFAULT_PROMPTS_DIR = Path("skills/pr-review/prompts")
PROMPT_FILENAME = "blast.md"

# Matches "def foo(", "async def foo(", or "class Foo" at the start of a diff
# line's content (after the leading +/- has already been stripped).
SIGNATURE_START = re.compile(r"^(async\s+def|def|class)\s+(\w+)\b")

# Module-level (unindented) constant assignment: NAME = value, optionally typed.
CONSTANT_PATTERN = re.compile(r"^([A-Z_][A-Z0-9_]*)\s*(?::\s*[^=]+)?=\s*(.+?)\s*$")

# Best-effort, permissive: JS/TS/Go/Rust-ish function declarations. There is no
# ast for arbitrary languages, so this cannot compute an exact before/after diff
# the way the Python path can — it only flags that a declaration line changed
# and lets git grep + the model do the real work. Over-matching here is by
# design; see the module docstring.
FALLBACK_SYMBOL_PATTERN = re.compile(
    r"^[+-]\s*(?:export\s+|public\s+|private\s+|static\s+|async\s+)*"
    r"(?:function|func|def|fn)\s+([A-Za-z_]\w*)\s*\("
)


class BlastRadiusError(RuntimeError):
    """The blast-radius check could not run at all (e.g. the prompt is missing)."""


@dataclass(frozen=True)
class ChangedSymbol:
    """One function, class, or constant whose definition changed in this diff.

    `kind_hint` is our own best guess at a P9 breakage kind (signature_change,
    return_shape_change, removal_or_rename, constant_change, semantic_change),
    used only to seed the prompt and as the label for a symbol with zero call
    sites. The model's own `kind` in its JSON response is authoritative for any
    symbol that actually gets judged.
    """

    name: str
    file: str
    language: str  # "python" | "other"
    old_signature: str
    new_signature: str
    change: str
    kind_hint: str = "signature_change"


@dataclass(frozen=True)
class CallSite:
    """One real reference to a changed symbol outside the diff, found by `git grep`."""

    file: str
    line: int
    text: str


def extract_changed_symbols(diff: str) -> tuple[ChangedSymbol, ...]:
    """Step 1: find every symbol whose definition changed, from the diff alone.

    Python files are parsed exactly with `ast`. Everything else falls back to a
    permissive regex, because there is no `ast` for arbitrary languages.
    """
    symbols: list[ChangedSymbol] = []
    for file_diff in split_by_file(diff):
        if file_diff.path.endswith(".py"):
            symbols.extend(_extract_python_symbols(file_diff))
        else:
            symbols.extend(_extract_fallback_symbols(file_diff))
    return tuple(symbols)


def analyze_blast_radius(
    pr: PullRequest,
    diff: str,
    model_fn: ModelFn,
    git_runner: GitRunner,
    repo_root: Path | str,
    prompts_dir: Path | str = DEFAULT_PROMPTS_DIR,
) -> tuple[BlastRadius, ...]:
    """Run all three P9 steps and return one `BlastRadius` per changed symbol.

    Step 3 makes one model call per changed symbol, carrying every real call
    site step 2 found for it — cheaper than one call per (symbol, caller) pair,
    while the judgment inside that call is still made pair by pair, over a list
    the model never had to guess at. A symbol with zero call sites never reaches
    the model at all: there is nothing to judge, and that is a valid result, not
    an error.
    """
    symbols = extract_changed_symbols(diff)
    if not symbols:
        return ()

    diff_paths = frozenset(f.path for f in split_by_file(diff))
    root = Path(repo_root)
    template = _read_prompt(Path(prompts_dir))

    return tuple(
        _judge_symbol(pr, symbol, diff_paths, model_fn, git_runner, root, template)
        for symbol in symbols
    )


def _judge_symbol(
    pr: PullRequest,
    symbol: ChangedSymbol,
    diff_paths: frozenset[str],
    model_fn: ModelFn,
    git_runner: GitRunner,
    repo_root: Path,
    template: str,
) -> BlastRadius:
    call_sites = _find_call_sites(symbol.name, diff_paths, git_runner, repo_root)
    if not call_sites:
        return BlastRadius(symbol=symbol.name, kind=symbol.kind_hint, change=symbol.change)

    prompt = build_blast_prompt(template, pr, symbol, call_sites)
    payload = _extract_json(model_fn(prompt))
    return _parse_blast(payload)


# --- Step 2: git grep, no model ----------------------------------------------


def _find_call_sites(
    symbol: str,
    diff_paths: frozenset[str],
    git_runner: GitRunner,
    repo_root: Path,
) -> tuple[CallSite, ...]:
    """Every real reference to `symbol` outside the files this diff already touches.

    A `GitError` here means the search itself failed, not that there are no
    callers. Treating it as "zero call sites found" is the safe direction: it
    costs a missed finding, never a fabricated one — the failure is absorbed
    here and never raised into the sweep.
    """
    try:
        raw = git_runner(["-C", str(repo_root), "grep", "-n", "--", symbol])
    except GitError:
        return ()
    sites = (_parse_grep_line(line) for line in raw.splitlines() if line.strip())
    return tuple(site for site in sites if site is not None and site.file not in diff_paths)


def _parse_grep_line(line: str) -> CallSite | None:
    parts = line.split(":", 2)
    if len(parts) != 3 or not parts[1].isdigit():
        return None
    path, line_no, text = parts
    return CallSite(file=path.replace("\\", "/"), line=int(line_no), text=text.strip())


# --- Step 3: model, prompt assembly and response parsing ---------------------


def build_blast_prompt(
    template: str,
    pr: PullRequest,
    symbol: ChangedSymbol,
    call_sites: tuple[CallSite, ...],
) -> str:
    """Assemble one P9 prompt: one changed symbol, plus every real call site found for it."""
    sites_block = "\n".join(f"- `{s.file}:{s.line}` — {s.text}" for s in call_sites)
    return (
        f"{template}\n\n"
        f"## Pull request #{pr.number}: {pr.title}\n\n"
        f"## Changed symbol\n\n"
        f"- name: `{symbol.name}`\n"
        f"- file: `{symbol.file}`\n"
        f"- language: {symbol.language}\n"
        f"- what changed: {symbol.change}\n\n"
        f"### Before\n\n```\n{symbol.old_signature or '(did not exist before this diff)'}\n```\n\n"
        f"### After\n\n```\n{symbol.new_signature or '(removed by this diff)'}\n```\n\n"
        f"## Call sites outside the diff — {len(call_sites)} total, every one must be accounted for\n\n"
        f"{sites_block}\n"
    )


def _read_prompt(directory: Path) -> str:
    path = directory / PROMPT_FILENAME
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BlastRadiusError(f"Could not read prompt {path}: {exc}") from exc


# --- Step 1: Python symbol extraction via ast ---------------------------------


def _extract_python_symbols(file_diff: FileDiff) -> list[ChangedSymbol]:
    old_lines = _side_lines(file_diff.body, "-")
    new_lines = _side_lines(file_diff.body, "+")

    old_sigs = _collect_signatures(old_lines)
    new_sigs = _collect_signatures(new_lines)
    old_consts = _collect_constants(old_lines)
    new_consts = _collect_constants(new_lines)

    changed: list[ChangedSymbol] = []
    changed.extend(_removed_symbols(file_diff.path, old_sigs, new_sigs))
    changed.extend(_modified_symbols(file_diff.path, old_sigs, new_sigs))
    changed.extend(_changed_constants(file_diff.path, old_consts, new_consts))
    return changed


def _side_lines(body: str, prefix: str) -> list[str]:
    """Content of every added (`+`) or removed (`-`) line, in file order.

    File header lines (`+++`/`---`) share the same first character as the
    lines we want, so they are excluded explicitly.
    """
    lines: list[str] = []
    for line in body.splitlines():
        if not line or line[0] != prefix or line.startswith(("+++", "---")):
            continue
        lines.append(line[1:])
    return lines


def _consume_signature(lines: Sequence[str], start: int) -> str:
    """From a `def`/`class` line, consume continuation lines until parens balance."""
    first = lines[start].lstrip()
    collected = [first]
    depth = first.count("(") - first.count(")")
    idx = start
    while depth > 0 and idx + 1 < len(lines):
        idx += 1
        collected.append(lines[idx])
        depth += lines[idx].count("(") - lines[idx].count(")")
    return "\n".join(collected)


def _collect_signatures(lines: Sequence[str]) -> dict[str, str]:
    """Map def/class name -> raw signature text. First occurrence wins."""
    found: dict[str, str] = {}
    i = 0
    while i < len(lines):
        match = SIGNATURE_START.match(lines[i].lstrip())
        if match is None:
            i += 1
            continue
        signature = _consume_signature(lines, i)
        found.setdefault(match.group(2), signature)
        i += signature.count("\n") + 1
    return found


def _collect_constants(lines: Sequence[str]) -> dict[str, str]:
    """Map UPPER_SNAKE_CASE module-level assignment name -> its value text."""
    found: dict[str, str] = {}
    for line in lines:
        if line[:1].isspace():
            continue  # indented — not module level
        match = CONSTANT_PATTERN.match(line)
        if match:
            found.setdefault(match.group(1), match.group(2).strip())
    return found


def _removed_symbols(path: str, old_sigs: dict[str, str], new_sigs: dict[str, str]) -> list[ChangedSymbol]:
    return [
        ChangedSymbol(
            name=name,
            file=path,
            language="python",
            old_signature=old_sigs[name],
            new_signature="",
            change=f"'{name}' was removed entirely",
            kind_hint="removal_or_rename",
        )
        for name in sorted(old_sigs.keys() - new_sigs.keys())
    ]


def _modified_symbols(path: str, old_sigs: dict[str, str], new_sigs: dict[str, str]) -> list[ChangedSymbol]:
    changed: list[ChangedSymbol] = []
    for name in sorted(old_sigs.keys() & new_sigs.keys()):
        described = _describe_signature_change(old_sigs[name], new_sigs[name])
        if described is None:
            continue
        kind_hint, change = described
        changed.append(
            ChangedSymbol(
                name=name,
                file=path,
                language="python",
                old_signature=old_sigs[name],
                new_signature=new_sigs[name],
                change=change,
                kind_hint=kind_hint,
            )
        )
    return changed


def _changed_constants(path: str, old_consts: dict[str, str], new_consts: dict[str, str]) -> list[ChangedSymbol]:
    changed: list[ChangedSymbol] = []
    for name in sorted(old_consts.keys() & new_consts.keys()):
        if old_consts[name] == new_consts[name]:
            continue
        changed.append(
            ChangedSymbol(
                name=name,
                file=path,
                language="python",
                old_signature=f"{name} = {old_consts[name]}",
                new_signature=f"{name} = {new_consts[name]}",
                change=f"constant value changed from {old_consts[name]!r} to {new_consts[name]!r}",
                kind_hint="constant_change",
            )
        )
    return changed


def _describe_signature_change(old_text: str, new_text: str) -> tuple[str, str] | None:
    """Compare a before/after `def`/`class` signature. `None` means no real change."""
    if old_text.strip() == new_text.strip():
        return None

    old_node = _parse_signature(old_text)
    new_node = _parse_signature(new_text)
    function_types = (ast.FunctionDef, ast.AsyncFunctionDef)

    if isinstance(old_node, function_types) and isinstance(new_node, function_types):
        return _describe_function_change(old_node, new_node)

    if old_node is None or new_node is None:
        return "signature_change", "signature text changed (could not fully parse for an exact diff)"

    return "semantic_change", "declaration changed"


def _parse_signature(signature_text: str) -> ast.AST | None:
    """Parse a possibly multi-line signature fragment via `ast`, for exactness.

    Wrapped with a dummy `pass` body so a bare `def foo(...):` is valid on its
    own, without needing the rest of the file.
    """
    snippet = f"{signature_text.rstrip()}\n    pass\n"
    try:
        tree = ast.parse(snippet)
    except SyntaxError:
        return None
    return tree.body[0] if tree.body else None


def _describe_function_change(
    old_node: ast.FunctionDef | ast.AsyncFunctionDef,
    new_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, str] | None:
    old_names = _param_names(old_node.args)
    new_names = _param_names(new_node.args)
    added_required = sorted(_required_params(new_node.args) - _required_params(old_node.args))
    removed = sorted(set(old_names) - set(new_names))
    old_positional = _positional_names(old_node.args)
    new_positional = _positional_names(new_node.args)
    reordered = old_positional != new_positional and set(old_positional) == set(new_positional)
    old_return = _return_annotation(old_node)
    new_return = _return_annotation(new_node)
    return_changed = old_return != new_return

    parts: list[str] = []
    kind = "signature_change"
    if added_required:
        parts.append(f"added required parameter(s) {', '.join(added_required)}")
    if removed:
        parts.append(f"removed parameter(s) {', '.join(removed)}")
    if reordered:
        parts.append("reordered positional parameters")
    if return_changed:
        parts.append(f"return annotation changed from {old_return!r} to {new_return!r}")
        if not (added_required or removed or reordered):
            kind = "return_shape_change"

    if not parts:
        return None
    return kind, "; ".join(parts)


def _positional_names(args: ast.arguments) -> list[str]:
    return [a.arg for a in (*args.posonlyargs, *args.args)]


def _param_names(args: ast.arguments) -> list[str]:
    names = _positional_names(args)
    if args.vararg:
        names.append(args.vararg.arg)
    names += [a.arg for a in args.kwonlyargs]
    if args.kwarg:
        names.append(args.kwarg.arg)
    return names


def _required_params(args: ast.arguments) -> set[str]:
    """Params with no default: mandatory positional/posonly args, plus bare kwonly args."""
    positional = (*args.posonlyargs, *args.args)
    n_defaults = len(args.defaults)
    required_positional = positional[: len(positional) - n_defaults] if n_defaults else positional
    required = {a.arg for a in required_positional}
    for kwarg, default in zip(args.kwonlyargs, args.kw_defaults):
        if default is None:
            required.add(kwarg.arg)
    return required


def _return_annotation(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    return ast.unparse(node.returns) if node.returns is not None else None


# --- Step 1 (fallback): non-Python regex --------------------------------------


def _extract_fallback_symbols(file_diff: FileDiff) -> list[ChangedSymbol]:
    removed: dict[str, str] = {}
    added: dict[str, str] = {}
    for raw_line in file_diff.body.splitlines():
        if not raw_line or raw_line.startswith(("+++", "---")):
            continue
        match = FALLBACK_SYMBOL_PATTERN.match(raw_line)
        if not match:
            continue
        name = match.group(1)
        target = removed if raw_line[0] == "-" else added
        target.setdefault(name, raw_line[1:].strip())

    return [
        ChangedSymbol(
            name=name,
            file=file_diff.path,
            language="other",
            old_signature=removed.get(name, ""),
            new_signature=added.get(name, ""),
            change="function declaration changed (non-Python; verify against the call sites below)",
            kind_hint="signature_change",
        )
        for name in sorted(removed.keys() | added.keys())
    ]
