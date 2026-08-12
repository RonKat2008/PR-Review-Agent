"""Repository context beyond the diff.

The reviewer normally sees only changed lines, so it cannot tell whether a renamed
parameter breaks a caller three files away, whether a "new" helper already exists
under another name, or whether the PR just violated a convention written down in
CONTRIBUTING.md. This module gathers that missing context for one PR: the full text
of changed files at the head SHA, call sites elsewhere in the repo, sibling tests,
and any convention documents present.

Every external call goes through an injected runner — `gh` for GitHub content, `git`
for local search — so this module is testable without a network, a token, or a git
repository. Sibling tests and convention files are read straight off disk under a
supplied repo root, which is exactly as testable with a `tmp_path` fixture.

Missing context is normal, not an error: a repo with no CONTRIBUTING.md, a file that
404s at that SHA, or a symbol nobody else calls all produce an empty section, never
an exception.
"""

from __future__ import annotations

import base64
import binascii
import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .diffs import split_by_file
from .github import GhRunner, GitHubError
from .github import default_runner as default_gh_runner

GitRunner = Callable[[Sequence[str]], str]

DEFAULT_MAX_BYTES = 40_000

FULL_FILES_SHARE = 0.40
CALL_SITES_SHARE = 0.25
SIBLING_TESTS_SHARE = 0.20
CONVENTIONS_SHARE = 0.15

GIT_TIMEOUT_SECONDS = 30
GIT_NO_MATCHES_EXIT_CODE = 1

SYMBOL_PATTERN = re.compile(r"^\s*(?:def|class)\s+(\w+)")

CONVENTION_FILENAMES = ("CONTRIBUTING.md", "CLAUDE.md", "AGENTS.md")
LINT_CONFIG_FILENAMES = (
    "ruff.toml",
    ".ruff.toml",
    ".eslintrc.json",
    ".eslintrc.yml",
    ".eslintrc.yaml",
    ".eslintrc.js",
    ".eslintrc.cjs",
    "eslint.config.js",
    "eslint.config.mjs",
    "eslint.config.cjs",
)
PYPROJECT_RUFF_MARKER = "[tool.ruff]"

SIBLING_TEST_PATTERNS = ("**/test_*{stem}*", "**/{stem}_test.*", "tests/**/*{stem}*")


class GitError(RuntimeError):
    """A local `git` invocation failed unexpectedly (not just "no matches")."""


@dataclass(frozen=True)
class ChangedFile:
    """Full content of one changed file, read at the PR head SHA."""

    path: str
    content: str

    @property
    def size(self) -> int:
        return len(self.content)


@dataclass(frozen=True)
class CallSite:
    """One place elsewhere in the repo that references a changed symbol."""

    symbol: str
    file: str
    line: int
    text: str

    @property
    def size(self) -> int:
        return len(self.text) + len(self.file) + len(self.symbol)


@dataclass(frozen=True)
class SiblingTest:
    """A test file discovered near a changed file by naming convention."""

    path: str
    content: str

    @property
    def size(self) -> int:
        return len(self.content)


@dataclass(frozen=True)
class Convention:
    """One convention or lint-config file's content."""

    path: str
    content: str

    @property
    def size(self) -> int:
        return len(self.content)


@dataclass(frozen=True)
class ReviewContext:
    changed_files: tuple[ChangedFile, ...] = ()
    call_sites: tuple[CallSite, ...] = ()
    sibling_tests: tuple[SiblingTest, ...] = ()
    conventions: tuple[Convention, ...] = ()
    # Human-readable notes on what the byte budget forced out, one per section.
    dropped: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not (
            self.changed_files or self.call_sites or self.sibling_tests or self.conventions
        )

    def render(self) -> str:
        """Render as the markdown block appended to the review prompt."""
        parts = [
            "## Repository context",
            "",
            _render_blob_section("Full changed files", self.changed_files),
            _render_call_sites(self.call_sites),
            _render_blob_section("Sibling tests", self.sibling_tests),
            _render_blob_section("Conventions", self.conventions),
            _render_dropped(self.dropped),
        ]
        return "\n".join(p for p in parts if p is not None).rstrip() + "\n"


def default_git_runner(args: Sequence[str]) -> str:
    """Run `git` in the current working directory and return stdout.

    `git grep` exits 1 when it simply finds no matches, which is not a failure.
    """
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            # Locale-independent: git emits UTF-8; cp1252 would crash on real diffs.
            encoding="utf-8",
            errors="replace",
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise GitError(f"`git {' '.join(args)}` failed: {exc}") from exc

    if result.returncode not in (0, GIT_NO_MATCHES_EXIT_CODE):
        raise GitError(
            f"`git {' '.join(args)}` failed with exit code {result.returncode}: "
            f"{result.stderr.strip()}"
        )
    return result.stdout


def gather_context(
    repo_slug: str,
    head_sha: str,
    diff: str,
    repo_root: Path | str,
    gh_runner: GhRunner = default_gh_runner,
    git_runner: GitRunner = default_git_runner,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> ReviewContext:
    """Collect everything outside the diff that the review needs for one PR."""
    root = Path(repo_root)
    diff_paths = tuple(f.path for f in split_by_file(diff))

    changed_files, dropped_files = _gather_changed_files(
        repo_slug, head_sha, diff_paths, gh_runner, _budget(max_bytes, FULL_FILES_SHARE)
    )
    call_sites, dropped_calls = _gather_call_sites(
        diff, diff_paths, git_runner, _budget(max_bytes, CALL_SITES_SHARE)
    )
    sibling_tests, dropped_siblings = _gather_sibling_tests(
        root, diff_paths, _budget(max_bytes, SIBLING_TESTS_SHARE)
    )
    conventions, dropped_conventions = _gather_conventions(
        root, _budget(max_bytes, CONVENTIONS_SHARE)
    )

    dropped = tuple(
        note
        for note in (dropped_files, dropped_calls, dropped_siblings, dropped_conventions)
        if note
    )

    return ReviewContext(
        changed_files=changed_files,
        call_sites=call_sites,
        sibling_tests=sibling_tests,
        conventions=conventions,
        dropped=dropped,
    )


def _budget(max_bytes: int, share: float) -> int:
    return int(max_bytes * share)


def _fit_within(items: Sequence, max_bytes: int) -> tuple[tuple, bool]:
    """Take items in order until the byte budget is spent. Never splits an item.

    Mirrors `diffs._fit_within`, generalized to any item exposing `.size`.
    """
    selected: list = []
    used = 0
    for item in items:
        if used + item.size > max_bytes:
            return tuple(selected), True
        selected.append(item)
        used += item.size
    return tuple(selected), False


# --- Full changed files -----------------------------------------------------


def _gather_changed_files(
    repo_slug: str,
    head_sha: str,
    paths: Sequence[str],
    runner: GhRunner,
    budget: int,
) -> tuple[tuple[ChangedFile, ...], str]:
    fetched = [
        ChangedFile(path=path, content=content)
        for path in paths
        for content in (_fetch_file_content(repo_slug, head_sha, path, runner),)
        if content is not None
    ]
    selected, truncated = _fit_within(fetched, budget)
    return selected, _drop_note("changed files", fetched, selected, truncated)


def _fetch_file_content(
    repo_slug: str, head_sha: str, path: str, runner: GhRunner
) -> str | None:
    """Full file content at the PR head SHA. `None` means unavailable — that is normal."""
    try:
        raw = runner(
            ["api", f"repos/{repo_slug}/contents/{path}?ref={head_sha}", "--jq", ".content"],
            None,
        )
    except GitHubError:
        return None

    encoded = raw.strip()
    if not encoded:
        return None
    try:
        return base64.b64decode(encoded).decode("utf-8", errors="replace")
    except (binascii.Error, ValueError):
        return None


# --- Call sites --------------------------------------------------------------


def _gather_call_sites(
    diff: str,
    diff_paths: Sequence[str],
    runner: GitRunner,
    budget: int,
) -> tuple[tuple[CallSite, ...], str]:
    in_diff = frozenset(diff_paths)
    found = [
        site
        for symbol in _extract_symbols(diff)
        for site in _grep_symbol(symbol, runner)
        if site.file not in in_diff
    ]
    selected, truncated = _fit_within(found, budget)
    return selected, _drop_note("call sites", found, selected, truncated)


def _extract_symbols(diff: str) -> tuple[str, ...]:
    """Names introduced by `def`/`class` on added or removed diff lines, in order."""
    symbols: list[str] = []
    seen: set[str] = set()
    for line in diff.splitlines():
        if not line or line[0] not in "+-" or line.startswith(("+++", "---")):
            continue
        match = SYMBOL_PATTERN.match(line[1:])
        if match and match.group(1) not in seen:
            seen.add(match.group(1))
            symbols.append(match.group(1))
    return tuple(symbols)


def _grep_symbol(symbol: str, runner: GitRunner) -> tuple[CallSite, ...]:
    try:
        raw = runner(["grep", "-n", "--", symbol])
    except GitError:
        return ()
    sites = (_parse_grep_line(symbol, line) for line in raw.splitlines() if line.strip())
    return tuple(site for site in sites if site is not None)


def _parse_grep_line(symbol: str, line: str) -> CallSite | None:
    parts = line.split(":", 2)
    if len(parts) != 3 or not parts[1].isdigit():
        return None
    path, line_no, text = parts
    return CallSite(symbol=symbol, file=path.replace("\\", "/"), line=int(line_no), text=text.strip())


# --- Sibling tests -------------------------------------------------------------


def _gather_sibling_tests(
    repo_root: Path,
    diff_paths: Sequence[str],
    budget: int,
) -> tuple[tuple[SiblingTest, ...], str]:
    excluded = frozenset(diff_paths)
    discovered: list[SiblingTest] = []
    seen: set[str] = set()

    for diff_path in diff_paths:
        stem = Path(diff_path).stem
        if not stem:
            continue
        for candidate in _sibling_test_candidates(repo_root, stem):
            rel = candidate.relative_to(repo_root).as_posix()
            if rel in excluded or rel in seen:
                continue
            content = _read_text_if_exists(candidate)
            if content is None:
                continue
            seen.add(rel)
            discovered.append(SiblingTest(path=rel, content=content))

    selected, truncated = _fit_within(discovered, budget)
    return selected, _drop_note("sibling tests", discovered, selected, truncated)


def _sibling_test_candidates(repo_root: Path, stem: str) -> tuple[Path, ...]:
    found: list[Path] = []
    seen: set[Path] = set()
    for template in SIBLING_TEST_PATTERNS:
        pattern = template.format(stem=stem)
        try:
            matches = sorted(repo_root.glob(pattern))
        except (OSError, ValueError):
            continue
        for path in matches:
            if path.is_file() and path not in seen:
                seen.add(path)
                found.append(path)
    return tuple(found)


# --- Conventions -----------------------------------------------------------


def _gather_conventions(repo_root: Path, budget: int) -> tuple[tuple[Convention, ...], str]:
    found: list[Convention] = []
    for name in (*CONVENTION_FILENAMES, *LINT_CONFIG_FILENAMES):
        content = _read_text_if_exists(repo_root / name)
        if content is not None:
            found.append(Convention(path=name, content=content))

    pyproject = _read_text_if_exists(repo_root / "pyproject.toml")
    if pyproject is not None and PYPROJECT_RUFF_MARKER in pyproject:
        found.append(Convention(path="pyproject.toml", content=pyproject))

    selected, truncated = _fit_within(found, budget)
    return selected, _drop_note("conventions", found, selected, truncated)


def _read_text_if_exists(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


# --- Shared helpers ----------------------------------------------------------


def _drop_note(label: str, found: Sequence, selected: Sequence, truncated: bool) -> str:
    if not truncated:
        return ""
    omitted = len(found) - len(selected)
    return f"{label}: {omitted} of {len(found)} omitted to fit the context budget"


def _render_blob_section(title: str, items: Sequence) -> str:
    lines = [f"### {title}", ""]
    if not items:
        lines.append("_None found._")
    else:
        for item in items:
            lines.append(f"#### `{item.path}`")
            lines.append("```")
            lines.append(item.content)
            lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _render_call_sites(sites: tuple[CallSite, ...]) -> str:
    lines = ["### Call sites outside the diff", ""]
    if not sites:
        lines.append("_None found._")
    else:
        for site in sites:
            lines.append(f"- `{site.file}:{site.line}` (`{site.symbol}`) — {site.text}")
    lines.append("")
    return "\n".join(lines)


def _render_dropped(dropped: tuple[str, ...]) -> str | None:
    if not dropped:
        return None
    lines = ["### Context omitted", ""]
    lines.extend(f"- {note}" for note in dropped)
    lines.append("")
    return "\n".join(lines)
