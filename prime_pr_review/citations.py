"""Citation validation: drop findings that cite a file or line that does not exist.

Go-live gate is "zero fabricated findings" — a finding citing a `file`/`line` the
model invented. Nothing upstream of this checks that a citation is real, so this
is the only place that does. Every drop is annotated with an auditable note;
nothing is silently discarded. This pass validates only `verdict.introduces` —
scope, blast-radius, and file-change citations are produced later in the
pipeline and are not covered here.

Three tiers of trust, cheapest first:
  1. The file must appear in the diff at all — after normalizing common prefix
     mismatches (a leading `./`, a leading `a/`/`b/`, or an absolute path
     resolved against `repo_root`) — or the finding is fabricated outright.
  2. A line inside a changed hunk (added or context) is directly verifiable
     against the diff itself — no repo checkout or API call needed.
  3. A line outside any hunk cannot be verified from the diff alone. An exact
     line count of the file *at the PR head*, fetched on demand only for files
     that need it (see `paths_needing_head_counts` / `head_line_counts`), is
     enough to catch a line number that could never exist. Without that count,
     such a finding is kept but marked unverified rather than dropped: recall
     is preferred when there is no way to disprove the citation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path

from .context import fetch_file_content
from .diffs import split_by_file
from .github import GhRunner
from .review import Finding
from .reviews_api import commentable_lines

CLAIM_TRUNCATE_LIMIT = 60


def validate_citations(
    findings: tuple[Finding, ...],
    diff: str,
    head_line_counts: Mapping[str, int] | None = None,
    repo_root: str | Path | None = None,
) -> tuple[tuple[Finding, ...], tuple[str, ...]]:
    """Drop fabricated citations; return the survivors (order preserved) and notes.

    `head_line_counts` maps a diff path to its exact line count at the PR head
    (see `head_line_counts()` below) — consulted only for a finding whose line
    falls outside every changed hunk. `repo_root`, when given, is used solely
    to resolve an absolute `finding.file` back to a diff-relative path.
    """
    if not findings:
        return (), ()

    diff_files = split_by_file(diff)
    diff_paths = {f.path for f in diff_files}
    deleted_paths = {f.path for f in diff_files if _is_deleted_file(f.body)}
    commentable = commentable_lines(diff)
    counts = head_line_counts or {}

    kept: list[Finding] = []
    notes: list[str] = []
    dropped = 0
    unverified = 0

    for finding in findings:
        canonical = _normalize_path(finding.file, diff_paths, repo_root)
        if canonical is None:
            notes.append(_not_in_diff_note(finding))
            dropped += 1
            continue
        if canonical != finding.file:
            finding = replace(finding, file=canonical)

        if finding.line is None:
            finding, line_end_note = _clear_stale_line_end(finding)
            if line_end_note:
                notes.append(line_end_note)
            kept.append(finding)
            continue

        outcome, bound = _classify_line(finding, commentable, counts, deleted_paths)
        if outcome == _DROPPED_NON_POSITIVE:
            notes.append(_non_positive_note(finding))
            dropped += 1
            continue
        if outcome == _DROPPED_DELETED:
            notes.append(_deleted_file_note(finding))
            dropped += 1
            continue
        if outcome == _DROPPED_BEYOND:
            notes.append(_beyond_bound_note(finding, bound))
            dropped += 1
            continue
        if outcome == _UNVERIFIED:
            unverified += 1

        finding, line_end_note = _clear_stale_line_end(finding)
        kept.append(finding)
        if line_end_note:
            notes.append(line_end_note)

    notes.append(
        f"citations: {len(findings)} checked, {dropped} dropped, {unverified} unverified"
    )

    return tuple(kept), tuple(notes)


def paths_needing_head_counts(
    findings: Iterable[Finding],
    diff: str,
    repo_root: str | Path | None = None,
) -> frozenset[str]:
    """Canonical diff paths that need an exact head-SHA line count to validate
    at least one finding on them — i.e. a finding whose line is outside every
    changed hunk. Empty on the common path (every citation verifiable from the
    diff alone), so the caller makes zero API calls."""
    diff_files = split_by_file(diff)
    diff_paths = {f.path for f in diff_files}
    commentable = commentable_lines(diff)

    needed: set[str] = set()
    for finding in findings:
        canonical = _normalize_path(finding.file, diff_paths, repo_root)
        if canonical is None or finding.line is None or finding.line <= 0:
            continue
        if (canonical, finding.line) not in commentable:
            needed.add(canonical)
    return frozenset(needed)


def head_line_counts(
    repo_slug: str, head_sha: str, paths: Iterable[str], runner: GhRunner
) -> dict[str, int]:
    """Exact line counts of `paths`, fetched at the PR head via the GitHub
    contents API. A path that cannot be fetched (deleted, renamed, or a
    transient API failure) is simply omitted — this never raises."""
    counts: dict[str, int] = {}
    for path in paths:
        try:
            text = fetch_file_content(repo_slug, head_sha, path, runner)
        except Exception:  # noqa: BLE001, S112 - one file's failure must never raise
            continue
        if text is None:
            continue
        counts[path] = text.count("\n") + (0 if text.endswith("\n") or not text else 1)
    return counts


def _normalize_path(
    file: str, diff_paths: set[str], repo_root: str | Path | None
) -> str | None:
    """Canonicalize `file` against `diff_paths`. Returns the canonical diff
    path, or `None` when nothing lines up — a fabricated file citation."""
    if file in diff_paths:
        return file

    candidates: list[str] = []
    if file.startswith("./"):
        candidates.append(file[2:])
    for prefix in ("a/", "b/"):
        if file.startswith(prefix):
            candidates.append(file[len(prefix) :])
    if repo_root and Path(file).is_absolute():
        try:
            rel = Path(file).resolve().relative_to(Path(repo_root).resolve())
        except (OSError, ValueError):
            rel = None
        if rel is not None:
            candidates.append(rel.as_posix())

    for candidate in candidates:
        if candidate in diff_paths:
            return candidate
    return None


# Sentinel outcomes for `_classify_line`.
_VERIFIED = "verified"
_UNVERIFIED = "unverified"
_DROPPED_NON_POSITIVE = "dropped_non_positive"
_DROPPED_DELETED = "dropped_deleted"
_DROPPED_BEYOND = "dropped_beyond"


def _classify_line(
    finding: Finding,
    commentable: frozenset[tuple[str, int]],
    counts: Mapping[str, int],
    deleted_paths: set[str],
) -> tuple[str, int | None]:
    """Classify `finding`'s line. Returns `(outcome, bound)`, where `bound` is
    the head-SHA line count when `outcome` is `_DROPPED_BEYOND`, else `None`."""
    if finding.line <= 0:
        return _DROPPED_NON_POSITIVE, None
    if finding.file in deleted_paths:
        return _DROPPED_DELETED, None
    if (finding.file, finding.line) in commentable:
        return _VERIFIED, None
    bound = counts.get(finding.file)
    if bound is not None and finding.line > bound:
        return _DROPPED_BEYOND, bound
    return _UNVERIFIED, None


def _is_deleted_file(body: str) -> bool:
    """Whether this file's diff body marks it deleted (`+++ /dev/null`).

    Checked only in the header section before the first hunk begins, never by
    scanning hunk content — a `+++ /dev/null`-looking line could otherwise be
    real file content on an added or context line.
    """
    for line in body.splitlines():
        if line.startswith("@@"):
            break
        if line.rstrip("\n") == "+++ /dev/null":
            return True
    return False


def _clear_stale_line_end(finding: Finding) -> tuple[Finding, str | None]:
    if finding.line_end is None:
        return finding, None
    if finding.line is None:
        note = f"citations: cleared line_end on {finding.file} — no line set"
        return replace(finding, line_end=None), note
    if finding.line_end >= finding.line:
        return finding, None
    note = f"citations: cleared line_end on {finding.file}:{finding.line} — precedes line"
    return replace(finding, line_end=None), note


def _not_in_diff_note(finding: Finding) -> str:
    location = f"{finding.file}:{finding.line if finding.line is not None else '?'}"
    claim = _truncate(finding.claim)
    return f'citations: dropped {location} — file not in diff ("{claim}")'


def _non_positive_note(finding: Finding) -> str:
    return f"citations: dropped {finding.file}:{finding.line} — line number must be positive"


def _deleted_file_note(finding: Finding) -> str:
    return f"citations: dropped {finding.file}:{finding.line} — file deleted by this PR"


def _beyond_bound_note(finding: Finding, bound: int | None) -> str:
    return (
        f"citations: dropped {finding.file}:{finding.line} "
        f"— line {finding.line} beyond file length ({bound} lines at head)"
    )


def _truncate(text: str, limit: int = CLAIM_TRUNCATE_LIMIT) -> str:
    stripped = text.strip()
    return stripped if len(stripped) <= limit else stripped[:limit].rstrip() + "..."
