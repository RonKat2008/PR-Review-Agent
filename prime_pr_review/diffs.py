"""Unified-diff parsing, noise filtering, and size guarding.

Lockfiles, snapshots, and build output are stripped before review: they are pure
token cost and produce nothing but false positives.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Sequence
from dataclasses import dataclass

FILE_HEADER_PREFIX = "diff --git "
TRUNCATION_NOTICE = "\n\n[diff truncated: exceeded max_diff_bytes]\n"


@dataclass(frozen=True)
class FileDiff:
    path: str
    body: str

    @property
    def size(self) -> int:
        return len(self.body)


@dataclass(frozen=True)
class FilteredDiff:
    text: str
    included: tuple[str, ...]
    ignored: tuple[str, ...]
    truncated: bool

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


def split_by_file(diff: str) -> tuple[FileDiff, ...]:
    """Split a unified diff into per-file chunks."""
    if not diff.strip():
        return ()

    chunks: list[FileDiff] = []
    current: list[str] = []
    current_path = ""

    for line in diff.splitlines(keepends=True):
        if line.startswith(FILE_HEADER_PREFIX):
            if current:
                chunks.append(FileDiff(path=current_path, body="".join(current)))
            current = [line]
            current_path = _path_from_header(line)
        elif current:
            current.append(line)

    if current:
        chunks.append(FileDiff(path=current_path, body="".join(current)))

    return tuple(chunks)


def filter_diff(
    diff: str,
    ignore_paths: Sequence[str] = (),
    max_bytes: int = 200_000,
) -> FilteredDiff:
    """Drop ignored paths, then truncate on a file boundary if still oversized."""
    files = split_by_file(diff)
    kept: list[FileDiff] = []
    ignored: list[str] = []

    for file_diff in files:
        if _is_ignored(file_diff.path, ignore_paths):
            ignored.append(file_diff.path)
        else:
            kept.append(file_diff)

    selected, truncated = _fit_within(kept, max_bytes)
    text = "".join(f.body for f in selected)
    if truncated:
        text += TRUNCATION_NOTICE

    return FilteredDiff(
        text=text,
        included=tuple(f.path for f in selected),
        ignored=tuple(ignored),
        truncated=truncated,
    )


def _fit_within(files: Sequence[FileDiff], max_bytes: int) -> tuple[tuple[FileDiff, ...], bool]:
    """Take files in order until the budget is spent. Never splits a file."""
    selected: list[FileDiff] = []
    used = 0

    for file_diff in files:
        if used + file_diff.size > max_bytes:
            return tuple(selected), True
        selected.append(file_diff)
        used += file_diff.size

    return tuple(selected), False


def _is_ignored(path: str, patterns: Sequence[str]) -> bool:
    for pattern in patterns:
        if fnmatch.fnmatch(path, pattern):
            return True
        # fnmatch treats "**/" as a single segment; also try the bare tail pattern
        # so "**/*.lock" matches a top-level "uv.lock".
        if pattern.startswith("**/") and fnmatch.fnmatch(path, pattern[3:]):
            return True
    return False


def _path_from_header(line: str) -> str:
    """Extract the b-side path from a `diff --git a/x b/y` header.

    Separators are normalized to forward slashes so paths are canonical everywhere
    downstream — ignore matching, review front matter, and the prompt itself.
    """
    remainder = line[len(FILE_HEADER_PREFIX):].strip().replace("\\", "/")
    marker = " b/"
    index = remainder.rfind(marker)
    if index == -1:
        return remainder
    return remainder[index + len(marker):]
