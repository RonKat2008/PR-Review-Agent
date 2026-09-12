"""A `gh` runner backed by one SWE-CARE row, so `sweep_lane` reviews the exact
mid-review diff humans commented on, with no GitHub listing or diff calls.

Only head-file content fetches (citation validation) reach the real `gh`; each
response is persisted so the offline scorer replays it without network."""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

from ..diffs import split_by_file
from ..github import GhRunner, GitHubError
from .corpus import Row

CONTENTS_RE = re.compile(r"^repos/[^/]+/[^/]+/contents/(?P<path>.+?)\?ref=")
# `gh api` turns any of these into a non-GET request. The eval runner is
# read-only by construction, so a write reaching it is a wiring bug, not a
# call to answer -- and the endpoints it *does* answer (issue comments) are
# exactly the ones a stray write would post to.
WRITE_FLAGS = frozenset({"-X", "--method", "-f", "-F"})
EVAL_AUTHOR = "swe-care"
EVAL_BASE_REF = "main"


class HeadFileStore:
    """`{path: raw gh response | None}` persisted as JSON after every put."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._data: dict[str, str | None] = (
            json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        )

    def get(self, path: str) -> str | None:
        return self._data.get(path)

    def paths(self) -> tuple[str, ...]:
        return tuple(self._data)

    def put(self, path: str, text: str | None) -> None:
        self._data = {**self._data, path: text}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data), encoding="utf-8")


def pr_list_json(row: Row) -> str:
    added = deleted = 0
    files = split_by_file(row.patch)
    for f in files:
        for line in f.body.splitlines():
            added += line.startswith("+") and not line.startswith("+++")
            deleted += line.startswith("-") and not line.startswith("---")
    return json.dumps([{
        "number": row.pull_number, "title": row.title, "author": {"login": EVAL_AUTHOR},
        "headRefOid": row.head_sha, "baseRefName": EVAL_BASE_REF,
        "url": f"https://github.com/{row.repo}/pull/{row.pull_number}",
        "additions": added, "deletions": deleted, "changedFiles": len(files), "mergedAt": None,
    }])


def corpus_runner(row: Row, fallback: GhRunner, head_files: HeadFileStore) -> GhRunner:
    listing = pr_list_json(row)

    def runner(args: Sequence[str], stdin: str | None = None) -> str:
        if _is_write(args):
            raise GitHubError(f"eval runner: refusing write gh call: {list(args)!r}")
        head = tuple(args[:2])
        if head == ("pr", "list"):
            return listing
        if head == ("pr", "diff"):
            return row.patch
        if head == ("pr", "checks") or (len(args) > 1 and args[0] == "api" and args[1].endswith("/comments")):
            return "[]"
        path = _contents_path(args)
        if path is not None:
            return _fetch_and_record(args, stdin, path, fallback, head_files)
        raise GitHubError(f"eval runner: unexpected gh call: {list(args)!r}")

    return runner


def replay_runner(store: HeadFileStore) -> GhRunner:
    def runner(args: Sequence[str], stdin: str | None = None) -> str:
        path = _contents_path(args)
        text = store.get(path) if path is not None else None
        if text is None:
            raise GitHubError(f"eval replay: no recorded head file for {list(args)!r}")
        return text
    return runner


def _is_write(args: Sequence[str]) -> bool:
    return bool(args) and args[0] == "api" and any(a in WRITE_FLAGS for a in args)


def _contents_path(args: Sequence[str]) -> str | None:
    if len(args) < 2 or args[0] != "api":
        return None
    match = CONTENTS_RE.match(args[1])
    return match.group("path") if match else None


def _fetch_and_record(args, stdin, path, fallback: GhRunner, store: HeadFileStore) -> str:
    try:
        text = fallback(args, stdin)
    except Exception:
        store.put(path, None)
        raise
    store.put(path, text)
    return text
