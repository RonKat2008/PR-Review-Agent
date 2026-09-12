# tests/test_eval_corpus.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prime_pr_review.evaluation.corpus import (
    ReferenceComment,
    Row,
    fetch_rows,
    parse_rows,
    select,
)

FIXTURE = Path(__file__).parent / "fixtures" / "swecare_rows.json"

@pytest.fixture
def rows() -> tuple[Row, ...]:
    return parse_rows(json.loads(FIXTURE.read_text(encoding="utf-8")))

def test_parse_rows_maps_every_field(rows):
    row = rows[0]
    assert row.instance_id and "/" in row.repo and row.pull_number > 0
    assert row.head_sha and row.patch.startswith("diff --git")
    assert isinstance(row.reference_comments, tuple)

def test_reference_comment_line_falls_back_to_original_line():
    payload = {"rows": [{"row": _row(comments=[{"path": "a.py", "line": None, "original_line": 7,
                                                 "start_line": None, "original_start_line": 5, "text": "x",
                                                 "diff_hunk": ""}])}]}
    (row,) = parse_rows(payload)
    assert row.reference_comments == (ReferenceComment(path="a.py", line=7, start_line=5, text="x"),)

def test_select_is_deterministic_and_filters(rows):
    a = select(rows, count=3, seed=0)
    b = select(rows, count=3, seed=0)
    assert a == b and len(a) == 3
    assert select(rows, count=3, seed=0, max_patch_bytes=10) == ()

def test_select_requires_a_comment_with_a_path():
    payload = {"rows": [{"row": _row(comments=[{"path": "", "line": 1, "original_line": 1,
                                                 "start_line": None, "original_start_line": None,
                                                 "text": "", "diff_hunk": ""}])}]}
    assert select(parse_rows(payload), count=1, seed=0) == ()

def test_fetch_rows_pages_then_caches(tmp_path):
    calls: list[tuple[int, int]] = []
    page = json.loads(FIXTURE.read_text(encoding="utf-8"))["rows"]
    def fetch_page(offset: int, length: int) -> dict:
        calls.append((offset, length))
        return {"rows": page if offset == 0 else [], "num_rows_total": len(page)}
    cache = tmp_path / "c.json"
    first = fetch_rows(cache, fetch_page)
    second = fetch_rows(cache, fetch_page)
    assert len(first) == len(page) and first == second
    assert calls == [(0, 100)]

def _row(*, comments):
    return {"instance_id": "o__r-1@abc", "repo": "o/r", "language": "Python", "pull_number": 1,
            "title": "t", "body": "b", "base_commit": "base",
            "commit_to_review": {"head_commit": "abc", "head_commit_message": "m",
                                 "patch_to_review": "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n"},
            "reference_review_comments": comments}
