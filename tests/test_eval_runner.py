from __future__ import annotations

import json

import pytest

from prime_pr_review.evaluation.corpus import ReferenceComment, Row
from prime_pr_review.evaluation.runner import (
    HeadFileStore,
    corpus_runner,
    pr_list_json,
    replay_runner,
)
from prime_pr_review.github import GitHubError, _parse_pr_list


def _row() -> Row:
    return Row(instance_id="o__r-7@abc", repo="o/r", language="Python", pull_number=7, title="T", body="B",
               base_commit="base", head_sha="abc123", head_commit_message="msg",
               patch="diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,2 +1,2 @@\n-x\n+y\n+z\n",
               reference_comments=(ReferenceComment("a.py", 1, None, "t"),))


def test_pr_list_json_parses_through_production_parser():
    (pr,) = _parse_pr_list(pr_list_json(_row()))
    assert (pr.number, pr.head_sha, pr.author, pr.base_ref) == (7, "abc123", "swe-care", "main")
    assert (pr.additions, pr.deletions, pr.changed_files) == (2, 1, 1)


def test_runner_answers_list_diff_checks_comments(tmp_path):
    run = corpus_runner(_row(), fallback=lambda a, s: "NEVER", head_files=HeadFileStore(tmp_path / "h.json"))
    assert json.loads(run(["pr", "list", "--repo", "o/r"], None))[0]["number"] == 7
    assert run(["pr", "diff", "7", "--repo", "o/r"], None) == _row().patch
    assert run(["pr", "checks", "7", "--repo", "o/r", "--json", "x"], None) == "[]"
    assert run(["api", "repos/o/r/issues/7/comments"], None) == "[]"


def test_runner_delegates_content_fetch_and_records(tmp_path):
    store = HeadFileStore(tmp_path / "h.json")
    run = corpus_runner(_row(), fallback=lambda a, s: "BASE64", head_files=store)
    assert run(["api", "repos/o/r/contents/a.py?ref=abc123", "--jq", ".content"], None) == "BASE64"
    assert HeadFileStore(tmp_path / "h.json").get("a.py") == "BASE64"


def test_runner_records_failed_fetch_as_none(tmp_path):
    store = HeadFileStore(tmp_path / "h.json")
    def boom(a, s): raise GitHubError("404")
    run = corpus_runner(_row(), fallback=boom, head_files=store)
    with pytest.raises(GitHubError):
        run(["api", "repos/o/r/contents/gone.py?ref=abc123", "--jq", ".content"], None)
    assert store.get("gone.py") is None and "gone.py" in store.paths()


def test_runner_rejects_unexpected_calls(tmp_path):
    run = corpus_runner(_row(), fallback=lambda a, s: "", head_files=HeadFileStore(tmp_path / "h.json"))
    with pytest.raises(GitHubError, match="unexpected gh call"):
        run(["pr", "comment", "7"], "body")


def test_runner_rejects_write_api_calls(tmp_path):
    """A write attempt must never reach the real `gh`, even on an endpoint the
    read path would otherwise answer (`.../comments`)."""
    run = corpus_runner(_row(), fallback=lambda a, s: "NEVER", head_files=HeadFileStore(tmp_path / "h.json"))
    for args in (["api", "repos/o/r/issues/7/comments", "-X", "POST"],
                 ["api", "repos/o/r/issues/7/comments", "--method", "POST"],
                 ["api", "repos/o/r/issues/7/comments", "-f", "body=hi"],
                 ["api", "repos/o/r/contents/a.py?ref=abc123", "-F", "body=@f"]):
        with pytest.raises(GitHubError, match="write"):
            run(args, None)


def test_replay_runner_serves_store(tmp_path):
    store = HeadFileStore(tmp_path / "h.json"); store.put("a.py", "B64"); store.put("gone.py", None)
    run = replay_runner(store)
    assert run(["api", "repos/o/r/contents/a.py?ref=abc123", "--jq", ".content"], None) == "B64"
    with pytest.raises(GitHubError):
        run(["api", "repos/o/r/contents/gone.py?ref=abc123", "--jq", ".content"], None)
