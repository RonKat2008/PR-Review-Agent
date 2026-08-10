"""Line-anchored PR review comments and committable suggestions (P4).

`commentable_lines` is the load-bearing filter here: GitHub rejects a review
comment anchored to a line outside the diff, and one rejected comment fails the
whole `POST .../reviews` call.
"""

from __future__ import annotations

import json

import pytest

from prime_pr_review.github import GitHubError
from prime_pr_review.review import BlastRadius, BrokenCaller, Finding, Severity, Verdict
from prime_pr_review.reviews_api import (
    EVENT_COMMENT,
    EVENT_REQUEST_CHANGES,
    ReviewComment,
    build_review_comments,
    commentable_lines,
    post_review,
    review_event_for,
)

from .conftest import FakeGh, SAMPLE_DIFF

# Two hunks in one file, with a gap (lines 4-9) that is not part of the diff —
# the case a naive "just count added lines" implementation gets wrong.
MULTI_HUNK_DIFF = """diff --git a/src/app.py b/src/app.py
index 1111111..2222222 100644
--- a/src/app.py
+++ b/src/app.py
@@ -1,3 +1,3 @@
-total = sum(values)
+total = sum(values[:-1])
 more_context
 trailing
@@ -10,2 +10,3 @@
 context_line
+new_line
 trailing_context
"""


def _finding(**overrides) -> Finding:
    defaults = dict(
        file="src/app.py",
        line=1,
        severity=Severity.HIGH,
        claim="Off-by-one",
        evidence="range(n) skips the final element",
    )
    defaults.update(overrides)
    return Finding(**defaults)


def _verdict(introduces=(), blast_radius=()) -> Verdict:
    return Verdict(introduces=introduces, fixes=(), confidence=0.9, blast_radius=blast_radius)


def _is_post_review(args):
    return args[0] == "api" and "reviews" in " ".join(args) and "--method" in args


# --------------------------------------------------------------------------
# commentable_lines — hunk-header parsing
# --------------------------------------------------------------------------


def test_added_and_context_lines_within_a_hunk_are_commentable():
    commentable = commentable_lines(MULTI_HUNK_DIFF)

    assert ("src/app.py", 1) in commentable  # added line
    assert ("src/app.py", 2) in commentable  # context line


def test_removed_only_lines_do_not_advance_the_right_side_counter():
    """The removed line must not create a phantom right-side entry."""
    commentable = commentable_lines(MULTI_HUNK_DIFF)

    app_lines = {line for path, line in commentable if path == "src/app.py"}
    assert app_lines == {1, 2, 3, 10, 11, 12}


def test_a_second_hunk_in_the_same_file_resumes_at_its_own_start_line():
    commentable = commentable_lines(MULTI_HUNK_DIFF)

    assert {10, 11, 12}.issubset({line for path, line in commentable if path == "src/app.py"})


def test_lines_outside_any_hunk_are_excluded():
    commentable = commentable_lines(MULTI_HUNK_DIFF)

    assert ("src/app.py", 5) not in commentable  # gap between hunk 1 and hunk 2
    assert ("src/app.py", 99) not in commentable  # far past any hunk


def test_multi_file_diffs_track_lines_per_file_independently():
    commentable = commentable_lines(SAMPLE_DIFF)

    assert ("src/app.py", 1) in commentable
    assert ("uv.lock", 1) in commentable
    assert ("uv.lock", 2) not in commentable  # only one line changed in that hunk


def test_a_bare_single_line_hunk_header_is_parsed_without_a_count():
    """`@@ -1 +1 @@` (no `,count`) must not be mistaken for malformed input."""
    commentable = commentable_lines(SAMPLE_DIFF)

    assert ("uv.lock", 1) in commentable


def test_blank_diff_yields_no_commentable_lines():
    assert commentable_lines("") == frozenset()


# --------------------------------------------------------------------------
# build_review_comments — anchoring
# --------------------------------------------------------------------------


def test_a_finding_on_a_commentable_line_is_anchored():
    commentable = commentable_lines(MULTI_HUNK_DIFF)
    finding = _finding(line=1)

    anchored, unanchored = build_review_comments([finding], commentable)

    assert len(anchored) == 1
    assert unanchored == ()
    assert anchored[0].path == "src/app.py"
    assert anchored[0].line == 1
    assert anchored[0].side == "RIGHT"


def test_a_finding_with_no_line_is_returned_unanchored_not_dropped():
    commentable = commentable_lines(MULTI_HUNK_DIFF)
    finding = _finding(line=None)

    anchored, unanchored = build_review_comments([finding], commentable)

    assert anchored == ()
    assert unanchored == (finding,)


def test_a_finding_on_a_line_outside_the_diff_is_returned_unanchored_not_dropped():
    commentable = commentable_lines(MULTI_HUNK_DIFF)
    finding = _finding(line=500)

    anchored, unanchored = build_review_comments([finding], commentable)

    assert anchored == ()
    assert unanchored == (finding,)


def test_mixed_findings_are_partitioned_correctly():
    commentable = commentable_lines(MULTI_HUNK_DIFF)
    good = _finding(line=1, claim="anchorable")
    bad = _finding(line=500, claim="not anchorable")

    anchored, unanchored = build_review_comments([good, bad], commentable)

    assert len(anchored) == 1
    assert unanchored == (bad,)


# --------------------------------------------------------------------------
# build_review_comments — body rendering
# --------------------------------------------------------------------------


def test_comment_body_includes_claim_evidence_and_corroboration():
    commentable = commentable_lines(MULTI_HUNK_DIFF)
    finding = _finding(line=1, corroboration="bandit:B608")

    anchored, _ = build_review_comments([finding], commentable)

    body = anchored[0].body
    assert "Off-by-one" in body
    assert "range(n) skips the final element" in body
    assert "bandit:B608" in body


def test_comment_body_renders_a_suggestion_fence_when_a_suggestion_is_present():
    commentable = commentable_lines(MULTI_HUNK_DIFF)
    finding = _finding(line=1, suggestion="total = sum(values)")

    anchored, _ = build_review_comments([finding], commentable)

    body = anchored[0].body
    assert "```suggestion" in body
    assert "total = sum(values)" in body
    assert body.count("```") == 2


def test_comment_body_omits_the_suggestion_fence_when_there_is_no_suggestion():
    commentable = commentable_lines(MULTI_HUNK_DIFF)
    finding = _finding(line=1, suggestion="")

    anchored, _ = build_review_comments([finding], commentable)

    assert "```suggestion" not in anchored[0].body


# --------------------------------------------------------------------------
# build_review_comments — multi-line suggestions
# --------------------------------------------------------------------------


def test_multi_line_finding_uses_start_line_and_line_for_the_range():
    commentable = commentable_lines(MULTI_HUNK_DIFF)
    finding = _finding(
        line=10, line_end=12, suggestion="context_line\nnew_line\ntrailing_context"
    )

    anchored, unanchored = build_review_comments([finding], commentable)

    assert unanchored == ()
    comment = anchored[0]
    assert comment.start_line == 10
    assert comment.line == 12
    assert comment.start_side == "RIGHT"
    assert comment.side == "RIGHT"


def test_line_end_outside_the_diff_degrades_to_a_single_line_comment():
    """GitHub rejects the ENTIRE review if any one comment is invalid.

    A model-supplied line_end past the end of the hunk would therefore cost every
    other finding in the sweep, so it degrades to a single-line anchor instead of
    being sent unverified.
    """
    commentable = commentable_lines(MULTI_HUNK_DIFF)
    finding = _finding(line=10, line_end=99)  # 99 is far past any hunk

    anchored, unanchored = build_review_comments([finding], commentable)

    assert unanchored == (), "the finding must be kept, not dropped"
    comment = anchored[0]
    assert comment.line == 10
    assert comment.start_line is None


def test_single_line_finding_leaves_start_line_unset():
    commentable = commentable_lines(MULTI_HUNK_DIFF)
    finding = _finding(line=1, line_end=None)

    anchored, _ = build_review_comments([finding], commentable)

    assert anchored[0].start_line is None
    assert anchored[0].start_side is None


def test_line_end_equal_to_line_is_treated_as_single_line():
    """`line_end == line` is not a range — the multi-line fields must stay unset."""
    commentable = commentable_lines(MULTI_HUNK_DIFF)
    finding = _finding(line=1, line_end=1)

    anchored, _ = build_review_comments([finding], commentable)

    assert anchored[0].start_line is None
    assert anchored[0].line == 1


# --------------------------------------------------------------------------
# post_review — payload shape
# --------------------------------------------------------------------------


def test_post_review_targets_the_reviews_endpoint_with_post():
    gh = FakeGh().on(_is_post_review, "")

    post_review("acme/widget", 7, "summary", (), EVENT_COMMENT, gh)

    args, _ = gh.calls[0]
    assert "repos/acme/widget/pulls/7/reviews" in " ".join(args)
    assert "--method" in args and "POST" in args
    assert "--input" in args and "-" in args


def test_post_review_payload_carries_body_event_and_comments():
    gh = FakeGh().on(_is_post_review, "")
    comment = ReviewComment(path="src/app.py", body="fix this", line=1)

    post_review("acme/widget", 7, "summary text", (comment,), EVENT_REQUEST_CHANGES, gh)

    _, stdin = gh.calls[0]
    payload = json.loads(stdin)
    assert payload["body"] == "summary text"
    assert payload["event"] == "REQUEST_CHANGES"
    assert payload["comments"] == [
        {"path": "src/app.py", "body": "fix this", "line": 1, "side": "RIGHT"}
    ]


def test_post_review_payload_includes_start_line_for_multi_line_comments():
    gh = FakeGh().on(_is_post_review, "")
    comment = ReviewComment(
        path="src/app.py", body="fix range", line=12, start_line=10, start_side="RIGHT"
    )

    post_review("acme/widget", 7, "summary", (comment,), EVENT_COMMENT, gh)

    _, stdin = gh.calls[0]
    payload = json.loads(stdin)
    assert payload["comments"][0]["start_line"] == 10
    assert payload["comments"][0]["start_side"] == "RIGHT"


def test_post_review_with_no_comments_sends_an_empty_list():
    gh = FakeGh().on(_is_post_review, "")

    post_review("acme/widget", 7, "summary", (), EVENT_COMMENT, gh)

    _, stdin = gh.calls[0]
    assert json.loads(stdin)["comments"] == []


def test_post_review_surfaces_gh_failures_as_github_error():
    def failing(args, stdin=None):
        raise GitHubError("boom")

    with pytest.raises(GitHubError, match="boom"):
        post_review("acme/widget", 7, "summary", (), EVENT_COMMENT, failing)


# --------------------------------------------------------------------------
# review_event_for
# --------------------------------------------------------------------------


def test_critical_finding_requests_changes_when_allowed():
    verdict = _verdict(introduces=(_finding(severity=Severity.CRITICAL),))

    assert review_event_for(verdict, allow_request_changes=True) == EVENT_REQUEST_CHANGES


def test_critical_finding_only_comments_when_not_allowed():
    verdict = _verdict(introduces=(_finding(severity=Severity.CRITICAL),))

    assert review_event_for(verdict, allow_request_changes=False) == EVENT_COMMENT


def test_broken_caller_requests_changes_when_allowed():
    blast = (
        BlastRadius(
            symbol="total_price",
            kind="signature_change",
            change="added required parameter",
            breaks=(BrokenCaller(file="a.py", line=5, severity=Severity.HIGH, claim="breaks"),),
        ),
    )
    verdict = _verdict(blast_radius=blast)

    assert review_event_for(verdict, allow_request_changes=True) == EVENT_REQUEST_CHANGES


def test_broken_caller_only_comments_when_not_allowed():
    blast = (
        BlastRadius(
            symbol="total_price",
            kind="signature_change",
            change="added required parameter",
            breaks=(BrokenCaller(file="a.py", line=5, severity=Severity.HIGH, claim="breaks"),),
        ),
    )
    verdict = _verdict(blast_radius=blast)

    assert review_event_for(verdict, allow_request_changes=False) == EVENT_COMMENT


def test_high_severity_alone_does_not_request_changes():
    """Only CRITICAL and broken callers are strong enough to justify blocking a merge."""
    verdict = _verdict(introduces=(_finding(severity=Severity.HIGH),))

    assert review_event_for(verdict, allow_request_changes=True) == EVENT_COMMENT


def test_empty_verdict_always_comments():
    assert review_event_for(_verdict(), allow_request_changes=True) == EVENT_COMMENT


# --------------------------------------------------------------------------
# immutability
# --------------------------------------------------------------------------


def test_review_comment_is_frozen():
    comment = ReviewComment(path="a.py", body="b", line=1)

    with pytest.raises(AttributeError):
        comment.line = 2  # type: ignore[misc]
