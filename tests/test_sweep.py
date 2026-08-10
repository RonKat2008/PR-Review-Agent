"""End-to-end sweep behavior against a fake gh and a stub reviewer."""

from __future__ import annotations

from datetime import datetime, timezone

from prime_pr_review.github import GitHubError
from prime_pr_review.state import (
    LANE_MERGED,
    LANE_OPEN,
    State,
    is_reviewed,
    mark_reviewed,
)
from prime_pr_review.sweep import sweep_lane

from .conftest import (
    FakeGh,
    SAMPLE_DIFF,
    VERDICT_EMPTY,
    VERDICT_LOW_CONFIDENCE,
    VERDICT_WITH_BUG,
    is_list_comments,
    is_pr_comment,
    is_pr_diff,
    is_pr_list,
    make_config,
    make_pr,
    pr_list_json,
)

NOW = datetime(2026, 8, 7, 12, 0, tzinfo=timezone.utc)


def reviewer_returning(raw: str):
    def reviewer(pr, diff, lane):
        return raw
    return reviewer


def gh_with(*prs, diff: str = SAMPLE_DIFF, comments: str = "") -> FakeGh:
    return (
        FakeGh()
        .on(is_pr_list, pr_list_json(*prs))
        .on(is_pr_diff, diff)
        .on(is_list_comments, comments)
        .on(is_pr_comment, "")
    )


def test_reviews_an_open_pr_end_to_end(tmp_path):
    # Arrange
    gh = gh_with(make_pr(number=1))

    # Act
    report, state = sweep_lane(
        make_config(), LANE_OPEN, reviewer_returning(VERDICT_WITH_BUG),
        State.empty(), gh, tmp_path, NOW,
    )

    # Assert
    assert report.considered == 1
    assert report.reviewed == 1
    assert report.posted == 1
    assert is_reviewed(state, LANE_OPEN, 1, "abcdef1234567890")


def test_writes_a_local_review_file(tmp_path):
    gh = gh_with(make_pr(number=7))

    sweep_lane(
        make_config(), LANE_OPEN, reviewer_returning(VERDICT_WITH_BUG),
        State.empty(), gh, tmp_path, NOW,
    )

    assert list(tmp_path.glob("PR-7-*.md"))


def test_skips_a_pr_already_reviewed_at_the_same_sha(tmp_path):
    gh = gh_with(make_pr(number=1, head_sha="same-sha"))
    state = mark_reviewed(State.empty(), LANE_OPEN, 1, "same-sha")

    report, _ = sweep_lane(
        make_config(), LANE_OPEN, reviewer_returning(VERDICT_WITH_BUG),
        state, gh, tmp_path, NOW,
    )

    assert report.reviewed == 0
    assert report.skipped == 1
    assert gh.calls_matching("pr diff") == [], "must not fetch a diff for a skipped PR"


def test_re_reviews_after_new_commits_change_the_head_sha(tmp_path):
    gh = gh_with(make_pr(number=1, head_sha="new-sha"))
    state = mark_reviewed(State.empty(), LANE_OPEN, 1, "old-sha")

    report, _ = sweep_lane(
        make_config(), LANE_OPEN, reviewer_returning(VERDICT_WITH_BUG),
        state, gh, tmp_path, NOW,
    )

    assert report.reviewed == 1


def test_skips_bot_authored_prs_before_spending_tokens(tmp_path):
    gh = gh_with(make_pr(number=1, author="prime-bot"))
    calls: list = []

    def tracking_reviewer(pr, diff, lane):
        calls.append(pr.number)
        return VERDICT_WITH_BUG

    report, _ = sweep_lane(
        make_config(bot_login="prime-bot"), LANE_OPEN, tracking_reviewer,
        State.empty(), gh, tmp_path, NOW,
    )

    assert report.skipped == 1
    assert calls == [], "reviewer must never run on a skipped PR"


def test_dry_run_writes_locally_but_never_posts(tmp_path):
    gh = gh_with(make_pr(number=1))

    report, _ = sweep_lane(
        make_config(dry_run=True), LANE_OPEN, reviewer_returning(VERDICT_WITH_BUG),
        State.empty(), gh, tmp_path, NOW,
    )

    assert report.reviewed == 1
    assert report.posted == 0
    assert gh.calls_matching("pr comment") == []
    assert list(tmp_path.glob("PR-1-*.md"))


def test_comment_budget_caps_posts_across_many_prs(tmp_path):
    gh = gh_with(*[make_pr(number=n, head_sha=f"sha{n}") for n in range(1, 6)])

    report, _ = sweep_lane(
        make_config(max_comments=2), LANE_OPEN, reviewer_returning(VERDICT_WITH_BUG),
        State.empty(), gh, tmp_path, NOW,
    )

    assert report.reviewed == 5
    assert report.posted == 2


def test_low_confidence_verdict_is_reviewed_but_not_posted(tmp_path):
    gh = gh_with(make_pr(number=1))

    report, _ = sweep_lane(
        make_config(min_confidence=0.7), LANE_OPEN,
        reviewer_returning(VERDICT_LOW_CONFIDENCE), State.empty(), gh, tmp_path, NOW,
    )

    assert report.reviewed == 1
    assert report.posted == 0
    assert list(tmp_path.glob("PR-1-*.md")), "held reviews still land in the audit trail"


def test_silent_verdict_is_recorded_but_not_posted(tmp_path):
    gh = gh_with(make_pr(number=1))

    report, _ = sweep_lane(
        make_config(), LANE_OPEN, reviewer_returning(VERDICT_EMPTY),
        State.empty(), gh, tmp_path, NOW,
    )

    assert report.reviewed == 1
    assert report.posted == 0


def test_skips_a_pr_whose_diff_is_all_ignored_paths(tmp_path):
    lock_only = "diff --git a/uv.lock b/uv.lock\n+version = 2\n"
    gh = gh_with(make_pr(number=1), diff=lock_only)

    report, _ = sweep_lane(
        make_config(ignore_paths=("**/*.lock",)), LANE_OPEN,
        reviewer_returning(VERDICT_WITH_BUG), State.empty(), gh, tmp_path, NOW,
    )

    assert report.reviewed == 0
    assert report.skipped == 1


def test_a_failing_reviewer_does_not_abort_the_sweep(tmp_path):
    gh = gh_with(make_pr(number=1, head_sha="s1"), make_pr(number=2, head_sha="s2"))

    def flaky(pr, diff, lane):
        if pr.number == 1:
            raise RuntimeError("subagent exploded")
        return VERDICT_WITH_BUG

    report, _ = sweep_lane(
        make_config(), LANE_OPEN, flaky, State.empty(), gh, tmp_path, NOW
    )

    assert report.errors == 1
    assert report.reviewed == 1, "the second PR must still be reviewed"


def test_a_failed_pr_is_not_marked_reviewed_so_it_retries_next_sweep(tmp_path):
    gh = gh_with(make_pr(number=1, head_sha="s1"))

    def always_fails(pr, diff, lane):
        raise RuntimeError("nope")

    _, state = sweep_lane(
        make_config(), LANE_OPEN, always_fails, State.empty(), gh, tmp_path, NOW
    )

    assert is_reviewed(state, LANE_OPEN, 1, "s1") is False


def test_unparseable_verdict_is_recorded_as_an_error(tmp_path):
    gh = gh_with(make_pr(number=1))

    report, _ = sweep_lane(
        make_config(), LANE_OPEN, reviewer_returning("I think it looks fine"),
        State.empty(), gh, tmp_path, NOW,
    )

    assert report.errors == 1
    assert "unusable verdict" in report.outcomes[0].summary_line()


def test_diff_fetch_failure_is_recorded_without_aborting(tmp_path):
    gh = FakeGh().on(is_pr_list, pr_list_json(make_pr(number=1)))  # no diff handler

    report, _ = sweep_lane(
        make_config(), LANE_OPEN, reviewer_returning(VERDICT_WITH_BUG),
        State.empty(), gh, tmp_path, NOW,
    )

    assert report.errors == 1
    assert "diff fetch failed" in report.outcomes[0].error


def test_merged_lane_queries_with_lookback_and_sets_cursor(tmp_path):
    gh = gh_with(make_pr(number=1, merged_at="2026-08-06T10:00:00Z"))

    report, state = sweep_lane(
        make_config(merged_lookback_days=7), LANE_MERGED,
        reviewer_returning(VERDICT_WITH_BUG), State.empty(), gh, tmp_path, NOW,
    )

    assert report.reviewed == 1
    assert state.merged_cursor == NOW.isoformat()
    # Filtering is client-side on mergedAt, not via --search: the search index is
    # eventually consistent and would hide a just-merged PR from this sweep.
    args = " ".join(gh.calls[0][0])
    assert "--state merged" in args
    assert "--search" not in args


def test_open_lane_does_not_set_a_merged_cursor(tmp_path):
    gh = gh_with(make_pr(number=1))

    _, state = sweep_lane(
        make_config(), LANE_OPEN, reviewer_returning(VERDICT_WITH_BUG),
        State.empty(), gh, tmp_path, NOW,
    )

    assert state.merged_cursor is None


def test_empty_repo_produces_an_empty_report(tmp_path):
    gh = FakeGh().on(is_pr_list, "[]")

    report, state = sweep_lane(
        make_config(), LANE_OPEN, reviewer_returning(VERDICT_WITH_BUG),
        State.empty(), gh, tmp_path, NOW,
    )

    assert report.considered == 0
    assert report.summaries() == ()


def test_summary_lines_describe_each_outcome(tmp_path):
    gh = gh_with(make_pr(number=11))

    report, _ = sweep_lane(
        make_config(), LANE_OPEN, reviewer_returning(VERDICT_WITH_BUG),
        State.empty(), gh, tmp_path, NOW,
    )

    line = report.summaries()[0]
    assert "PR #11" in line
    assert "HIGH" in line
    assert "posted" in line
