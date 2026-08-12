"""The gh CLI wrapper, exercised entirely against recorded JSON."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone

import pytest

from prime_pr_review import github
from prime_pr_review.github import (
    GitHubError,
    authenticated_login,
    fetch_diff,
    list_comments,
    list_merged_prs,
    list_open_prs,
    lookback_cutoff,
    post_comment,
)

from .conftest import (
    FakeGh,
    SAMPLE_DIFF,
    is_list_comments,
    is_pr_comment,
    is_pr_diff,
    is_pr_list,
    make_pr,
    pr_list_json,
)


def _limit_value(args: Sequence[str]) -> str:
    """The value passed to `--limit` in a recorded `gh` call."""
    return args[args.index("--limit") + 1]


def pr_list_with_limit(limit: int) -> Callable[[Sequence[str]], bool]:
    """A FakeGh predicate matching a `pr list` call requesting exactly `limit`."""

    def predicate(args: Sequence[str]) -> bool:
        return is_pr_list(args) and _limit_value(args) == str(limit)

    return predicate


def test_parses_open_pull_requests():
    gh = FakeGh().on(is_pr_list, pr_list_json(make_pr(number=1), make_pr(number=2)))

    prs = list_open_prs("acme/widget", gh)

    assert [p.number for p in prs] == [1, 2]


def test_extracts_nested_author_login():
    gh = FakeGh().on(is_pr_list, pr_list_json(make_pr(author="bob")))

    assert list_open_prs("acme/widget", gh)[0].author == "bob"


def test_empty_pr_list_yields_no_pull_requests():
    gh = FakeGh().on(is_pr_list, "[]")

    assert list_open_prs("acme/widget", gh) == ()


def test_blank_output_yields_no_pull_requests():
    gh = FakeGh().on(is_pr_list, "")

    assert list_open_prs("acme/widget", gh) == ()


def test_raises_on_non_json_pr_list():
    gh = FakeGh().on(is_pr_list, "not json at all")

    with pytest.raises(GitHubError, match="non-JSON"):
        list_open_prs("acme/widget", gh)


def test_raises_when_pr_list_is_not_an_array():
    gh = FakeGh().on(is_pr_list, '{"number": 1}')

    with pytest.raises(GitHubError, match="JSON array"):
        list_open_prs("acme/widget", gh)


def test_raises_on_malformed_pr_entry():
    gh = FakeGh().on(is_pr_list, '[{"number": "not-a-number"}]')

    with pytest.raises(GitHubError, match="Malformed"):
        list_open_prs("acme/widget", gh)


def test_open_pr_query_targets_the_right_repo_and_state():
    gh = FakeGh().on(is_pr_list, "[]")

    list_open_prs("acme/widget", gh)

    args = " ".join(gh.calls[0][0])
    assert "--repo acme/widget" in args
    assert "--state open" in args


def test_merged_pr_query_avoids_the_eventually_consistent_search_index():
    """`--search` lags merges by up to a minute; a just-merged PR must not be missed."""
    gh = FakeGh().on(is_pr_list, "[]")
    since = datetime(2026, 8, 1, tzinfo=timezone.utc)

    list_merged_prs("acme/widget", since, gh)

    args = " ".join(gh.calls[0][0])
    assert "--state merged" in args
    assert "--search" not in args


def test_merged_prs_are_filtered_by_merge_timestamp():
    recent = make_pr(number=1, merged_at="2026-08-09T12:00:00Z")
    stale = make_pr(number=2, merged_at="2026-07-01T12:00:00Z")
    gh = FakeGh().on(is_pr_list, pr_list_json(recent, stale))
    since = datetime(2026, 8, 1, tzinfo=timezone.utc)

    result = list_merged_prs("acme/widget", since, gh)

    assert [p.number for p in result] == [1]


def test_merged_pr_exactly_at_the_cutoff_is_included():
    pr = make_pr(number=1, merged_at="2026-08-01T00:00:00Z")
    gh = FakeGh().on(is_pr_list, pr_list_json(pr))

    result = list_merged_prs("acme/widget", datetime(2026, 8, 1, tzinfo=timezone.utc), gh)

    assert len(result) == 1


def test_pr_without_a_merge_timestamp_is_excluded():
    gh = FakeGh().on(is_pr_list, pr_list_json(make_pr(number=1, merged_at=None)))

    result = list_merged_prs("acme/widget", datetime(2026, 8, 1, tzinfo=timezone.utc), gh)

    assert result == ()


def test_unparseable_merge_timestamp_is_kept_rather_than_dropped():
    """Reviewing a PR twice is cheap. Silently dropping one is not."""
    gh = FakeGh().on(is_pr_list, pr_list_json(make_pr(number=1, merged_at="not-a-date")))

    result = list_merged_prs("acme/widget", datetime(2026, 8, 1, tzinfo=timezone.utc), gh)

    assert len(result) == 1


def test_fetches_a_raw_diff():
    gh = FakeGh().on(is_pr_diff, SAMPLE_DIFF)

    assert fetch_diff("acme/widget", 5, gh) == SAMPLE_DIFF


def test_comment_body_is_sent_via_stdin():
    """Bodies go through stdin so markdown, quotes, and newlines survive intact."""
    gh = FakeGh().on(is_pr_comment, "")

    post_comment("acme/widget", 9, "## body with `quotes` and\nnewlines", gh)

    args, stdin = gh.calls[0]
    assert "--body-file" in args and "-" in args
    assert stdin == "## body with `quotes` and\nnewlines"


def test_lists_existing_comment_bodies():
    gh = FakeGh().on(is_list_comments, "first comment\nsecond comment\n")

    assert list_comments("acme/widget", 3, gh) == ("first comment", "second comment")


def test_blank_lines_are_dropped_from_comment_list():
    gh = FakeGh().on(is_list_comments, "one\n\n  \ntwo\n")

    assert list_comments("acme/widget", 3, gh) == ("one", "two")


def test_reads_authenticated_login():
    gh = FakeGh().on(lambda a: a[0] == "api", "prime-bot\n")

    assert authenticated_login(gh) == "prime-bot"


def test_unhandled_call_surfaces_as_a_github_error():
    gh = FakeGh()

    with pytest.raises(GitHubError, match="unexpected gh call"):
        list_open_prs("acme/widget", gh)


def test_lookback_cutoff_subtracts_the_requested_days():
    now = datetime(2026, 8, 7, 12, 0, tzinfo=timezone.utc)

    assert lookback_cutoff(7, now) == now - timedelta(days=7)


def test_pull_request_size_sums_additions_and_deletions():
    assert make_pr(additions=10, deletions=4).size == 14


def test_default_limit_used_when_none_specified_is_two_hundred():
    gh = FakeGh().on(pr_list_with_limit(200), "[]")

    list_open_prs("acme/widget", gh)

    assert len(gh.calls) == 1
    assert _limit_value(gh.calls[0][0]) == "200"


def test_hard_cap_is_two_thousand():
    assert github.HARD_CAP == 2000


def test_full_first_page_triggers_a_second_request_with_doubled_limit():
    """A page exactly as full as requested is the signature of possible
    truncation, so the retry must ask for double."""
    gh = (
        FakeGh()
        .on(
            pr_list_with_limit(3),
            pr_list_json(make_pr(number=1), make_pr(number=2), make_pr(number=3)),
        )
        .on(pr_list_with_limit(6), pr_list_json(make_pr(number=4)))
    )

    prs = list_open_prs("acme/widget", gh, limit=3)

    assert [p.number for p in prs] == [4]
    assert [_limit_value(c[0]) for c in gh.calls] == ["3", "6"]


def test_short_first_page_does_not_trigger_a_second_request():
    """Fewer PRs than requested means nothing was truncated -- stop immediately."""
    gh = FakeGh().on(
        pr_list_with_limit(5), pr_list_json(make_pr(number=1), make_pr(number=2))
    )

    prs = list_open_prs("acme/widget", gh, limit=5)

    assert [p.number for p in prs] == [1, 2]
    assert len(gh.calls) == 1


def test_escalation_keeps_doubling_across_multiple_full_pages():
    """A full page at 2x must escalate again to 4x -- doubling repeats, not just
    once."""
    gh = (
        FakeGh()
        .on(pr_list_with_limit(2), pr_list_json(make_pr(number=1), make_pr(number=2)))
        .on(
            pr_list_with_limit(4),
            pr_list_json(*[make_pr(number=n) for n in range(1, 5)]),
        )
        .on(
            pr_list_with_limit(8),
            pr_list_json(*[make_pr(number=n) for n in range(1, 6)]),
        )
    )

    prs = list_open_prs("acme/widget", gh, limit=2)

    assert [p.number for p in prs] == [1, 2, 3, 4, 5]
    assert [_limit_value(c[0]) for c in gh.calls] == ["2", "4", "8"]


def test_escalated_call_preserves_repo_state_and_json_arguments():
    gh = (
        FakeGh()
        .on(pr_list_with_limit(2), pr_list_json(make_pr(number=1), make_pr(number=2)))
        .on(pr_list_with_limit(4), pr_list_json(make_pr(number=3)))
    )

    list_open_prs("acme/widget", gh, limit=2)

    assert len(gh.calls) == 2
    escalated = gh.calls[1][0]
    assert escalated[0] == "pr" and escalated[1] == "list"
    joined = " ".join(escalated)
    assert "--repo acme/widget" in joined
    assert "--state open" in joined
    assert f"--json {github.PR_FIELDS}" in joined
    assert "--limit 4" in joined


def test_escalation_stops_at_hard_cap(monkeypatch):
    """Doubling must clamp to HARD_CAP rather than requesting past it."""
    monkeypatch.setattr(github, "HARD_CAP", 4)
    gh = (
        FakeGh()
        .on(
            pr_list_with_limit(3),
            pr_list_json(*[make_pr(number=n) for n in range(1, 4)]),
        )
        .on(pr_list_with_limit(4), pr_list_json(make_pr(number=1), make_pr(number=2)))
    )

    prs = list_open_prs("acme/widget", gh, limit=3)

    assert [_limit_value(c[0]) for c in gh.calls] == ["3", "4"]
    assert [p.number for p in prs] == [1, 2]


def test_full_hard_cap_response_still_returns_results_with_a_warning(monkeypatch):
    """Even a still-full page at HARD_CAP must not raise -- return what we have
    and make the truncation visible via a warning instead of print."""
    monkeypatch.setattr(github, "HARD_CAP", 4)
    gh = FakeGh().on(
        pr_list_with_limit(4), pr_list_json(*[make_pr(number=n) for n in range(1, 5)])
    )

    with pytest.warns(UserWarning, match="HARD_CAP"):
        prs = list_open_prs("acme/widget", gh, limit=4)

    assert len(gh.calls) == 1, "must not request past HARD_CAP even when the page is still full"
    assert [p.number for p in prs] == [1, 2, 3, 4]


def test_merged_lane_since_filter_is_applied_after_escalation():
    """Escalation must run to completion before the since-filter narrows results,
    so a PR only visible on the larger page is still considered."""
    since = datetime(2026, 8, 1, tzinfo=timezone.utc)
    stale = [make_pr(number=n, merged_at="2026-07-01T12:00:00Z") for n in range(1, 4)]
    recent = make_pr(number=5, merged_at="2026-08-09T12:00:00Z")
    gh = (
        FakeGh()
        .on(pr_list_with_limit(3), pr_list_json(*stale))
        .on(pr_list_with_limit(6), pr_list_json(*stale, recent))
    )

    result = list_merged_prs("acme/widget", since, gh, limit=3)

    assert [p.number for p in result] == [5]
    assert len(gh.calls) == 2


def test_merged_lane_escalated_call_still_avoids_the_search_index():
    since = datetime(2026, 8, 1, tzinfo=timezone.utc)
    gh = (
        FakeGh()
        .on(
            pr_list_with_limit(2),
            pr_list_json(
                *[make_pr(number=n, merged_at="2026-08-05T00:00:00Z") for n in range(1, 3)]
            ),
        )
        .on(
            pr_list_with_limit(4),
            pr_list_json(make_pr(number=9, merged_at="2026-08-05T00:00:00Z")),
        )
    )

    list_merged_prs("acme/widget", since, gh, limit=2)

    escalated = " ".join(gh.calls[1][0])
    assert "--state merged" in escalated
    assert "--search" not in escalated
