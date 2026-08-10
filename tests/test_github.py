"""The gh CLI wrapper, exercised entirely against recorded JSON."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

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
