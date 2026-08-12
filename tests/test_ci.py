"""CI status awareness (P12): bucket/state mapping, tolerant parsing, the
best-effort failure excerpt, and rendering -- exercised entirely against
recorded `gh` JSON via FakeGh. No real `gh` call is ever made.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from prime_pr_review import ci
from prime_pr_review.ci import (
    CheckResult,
    CIStatus,
    activity_note,
    fetch_ci_status,
    fetch_failure_excerpt,
    render,
)

from .conftest import FakeGh

# --- local gh-call predicates ---------------------------------------------------
# `conftest`'s shared predicates cover calls every test file needs (pr list,
# pr diff, ...); the `pr checks` / `run view` / `run list` calls are specific
# to CI awareness, so they live here.


def is_pr_checks(args: Sequence[str]) -> bool:
    return args[0] == "pr" and args[1] == "checks"


def is_run_view(args: Sequence[str]) -> bool:
    return args[0] == "run" and args[1] == "view"


def is_run_list(args: Sequence[str]) -> bool:
    return args[0] == "run" and args[1] == "list"


def checks_payload(*entries: dict) -> str:
    return json.dumps(list(entries))


# --- CIStatus.state / .failing ---------------------------------------------------


def test_state_is_unknown_when_there_are_no_checks():
    assert CIStatus().state == "unknown"


def test_state_is_passing_when_no_checks_fail_or_pend():
    status = CIStatus(
        checks=(
            CheckResult("lint", "pass", False),
            CheckResult("legacy", "skipped", False),
            CheckResult("flaky", "cancelled", False),
        )
    )

    assert status.state == "passing"


def test_state_is_pending_when_none_fail_but_one_is_pending():
    status = CIStatus(
        checks=(CheckResult("lint", "pass", False), CheckResult("tests", "pending", False))
    )

    assert status.state == "pending"


def test_state_is_failing_when_any_check_fails_even_alongside_pending_ones():
    status = CIStatus(
        checks=(CheckResult("lint", "fail", True), CheckResult("tests", "pending", False))
    )

    assert status.state == "failing"


def test_failing_property_returns_only_the_checks_with_fail_status():
    status = CIStatus(
        checks=(
            CheckResult("a", "fail", False),
            CheckResult("b", "pass", False),
            CheckResult("c", "fail", False),
        )
    )

    assert [c.name for c in status.failing] == ["a", "c"]


def test_check_result_is_frozen():
    check = CheckResult("tests", "pass", False)

    with pytest.raises(AttributeError):
        check.status = "fail"  # type: ignore[misc]


def test_ci_status_is_frozen():
    status = CIStatus()

    with pytest.raises(AttributeError):
        status.checks = ()  # type: ignore[misc]


# --- bucket / state -> status mapping --------------------------------------------


@pytest.mark.parametrize(
    "item,expected",
    [
        ({"bucket": "pass"}, "pass"),
        ({"bucket": "fail"}, "fail"),
        ({"bucket": "pending"}, "pending"),
        ({"bucket": "skipping"}, "skipped"),
        ({"bucket": "cancel"}, "cancelled"),
        ({"bucket": "PASS"}, "pass"),  # case-insensitive
        ({"bucket": "unrecognized-bucket", "state": "SUCCESS"}, "pass"),  # falls back
        ({"state": "SUCCESS"}, "pass"),
        ({"state": "failure"}, "fail"),
        ({"state": "cancelled"}, "cancelled"),
        ({"state": "skipped"}, "skipped"),
        ({}, "pending"),  # neither present -> conservative default, never a guess
        ({"bucket": "??", "state": "??"}, "pending"),  # both unrecognized
    ],
)
def test_normalize_status_mapping(item, expected):
    assert ci._normalize_status(item) == expected


@pytest.mark.parametrize(
    "item,expected",
    [
        ({}, False),
        ({"required": True}, True),
        ({"required": False}, False),
        ({"isRequired": True}, True),
    ],
)
def test_parse_required_mapping(item, expected):
    assert ci._parse_required(item) == expected


# --- fetch_ci_status ---------------------------------------------------------------


def test_fetch_ci_status_requests_the_documented_fields_repo_and_pr_number():
    gh = FakeGh().on(is_pr_checks, "[]")

    fetch_ci_status("acme/widget", 42, gh)

    args = " ".join(gh.calls[0][0])
    assert "pr checks 42" in args
    assert "--repo acme/widget" in args
    assert f"--json {ci.CHECK_FIELDS}" in args


def test_check_fields_is_the_documented_field_set():
    assert ci.CHECK_FIELDS == "name,state,bucket,link,description"


def test_fetch_ci_status_parses_a_realistic_passing_payload():
    payload = checks_payload(
        {"name": "lint", "state": "SUCCESS", "bucket": "pass", "link": "", "description": ""},
        {"name": "tests", "state": "SUCCESS", "bucket": "pass", "link": "", "description": ""},
    )
    gh = FakeGh().on(is_pr_checks, payload)

    status = fetch_ci_status("acme/widget", 42, gh)

    assert status.state == "passing"
    assert len(status.checks) == 2


def test_fetch_ci_status_parses_a_realistic_failing_payload_matching_the_demo_scenario():
    payload = checks_payload(
        {
            "name": "contract-and-static",
            "state": "FAILURE",
            "bucket": "fail",
            "link": "https://github.com/acme/widget/actions/runs/1/job/2",
            "description": "ruff F401 `os` imported but unused",
        },
        {
            "name": "check-integrity",
            "state": "FAILURE",
            "bucket": "fail",
            "link": "",
            "description": "dependency job failed",
        },
    )
    gh = FakeGh().on(is_pr_checks, payload)

    status = fetch_ci_status("acme/widget", 42, gh)

    assert status.state == "failing"
    assert {c.name for c in status.failing} == {"contract-and-static", "check-integrity"}
    summaries = {c.name: c.summary for c in status.checks}
    assert summaries["contract-and-static"] == "ruff F401 `os` imported but unused"


def test_fetch_ci_status_returns_unknown_when_the_gh_call_fails():
    gh = FakeGh()  # no handler registered -> FakeGh raises GitHubError

    status = fetch_ci_status("acme/widget", 42, gh)

    assert status.state == "unknown"
    assert status.checks == ()


def test_fetch_ci_status_returns_unknown_on_malformed_json():
    gh = FakeGh().on(is_pr_checks, "not json {{{")

    assert fetch_ci_status("acme/widget", 42, gh).state == "unknown"


def test_fetch_ci_status_returns_unknown_when_payload_is_not_a_list():
    gh = FakeGh().on(is_pr_checks, '{"name": "lint"}')

    assert fetch_ci_status("acme/widget", 42, gh).state == "unknown"


def test_fetch_ci_status_returns_unknown_on_blank_output():
    gh = FakeGh().on(is_pr_checks, "")

    assert fetch_ci_status("acme/widget", 42, gh).state == "unknown"


def test_fetch_ci_status_skips_entries_with_no_usable_name_without_raising():
    payload = checks_payload({"state": "SUCCESS", "bucket": "pass"}, {"name": "tests", "bucket": "pass"})
    gh = FakeGh().on(is_pr_checks, payload)

    status = fetch_ci_status("acme/widget", 42, gh)

    assert [c.name for c in status.checks] == ["tests"]


def test_fetch_ci_status_skips_non_dict_entries_without_raising():
    gh = FakeGh().on(is_pr_checks, json.dumps([{"name": "tests", "bucket": "pass"}, "garbage", 5]))

    status = fetch_ci_status("acme/widget", 42, gh)

    assert [c.name for c in status.checks] == ["tests"]


def test_fetch_ci_status_required_flag_defaults_false_when_gh_does_not_send_it():
    gh = FakeGh().on(is_pr_checks, checks_payload({"name": "tests", "bucket": "pass"}))

    status = fetch_ci_status("acme/widget", 42, gh)

    assert status.checks[0].required is False


def test_fetch_ci_status_required_flag_is_read_when_present_in_the_payload():
    gh = FakeGh().on(
        is_pr_checks, checks_payload({"name": "tests", "bucket": "pass", "required": True})
    )

    status = fetch_ci_status("acme/widget", 42, gh)

    assert status.checks[0].required is True


# --- fetch_failure_excerpt ---------------------------------------------------------


def test_fetch_failure_excerpt_uses_the_run_id_from_the_failing_checks_link():
    payload = checks_payload(
        {
            "name": "contract-and-static",
            "bucket": "fail",
            "link": "https://github.com/acme/widget/actions/runs/555/job/1",
        }
    )
    log_text = "\n".join(f"line{i}" for i in range(1, 6))
    gh = FakeGh().on(is_pr_checks, payload).on(is_run_view, log_text)

    excerpt = fetch_failure_excerpt("acme/widget", 42, gh)

    assert excerpt == log_text
    run_view_call = next(c[0] for c in gh.calls if c[0][0] == "run" and c[0][1] == "view")
    assert run_view_call[2] == "555"
    assert "--log-failed" in run_view_call
    assert "--repo" in run_view_call and "acme/widget" in run_view_call


def test_fetch_failure_excerpt_truncates_to_the_last_max_lines():
    payload = checks_payload(
        {"name": "x", "bucket": "fail", "link": "https://x/actions/runs/9/job/1"}
    )
    log_text = "\n".join(f"line{i}" for i in range(1, 101))  # 100 lines
    gh = FakeGh().on(is_pr_checks, payload).on(is_run_view, log_text)

    excerpt = fetch_failure_excerpt("acme/widget", 42, gh, max_lines=10)

    assert excerpt.splitlines() == [f"line{i}" for i in range(91, 101)]


def test_fetch_failure_excerpt_uses_forty_lines_by_default():
    payload = checks_payload(
        {"name": "x", "bucket": "fail", "link": "https://x/actions/runs/9/job/1"}
    )
    log_text = "\n".join(f"line{i}" for i in range(1, 101))
    gh = FakeGh().on(is_pr_checks, payload).on(is_run_view, log_text)

    excerpt = fetch_failure_excerpt("acme/widget", 42, gh)

    assert len(excerpt.splitlines()) == 40
    assert excerpt.splitlines()[-1] == "line100"


def test_fetch_failure_excerpt_is_empty_when_there_is_no_failing_check():
    gh = FakeGh().on(is_pr_checks, checks_payload({"name": "tests", "bucket": "pass"}))

    excerpt = fetch_failure_excerpt("acme/widget", 42, gh)

    assert excerpt == ""
    assert not [c for c in gh.calls if c[0][0] == "run"]


def test_fetch_failure_excerpt_is_empty_when_the_checks_call_itself_fails():
    gh = FakeGh()  # unhandled -> raises

    assert fetch_failure_excerpt("acme/widget", 42, gh) == ""


def test_fetch_failure_excerpt_falls_back_to_run_list_when_the_link_has_no_run_id():
    payload = checks_payload(
        {"name": "external-ci", "bucket": "fail", "link": "https://example.com/status/abc"}
    )
    run_list_payload = json.dumps([{"databaseId": 777, "headSha": "abc123"}])
    log_text = "boom\ntraceback"
    gh = (
        FakeGh()
        .on(is_pr_checks, payload)
        .on(is_run_list, run_list_payload)
        .on(is_run_view, log_text)
    )

    excerpt = fetch_failure_excerpt("acme/widget", 42, gh)

    assert excerpt == log_text
    run_view_call = next(c[0] for c in gh.calls if c[0][0] == "run" and c[0][1] == "view")
    assert run_view_call[2] == "777"


def test_fetch_failure_excerpt_run_list_fallback_requests_the_documented_fields():
    payload = checks_payload({"name": "external-ci", "bucket": "fail", "link": "unmatched"})
    gh = FakeGh().on(is_pr_checks, payload).on(is_run_list, "[]")

    fetch_failure_excerpt("acme/widget", 42, gh)

    run_list_call = next(c[0] for c in gh.calls if c[0][0] == "run" and c[0][1] == "list")
    args = " ".join(run_list_call)
    assert "--repo acme/widget" in args
    assert "--status failure" in args
    assert f"--json {ci.RUN_LIST_FIELDS}" in args


def test_run_list_fields_is_the_documented_field_set():
    assert ci.RUN_LIST_FIELDS == "databaseId,headSha"


def test_fetch_failure_excerpt_is_empty_when_link_and_run_list_both_fail_to_resolve():
    payload = checks_payload({"name": "external-ci", "bucket": "fail", "link": ""})
    gh = FakeGh().on(is_pr_checks, payload).on(is_run_list, "[]")

    assert fetch_failure_excerpt("acme/widget", 42, gh) == ""


def test_fetch_failure_excerpt_is_empty_when_the_run_list_fallback_call_fails():
    payload = checks_payload({"name": "external-ci", "bucket": "fail", "link": ""})
    gh = FakeGh().on(is_pr_checks, payload)  # no run-list handler -> raises

    assert fetch_failure_excerpt("acme/widget", 42, gh) == ""


def test_fetch_failure_excerpt_is_empty_when_run_view_itself_fails():
    payload = checks_payload(
        {"name": "x", "bucket": "fail", "link": "https://x/actions/runs/9/job/1"}
    )
    gh = FakeGh().on(is_pr_checks, payload)  # no run-view handler -> raises

    assert fetch_failure_excerpt("acme/widget", 42, gh) == ""


def test_fetch_failure_excerpt_is_empty_on_malformed_checks_json():
    gh = FakeGh().on(is_pr_checks, "not json {{{")

    assert fetch_failure_excerpt("acme/widget", 42, gh) == ""


# --- _run_id_from_link / _tail_lines (unit-level) -----------------------------------


def test_run_id_from_link_extracts_the_numeric_run_id():
    link = "https://github.com/acme/widget/actions/runs/123456/job/789"

    assert ci._run_id_from_link(link) == "123456"


def test_run_id_from_link_returns_none_when_the_link_does_not_match():
    assert ci._run_id_from_link("https://example.com/not-actions") is None


def test_run_id_from_link_returns_none_for_an_empty_link():
    assert ci._run_id_from_link("") is None


def test_tail_lines_returns_everything_when_there_are_fewer_lines_than_the_limit():
    assert ci._tail_lines("a\nb\nc", 40) == "a\nb\nc"


def test_tail_lines_truncates_to_the_last_max_lines():
    text = "\n".join(f"line{i}" for i in range(1, 51))  # 50 lines

    result = ci._tail_lines(text, 40)

    assert result.splitlines() == [f"line{i}" for i in range(11, 51)]


def test_tail_lines_returns_empty_string_for_zero_max_lines():
    """`lines[-0:]` is a Python gotcha that returns everything; must not leak through."""
    assert ci._tail_lines("a\nb\nc", 0) == ""


def test_tail_lines_returns_empty_string_for_negative_max_lines():
    assert ci._tail_lines("a\nb\nc", -5) == ""


def test_tail_lines_returns_empty_string_for_empty_text():
    assert ci._tail_lines("", 40) == ""


# --- render --------------------------------------------------------------------


def test_render_is_empty_when_state_is_unknown():
    assert render(CIStatus(), "") == ""


def test_render_includes_the_state_line_for_passing():
    status = CIStatus(checks=(CheckResult("tests", "pass", False),))

    markdown = render(status, "")

    assert "CI status: passing" in markdown


def test_render_includes_a_failing_checks_table_with_name_and_summary():
    status = CIStatus(
        checks=(
            CheckResult("contract-and-static", "fail", True, "ruff F401 unused import"),
            CheckResult("check-integrity", "fail", True, "dependency failed"),
        )
    )

    markdown = render(status, "")

    assert "contract-and-static" in markdown
    assert "ruff F401 unused import" in markdown
    assert "check-integrity" in markdown
    assert "dependency failed" in markdown


def test_render_omits_the_failing_table_when_nothing_is_failing():
    status = CIStatus(checks=(CheckResult("tests", "pass", False),))

    markdown = render(status, "")

    assert "| Check | Summary |" not in markdown


def test_render_includes_the_fenced_excerpt_when_non_empty():
    status = CIStatus(checks=(CheckResult("tests", "fail", False),))

    markdown = render(status, "Traceback (most recent call last):\n  boom")

    assert "```" in markdown
    assert "Traceback (most recent call last):" in markdown


def test_render_omits_the_excerpt_block_when_excerpt_is_empty():
    status = CIStatus(checks=(CheckResult("tests", "fail", False),))

    markdown = render(status, "")

    assert "```" not in markdown


def test_render_includes_the_guidance_line_naming_the_state():
    status = CIStatus(checks=(CheckResult("tests", "fail", False),))

    markdown = render(status, "")

    assert "Do not re-report what CI already reports" in markdown
    assert "CI is failing" in markdown


def test_render_guidance_line_reflects_a_passing_state_too():
    status = CIStatus(checks=(CheckResult("tests", "pass", False),))

    markdown = render(status, "")

    assert "CI is passing" in markdown


# --- activity_note --------------------------------------------------------------


def test_activity_note_for_failing_state_includes_the_failing_check_count():
    status = CIStatus(
        checks=(
            CheckResult("a", "fail", False),
            CheckResult("b", "fail", False),
            CheckResult("c", "pass", False),
        )
    )

    assert activity_note(status) == "ci: failing (2 checks)"


def test_activity_note_for_passing_state():
    status = CIStatus(checks=(CheckResult("a", "pass", False),))

    assert activity_note(status) == "ci: passing"


def test_activity_note_for_pending_state():
    status = CIStatus(checks=(CheckResult("a", "pending", False),))

    assert activity_note(status) == "ci: pending"


def test_activity_note_is_empty_for_unknown_state():
    assert activity_note(CIStatus()) == ""
