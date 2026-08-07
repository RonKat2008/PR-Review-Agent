"""The safety gates.

These are the tests that matter most: they are what stands between a miscalibrated
run and a public comment on someone else's pull request.
"""

from __future__ import annotations

import json

import httpx
import pytest

from prime_pr_review.review import parse_verdict
from prime_pr_review.sinks import (
    CommentBudget,
    SinkError,
    evaluate_comment_gates,
    post_pr_comment,
    send_webhook,
    write_local,
)
from prime_pr_review.state import LANE_OPEN, build_marker

from .conftest import (
    FakeGh,
    VERDICT_EMPTY,
    VERDICT_LOW_CONFIDENCE,
    VERDICT_WITH_BUG,
    is_list_comments,
    is_pr_comment,
    make_config,
    make_pr,
)

BUDGET = CommentBudget(limit=5)


def _verdict(raw: str = VERDICT_WITH_BUG):
    return parse_verdict(raw)


# --------------------------------------------------------------------------
# Gate decisions (pure)
# --------------------------------------------------------------------------


def test_allows_a_confident_verdict_when_everything_is_configured():
    decision = evaluate_comment_gates(make_config(), make_pr(), _verdict(), BUDGET, [])

    assert decision.allowed is True


def test_dry_run_blocks_posting():
    decision = evaluate_comment_gates(
        make_config(dry_run=True), make_pr(), _verdict(), BUDGET, []
    )

    assert decision.allowed is False
    assert "dry_run" in decision.reason


def test_disabled_sink_blocks_posting():
    decision = evaluate_comment_gates(
        make_config(pr_comment=False), make_pr(), _verdict(), BUDGET, []
    )

    assert decision.allowed is False
    assert "disabled" in decision.reason


def test_low_confidence_verdict_is_blocked():
    decision = evaluate_comment_gates(
        make_config(min_confidence=0.7), make_pr(), _verdict(VERDICT_LOW_CONFIDENCE), BUDGET, []
    )

    assert decision.allowed is False
    assert "confidence" in decision.reason


def test_silent_verdict_is_blocked():
    decision = evaluate_comment_gates(make_config(), make_pr(), _verdict(VERDICT_EMPTY), BUDGET, [])

    assert decision.allowed is False
    assert "empty" in decision.reason


def test_self_authored_pr_is_blocked():
    """Prevents the agent reviewing its own pull requests in a loop."""
    decision = evaluate_comment_gates(
        make_config(bot_login="prime-bot"), make_pr(author="prime-bot"), _verdict(), BUDGET, []
    )

    assert decision.allowed is False
    assert "bot account" in decision.reason


def test_self_exclusion_is_off_when_bot_login_is_unset():
    decision = evaluate_comment_gates(
        make_config(bot_login=""), make_pr(author="anyone"), _verdict(), BUDGET, []
    )

    assert decision.allowed is True


def test_exhausted_budget_blocks_posting():
    decision = evaluate_comment_gates(
        make_config(), make_pr(), _verdict(), CommentBudget(limit=2, used=2), []
    )

    assert decision.allowed is False
    assert "budget exhausted" in decision.reason


def test_existing_marker_for_same_sha_blocks_a_duplicate():
    existing = [f"{build_marker('abcdef1234567890')}\nprevious review"]

    decision = evaluate_comment_gates(make_config(), make_pr(), _verdict(), BUDGET, existing)

    assert decision.allowed is False
    assert "already commented" in decision.reason


def test_marker_for_a_different_sha_does_not_block():
    """New commits mean a new head SHA, which must be reviewable again."""
    existing = [f"{build_marker('an-older-sha')}\nprevious review"]

    decision = evaluate_comment_gates(make_config(), make_pr(), _verdict(), BUDGET, existing)

    assert decision.allowed is True


def test_zero_budget_blocks_every_comment():
    decision = evaluate_comment_gates(
        make_config(max_comments=0), make_pr(), _verdict(), CommentBudget(limit=0), []
    )

    assert decision.allowed is False


def test_budget_spend_is_immutable():
    original = CommentBudget(limit=3)

    spent = original.spend()

    assert original.used == 0
    assert spent.used == 1


# --------------------------------------------------------------------------
# Posting (side effects)
# --------------------------------------------------------------------------


def test_posts_and_spends_budget_when_allowed():
    gh = FakeGh().on(is_list_comments, "").on(is_pr_comment, "")

    outcome = post_pr_comment(make_config(), make_pr(), _verdict(), "body", BUDGET, gh)

    assert outcome.posted is True
    assert outcome.budget.used == 1
    assert len(gh.calls_matching("pr comment")) == 1


def test_dry_run_never_calls_github():
    gh = FakeGh()

    outcome = post_pr_comment(
        make_config(dry_run=True), make_pr(), _verdict(), "body", BUDGET, gh
    )

    assert outcome.posted is False
    assert gh.calls == []


def test_budget_is_not_spent_when_a_gate_blocks():
    gh = FakeGh().on(is_list_comments, "").on(is_pr_comment, "")

    outcome = post_pr_comment(
        make_config(min_confidence=0.99), make_pr(), _verdict(), "body", BUDGET, gh
    )

    assert outcome.posted is False
    assert outcome.budget.used == 0


def test_refuses_to_post_when_existing_comments_cannot_be_read():
    """Unable to verify idempotency means we must not risk a duplicate."""
    gh = FakeGh().on(is_pr_comment, "")  # no handler for listing comments

    outcome = post_pr_comment(make_config(), make_pr(), _verdict(), "body", BUDGET, gh)

    assert outcome.posted is False
    assert "could not read existing comments" in outcome.reason
    assert gh.calls_matching("pr comment") == []


def test_reports_a_failed_post_without_raising():
    from prime_pr_review.github import GitHubError

    def failing(args, stdin=None):
        if args[0] == "api":
            return ""
        raise GitHubError("boom")

    outcome = post_pr_comment(make_config(), make_pr(), _verdict(), "body", BUDGET, failing)

    assert outcome.posted is False
    assert "post failed" in outcome.reason


# --------------------------------------------------------------------------
# Local audit trail
# --------------------------------------------------------------------------


def test_writes_review_to_disk(tmp_path):
    path = write_local(make_pr(number=42), _verdict(), "the body", LANE_OPEN, tmp_path)

    assert path.is_file()
    assert "the body" in path.read_text(encoding="utf-8")


def test_local_filename_includes_pr_number_and_sha(tmp_path):
    path = write_local(make_pr(number=42, head_sha="abcdef1234"), _verdict(), "b", LANE_OPEN, tmp_path)

    assert path.name == "PR-42-abcdef12.md"


def test_local_file_carries_machine_readable_front_matter(tmp_path):
    path = write_local(make_pr(number=42), _verdict(), "b", LANE_OPEN, tmp_path)

    content = path.read_text(encoding="utf-8")
    payload = json.loads(content.split("<!--")[1].split("-->")[0])
    assert payload["pr"] == 42
    assert payload["introduces"] == 1
    assert payload["lane"] == LANE_OPEN


def test_creates_reviews_directory_when_absent(tmp_path):
    target = tmp_path / "does" / "not" / "exist"

    write_local(make_pr(), _verdict(), "b", LANE_OPEN, target)

    assert target.is_dir()


# --------------------------------------------------------------------------
# Webhook
# --------------------------------------------------------------------------


def _client(capture: list) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        capture.append(json.loads(request.content))
        return httpx.Response(200)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_sends_slack_shaped_payload(secrets):
    captured: list = []

    sent = send_webhook(make_config(webhook_kind="slack"), secrets, ["PR #1 — ok"], _client(captured))

    assert sent is True
    assert "text" in captured[0]
    assert "PR #1" in captured[0]["text"]


def test_sends_discord_shaped_payload(secrets):
    captured: list = []

    send_webhook(make_config(webhook_kind="discord"), secrets, ["PR #1 — ok"], _client(captured))

    assert "content" in captured[0]


def test_generic_payload_carries_structured_items(secrets):
    captured: list = []

    send_webhook(make_config(webhook_kind="generic"), secrets, ["a", "b"], _client(captured))

    assert captured[0]["count"] == 2
    assert captured[0]["items"] == ["a", "b"]


def test_long_digests_are_truncated_with_an_overflow_note(secrets):
    captured: list = []
    summaries = [f"PR #{i}" for i in range(25)]

    send_webhook(make_config(), secrets, summaries, _client(captured))

    assert "and 15 more" in captured[0]["text"]


def test_skips_delivery_when_there_is_nothing_to_report(secrets):
    assert send_webhook(make_config(), secrets, [], _client([])) is False


def test_skips_delivery_when_webhook_sink_is_disabled(secrets):
    assert send_webhook(make_config(webhook=False), secrets, ["x"], _client([])) is False


def test_raises_when_webhook_enabled_but_url_missing():
    from prime_pr_review.config import Secrets

    no_url = Secrets(github_token="t", webhook_url=None)

    with pytest.raises(SinkError, match="no webhook URL"):
        send_webhook(make_config(), no_url, ["x"], _client([]))


def test_raises_on_webhook_error_status(secrets):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server exploded")

    client = httpx.Client(transport=httpx.MockTransport(handler))

    with pytest.raises(SinkError, match="500"):
        send_webhook(make_config(), secrets, ["x"], client)
