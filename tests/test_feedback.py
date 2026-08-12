"""The false-positive feedback loop (P6): fingerprinting, gh-derived rejections,
persistence, filtering, and prompt guidance."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from prime_pr_review.feedback import (
    DEFAULT_DISMISSAL_PHRASES,
    FeedbackError,
    Rejection,
    claim_fingerprint,
    fetch_rejections,
    filter_rejected,
    load_rejections,
    render_rejection_guidance,
    save_rejections,
)
from prime_pr_review.review import Finding, Severity, Verdict, render_markdown
from prime_pr_review.state import LANE_OPEN
from prime_pr_review.template import render_review

from .conftest import FakeGh, make_pr

REPO = "acme/widget"
PR = 42
BOT = "prime-bot"
NOW = datetime(2026, 8, 10, tzinfo=timezone.utc)


def _finding(
    *,
    file: str = "shop/orders.py",
    line: int | None = 13,
    severity: Severity = Severity.HIGH,
    claim: str = "Bare except returns None implicitly",
    evidence: str = "",
) -> Finding:
    return Finding(file=file, line=line, severity=severity, claim=claim, evidence=evidence)


def _comment(id_: int, login: str, body: str, created_at: str) -> dict:
    return {"id": id_, "user": {"login": login}, "body": body, "created_at": created_at}


def _comments_json(*comments: dict) -> str:
    return json.dumps(list(comments))


def _reactions_json(*contents: str) -> str:
    return json.dumps([{"content": c} for c in contents])


def _is_comments_call(args) -> bool:
    return args[0] == "api" and args[1] == f"repos/{REPO}/issues/{PR}/comments"


def _is_reactions_call(comment_id: int):
    target = f"repos/{REPO}/issues/comments/{comment_id}/reactions"

    def predicate(args) -> bool:
        return args[0] == "api" and args[1] == target

    return predicate


# --- claim_fingerprint ----------------------------------------------------------


def test_fingerprint_is_stable_across_punctuation_case_and_word_order():
    a = claim_fingerprint("Bare `except:` returns None implicitly.")
    b = claim_fingerprint("returns none implicitly, bare except")

    assert a == b


def test_fingerprint_differs_for_materially_different_claims():
    a = claim_fingerprint("SQL injection in the customer search query")
    b = claim_fingerprint("Off-by-one error in the total price calculation")

    assert a != b


def test_fingerprint_is_a_sixteen_character_hex_digest():
    fingerprint = claim_fingerprint("anything at all")

    assert len(fingerprint) == 16
    int(fingerprint, 16)  # raises ValueError if it isn't hex


def test_fingerprint_of_empty_text_does_not_raise():
    assert claim_fingerprint("") != ""


# --- fetch_rejections: reaction and dismissal detection --------------------------


def test_thumbs_down_reaction_marks_the_bot_finding_rejected():
    body = "**HIGH · `shop/orders.py:13` · Bare except returns None implicitly**"
    gh = (
        FakeGh()
        .on(_is_comments_call, _comments_json(_comment(1, BOT, body, "2026-08-01T00:00:00Z")))
        .on(_is_reactions_call(1), _reactions_json("-1"))
    )

    rejections = fetch_rejections(REPO, PR, BOT, gh, now=NOW)

    assert len(rejections) == 1
    assert rejections[0].file == "shop/orders.py"
    assert rejections[0].reason == "thumbs-down reaction"
    assert rejections[0].claim_fingerprint == claim_fingerprint(
        "Bare except returns None implicitly"
    )


def test_dismissal_phrase_is_detected_case_insensitively():
    body = "**MEDIUM · `shop/orders.py:13` · Bare except returns None implicitly**"
    gh = (
        FakeGh()
        .on(
            _is_comments_call,
            _comments_json(
                _comment(1, BOT, body, "2026-08-01T00:00:00Z"),
                _comment(2, "alice", "Actually this is a FALSE POSITIVE, ignore it.", "2026-08-01T01:00:00Z"),
            ),
        )
        .on(_is_reactions_call(1), _reactions_json())
    )

    rejections = fetch_rejections(REPO, PR, BOT, gh, now=NOW)

    assert len(rejections) == 1
    assert "false positive" in rejections[0].reason.lower()


def test_bot_own_later_reply_does_not_count_as_a_dismissal():
    finding_body = "**MEDIUM · `shop/orders.py:13` · Bare except returns None implicitly**"
    gh = (
        FakeGh()
        .on(
            _is_comments_call,
            _comments_json(
                _comment(1, BOT, finding_body, "2026-08-01T00:00:00Z"),
                _comment(2, BOT, "Re-checked -- not a bug after all, disregard.", "2026-08-01T01:00:00Z"),
            ),
        )
        .on(_is_reactions_call(1), _reactions_json())
        .on(_is_reactions_call(2), _reactions_json())
    )

    rejections = fetch_rejections(REPO, PR, BOT, gh, now=NOW)

    assert rejections == ()


def test_reactions_on_non_bot_comments_are_ignored():
    """A -1 on someone else's comment must never be read as rejecting a bot finding,
    even if that comment happens to look like a finding line itself."""
    alice_body = "**MEDIUM · `shop/orders.py:99` · A comment that happens to look like a finding**"
    gh = (
        FakeGh()
        .on(
            _is_comments_call,
            _comments_json(
                _comment(1, BOT, "no findings here, all clear", "2026-08-01T00:00:00Z"),
                _comment(2, "alice", alice_body, "2026-08-01T01:00:00Z"),
            ),
        )
        .on(_is_reactions_call(1), _reactions_json())
        .on(_is_reactions_call(2), _reactions_json("-1"))
    )

    rejections = fetch_rejections(REPO, PR, BOT, gh, now=NOW)

    assert rejections == ()


def test_dismissal_comment_before_the_finding_does_not_count():
    body = "**HIGH · `shop/orders.py:13` · Bare except returns None implicitly**"
    gh = (
        FakeGh()
        .on(
            _is_comments_call,
            _comments_json(
                _comment(1, "alice", "false positive, unrelated to what follows", "2026-07-01T00:00:00Z"),
                _comment(2, BOT, body, "2026-08-01T00:00:00Z"),
            ),
        )
        .on(_is_reactions_call(2), _reactions_json())
    )

    rejections = fetch_rejections(REPO, PR, BOT, gh, now=NOW)

    assert rejections == ()


def test_dismissal_phrases_param_is_configurable():
    body = "**HIGH · `shop/orders.py:13` · Bare except returns None implicitly**"
    gh = (
        FakeGh()
        .on(
            _is_comments_call,
            _comments_json(
                _comment(1, BOT, body, "2026-08-01T00:00:00Z"),
                _comment(2, "alice", "meh, works as designed", "2026-08-01T01:00:00Z"),
            ),
        )
        .on(_is_reactions_call(1), _reactions_json())
    )

    rejections = fetch_rejections(
        REPO, PR, BOT, gh, dismissal_phrases=("works as designed",), now=NOW
    )

    assert len(rejections) == 1


def test_default_dismissal_phrases_do_not_match_arbitrary_text():
    body = "**HIGH · `shop/orders.py:13` · Bare except returns None implicitly**"
    gh = (
        FakeGh()
        .on(
            _is_comments_call,
            _comments_json(
                _comment(1, BOT, body, "2026-08-01T00:00:00Z"),
                _comment(2, "alice", "meh, works as designed", "2026-08-01T01:00:00Z"),
            ),
        )
        .on(_is_reactions_call(1), _reactions_json())
    )

    rejections = fetch_rejections(REPO, PR, BOT, gh, now=NOW)

    assert rejections == ()


def test_default_dismissal_phrases_match_the_spec():
    assert DEFAULT_DISMISSAL_PHRASES == ("not a bug", "false positive", "intentional", "wontfix")


# --- fetch_rejections: tolerance and failure containment -------------------------


def test_any_gh_failure_yields_no_rejections():
    gh = FakeGh()  # no handlers registered -> every call raises GitHubError

    rejections = fetch_rejections(REPO, PR, BOT, gh, now=NOW)

    assert rejections == ()


def test_gh_failure_on_the_reactions_call_still_yields_no_rejections():
    body = "**HIGH · `shop/orders.py:13` · Bare except returns None implicitly**"
    gh = FakeGh().on(
        _is_comments_call, _comments_json(_comment(1, BOT, body, "2026-08-01T00:00:00Z"))
    )
    # reactions endpoint deliberately left unregistered

    rejections = fetch_rejections(REPO, PR, BOT, gh, now=NOW)

    assert rejections == ()


def test_no_comments_yields_no_rejections():
    gh = FakeGh().on(_is_comments_call, _comments_json())

    assert fetch_rejections(REPO, PR, BOT, gh, now=NOW) == ()


def test_blank_comments_response_yields_no_rejections():
    gh = FakeGh().on(_is_comments_call, "")

    assert fetch_rejections(REPO, PR, BOT, gh, now=NOW) == ()


def test_unexpected_comments_payload_shape_yields_no_rejections():
    gh = FakeGh().on(_is_comments_call, json.dumps({"not": "a list"}))

    assert fetch_rejections(REPO, PR, BOT, gh, now=NOW) == ()


def test_blank_reactions_response_is_treated_as_no_reactions():
    body = "**HIGH · `shop/orders.py:13` · Bare except returns None implicitly**"
    gh = (
        FakeGh()
        .on(_is_comments_call, _comments_json(_comment(1, BOT, body, "2026-08-01T00:00:00Z")))
        .on(_is_reactions_call(1), "")
    )

    assert fetch_rejections(REPO, PR, BOT, gh, now=NOW) == ()


def test_non_list_reactions_payload_is_treated_as_no_reactions():
    body = "**HIGH · `shop/orders.py:13` · Bare except returns None implicitly**"
    gh = (
        FakeGh()
        .on(_is_comments_call, _comments_json(_comment(1, BOT, body, "2026-08-01T00:00:00Z")))
        .on(_is_reactions_call(1), json.dumps({"unexpected": "shape"}))
    )

    assert fetch_rejections(REPO, PR, BOT, gh, now=NOW) == ()


def test_comment_with_missing_user_field_is_not_treated_as_the_bot():
    body = "**HIGH · `shop/orders.py:13` · Bare except returns None implicitly**"
    comment_without_user = {"id": 1, "body": body, "created_at": "2026-08-01T00:00:00Z"}
    gh = FakeGh().on(_is_comments_call, json.dumps([comment_without_user]))

    assert fetch_rejections(REPO, PR, BOT, gh, now=NOW) == ()


def test_finding_line_without_a_location_is_skipped_not_raised():
    body = "\n".join(
        [
            "**HIGH** no backticked location here at all",
            "**HIGH · `shop/orders.py:13` · Bare except returns None implicitly**",
        ]
    )
    gh = (
        FakeGh()
        .on(_is_comments_call, _comments_json(_comment(1, BOT, body, "2026-08-01T00:00:00Z")))
        .on(_is_reactions_call(1), _reactions_json("-1"))
    )

    rejections = fetch_rejections(REPO, PR, BOT, gh, now=NOW)

    assert len(rejections) == 1
    assert rejections[0].file == "shop/orders.py"


# --- fetch_rejections: round trip against the real templates ---------------------


def test_end_to_end_against_the_live_template_suppresses_a_rejected_finding():
    """The exact loop P6 exists for: render a real finding with `render_review`,
    simulate a maintainer thumbs-down, fetch the rejection back, and confirm
    `filter_rejected` would hold the same finding back on the next sweep."""
    finding = _finding(file="shop/orders.py", claim="Bare except returns None implicitly")
    verdict = Verdict(introduces=(finding,), fixes=(), confidence=0.9)
    body = render_review(make_pr(number=PR), verdict, LANE_OPEN)

    gh = (
        FakeGh()
        .on(_is_comments_call, _comments_json(_comment(1, BOT, body, "2026-08-01T00:00:00Z")))
        .on(_is_reactions_call(1), _reactions_json("-1"))
    )

    rejections = fetch_rejections(REPO, PR, BOT, gh, now=NOW)
    kept, suppressed = filter_rejected([finding], rejections)

    assert kept == ()
    assert suppressed == (finding,)


def test_end_to_end_against_the_legacy_render_markdown_format():
    finding = _finding(file="shop/orders.py", claim="Bare except returns None implicitly")
    verdict = Verdict(introduces=(finding,), fixes=(), confidence=0.9)
    body = render_markdown(make_pr(number=PR), verdict, LANE_OPEN)

    gh = (
        FakeGh()
        .on(_is_comments_call, _comments_json(_comment(1, BOT, body, "2026-08-01T00:00:00Z")))
        .on(_is_reactions_call(1), _reactions_json("-1"))
    )

    rejections = fetch_rejections(REPO, PR, BOT, gh, now=NOW)
    kept, suppressed = filter_rejected([finding], rejections)

    assert kept == ()
    assert suppressed == (finding,)


# --- persistence: load_rejections / save_rejections -------------------------------


def test_missing_rejections_file_is_a_cold_start_not_an_error(tmp_path):
    assert load_rejections(tmp_path / "absent.json") == ()


def test_round_trips_through_disk(tmp_path):
    # Arrange
    path = tmp_path / "state" / "rejections.json"
    rejection = Rejection(
        file="shop/orders.py",
        claim_fingerprint="abc123",
        reason="thumbs-down reaction",
        pr_number=7,
        rejected_at="2026-08-01T00:00:00+00:00",
    )

    # Act
    save_rejections([rejection], path)
    reloaded = load_rejections(path)

    # Assert
    assert reloaded == (rejection,)


def test_save_creates_parent_directory(tmp_path):
    path = tmp_path / "deeply" / "nested" / "rejections.json"

    save_rejections([], path)

    assert path.is_file()


def test_rejects_file_that_is_not_a_json_array(tmp_path):
    path = tmp_path / "rejections.json"
    path.write_text(json.dumps({"file": "a.py"}), encoding="utf-8")

    with pytest.raises(FeedbackError, match="JSON array"):
        load_rejections(path)


def test_rejects_malformed_json(tmp_path):
    path = tmp_path / "rejections.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(FeedbackError, match="Could not read"):
        load_rejections(path)


def test_rejects_entry_missing_a_required_field(tmp_path):
    path = tmp_path / "rejections.json"
    path.write_text(json.dumps([{"file": "a.py"}]), encoding="utf-8")

    with pytest.raises(FeedbackError, match="malformed entry"):
        load_rejections(path)


def test_save_with_malformed_existing_file_raises_rather_than_overwriting(tmp_path):
    path = tmp_path / "rejections.json"
    path.write_text("not json", encoding="utf-8")

    with pytest.raises(FeedbackError):
        save_rejections([], path)


def test_merge_prefers_the_newest_rejected_at_on_conflict(tmp_path):
    path = tmp_path / "rejections.json"
    old = Rejection("shop/orders.py", "fp1", "thumbs-down reaction", 7, "2026-08-01T00:00:00+00:00")
    new = Rejection("shop/orders.py", "fp1", "dismissed: 'wontfix'", 7, "2026-08-05T00:00:00+00:00")

    save_rejections([old], path)
    save_rejections([new], path)

    assert load_rejections(path) == (new,)


def test_merge_keeps_the_newer_entry_when_the_incoming_write_is_stale(tmp_path):
    path = tmp_path / "rejections.json"
    newer = Rejection("shop/orders.py", "fp1", "thumbs-down reaction", 7, "2026-08-05T00:00:00+00:00")
    stale = Rejection("shop/orders.py", "fp1", "dismissed: 'wontfix'", 7, "2026-08-01T00:00:00+00:00")

    save_rejections([newer], path)
    save_rejections([stale], path)

    assert load_rejections(path) == (newer,)


def test_merge_unions_disjoint_keys_rather_than_overwriting(tmp_path):
    path = tmp_path / "rejections.json"
    first = Rejection("a.py", "fp1", "thumbs-down reaction", 1, "2026-08-01T00:00:00+00:00")
    second = Rejection("b.py", "fp2", "thumbs-down reaction", 2, "2026-08-02T00:00:00+00:00")

    save_rejections([first], path)
    save_rejections([second], path)

    assert set(load_rejections(path)) == {first, second}


# --- filter_rejected ---------------------------------------------------------------


def test_filter_rejected_matches_on_file_and_claim_fingerprint():
    finding = _finding(file="shop/orders.py", claim="Bare except returns None implicitly")
    rejection = Rejection(
        file="shop/orders.py",
        claim_fingerprint=claim_fingerprint("Bare except returns None implicitly"),
        reason="thumbs-down reaction",
        pr_number=1,
        rejected_at="2026-08-01T00:00:00+00:00",
    )

    kept, suppressed = filter_rejected([finding], [rejection])

    assert kept == ()
    assert suppressed == (finding,)


def test_filter_rejected_requires_the_same_file_not_just_the_same_claim():
    finding = _finding(file="shop/other.py", claim="Bare except returns None implicitly")
    rejection = Rejection(
        file="shop/orders.py",
        claim_fingerprint=claim_fingerprint("Bare except returns None implicitly"),
        reason="thumbs-down reaction",
        pr_number=1,
        rejected_at="2026-08-01T00:00:00+00:00",
    )

    kept, suppressed = filter_rejected([finding], [rejection])

    assert kept == (finding,)
    assert suppressed == ()


def test_filter_rejected_keeps_findings_with_no_matching_rejection():
    finding = _finding()

    kept, suppressed = filter_rejected([finding], [])

    assert kept == (finding,)
    assert suppressed == ()


def test_filter_rejected_handles_empty_findings():
    rejection = Rejection("a.py", "fp", "thumbs-down reaction", 1, "2026-08-01T00:00:00+00:00")

    kept, suppressed = filter_rejected([], [rejection])

    assert kept == ()
    assert suppressed == ()


# --- render_rejection_guidance -----------------------------------------------------


def test_guidance_is_empty_when_there_are_no_rejections():
    assert render_rejection_guidance([]) == ""


def test_guidance_includes_file_and_reason():
    rejection = Rejection("shop/orders.py", "fp1", "thumbs-down reaction", 7, "2026-08-01T00:00:00+00:00")

    guidance = render_rejection_guidance([rejection])

    assert "shop/orders.py" in guidance
    assert "thumbs-down reaction" in guidance
    assert "Do not re-report" in guidance


def test_guidance_is_capped_at_the_limit():
    rejections = [
        Rejection(f"file{i}.py", f"fp{i}", "thumbs-down reaction", 1, f"2026-08-{i + 1:02d}T00:00:00+00:00")
        for i in range(25)
    ]

    guidance = render_rejection_guidance(rejections, limit=20)

    assert guidance.count("- `file") == 20


def test_guidance_orders_newest_rejection_first():
    older = Rejection("a.py", "fp1", "thumbs-down reaction", 1, "2026-08-01T00:00:00+00:00")
    newer = Rejection("b.py", "fp2", "thumbs-down reaction", 1, "2026-08-05T00:00:00+00:00")

    guidance = render_rejection_guidance([older, newer])

    assert guidance.index("b.py") < guidance.index("a.py")
