"""Downstream guarantees of the skeptic pass (P14): a refuted finding is
challenged everywhere it matters — sweep wiring, blocking status, review
events, rendering, inline delivery — and dropped nowhere."""

from __future__ import annotations

import json
from dataclasses import replace

from prime_pr_review.config import _build_config
from prime_pr_review.review import Finding, Severity, Verdict
from prime_pr_review.reviews_api import (
    EVENT_COMMENT,
    EVENT_REQUEST_CHANGES,
    review_event_for,
)
from prime_pr_review.sinks import CommentBudget, post_pr_comment
from prime_pr_review.sweep import Enrichment, _apply_refutation
from prime_pr_review.template import render_review

from .conftest import FakeGh, is_list_comments, is_post_review, make_config, make_pr

PROMPTS_DIR = "skills/pr-review/prompts"
DIFF = (
    "diff --git a/src/app.py b/src/app.py\n"
    "index 1111111..2222222 100644\n"
    "--- a/src/app.py\n"
    "+++ b/src/app.py\n"
    "@@ -8,5 +8,5 @@\n"
    " context\n"
    " context\n"
    "-old\n"
    "+new\n"
    " context\n"
)


def _finding(
    *,
    line: int = 10,
    severity: Severity = Severity.HIGH,
    claim: str = "Off-by-one in the loop bound",
    suggestion: str = "",
    refuted: bool = False,
    refutation: str = "",
) -> Finding:
    return Finding(
        file="src/app.py",
        line=line,
        severity=severity,
        claim=claim,
        evidence="the loop runs one extra time",
        suggestion=suggestion,
        refuted=refuted,
        refutation=refutation,
    )


def _verdict(*findings: Finding) -> Verdict:
    return Verdict(introduces=tuple(findings), fixes=(), confidence=0.9)


def _skeptic(response: str):
    calls: list[str] = []

    def fn(prompt: str) -> str:
        calls.append(prompt)
        return response

    fn.calls = calls  # type: ignore[attr-defined]
    return fn


REFUTED_JSON = json.dumps({"refuted": True, "reasoning": "the path is guarded at line 5"})


class TestSweepWiring:
    def test_apply_refutation_annotates_and_notes(self) -> None:
        enrichment = Enrichment(skeptic_fn=_skeptic(REFUTED_JSON), prompts_dir=PROMPTS_DIR)

        verdict, notes = _apply_refutation(
            make_config(), DIFF, _verdict(_finding()), enrichment
        )

        assert verdict.introduces[0].refuted is True
        assert "guarded at line 5" in verdict.introduces[0].refutation
        assert any("1/1" in note for note in notes)

    def test_check_refute_false_makes_no_calls(self) -> None:
        skeptic = _skeptic(REFUTED_JSON)
        config = make_config()
        config = replace(config, review=replace(config.review, check_refute=False))
        enrichment = Enrichment(skeptic_fn=skeptic, prompts_dir=PROMPTS_DIR)

        verdict, notes = _apply_refutation(config, DIFF, _verdict(_finding()), enrichment)

        assert skeptic.calls == []
        assert verdict.introduces[0].refuted is False
        assert notes == ()

    def test_falls_back_to_model_fn_when_no_skeptic_fn(self) -> None:
        skeptic = _skeptic(REFUTED_JSON)
        enrichment = Enrichment(model_fn=skeptic, prompts_dir=PROMPTS_DIR)

        verdict, _ = _apply_refutation(make_config(), DIFF, _verdict(_finding()), enrichment)

        assert len(skeptic.calls) == 1
        assert verdict.introduces[0].refuted is True

    def test_no_model_at_all_is_a_silent_skip(self) -> None:
        verdict, notes = _apply_refutation(
            make_config(), DIFF, _verdict(_finding()), Enrichment(prompts_dir=PROMPTS_DIR)
        )

        assert verdict.introduces[0].refuted is False
        assert notes == ()


class TestBlockingAndEvents:
    def test_refuted_high_finding_does_not_block(self) -> None:
        assert _verdict(_finding(refuted=True, refutation="guarded")).has_blocking is False
        assert _verdict(_finding()).has_blocking is True

    def test_refuted_critical_never_requests_changes(self) -> None:
        refuted = _verdict(
            _finding(severity=Severity.CRITICAL, refuted=True, refutation="guarded")
        )
        upheld = _verdict(_finding(severity=Severity.CRITICAL))

        assert review_event_for(refuted, allow_request_changes=True) == EVENT_COMMENT
        assert review_event_for(upheld, allow_request_changes=True) == EVENT_REQUEST_CHANGES


class TestRendering:
    def test_refuted_finding_renders_as_challenged_with_reasoning(self) -> None:
        body = render_review(
            make_pr(),
            _verdict(_finding(refuted=True, refutation="the path is guarded")),
            "open",
        )

        assert "challenged by the skeptic pass" in body
        assert "the path is guarded" in body
        # It left the blocking headline: the callout reads clean.
        assert "✅" in body

    def test_refuted_suggestion_is_not_proposed(self) -> None:
        body = render_review(
            make_pr(),
            _verdict(_finding(suggestion="fixed = 1", refuted=True, refutation="guarded")),
            "open",
        )

        assert "No committable suggestions" in body
        assert "```suggestion" not in body


class TestInlineDelivery:
    def test_refuted_finding_never_posts_as_inline_comment(self) -> None:
        gh = FakeGh()
        gh.on(is_list_comments, "[]").on(is_post_review, "{}")
        verdict = _verdict(
            _finding(refuted=True, refutation="guarded", suggestion="fixed = 1"),
            _finding(line=11, claim="A real bug"),
        )

        outcome = post_pr_comment(
            make_config(min_confidence=0.5),
            make_pr(),
            verdict,
            "body",
            CommentBudget(limit=5),
            gh,
            diff=DIFF,
        )

        assert outcome.posted is True
        (call,) = gh.calls_matching("/reviews")
        payload = json.loads(call[1])
        assert len(payload["comments"]) == 1
        assert "A real bug" in payload["comments"][0]["body"]


class TestConfigKnobs:
    def test_knobs_default_true_and_parse_from_toml(self) -> None:
        assert _build_config({}).review.check_refute is True
        assert _build_config({}).review.judge_merge is True

        built = _build_config({"review": {"check_refute": False, "judge_merge": False}})
        assert built.review.check_refute is False
        assert built.review.judge_merge is False
