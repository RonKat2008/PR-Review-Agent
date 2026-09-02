"""Ensemble agreement (P2): matching, agreement-ratio confidence, and the
size=1 off switch."""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from prime_pr_review.ensemble import (
    LINE_BUCKET,
    ensemble_review,
    ensemble_review_detailed,
)
from prime_pr_review.github import PullRequest
from prime_pr_review.review import VerdictError, parse_verdict
from prime_pr_review.sweep import Reviewer

from .conftest import VERDICT_EMPTY, VERDICT_WITH_BUG, make_pr

PAYLOAD = "diff --git a/src/app.py b/src/app.py\n@@ -1,3 +1,3 @@\n-old\n+new\n"
LANE = "open"


def _finding(
    *,
    file: str = "src/app.py",
    line: int | None = 10,
    severity: str = "HIGH",
    claim: str = "Off-by-one in the loop bound",
    evidence: str = "",
    corroboration: str = "",
) -> dict:
    entry: dict = {
        "file": file,
        "line": line,
        "severity": severity,
        "claim": claim,
        "evidence": evidence,
    }
    if corroboration:
        entry["corroboration"] = corroboration
    return entry


def _fix(claim: str = "Guards against a null user", evidence: str = "") -> dict:
    return {"claim": claim, "evidence": evidence}


def _raw(
    *,
    introduces: Sequence[dict] = (),
    fixes: Sequence[dict] = (),
    confidence: float = 0.9,
    scope: dict | None = None,
    blast_radius: Sequence[dict] = (),
    files: Sequence[dict] = (),
    manual_checks: Sequence[dict] = (),
) -> str:
    payload: dict = {
        "introduces": list(introduces),
        "fixes": list(fixes),
        "confidence": confidence,
    }
    if scope is not None:
        payload["scope"] = scope
    if blast_radius:
        payload["blast_radius"] = list(blast_radius)
    if files:
        payload["files"] = list(files)
    if manual_checks:
        payload["manual_checks"] = list(manual_checks)
    return json.dumps(payload)


def scripted_reviewer(*responses: str | Exception) -> Reviewer:
    """A Reviewer stub returning (or raising) the next scripted response, in
    order. Running out of scripted responses is a test bug, not a case
    `ensemble_review`'s own failure tolerance should absorb -- so it asserts."""
    queue = list(responses)

    def reviewer(pr: PullRequest, payload: str, lane: str) -> str:
        assert queue, "reviewer called more times than scripted"
        value = queue.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    return reviewer


# --- basic agreement ----------------------------------------------------------------


def test_a_finding_all_three_runs_agree_on_survives_with_full_confidence():
    # Arrange
    raw = _raw(introduces=[_finding()])
    reviewer = scripted_reviewer(raw, raw, raw)

    # Act
    verdict = ensemble_review(make_pr(), PAYLOAD, LANE, reviewer, size=3, min_agreement=2)

    # Assert
    assert len(verdict.introduces) == 1
    assert verdict.confidence == 1.0
    assert verdict.introduces[0].corroboration == "3/3 reviewers"


def test_a_finding_only_one_of_three_runs_reports_is_dropped():
    # Arrange
    reviewer = scripted_reviewer(
        _raw(introduces=[_finding()]),
        _raw(introduces=[]),
        _raw(introduces=[]),
    )

    # Act
    verdict = ensemble_review(make_pr(), PAYLOAD, LANE, reviewer, size=3, min_agreement=2)

    # Assert
    assert verdict.introduces == ()


def test_min_agreement_of_one_keeps_a_finding_seen_by_only_one_run():
    # Arrange
    reviewer = scripted_reviewer(
        _raw(introduces=[_finding()]),
        _raw(introduces=[]),
        _raw(introduces=[]),
    )

    # Act
    verdict = ensemble_review(make_pr(), PAYLOAD, LANE, reviewer, size=3, min_agreement=1)

    # Assert
    assert len(verdict.introduces) == 1
    assert verdict.introduces[0].corroboration == "1/3 reviewers"


# --- line-bucket matching ------------------------------------------------------------


def test_findings_reported_at_nearby_lines_in_the_same_bucket_are_matched():
    # Arrange: lines 10 and 14 both floor-divide to bucket 2 (LINE_BUCKET == 5)
    reviewer = scripted_reviewer(
        _raw(introduces=[_finding(line=10)]),
        _raw(introduces=[_finding(line=14)]),
    )

    # Act
    verdict = ensemble_review(make_pr(), PAYLOAD, LANE, reviewer, size=2, min_agreement=2)

    # Assert: merged into one finding, not two
    assert len(verdict.introduces) == 1
    assert verdict.introduces[0].corroboration == "2/2 reviewers"


def test_findings_reported_in_different_line_buckets_are_not_matched():
    # Arrange: line 10 (bucket 2) and line 18 (bucket 3) land in different buckets
    reviewer = scripted_reviewer(
        _raw(introduces=[_finding(line=10)]),
        _raw(introduces=[_finding(line=18)]),
    )

    # Act
    verdict = ensemble_review(make_pr(), PAYLOAD, LANE, reviewer, size=2, min_agreement=2)

    # Assert: two separate groups of one, neither reaches min_agreement
    assert verdict.introduces == ()


def test_findings_with_no_line_number_still_match_on_file_and_severity():
    # Arrange
    reviewer = scripted_reviewer(
        _raw(introduces=[_finding(line=None)]),
        _raw(introduces=[_finding(line=None)]),
    )

    # Act
    verdict = ensemble_review(make_pr(), PAYLOAD, LANE, reviewer, size=2, min_agreement=2)

    # Assert
    assert len(verdict.introduces) == 1
    assert verdict.introduces[0].line is None


def test_severity_mismatch_prevents_two_reports_from_matching():
    # Arrange: same file, same line, different severity
    reviewer = scripted_reviewer(
        _raw(introduces=[_finding(severity="HIGH")]),
        _raw(introduces=[_finding(severity="MEDIUM")]),
    )

    # Act
    verdict = ensemble_review(make_pr(), PAYLOAD, LANE, reviewer, size=2, min_agreement=2)

    # Assert: two separate groups of one, neither reaches min_agreement
    assert verdict.introduces == ()


def test_line_bucket_constant_is_five():
    assert LINE_BUCKET == 5


# --- representative selection ---------------------------------------------------------


def test_representative_for_a_surviving_group_carries_the_longest_evidence():
    # Arrange
    reviewer = scripted_reviewer(
        _raw(introduces=[_finding(line=10, evidence="short reason")]),
        _raw(
            introduces=[
                _finding(line=11, evidence="a considerably longer and more detailed reason")
            ]
        ),
    )

    # Act
    verdict = ensemble_review(make_pr(), PAYLOAD, LANE, reviewer, size=2, min_agreement=2)

    # Assert
    assert verdict.introduces[0].evidence == "a considerably longer and more detailed reason"


def test_linter_corroboration_is_preserved_with_the_agreement_ratio_appended():
    # Arrange
    raw = _raw(introduces=[_finding(corroboration="bandit:B608")])
    reviewer = scripted_reviewer(raw, raw, raw)

    # Act
    verdict = ensemble_review(make_pr(), PAYLOAD, LANE, reviewer, size=3, min_agreement=2)

    # Assert: appended, never overwritten
    assert verdict.introduces[0].corroboration == "bandit:B608 · 3/3 reviewers"


# --- failure tolerance -----------------------------------------------------------------


def test_one_run_raising_an_exception_is_tolerated_when_others_agree():
    # Arrange
    reviewer = scripted_reviewer(
        _raw(introduces=[_finding()]),
        RuntimeError("subagent timed out"),
        _raw(introduces=[_finding()]),
    )

    # Act
    verdict = ensemble_review(make_pr(), PAYLOAD, LANE, reviewer, size=3, min_agreement=2)

    # Assert
    assert len(verdict.introduces) == 1
    assert verdict.introduces[0].corroboration == "2/3 reviewers"
    assert verdict.confidence == 2 / 3


def test_one_run_producing_an_unparsable_response_is_tolerated_when_others_agree():
    # Arrange
    reviewer = scripted_reviewer(
        _raw(introduces=[_finding()]),
        "not a verdict, just prose with no JSON in it",
        _raw(introduces=[_finding()]),
    )

    # Act
    verdict = ensemble_review(make_pr(), PAYLOAD, LANE, reviewer, size=3, min_agreement=2)

    # Assert
    assert len(verdict.introduces) == 1
    assert verdict.confidence == 2 / 3


def test_all_runs_failing_raises_verdict_error_naming_the_failure_count():
    # Arrange
    reviewer = scripted_reviewer(
        RuntimeError("boom"),
        "not json",
        VerdictError("already unusable"),
    )

    # Act / Assert
    with pytest.raises(VerdictError, match="3/3"):
        ensemble_review(make_pr(), PAYLOAD, LANE, reviewer, size=3, min_agreement=2)


# --- confidence: strongest finding and clean consensus ----------------------------------


def test_confidence_follows_the_most_corroborated_surviving_finding_not_the_weakest():
    # Arrange: finding A is agreed on by all 3 runs, finding B only by 2 -- both
    # survive min_agreement=2, but overall confidence must reflect the strongest.
    reviewer = scripted_reviewer(
        _raw(
            introduces=[
                _finding(file="a.py", line=10),
                _finding(file="b.py", line=50, severity="MEDIUM"),
            ]
        ),
        _raw(
            introduces=[
                _finding(file="a.py", line=10),
                _finding(file="b.py", line=50, severity="MEDIUM"),
            ]
        ),
        _raw(introduces=[_finding(file="a.py", line=10)]),
    )

    # Act
    verdict = ensemble_review(make_pr(), PAYLOAD, LANE, reviewer, size=3, min_agreement=2)

    # Assert
    assert len(verdict.introduces) == 2
    assert verdict.confidence == 1.0


def test_confidence_falls_back_to_clean_consensus_when_nothing_survives():
    # Arrange: two runs report a clean diff, one reports a finding no one else saw
    reviewer = scripted_reviewer(
        _raw(introduces=[]),
        _raw(introduces=[]),
        _raw(introduces=[_finding()]),
    )

    # Act
    verdict = ensemble_review(make_pr(), PAYLOAD, LANE, reviewer, size=3, min_agreement=2)

    # Assert
    assert verdict.introduces == ()
    assert verdict.confidence == 2 / 3


def test_confidence_is_zero_when_no_run_is_clean_and_nothing_survives():
    # Arrange: three runs, three mutually distinct unmatched findings
    reviewer = scripted_reviewer(
        _raw(introduces=[_finding(file="a.py")]),
        _raw(introduces=[_finding(file="b.py")]),
        _raw(introduces=[_finding(file="c.py")]),
    )

    # Act
    verdict = ensemble_review(make_pr(), PAYLOAD, LANE, reviewer, size=3, min_agreement=2)

    # Assert
    assert verdict.introduces == ()
    assert verdict.confidence == 0.0


# --- fixes: union at effective min_agreement=1 -------------------------------------------


def test_fixes_reported_by_only_one_run_still_survive():
    # Arrange
    reviewer = scripted_reviewer(
        _raw(fixes=[_fix(claim="Guards against a null user")]),
        _raw(fixes=[]),
        _raw(fixes=[]),
    )

    # Act
    verdict = ensemble_review(make_pr(), PAYLOAD, LANE, reviewer, size=3, min_agreement=2)

    # Assert: unlike introduces, one report is enough
    assert len(verdict.fixes) == 1
    assert verdict.fixes[0].claim == "Guards against a null user"


def test_fixes_are_unioned_and_near_duplicate_phrasing_collapses_to_one():
    # Arrange
    reviewer = scripted_reviewer(
        _raw(fixes=[_fix(claim="Guards against a null user")]),
        _raw(fixes=[_fix(claim="guards against a NULL user!")]),
        _raw(fixes=[_fix(claim="Fixes a memory leak in the cache")]),
    )

    # Act
    verdict = ensemble_review(make_pr(), PAYLOAD, LANE, reviewer, size=3, min_agreement=2)

    # Assert: two distinct fixes; the reworded duplicate collapsed into the first
    assert len(verdict.fixes) == 2
    claims = {fix.claim for fix in verdict.fixes}
    assert "Guards against a null user" in claims
    assert "Fixes a memory leak in the cache" in claims


# --- scope / blast_radius / files / manual_checks: first success, never merged -------------


def test_scope_blast_radius_files_and_manual_checks_come_from_the_first_run():
    # Arrange
    first = _raw(
        scope={"intent": "Fix the off-by-one", "aligned": True},
        blast_radius=[
            {"symbol": "total_price", "kind": "signature_change", "change": "added tax_rate"}
        ],
        files=[{"file": "src/app.py", "summary": "Fixes the bound", "relation": "serves_intent"}],
        manual_checks=[
            {"feature": "Checkout total", "files": ["src/app.py"], "steps": "Open cart, check total."}
        ],
    )
    second = _raw(
        scope={"intent": "A different intent entirely", "aligned": False},
        blast_radius=[{"symbol": "unrelated_symbol", "kind": "removal", "change": "deleted"}],
        files=[{"file": "other.py", "summary": "Different file", "relation": "unrelated"}],
        manual_checks=[
            {"feature": "Different feature", "files": ["other.py"], "steps": "Do something else."}
        ],
    )
    reviewer = scripted_reviewer(first, second, second)

    # Act
    verdict = ensemble_review(make_pr(), PAYLOAD, LANE, reviewer, size=3, min_agreement=2)

    # Assert
    assert verdict.scope.intent == "Fix the off-by-one"
    assert verdict.blast_radius[0].symbol == "total_price"
    assert verdict.files[0].file == "src/app.py"
    assert verdict.manual_checks[0].feature == "Checkout total"


def test_scope_and_friends_come_from_the_first_run_that_actually_succeeded():
    # Arrange: the first attempt fails outright, so "first" must mean the first
    # *successful* parse, not attempt index 0.
    reviewer = scripted_reviewer(
        RuntimeError("subagent timed out"),
        _raw(scope={"intent": "From the second attempt", "aligned": True}),
        _raw(scope={"intent": "From the third attempt", "aligned": True}),
    )

    # Act
    verdict = ensemble_review(make_pr(), PAYLOAD, LANE, reviewer, size=3, min_agreement=2)

    # Assert
    assert verdict.scope.intent == "From the second attempt"


# --- size=1: the off switch, exact passthrough --------------------------------------------


def test_size_one_makes_exactly_one_call_and_returns_parse_verdict_unchanged():
    # Arrange
    reviewer = scripted_reviewer(VERDICT_WITH_BUG)  # a second call would assert

    # Act
    verdict = ensemble_review(make_pr(), PAYLOAD, LANE, reviewer, size=1)

    # Assert
    assert verdict == parse_verdict(VERDICT_WITH_BUG)


def test_size_one_propagates_a_verdict_error_unchanged_rather_than_batching_it():
    # Arrange
    reviewer = scripted_reviewer("not valid json at all")

    # Act / Assert: the original parse_verdict error, not an ensemble "all failed" message
    with pytest.raises(VerdictError, match="not valid JSON"):
        ensemble_review(make_pr(), PAYLOAD, LANE, reviewer, size=1)


def test_size_one_does_not_touch_confidence_even_though_it_is_self_reported():
    # Arrange
    reviewer = scripted_reviewer(VERDICT_EMPTY)

    # Act
    verdict = ensemble_review(make_pr(), PAYLOAD, LANE, reviewer, size=1)

    # Assert: passthrough means the lone self-reported confidence is untouched
    assert verdict.confidence == 0.95


# --- judge-merge (P15) --------------------------------------------------------------

PROMPTS_DIR = "skills/pr-review/prompts"


def _judge(response: str | Exception):
    calls: list[str] = []

    def fn(prompt: str) -> str:
        calls.append(prompt)
        if isinstance(response, Exception):
            raise response
        return response

    fn.calls = calls  # type: ignore[attr-defined]
    return fn


def test_judge_merges_two_descriptions_of_one_bug_into_shared_agreement():
    # Arrange: two seats report the same defect far apart in the same file --
    # different buckets, so deterministic grouping sees two 1/3 singletons.
    reviewer = scripted_reviewer(
        _raw(introduces=[_finding(line=10)]),
        _raw(introduces=[_finding(line=48, claim="Loop bound is off by one")]),
        _raw(introduces=[]),
    )
    judge = _judge(json.dumps({"clusters": [[0, 1]]}))

    # Act
    verdict, notes = ensemble_review_detailed(
        make_pr(), PAYLOAD, LANE, reviewer,
        size=3, min_agreement=2, judge_fn=judge, prompts_dir=PROMPTS_DIR,
    )

    # Assert: one finding, corroborated 2/3 -- it would have been DROPPED
    # entirely at min_agreement=2 without the judge.
    assert len(verdict.introduces) == 1
    assert "2/3 reviewers" in verdict.introduces[0].corroboration
    assert verdict.confidence == pytest.approx(2 / 3)
    assert notes == ("judge-merge: 1 duplicate group(s) merged",)


def test_judge_failure_falls_back_to_deterministic_grouping_with_a_note():
    reviewer = scripted_reviewer(
        _raw(introduces=[_finding(line=10)]),
        _raw(introduces=[_finding(line=48)]),
        _raw(introduces=[]),
    )
    judge = _judge(RuntimeError("judge died"))

    verdict, notes = ensemble_review_detailed(
        make_pr(), PAYLOAD, LANE, reviewer,
        size=3, min_agreement=1, judge_fn=judge, prompts_dir=PROMPTS_DIR,
    )

    # Both singletons survive (min_agreement=1) exactly as without a judge.
    assert len(verdict.introduces) == 2
    assert any("judge-merge failed" in note for note in notes)


def test_judge_is_not_consulted_when_no_two_groups_share_a_file():
    reviewer = scripted_reviewer(
        _raw(introduces=[_finding(file="src/a.py")]),
        _raw(introduces=[_finding(file="src/b.py")]),
        _raw(introduces=[]),
    )
    judge = _judge(json.dumps({"clusters": []}))

    _, notes = ensemble_review_detailed(
        make_pr(), PAYLOAD, LANE, reviewer,
        size=3, min_agreement=1, judge_fn=judge, prompts_dir=PROMPTS_DIR,
    )

    assert judge.calls == []
    assert notes == ()


def test_judge_finding_no_duplicates_leaves_groups_alone_and_says_so():
    reviewer = scripted_reviewer(
        _raw(introduces=[_finding(line=10)]),
        _raw(introduces=[_finding(line=48, claim="A different bug entirely")]),
        _raw(introduces=[]),
    )
    judge = _judge(json.dumps({"clusters": []}))

    verdict, notes = ensemble_review_detailed(
        make_pr(), PAYLOAD, LANE, reviewer,
        size=3, min_agreement=1, judge_fn=judge, prompts_dir=PROMPTS_DIR,
    )

    assert len(verdict.introduces) == 2
    assert notes == ("judge-merge: no duplicates found",)


def test_no_judge_fn_means_identical_behavior_to_the_wrapper():
    reviewer_a = scripted_reviewer(
        _raw(introduces=[_finding()]), _raw(introduces=[_finding()]), _raw()
    )
    reviewer_b = scripted_reviewer(
        _raw(introduces=[_finding()]), _raw(introduces=[_finding()]), _raw()
    )

    detailed, notes = ensemble_review_detailed(
        make_pr(), PAYLOAD, LANE, reviewer_a, size=3, min_agreement=2
    )
    wrapped = ensemble_review(make_pr(), PAYLOAD, LANE, reviewer_b, size=3, min_agreement=2)

    assert detailed == wrapped
    assert notes == ()
