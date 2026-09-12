"""The offline scorer: per-arm cost attribution, the derived `single` baseline,
and the exclusion counts that keep an unscorable PR out of every metric."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prime_pr_review.ensemble import ensemble_review_detailed
from prime_pr_review.evaluation.corpus import ReferenceComment, Row
from prime_pr_review.evaluation.recording import (
    Recorder,
    recording_model_fn,
    recording_reviewer,
)
from prime_pr_review.evaluation.scorer import (
    Accumulator,
    build_aggregates,
    drifted,
    load_pricing,
    score_instance_dir,
)
from prime_pr_review.evaluation.scoring import InstanceScore
from prime_pr_review.refute import build_refute_prompt
from prime_pr_review.review import Finding, Severity, Verdict

from .conftest import make_pr

PROMPTS = Path("skills/pr-review/prompts")
DIFF = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,2 +1,2 @@\n-x\n+y\n+z\n"
SEAT_MODELS = ("m/a", "m/b", "m/c")
AUX_MODEL, JUDGE_MODEL, SKEPTIC_MODEL = "m/aux", "m/judge", "m/skeptic"
# 1 USD per token in and out, so a call's cost is exactly its token count and
# the arithmetic under test is visible in the assertions.
PRICING = {m: (1e6, 1e6) for m in (*SEAT_MODELS, AUX_MODEL, JUDGE_MODEL, SKEPTIC_MODEL)}
TOKENS = {"m/a": (100, 1), "m/b": (200, 2), "m/c": (300, 3),
          AUX_MODEL: (10, 1), JUDGE_MODEL: (40, 4), SKEPTIC_MODEL: (50, 5)}
COST = {model: float(sum(t)) for model, t in TOKENS.items()}


DEFAULT_COMMENTS = (ReferenceComment("a.py", 1, None, "t"),)


def _finding(line, claim):
    return {"file": "a.py", "line": line, "severity": "HIGH", "claim": claim, "evidence": "e"}


def _verdict(*fs):
    return json.dumps({"introduces": list(fs), "fixes": [], "confidence": 0.9})


# Two seats report the same file at different lines, so the judge pass has
# same-file groups to consider and its call is actually recorded.
SEAT_RESPONSES = [_verdict(_finding(1, "bug one")), _verdict(_finding(7, "bug two")), _verdict()]


def _row(comments=DEFAULT_COMMENTS) -> Row:
    return Row(instance_id="o__r-1@abc", repo="o/r", language="Python", pull_number=1, title="t",
               body="", base_commit="base", head_sha="abc", head_commit_message="m", patch=DIFF,
               reference_comments=tuple(comments))


@pytest.fixture
def instance(tmp_path):
    """One recorded instance directory, as `run` would leave it: three seats, an
    aux (intent) pass, a judge merge and a skeptic pass, each with known tokens."""
    inst = tmp_path / "o__r-1@abc"
    inst.mkdir(parents=True)
    rec = Recorder(inst)
    usage = {"v": (0, 0)}
    outputs = iter(SEAT_RESPONSES)

    def make_seat_fn(model):
        def fn(_prompt):
            usage["v"] = TOKENS[model]
            return next(outputs)
        return fn

    def staged(role, model, response):
        def fn(_prompt):
            usage["v"] = TOKENS[model]
            return response
        return recording_model_fn(role, model, fn, rec, lambda: usage["v"])

    staged("aux", AUX_MODEL, '{"summary":"s","claims":[]}')("intent prompt")
    reviewer = recording_reviewer(SEAT_MODELS, make_seat_fn, rec, PROMPTS, lambda: usage["v"])
    live, _ = ensemble_review_detailed(make_pr(), DIFF, "open", reviewer, size=3, min_agreement=1,
                                       judge_fn=staged("judge", JUDGE_MODEL, '{"clusters": []}'),
                                       prompts_dir=PROMPTS)
    skeptic = staged("skeptic", SKEPTIC_MODEL, '{"refuted": false, "reasoning": "r"}')
    template = (PROMPTS / "refute.md").read_text(encoding="utf-8")
    for f in live.introduces:
        skeptic(build_refute_prompt(template, f, DIFF))
    (inst / "outcome.json").write_text(json.dumps(
        {"instance_id": "o__r-1@abc", "error": None, "seconds": 12.0, "cost_usd": 0.25,
         "verdict": {"introduces": [{"file": f.file, "line": f.line, "claim": f.claim}
                                    for f in live.introduces]}}))
    return inst


def _score(instance, row=None, prompts=PROMPTS, pricing=PRICING):
    acc = Accumulator()
    score_instance_dir(instance, row or _row(), prompts, acc, SEAT_MODELS, pricing)
    return acc


def test_arm_cost_counts_only_the_calls_that_arm_consumes(instance):
    acc = _score(instance)
    cost = {arm: acc.arm_cost[(arm, "on")] for arm, _ in acc.arm_cost}
    assert cost["seat-1"] == pytest.approx(COST[AUX_MODEL] + COST["m/a"])
    assert cost["seat-2"] == pytest.approx(COST[AUX_MODEL] + COST["m/b"])
    seats = COST["m/a"] + COST["m/b"] + COST["m/c"]
    assert cost["ensemble"] == pytest.approx(COST[AUX_MODEL] + seats)
    assert cost["ensemble+judge"] == pytest.approx(COST[AUX_MODEL] + seats + COST[JUDGE_MODEL])
    skeptics = sum(c.role == "skeptic" for c in Recorder(instance).calls())
    assert cost["full"] == pytest.approx(cost["ensemble+judge"] + skeptics * COST[SKEPTIC_MODEL])


def test_arm_seconds_are_the_wall_time_of_those_same_calls(instance):
    acc = _score(instance)
    calls = {c.role: c.seconds for c in Recorder(instance).calls()}
    assert acc.arm_seconds[("seat-1", "on")] == pytest.approx(calls["aux"] + calls["seat"], abs=1.0)
    assert acc.arm_seconds[("full", "on")] >= acc.arm_seconds[("ensemble", "on")]


def test_missing_pricing_snapshot_reports_zero_cost_with_a_note(instance, tmp_path):
    pricing, notes = load_pricing(tmp_path)
    assert pricing == {} and notes
    acc = _score(instance, pricing=pricing)
    assert all(v == 0.0 for v in acc.arm_cost.values())


def test_pricing_snapshot_is_read_from_the_run_dir(tmp_path):
    (tmp_path / "pricing.json").write_text(json.dumps({"m/a": [1.5, 2.5]}))
    assert load_pricing(tmp_path) == ({"m/a": (1.5, 2.5)}, [])


def test_per_instance_rows_carry_scores_and_cost(instance):
    acc = _score(instance)
    row = next(r for r in acc.per_instance if r["arm"] == "seat-1" and r["mode"] == "on")
    assert set(row) == {"instance_id", "arm", "mode", "findings", "matched_findings", "refs",
                        "matched_refs", "dropped", "cost_usd", "seconds"}
    assert row["instance_id"] == "o__r-1@abc" and row["refs"] == 1


def test_instance_without_line_anchored_refs_is_excluded_everywhere(instance):
    """A PR whose only reference comment is file-level can neither be matched
    nor missed, so scoring it would only dilute precision."""
    acc = _score(instance, row=_row(comments=(ReferenceComment("a.py", None, None, "t"),)))
    assert acc.excluded["no_anchored_refs"] == 1
    assert acc.instances == 0 and acc.per_arm == {} and acc.per_instance == []
    assert acc.cost == pytest.approx(0.25)


def test_arm_build_error_is_counted_and_noted(instance, tmp_path_factory):
    empty = tmp_path_factory.mktemp("empty-prompts")
    acc = _score(instance, prompts=empty)
    assert acc.excluded["error"] >= 1
    assert any("full" in key for key in acc.arm_notes)
    assert all(any(k.endswith(m) for m in ("|on", "|off")) for k in acc.arm_notes)


def test_replay_miss_is_counted_as_an_exclusion(instance):
    for path in (instance / "calls").glob("*-seat.json"):
        if json.loads(path.read_text())["model"] == "m/a":
            path.unlink()
    acc = _score(instance)
    assert acc.excluded["replay_miss"] == 4  # seat-1 plus the three ensemble arms
    assert ("seat-1", "on") not in acc.per_arm and ("seat-2", "on") in acc.per_arm


def _seat_aggregates(mode, precisions):
    acc = Accumulator()
    for arm, precision in zip(("seat-1", "seat-2", "seat-3"), precisions, strict=True):
        found = 4
        acc.per_arm[(arm, mode)] = [InstanceScore(found, round(precision * found), 0, 2, 1, 0)]
        acc.fabricated[(arm, mode)] = 0
        acc.arm_cost[(arm, mode)] = 1.0
        acc.arm_seconds[(arm, mode)] = 10.0
    return acc


def test_single_arm_is_the_mean_of_the_three_seat_arms():
    acc = _seat_aggregates("on", (0.25, 0.5, 0.75))
    acc.per_arm[("seat-3", "on")].append(InstanceScore(0, 0, 0, 0, 0, 0))
    single = next(a for a in build_aggregates(acc) if a.arm == "single")
    assert single.mode == "on"
    assert single.precision == pytest.approx(0.5)
    assert single.cost_usd_per_pr == pytest.approx((1.0 + 1.0 + 0.5) / 3)
    assert single.instances == 1  # the smallest of the three seat arms


def test_single_arm_does_not_measure_fabrication_on_the_off_row():
    acc = _seat_aggregates("off", (0.25, 0.5, 0.75))
    single = next(a for a in build_aggregates(acc) if a.arm == "single")
    assert single.fabrication_rate is None


def test_drifted_handles_line_less_findings():
    live = {"introduces": [{"file": "a.py", "line": None, "claim": "c1"},
                           {"file": "a.py", "line": 5, "claim": "c2"}]}
    replayed = Verdict(introduces=(
        Finding(file="a.py", line=None, severity=Severity.HIGH, claim="c1", evidence="e"),
        Finding(file="a.py", line=5, severity=Severity.HIGH, claim="c2", evidence="e"),
    ), fixes=(), confidence=0.9)
    assert drifted(live, replayed) == 0
    assert drifted(None, replayed) == 1
