"""`repair_passes`: re-issue the judge/skeptic calls a rebuilt verdict needs.

After `--repair` adds a missing seat recording, the offline `ensemble+judge`
and `full` arms rebuild a DIFFERENT verdict than the live run produced, so the
judge (and skeptic) prompts no longer match anything recorded. These tests pin
that the missing calls are re-issued live, recorded, and replay cleanly after.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from prime_pr_review.evaluation.arms import build_arm
from prime_pr_review.evaluation.recording import Recorder, mark_done, recording_model_fn
from prime_pr_review.evaluation.repair import repair_passes
from prime_pr_review.providers import BudgetExceeded, CostMeter, MeterBox, ProviderError

from .conftest import make_pr

PROMPTS = Path("skills/pr-review/prompts")
DIFF = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,2 +1,2 @@\n-x\n+y\n+z\n"
SEAT_MODELS = ("m/a", "m/b", "m/c")
JUDGE_RESPONSE = '{"clusters": [[0, 1]]}'
SKEPTIC_RESPONSE = '{"refuted": false, "reasoning": "r"}'


def _verdict(*findings):
    return json.dumps({"introduces": list(findings), "fixes": [], "confidence": 0.9})


def _finding(line, claim):
    return {"file": "a.py", "line": line, "severity": "HIGH", "claim": claim, "evidence": "e"}


# Two same-file findings in different LINE_BUCKETs: the judge is consulted only
# when two groups share a file, so this is the shape that needs a judge call.
SEAT_RESPONSES = {
    "m/a": _verdict(_finding(1, "bug one")),
    "m/b": _verdict(_finding(7, "bug one again")),
    "m/c": _verdict(),
}


@dataclass
class FakeProvider:
    """Structurally what `repair_passes` needs of `eval_swecare.Provider`."""

    box: MeterBox
    judge_fn: object
    skeptic_fn: object
    judge_model: str = "m/judge"
    skeptic_model: str = "m/skeptic"
    seat_models: tuple[str, ...] = SEAT_MODELS
    calls: list[str] = field(default_factory=list)


def _provider(judge=JUDGE_RESPONSE, skeptic=SKEPTIC_RESPONSE):
    pricing = {m: (1.0, 1.0) for m in (*SEAT_MODELS, "m/judge", "m/skeptic")}
    box = MeterBox(CostMeter(cap_usd=10, pricing=pricing))
    calls: list[str] = []

    def make(role, response):
        def fn(_prompt):
            calls.append(role)
            if isinstance(response, Exception):
                raise response
            return response
        return fn

    return FakeProvider(box=box, judge_fn=make("judge", judge),
                        skeptic_fn=make("skeptic", skeptic), calls=calls)


@pytest.fixture
def run_dir(tmp_path):
    """One `done` instance with all three seats recorded and no judge/skeptic
    -- exactly what an instance looks like after `--repair` re-issued a seat."""
    inst = tmp_path / "inst-0"
    rec = Recorder(inst)
    for model in SEAT_MODELS:
        recording_model_fn("seat", model, lambda p, m=model: SEAT_RESPONSES[m], rec)("shared prompt")
    mark_done(inst)
    return tmp_path


def _repair(run_dir, provider):
    return repair_passes(run_dir, provider, SEAT_MODELS, PROMPTS,
                         lambda _id: DIFF, lambda _id: make_pr())


def test_repair_passes_records_the_missing_judge_and_skeptic_calls(run_dir):
    provider = _provider()
    counts = _repair(run_dir, provider)

    calls = run_dir / "inst-0" / "calls"
    assert len(list(calls.glob("*-judge.json"))) == 1
    assert len(list(calls.glob("*-skeptic.json"))) == 1
    assert counts == {"instances": 1, "repaired": 1, "judge_calls": 1,
                      "skeptic_calls": 1, "failed": 0}


def test_repaired_instance_replays_without_live_fallbacks(run_dir):
    _repair(run_dir, _provider())
    built = build_arm("full", run_dir / "inst-0", make_pr(), DIFF, "open", PROMPTS, SEAT_MODELS)
    assert not built.replay_miss and built.verdict is not None, built.error


def test_a_cleanly_replaying_instance_makes_no_live_calls(run_dir):
    _repair(run_dir, _provider())
    second = _provider()
    counts = _repair(run_dir, second)
    assert second.calls == []
    assert counts == {"instances": 1, "repaired": 0, "judge_calls": 0,
                      "skeptic_calls": 0, "failed": 0}


def test_a_failed_live_call_is_counted_and_the_instance_skipped(run_dir):
    provider = _provider(judge=ProviderError("empty content (finish_reason=length)"))
    counts = _repair(run_dir, provider)
    assert counts["failed"] == 1 and counts["repaired"] == 0
    assert not list((run_dir / "inst-0" / "calls").glob("*-judge.json"))


def test_missing_run_dir_is_not_an_error(tmp_path):
    counts = _repair(tmp_path / "nope", _provider())
    assert counts == {"instances": 0, "repaired": 0, "judge_calls": 0,
                      "skeptic_calls": 0, "failed": 0}


def test_budget_exceeded_stops_the_repair(run_dir):
    """`_judge_merge` swallows the exception, so a cap crossed mid-instance
    would otherwise look like a judge that merely declined to merge."""
    provider = _provider(judge=BudgetExceeded("spent $11.00 > cap $10.00"))
    with pytest.raises(BudgetExceeded):
        _repair(run_dir, provider)


def test_an_instance_missing_from_rows_is_skipped(run_dir):
    def missing(instance_id):
        raise KeyError(instance_id)

    provider = _provider()
    counts = repair_passes(run_dir, provider, SEAT_MODELS, PROMPTS, missing, missing)
    assert counts["instances"] == 1 and counts["repaired"] == 0
    assert provider.calls == []


def test_a_failing_skeptic_is_called_once_not_once_per_finding(run_dir):
    """`refute_findings` calls the skeptic once per finding and fails open, so
    a failing provider would be billed for every remaining finding of an
    instance already being skipped."""
    provider = _provider(judge='{"clusters": []}',
                         skeptic=ProviderError("empty content (finish_reason=length)"))
    counts = _repair(run_dir, provider)
    assert counts["failed"] == 1
    assert provider.calls.count("skeptic") == 1
