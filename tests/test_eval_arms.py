from __future__ import annotations

import base64
import json
from dataclasses import replace
from pathlib import Path

import pytest

from prime_pr_review.ensemble import ensemble_review_detailed
from prime_pr_review.evaluation.arms import (
    ARMS,
    HEAD_FILES,
    apply_citations,
    arm_calls,
    build_arm,
)
from prime_pr_review.evaluation.recording import (
    Recorder,
    recording_model_fn,
    recording_reviewer,
)
from prime_pr_review.evaluation.runner import HeadFileStore
from prime_pr_review.refute import build_refute_prompt
from prime_pr_review.review import Finding, Severity

from .conftest import make_pr

PROMPTS = Path("skills/pr-review/prompts")
DIFF = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,2 +1,2 @@\n-x\n+y\n+z\n"


def finding(line, claim):
    return {"file": "a.py", "line": line, "severity": "HIGH", "claim": claim, "evidence": "e"}


def verdict(*fs):
    return json.dumps({"introduces": list(fs), "fixes": [], "confidence": 0.9})


SEATS = [verdict(finding(1, "bug one")), verdict(finding(7, "bug one again")), verdict()]
SEAT_MODELS = ("m/a", "m/b", "m/c")


def build(arm, d, prompts=PROMPTS, seat_models=SEAT_MODELS):
    return build_arm(arm, d, make_pr(), DIFF, "open", prompts, seat_models)


def drop_seat(d, model):
    """Delete the recording of the seat that ran `model`, as if it had failed
    live and written nothing."""
    for path in (d / "calls").glob("*-seat.json"):
        if json.loads(path.read_text())["model"] == model:
            path.unlink()


@pytest.fixture
def recorded(tmp_path):
    rec = Recorder(tmp_path)
    outputs = iter(SEATS)
    reviewer = recording_reviewer(
        ["m/a", "m/b", "m/c"], lambda m: (lambda p: next(outputs)), rec, PROMPTS
    )
    judge = recording_model_fn("judge", "m/a", lambda p: '{"clusters": [[0, 1]]}', rec)
    live, _ = ensemble_review_detailed(
        make_pr(), DIFF, "open", reviewer, size=3, min_agreement=1,
        judge_fn=judge, prompts_dir=PROMPTS
    )
    skeptic = recording_model_fn("skeptic", "m/a", lambda p: '{"refuted": true, "reasoning": "no"}', rec)
    template = (PROMPTS / "refute.md").read_text(encoding="utf-8")
    for f in live.introduces:
        skeptic(build_refute_prompt(template, f, DIFF))
    return tmp_path, live


def test_all_arms_are_buildable(recorded):
    d, _ = recorded
    for arm in ARMS:
        r = build(arm, d)
        assert r.arm == arm and not r.replay_miss and r.verdict is not None, r.error


def test_seat_and_ensemble_arms_differ_in_grouping(recorded):
    d, _ = recorded
    assert len(build("seat-1", d).verdict.introduces) == 1
    assert len(build("ensemble", d).verdict.introduces) == 2
    assert len(build("ensemble+judge", d).verdict.introduces) == 1


def test_full_arm_marks_refuted_and_matches_live(recorded):
    d, live = recorded
    full = build("full", d).verdict
    assert all(f.refuted for f in full.introduces)
    assert [f.claim for f in full.introduces] == [f.claim for f in live.introduces]


def test_replay_miss_is_reported_not_raised(tmp_path):
    Recorder(tmp_path)  # no calls at all
    r = build("ensemble", tmp_path)
    assert r.replay_miss and r.verdict is None


def test_apply_citations_uses_recorded_head_files(recorded):
    d, live = recorded
    HeadFileStore(d / HEAD_FILES).put("a.py", base64.b64encode(b"y\nz\n").decode())
    bad = Finding(file="a.py", line=99, severity=Severity.LOW, claim="beyond", evidence="e")
    v = replace(live, introduces=(*live.introduces, bad))
    kept, dropped = apply_citations(v, DIFF, d, "o/r", "abc")
    assert dropped == 1 and all(f.line != 99 for f in kept.introduces)


def test_full_arm_reports_skeptic_replay_miss(recorded):
    d, _ = recorded
    for p in (d / "calls").glob("*-skeptic.json"):
        p.unlink()
    r = build("full", d)
    assert r.replay_miss and r.verdict is None


def test_build_arm_reports_missing_prompt_file_as_error(recorded, tmp_path_factory):
    d, _ = recorded
    empty_prompts = tmp_path_factory.mktemp("empty")
    r = build("full", d, prompts=empty_prompts)
    assert r.verdict is None and not r.replay_miss and r.error


def test_full_arm_validates_citations_before_refuting(recorded):
    """A dropped (fabricated-file) finding must never reach the skeptic: its
    prompt was never recorded, so refuting it first would raise ReplayMiss and
    lose the whole instance. Replace seat 1's recorded response with a single
    fabricated-file finding (on a distinct file, so no judge merge is needed
    and the surviving `a.py` finding's skeptic prompt is unchanged from what
    the fixture already recorded)."""
    d, _ = recorded
    seat_files = sorted((d / "calls").glob("*-seat.json"))
    payload = json.loads(seat_files[1].read_text())
    payload["response"] = json.dumps({
        "introduces": [{"file": "nope.py", "line": 1, "severity": "HIGH",
                        "claim": "fabricated", "evidence": "e"}],
        "fixes": [], "confidence": 0.9,
    })
    seat_files[1].write_text(json.dumps(payload))
    r = build("full", d)
    assert r.verdict is not None and not r.replay_miss and r.dropped == 1


def test_seat_arms_are_attributed_by_model_when_a_seat_failed(recorded):
    """Seat `m/a` failed live and recorded nothing. The later seats must keep
    their own identity instead of sliding up into the empty slot."""
    d, _ = recorded
    drop_seat(d, "m/a")
    first = build("seat-1", d)
    assert first.replay_miss and first.verdict is None
    assert [f.line for f in build("seat-2", d).verdict.introduces] == [7]
    assert build("seat-3", d).verdict.introduces == ()


def test_ensemble_arms_report_a_missing_seat_recording(recorded):
    """`_collect_runs` swallows every reviewer exception, so a replay miss on
    one seat would otherwise render as a clean two-seat ensemble."""
    d, _ = recorded
    drop_seat(d, "m/a")
    for arm in ("ensemble", "ensemble+judge", "full"):
        r = build(arm, d)
        assert r.replay_miss and r.verdict is None, arm


def test_arm_calls_charge_aux_to_every_arm_and_stage_calls_to_their_arm(recorded):
    d, _ = recorded
    rec = Recorder(d)
    recording_model_fn("aux", "m/x", lambda p: "intent", rec)("p")
    roles = {arm: sorted(c.role for c in arm_calls(rec, arm, SEAT_MODELS)) for arm in ARMS}
    assert roles["seat-1"] == ["aux", "seat"]
    assert roles["ensemble"] == ["aux", "seat", "seat", "seat"]
    assert roles["ensemble+judge"] == ["aux", "judge", "seat", "seat", "seat"]
    assert roles["full"] == ["aux", "judge", "seat", "seat", "seat", "skeptic"]
    assert [c.model for c in arm_calls(rec, "seat-2", SEAT_MODELS) if c.role == "seat"] == ["m/b"]


def test_arm_calls_skip_a_seat_with_no_recording(recorded):
    d, _ = recorded
    drop_seat(d, "m/a")
    assert arm_calls(Recorder(d), "seat-1", SEAT_MODELS) == ()
    assert len(arm_calls(Recorder(d), "ensemble", SEAT_MODELS)) == 2


def test_unparseable_seats_are_an_error_not_a_replay_miss(recorded):
    """Every seat recorded, none of them usable: the recordings are present, so
    this is a bad arm build, not a gap in the corpus."""
    d, _ = recorded
    for path in (d / "calls").glob("*-seat.json"):
        payload = json.loads(path.read_text())
        payload["response"] = "not a verdict at all"
        path.write_text(json.dumps(payload))
    r = build("ensemble", d)
    assert r.verdict is None and not r.replay_miss and r.error
