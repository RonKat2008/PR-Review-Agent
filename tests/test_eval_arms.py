from __future__ import annotations

import base64
import json
from dataclasses import replace
from pathlib import Path

import pytest

from prime_pr_review.ensemble import ensemble_review_detailed
from prime_pr_review.evaluation.arms import ARMS, HEAD_FILES, apply_citations, build_arm
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
        r = build_arm(arm, d, make_pr(), DIFF, "open", PROMPTS)
        assert r.arm == arm and not r.replay_miss and r.verdict is not None, r.error


def test_seat_and_ensemble_arms_differ_in_grouping(recorded):
    d, _ = recorded
    assert len(build_arm("seat-1", d, make_pr(), DIFF, "open", PROMPTS).verdict.introduces) == 1
    assert len(build_arm("ensemble", d, make_pr(), DIFF, "open", PROMPTS).verdict.introduces) == 2
    assert len(build_arm("ensemble+judge", d, make_pr(), DIFF, "open", PROMPTS).verdict.introduces) == 1


def test_full_arm_marks_refuted_and_matches_live(recorded):
    d, live = recorded
    full = build_arm("full", d, make_pr(), DIFF, "open", PROMPTS).verdict
    assert all(f.refuted for f in full.introduces)
    assert [f.claim for f in full.introduces] == [f.claim for f in live.introduces]


def test_replay_miss_is_reported_not_raised(tmp_path):
    Recorder(tmp_path)  # no calls at all
    r = build_arm("ensemble", tmp_path, make_pr(), DIFF, "open", PROMPTS)
    assert r.replay_miss and r.verdict is None


def test_apply_citations_uses_recorded_head_files(recorded):
    d, live = recorded
    HeadFileStore(d / HEAD_FILES).put("a.py", base64.b64encode(b"y\nz\n").decode())
    bad = Finding(file="a.py", line=99, severity=Severity.LOW, claim="beyond", evidence="e")
    v = replace(live, introduces=(*live.introduces, bad))
    kept, dropped = apply_citations(v, DIFF, d, "o/r", "abc")
    assert dropped == 1 and all(f.line != 99 for f in kept.introduces)
