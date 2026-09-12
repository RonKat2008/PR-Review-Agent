from __future__ import annotations

import pytest

from prime_pr_review.evaluation.recording import (
    Recorder,
    ReplayMiss,
    is_done,
    mark_done,
    recording_model_fn,
    recording_reviewer,
    replay_model_fn,
    replay_reviewer,
)

from .conftest import make_pr


def test_recorder_persists_calls_in_sequence(tmp_path):
    rec = Recorder(tmp_path)
    fn = recording_model_fn("skeptic", "m/a", lambda p: "R:" + p, rec)
    assert fn("one") == "R:one" and fn("two") == "R:two"
    calls = Recorder(tmp_path).calls("skeptic")
    assert [c.seq for c in calls] == [0, 1] and calls[1].response == "R:two"
    assert calls[0].prompt_sha256 != calls[1].prompt_sha256


def test_recording_reviewer_assigns_seats_round_robin(tmp_path):
    (tmp_path / "open_pr.md").write_text("T", encoding="utf-8")
    rec = Recorder(tmp_path)
    reviewer = recording_reviewer(["m/a", "m/b", "m/c"], lambda model: (lambda p: model), rec, tmp_path)
    pr = make_pr()
    assert [reviewer(pr, "D", "open") for _ in range(3)] == ["m/a", "m/b", "m/c"]
    assert [c.model for c in rec.calls("seat")] == ["m/a", "m/b", "m/c"]
    assert rec.calls("seat")[0].prompt.startswith("T")


def test_replay_model_fn_hits_by_prompt_hash_and_misses_loudly(tmp_path):
    rec = Recorder(tmp_path)
    recording_model_fn("judge", "m/a", lambda p: "J", rec)("the prompt")
    replay = replay_model_fn(Recorder(tmp_path), "judge")
    assert replay("the prompt") == "J"
    with pytest.raises(ReplayMiss):
        replay("a different prompt")


def test_replay_reviewer_returns_seats_in_order(tmp_path):
    (tmp_path / "open_pr.md").write_text("T", encoding="utf-8")
    rec = Recorder(tmp_path)
    reviewer = recording_reviewer(["m/a", "m/b"], lambda model: (lambda p: "out-" + model), rec, tmp_path)
    pr = make_pr()
    [reviewer(pr, "D", "open") for _ in range(2)]
    replay = replay_reviewer(Recorder(tmp_path))
    assert [replay(pr, "D", "open"), replay(pr, "D", "open")] == ["out-m/a", "out-m/b"]
    with pytest.raises(ReplayMiss):
        replay(pr, "D", "open")


def test_done_marker(tmp_path):
    assert not is_done(tmp_path)
    mark_done(tmp_path)
    assert is_done(tmp_path)
