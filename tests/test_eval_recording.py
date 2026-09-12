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
    seat_calls,
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


def test_replay_model_fn_serves_duplicate_prompts_in_order(tmp_path):
    rec = Recorder(tmp_path)
    fn = recording_model_fn("skeptic", "m/a", lambda p: "first", rec)
    fn("same prompt")
    fn2 = recording_model_fn("skeptic", "m/a", lambda p: "second", rec)
    fn2("same prompt")
    replay = replay_model_fn(Recorder(tmp_path), "skeptic")
    assert replay("same prompt") == "first"
    assert replay("same prompt") == "second"
    with pytest.raises(ReplayMiss):
        replay("same prompt")


def test_replay_reviewer_returns_seats_in_order(tmp_path):
    (tmp_path / "open_pr.md").write_text("T", encoding="utf-8")
    rec = Recorder(tmp_path)
    reviewer = recording_reviewer(["m/a", "m/b"], lambda model: (lambda p: "out-" + model), rec, tmp_path)
    pr = make_pr()
    [reviewer(pr, "D", "open") for _ in range(2)]
    replay = replay_reviewer(Recorder(tmp_path), ["m/a", "m/b"])
    assert [replay(pr, "D", "open"), replay(pr, "D", "open")] == ["out-m/a", "out-m/b"]
    with pytest.raises(ReplayMiss):
        replay(pr, "D", "open")


def test_replay_reviewer_serves_seats_by_model_not_recorded_position(tmp_path):
    """Seat 1 failed live, so nothing was recorded for it. Seat 2's response
    must still be served as seat 2, not promoted into seat 1's slot."""
    (tmp_path / "open_pr.md").write_text("T", encoding="utf-8")
    rec = Recorder(tmp_path)
    recording_model_fn("seat", "m/b", lambda p: "out-m/b", rec)("p")
    recording_model_fn("seat", "m/c", lambda p: "out-m/c", rec)("p")
    replay = replay_reviewer(Recorder(tmp_path), ["m/a", "m/b", "m/c"])
    pr = make_pr()
    with pytest.raises(ReplayMiss):
        replay(pr, "D", "open")
    assert [replay(pr, "D", "open"), replay(pr, "D", "open")] == ["out-m/b", "out-m/c"]


def test_seat_calls_indexes_by_model_and_repeats_positionally(tmp_path):
    rec = Recorder(tmp_path)
    for model, out in (("m/a", "one"), ("m/a", "two")):
        recording_model_fn("seat", model, lambda p, o=out: o, rec)("p")
    got = seat_calls(Recorder(tmp_path), ["m/a", "m/a", "m/b"])
    assert [c.response if c else None for c in got] == ["one", "two", None]


def test_recording_model_fn_records_usage_from_source(tmp_path):
    rec = Recorder(tmp_path)
    fn = recording_model_fn("seat", "m/a", lambda p: "R", rec, usage_source=lambda: (13, 7))
    fn("p")
    (call,) = Recorder(tmp_path).calls("seat")
    assert (call.prompt_tokens, call.completion_tokens) == (13, 7)


def test_recording_reviewer_records_seat_usage(tmp_path):
    (tmp_path / "open_pr.md").write_text("T", encoding="utf-8")
    rec = Recorder(tmp_path)
    reviewer = recording_reviewer(["m/a"], lambda model: (lambda p: "x"), rec, tmp_path,
                                  usage_source=lambda: (3, 4))
    reviewer(make_pr(), "D", "open")
    assert [(c.prompt_tokens, c.completion_tokens) for c in rec.calls("seat")] == [(3, 4)]


def test_done_marker(tmp_path):
    assert not is_done(tmp_path)
    mark_done(tmp_path)
    assert is_done(tmp_path)
