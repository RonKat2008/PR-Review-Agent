from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import httpx

from prime_pr_review.evaluation.corpus import parse_rows
from prime_pr_review.evaluation.recording import (
    Recorder,
    is_done,
    mark_done,
    recording_model_fn,
)
from prime_pr_review.providers import (
    BASE_URL,
    MAX_COMPLETION_TOKENS,
    CostMeter,
    MeterBox,
    ProviderError,
    Usage,
)

from .conftest import make_config

FIXTURE = Path("tests/fixtures/swecare_rows.json")
PROMPTS = Path("skills/pr-review/prompts")
CLEAN = '{"introduces":[],"fixes":[],"confidence":0.9}'
MINI_DIFF = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,2 +1,2 @@\n-x\n+y\n+z\n"


def _load():
    spec = importlib.util.spec_from_file_location("eval_swecare", Path("scripts/eval_swecare.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _fake_provider(mod, response=CLEAN):
    box = MeterBox(CostMeter(cap_usd=10, pricing={"m": (1.0, 1.0)}))
    return mod.Provider(make_reviewer_model_fn=lambda model: (lambda p: response),
                        aux_fn=lambda p: '{"summary":"s","claims":[]}',
                        skeptic_fn=lambda p: '{"refuted": false, "reasoning": "r"}',
                        judge_fn=lambda p: '{"clusters": []}', box=box, seat_models=("m", "m", "m"))


def _fail(a, s):
    raise RuntimeError("no network in tests")


def _raising_provider(mod):
    box = MeterBox(CostMeter(cap_usd=10, pricing={"m": (1.0, 1.0)}))

    def boom(_prompt):
        raise RuntimeError("seat exploded")

    return mod.Provider(make_reviewer_model_fn=lambda model: boom,
                        aux_fn=lambda p: '{"summary":"s","claims":[]}',
                        skeptic_fn=lambda p: '{"refuted": false, "reasoning": "r"}',
                        judge_fn=lambda p: '{"clusters": []}', box=box, seat_models=("m", "m", "m"))


def _mini_row_payload(idx):
    return {"row_idx": idx, "row": {
        "instance_id": f"fake-{idx}", "repo": "o/r", "language": "Python", "pull_number": idx,
        "title": "t", "body": "", "base_commit": "base",
        "commit_to_review": {"head_commit": "head", "head_commit_message": "m", "patch_to_review": MINI_DIFF},
        "reference_review_comments": [{"path": "a.py", "line": 1, "original_line": 1,
                                       "start_line": None, "original_start_line": None,
                                       "text": "t", "diff_hunk": ""}],
    }, "truncated_cells": []}


def test_eval_config_locks_down_and_targets_row():
    mod = _load()
    (row, *_) = parse_rows(json.loads(FIXTURE.read_text()))
    cfg = mod.eval_config(make_config(), row)
    assert cfg.review.dry_run and cfg.repo.read_only and not cfg.sinks.pr_comment
    assert (cfg.repo.owner, cfg.repo.name) == tuple(row.repo.split("/"))
    assert cfg.review.ensemble_size == 3 and cfg.review.min_agreement == 1 and cfg.review.repo_root == ""


def test_build_provider_forwards_seat_options_for_gpt_5_4_mini():
    mod = _load()
    seen = []

    def handler(req):
        seen.append(json.loads(req.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "v"}}],
                                         "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url=BASE_URL)
    pricing = {m: (1.0, 1.0) for m in mod.SEATS}
    box = MeterBox(CostMeter(cap_usd=10, pricing=pricing))
    provider = mod.build_provider(client, box)

    provider.make_reviewer_model_fn("openai/gpt-5.4-mini")("p")
    assert seen[-1]["reasoning"] == {"effort": "medium"}

    provider.make_reviewer_model_fn("z-ai/glm-5.2")("p")
    assert "reasoning" not in seen[-1]


def test_build_provider_caps_deepseek_seat_skeptic_and_judge_at_64k():
    mod = _load()
    assert mod.SEAT_OPTIONS["deepseek/deepseek-v4-pro"] == {"max_tokens": 64_000, "_timeout": 900}
    assert mod.SEAT_OPTIONS["openai/gpt-5.4-mini"] == {"reasoning": {"effort": "medium"}}
    seen = []
    timeouts = []

    def handler(req):
        seen.append(json.loads(req.content))
        timeouts.append(req.extensions["timeout"])
        return httpx.Response(200, json={"choices": [{"message": {"content": "v"}}],
                                         "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url=BASE_URL)
    pricing = {m: (1.0, 1.0) for m in mod.SEATS}
    box = MeterBox(CostMeter(cap_usd=10, pricing=pricing))
    provider = mod.build_provider(client, box)

    provider.make_reviewer_model_fn("deepseek/deepseek-v4-pro")("p")
    assert seen[-1]["max_tokens"] == 64_000
    assert "_timeout" not in seen[-1]
    assert timeouts[-1]["read"] == 900

    provider.make_reviewer_model_fn("z-ai/glm-5.2")("p")
    assert seen[-1]["max_tokens"] == MAX_COMPLETION_TOKENS
    assert timeouts[-1]["read"] != 900

    provider.skeptic_fn("p")
    assert seen[-1]["max_tokens"] == 64_000
    assert timeouts[-1]["read"] == 900

    provider.judge_fn("p")
    assert seen[-1]["max_tokens"] == 64_000
    assert timeouts[-1]["read"] == 900


def test_run_one_writes_layout_and_is_resumable(tmp_path):
    mod = _load()
    (row, *_) = parse_rows(json.loads(FIXTURE.read_text()))
    run_dir = tmp_path / "run"
    out = mod.run_one(row, mod.eval_config(make_config(), row), _fake_provider(mod), run_dir, PROMPTS, _fail)
    inst = run_dir / row.instance_id
    assert (inst / "outcome.json").is_file() and (inst / "done").is_file() and (inst / "calls").is_dir()
    assert out["error"] is None and len(list((inst / "calls").glob("*-seat.json"))) == 3
    again = mod.run_one(row, mod.eval_config(make_config(), row), _fake_provider(mod), run_dir, PROMPTS, _fail)
    assert again["skipped"] is True


def test_score_and_report_on_fake_run(tmp_path):
    mod = _load()
    raw = json.loads(FIXTURE.read_text())["rows"][:2]
    rows = parse_rows({"rows": raw})
    run_dir = tmp_path / "run"
    for row in rows:
        mod.run_one(row, mod.eval_config(make_config(), row), _fake_provider(mod), run_dir, PROMPTS, _fail)
    (run_dir / "rows.json").write_text(json.dumps({"rows": raw}))
    (run_dir / "config.json").write_text(
        json.dumps({"count": 2, "seed": 0, "run_id": "run", "seats": ["m", "m", "m"]}))
    summary = mod.score_run(run_dir, PROMPTS)
    assert summary["instances"] == 2 and any(a["arm"] == "full" for a in summary["aggregates"])
    path = mod.write_report(run_dir, tmp_path / "docs")
    assert path.read_text().startswith("# SWE-CARE ablation")


def test_run_one_clears_stale_recordings_without_done_marker(tmp_path):
    mod = _load()
    (row, *_) = parse_rows(json.loads(FIXTURE.read_text()))
    run_dir = tmp_path / "run"
    inst = run_dir / row.instance_id
    (inst / "calls").mkdir(parents=True)
    (inst / "calls" / "000-seat.json").write_text("junk")
    mod.run_one(row, mod.eval_config(make_config(), row), _fake_provider(mod), run_dir, PROMPTS, _fail)
    assert len(list((inst / "calls").glob("*-seat.json"))) == 3


def test_run_one_does_not_mark_done_on_error(tmp_path):
    mod = _load()
    (row, *_) = parse_rows(json.loads(FIXTURE.read_text()))
    run_dir = tmp_path / "run"
    provider = _raising_provider(mod)
    result = mod.run_one(row, mod.eval_config(make_config(), row), provider, run_dir, PROMPTS, _fail)
    inst = run_dir / row.instance_id
    assert result["error"] is not None
    assert not (inst / "done").is_file()
    again = mod.run_one(row, mod.eval_config(make_config(), row), provider, run_dir, PROMPTS, _fail)
    assert again["skipped"] is False


def test_fabrication_rate_only_applies_to_on_row(tmp_path):
    mod = _load()
    row = parse_rows({"rows": [_mini_row_payload(0)]})[0]
    response = json.dumps({"introduces": [
        {"file": "a.py", "line": 1, "severity": "HIGH", "claim": "real bug", "evidence": "e"},
        {"file": "nope.py", "line": 1, "severity": "HIGH", "claim": "fake bug", "evidence": "e"},
    ], "fixes": [], "confidence": 0.9})
    run_dir = tmp_path / "run"
    mod.run_one(row, mod.eval_config(make_config(), row), _fake_provider(mod, response=response),
               run_dir, PROMPTS, _fail)
    (run_dir / "rows.json").write_text(json.dumps({"rows": [_mini_row_payload(0)]}))
    (run_dir / "config.json").write_text(
        json.dumps({"count": 1, "seed": 0, "run_id": "run", "seats": ["m", "m", "m"]}))
    summary = mod.score_run(run_dir, PROMPTS)
    on_row = next(a for a in summary["aggregates"] if a["arm"] == "ensemble" and a["mode"] == "on")
    off_row = next(a for a in summary["aggregates"] if a["arm"] == "ensemble" and a["mode"] == "off")
    assert on_row["fabrication_rate"] == 0.5
    assert off_row["fabrication_rate"] is None


def _repair_provider(mod, pricing=None):
    pricing = pricing or {m: (1.0, 1.0) for m in ("m/a", "m/b", "m/c")}
    box = MeterBox(CostMeter(cap_usd=10, pricing=pricing))
    return mod.Provider(make_reviewer_model_fn=lambda model: (lambda p: f"resp-{model}"),
                        aux_fn=lambda p: "", skeptic_fn=lambda p: "", judge_fn=lambda p: "",
                        box=box, seat_models=("m/a", "m/b", "m/c"))


def test_repair_run_reissues_only_the_missing_seat(tmp_path):
    mod = _load()
    run_dir = tmp_path / "run"
    inst = run_dir / "inst-0"
    rec = Recorder(inst)
    recording_model_fn("seat", "m/b", lambda p, o="out-b": o, rec)("shared prompt")
    recording_model_fn("seat", "m/c", lambda p, o="out-c": o, rec)("shared prompt")
    mark_done(inst)

    counts = mod.repair_run(run_dir, _repair_provider(mod))

    seat_files = sorted((inst / "calls").glob("*-seat.json"))
    assert len(seat_files) == 3
    new_files = [p for p in seat_files if json.loads(p.read_text())["model"] == "m/a"]
    assert len(new_files) == 1
    assert json.loads(new_files[0].read_text())["prompt"] == "shared prompt"
    assert is_done(inst)
    assert counts == {"instances": 1, "repaired": 1, "calls": 1, "unrepairable": 0, "failed": 0}


def test_repair_run_leaves_a_fully_recorded_instance_untouched(tmp_path):
    mod = _load()
    run_dir = tmp_path / "run"
    inst = run_dir / "inst-0"
    rec = Recorder(inst)
    for model in ("m/a", "m/b", "m/c"):
        recording_model_fn("seat", model, lambda p, o=f"out-{model}": o, rec)("shared prompt")
    mark_done(inst)

    counts = mod.repair_run(run_dir, _repair_provider(mod))

    assert len(list((inst / "calls").glob("*-seat.json"))) == 3
    assert counts == {"instances": 1, "repaired": 0, "calls": 0, "unrepairable": 0, "failed": 0}


def test_repair_run_counts_an_instance_with_no_seat_files_as_unrepairable(tmp_path):
    mod = _load()
    run_dir = tmp_path / "run"
    inst = run_dir / "inst-0"
    inst.mkdir(parents=True)
    mark_done(inst)

    counts = mod.repair_run(run_dir, _repair_provider(mod))

    assert counts == {"instances": 1, "repaired": 0, "calls": 0, "unrepairable": 1, "failed": 0}
    assert not list((inst / "calls").glob("*.json"))


def test_repair_run_records_a_failed_seat_and_continues_to_the_next(tmp_path):
    mod = _load()
    run_dir = tmp_path / "run"
    inst = run_dir / "inst-0"
    rec = Recorder(inst)
    recording_model_fn("seat", "m/b", lambda p, o="out-b": o, rec)("shared prompt")
    mark_done(inst)

    def make_model_fn(model):
        if model == "m/a":
            def boom(_prompt):
                raise ProviderError("empty content (finish_reason=length)")
            return boom
        return lambda p, o=f"resp-{model}": o

    pricing = {m: (1.0, 1.0) for m in ("m/a", "m/b", "m/c")}
    box = MeterBox(CostMeter(cap_usd=10, pricing=pricing))
    provider = mod.Provider(make_reviewer_model_fn=make_model_fn,
                            aux_fn=lambda p: "", skeptic_fn=lambda p: "", judge_fn=lambda p: "",
                            box=box, seat_models=("m/a", "m/b", "m/c"))

    counts = mod.repair_run(run_dir, provider)

    seat_files = sorted((inst / "calls").glob("*-seat.json"))
    recorded_models = {json.loads(p.read_text())["model"] for p in seat_files}
    assert recorded_models == {"m/b", "m/c"}
    assert counts == {"instances": 1, "repaired": 1, "calls": 1, "unrepairable": 0, "failed": 1}


def test_run_repair_flag_skips_the_normal_loop_and_never_calls_run_one(tmp_path, monkeypatch):
    mod = _load()
    _monkeypatch_run(mod, tmp_path, monkeypatch)

    def boom(*_a, **_k):
        raise AssertionError("run_one must not be called in --repair mode")

    monkeypatch.setattr(mod, "run_one", boom)
    exit_code = mod.main(["run", "--run-id", "t", "--repair"])
    assert exit_code == 0


def test_cmd_run_stops_on_budget_and_persists_meter(tmp_path, monkeypatch):
    mod = _load()
    pages = {0: {"rows": [_mini_row_payload(0), _mini_row_payload(1)], "num_rows_total": 2}}
    monkeypatch.setattr(mod, "_fetch_page",
                        lambda offset, length: pages.get(offset, {"rows": [], "num_rows_total": 2}))
    monkeypatch.setattr(mod, "resolve_prime_key", lambda: "k")
    monkeypatch.setattr(mod, "make_client", lambda key: object())
    monkeypatch.setattr(mod, "fetch_pricing",
                        lambda client, models: {m: (1.0, 1.0) for m in [*mod.SEATS, mod.AUX_MODEL]})

    def fake_build_provider(client, box):
        provider = _fake_provider(mod)
        provider.box.meter = mod.CostMeter(cap_usd=0.0, pricing=provider.box.meter.pricing, spent_usd=1.0)
        return provider

    monkeypatch.setattr(mod, "build_provider", fake_build_provider)
    monkeypatch.setattr(mod, "EVAL_ROOT", tmp_path)
    monkeypatch.setattr(mod, "load_config", lambda path: make_config())

    exit_code = mod.main(["run", "--count", "2", "--seed", "0", "--cap-usd", "0", "--run-id", "t"])
    assert exit_code == 3
    assert (tmp_path / "runs" / "t" / "meter.json").is_file()


def _monkeypatch_run(mod, tmp_path, monkeypatch, provider_factory=None):
    pages = {0: {"rows": [_mini_row_payload(0), _mini_row_payload(1)], "num_rows_total": 2}}
    monkeypatch.setattr(mod, "_fetch_page",
                        lambda offset, length: pages.get(offset, {"rows": [], "num_rows_total": 2}))
    monkeypatch.setattr(mod, "resolve_prime_key", lambda: "k")
    monkeypatch.setattr(mod, "make_client", lambda key: object())
    monkeypatch.setattr(mod, "fetch_pricing",
                        lambda client, models: {m: (1.0, 2.0) for m in models})
    monkeypatch.setattr(mod, "build_provider",
                        provider_factory or (lambda client, box: _fake_provider(mod)))
    monkeypatch.setattr(mod, "EVAL_ROOT", tmp_path)
    monkeypatch.setattr(mod, "load_config", lambda path: make_config())


def test_run_refuses_to_reuse_a_run_dir_without_resume(tmp_path, capsys):
    mod = _load()
    mod.EVAL_ROOT = tmp_path
    (tmp_path / "runs" / "t" / "fake-0" / "calls").mkdir(parents=True)
    assert mod.main(["run", "--count", "1", "--run-id", "t"]) == 2
    assert "--resume" in capsys.readouterr().out


def test_run_resumes_finished_instances_with_the_flag(tmp_path, monkeypatch):
    mod = _load()
    _monkeypatch_run(mod, tmp_path, monkeypatch)
    inst = tmp_path / "runs" / "t" / "fake-0"
    (inst / "calls").mkdir(parents=True)
    (inst / "done").write_text("ok")
    assert mod.main(["run", "--count", "2", "--cap-usd", "10", "--run-id", "t", "--resume"]) == 0
    assert not list((inst / "calls").glob("*.json"))  # untouched, not re-reviewed


def test_run_persists_the_pricing_snapshot_and_filter_counts(tmp_path, monkeypatch):
    mod = _load()
    _monkeypatch_run(mod, tmp_path, monkeypatch)
    assert mod.main(["run", "--count", "2", "--cap-usd", "10", "--run-id", "t"]) == 0
    run_dir = tmp_path / "runs" / "t"
    pricing = json.loads((run_dir / "pricing.json").read_text())
    assert pricing[mod.SEATS[0]] == [1.0, 2.0]
    filters = json.loads((run_dir / "config.json").read_text())["filters"]
    assert filters["total"] == 2 and filters["selected"] == 2


def test_run_records_seat_token_usage_from_the_meter_box(tmp_path):
    mod = _load()
    (row, *_) = parse_rows(json.loads(FIXTURE.read_text()))
    provider = _fake_provider(mod)

    def metered(_model):
        def fn(prompt):
            provider.box.last_usage = Usage(9, 4)
            return CLEAN
        return fn

    provider = mod.replace(provider, make_reviewer_model_fn=metered)
    run_dir = tmp_path / "run"
    mod.run_one(row, mod.eval_config(make_config(), row), provider, run_dir, PROMPTS, _fail)
    calls = [json.loads(p.read_text()) for p in sorted((run_dir / row.instance_id / "calls").glob("*-seat.json"))]
    assert [(c["prompt_tokens"], c["completion_tokens"]) for c in calls] == [(9, 4)] * 3


def test_summary_carries_pricing_per_instance_and_exclusions(tmp_path):
    mod = _load()
    raw = json.loads(FIXTURE.read_text())["rows"][:2]
    rows = parse_rows({"rows": raw})
    run_dir = tmp_path / "run"
    for row in rows:
        mod.run_one(row, mod.eval_config(make_config(), row), _fake_provider(mod), run_dir, PROMPTS, _fail)
    (run_dir / "rows.json").write_text(json.dumps({"rows": raw}))
    (run_dir / "config.json").write_text(
        json.dumps({"count": 2, "seed": 0, "run_id": "run", "seats": ["m", "m", "m"]}))
    (run_dir / "pricing.json").write_text(json.dumps({"m": [1.0, 1.0]}))
    summary = mod.score_run(run_dir, PROMPTS)
    assert summary["pricing"] == {"m": [1.0, 1.0]}
    assert set(summary["excluded"]) == {"replay_miss", "error", "no_anchored_refs"}
    assert summary["per_instance"] and set(summary["per_instance"][0]) == {
        "instance_id", "arm", "mode", "findings", "matched_findings", "refs", "matched_refs",
        "dropped", "cost_usd", "seconds"}
    assert any(a["arm"] == "single" for a in summary["aggregates"])
    md = mod.write_report(run_dir, tmp_path / "docs").read_text()
    assert f"Instances: {summary['instances']}" in md


def _record_seats(inst, responses):
    rec = Recorder(inst)
    for response in responses:
        recording_model_fn("seat", "m", lambda p, r=response: r, rec)("shared prompt")
    mark_done(inst)


def _two_finding_seats():
    """Two same-file findings in different LINE_BUCKETs, so the rebuilt verdict
    needs a judge call, plus a clean third seat."""
    def verdict(*findings):
        return json.dumps({"introduces": list(findings), "fixes": [], "confidence": 0.9})

    def finding(line, claim):
        return {"file": "a.py", "line": line, "severity": "HIGH", "claim": claim, "evidence": "e"}

    return [verdict(finding(1, "bug one")), verdict(finding(7, "bug one again")), verdict()]


def test_run_repair_passes_flag_reissues_judge_and_skeptic_without_run_one(tmp_path, monkeypatch):
    mod = _load()
    _monkeypatch_run(mod, tmp_path, monkeypatch)
    run_dir = tmp_path / "runs" / "t"
    run_dir.mkdir(parents=True)
    (run_dir / "rows.json").write_text(json.dumps({"rows": [_mini_row_payload(0)]}))
    (run_dir / "config.json").write_text(
        json.dumps({"count": 1, "seed": 0, "run_id": "t", "seats": ["m", "m", "m"]}))
    _record_seats(run_dir / "fake-0", _two_finding_seats())

    def boom(*_a, **_k):
        raise AssertionError("run_one must not be called in --repair-passes mode")

    monkeypatch.setattr(mod, "run_one", boom)
    assert mod.main(["run", "--run-id", "t", "--repair-passes"]) == 0

    calls = run_dir / "fake-0" / "calls"
    assert len(list(calls.glob("*-judge.json"))) == 1
    assert list(calls.glob("*-skeptic.json"))
    assert (run_dir / "meter.json").is_file()
