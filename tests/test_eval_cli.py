from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import httpx

from prime_pr_review.evaluation.corpus import parse_rows
from prime_pr_review.providers import BASE_URL, CostMeter, MeterBox
from prime_pr_review.review import Finding, Severity, Verdict

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
    (run_dir / "config.json").write_text(json.dumps({"count": 2, "seed": 0, "run_id": "run"}))
    summary = mod.score_run(run_dir, PROMPTS)
    assert summary["instances"] == 2 and any(a["arm"] == "full" for a in summary["aggregates"])
    path = mod.write_report(run_dir, tmp_path / "docs")
    assert path.read_text().startswith("# SWE-CARE ablation")


def test_drifted_handles_line_less_findings():
    mod = _load()
    live = {"introduces": [{"file": "a.py", "line": None, "claim": "c1"},
                           {"file": "a.py", "line": 5, "claim": "c2"}]}
    replayed = Verdict(introduces=(
        Finding(file="a.py", line=None, severity=Severity.HIGH, claim="c1", evidence="e"),
        Finding(file="a.py", line=5, severity=Severity.HIGH, claim="c2", evidence="e"),
    ), fixes=(), confidence=0.9)
    assert mod._drifted(live, replayed) == 0


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
    (run_dir / "config.json").write_text(json.dumps({"count": 1, "seed": 0, "run_id": "run"}))
    summary = mod.score_run(run_dir, PROMPTS)
    on_row = next(a for a in summary["aggregates"] if a["arm"] == "ensemble" and a["mode"] == "on")
    off_row = next(a for a in summary["aggregates"] if a["arm"] == "ensemble" and a["mode"] == "off")
    assert on_row["fabrication_rate"] == 0.5
    assert off_row["fabrication_rate"] == 0.0


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
