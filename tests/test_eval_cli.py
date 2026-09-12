from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from prime_pr_review.evaluation.corpus import parse_rows
from prime_pr_review.providers import CostMeter, MeterBox

from .conftest import make_config

FIXTURE = Path("tests/fixtures/swecare_rows.json")
PROMPTS = Path("skills/pr-review/prompts")
CLEAN = '{"introduces":[],"fixes":[],"confidence":0.9}'


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


def test_eval_config_locks_down_and_targets_row():
    mod = _load()
    (row, *_) = parse_rows(json.loads(FIXTURE.read_text()))
    cfg = mod.eval_config(make_config(), row)
    assert cfg.review.dry_run and cfg.repo.read_only and not cfg.sinks.pr_comment
    assert (cfg.repo.owner, cfg.repo.name) == tuple(row.repo.split("/"))
    assert cfg.review.ensemble_size == 3 and cfg.review.min_agreement == 1 and cfg.review.repo_root == ""


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
