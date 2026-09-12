"""SWE-CARE ablation harness: `run` reviews sampled PRs live on Prime Inference
and records every model call; `score` rebuilds each ablation arm offline and
scores it; `report` renders docs/eval/<run_id>.md.
Spec: docs/superpowers/specs/2026-09-11-swecare-ablation-harness-design.md."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import httpx

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from prime_pr_review import github
from prime_pr_review.config import Config, load_config
from prime_pr_review.evaluation import arms as arms_mod
from prime_pr_review.evaluation.corpus import (
    Row,
    fetch_rows,
    parse_rows,
    select,
)
from prime_pr_review.evaluation.recording import (
    Recorder,
    is_done,
    mark_done,
    recording_model_fn,
    recording_reviewer,
)
from prime_pr_review.evaluation.report import render_markdown
from prime_pr_review.evaluation.runner import (
    HeadFileStore,
    corpus_runner,
    pr_list_json,
)
from prime_pr_review.evaluation.scoring import (
    Aggregate,
    aggregate,
    match,
    score_instance,
)
from prime_pr_review.providers import (
    BudgetExceeded,
    CostMeter,
    MeterBox,
    fetch_pricing,
    make_client,
    prime_model_fn,
    resolve_prime_key,
)
from prime_pr_review.review import Verdict
from prime_pr_review.state import LANE_OPEN, State
from prime_pr_review.sweep import Enrichment, sweep_lane

DATASET_URL = "https://datasets-server.huggingface.co/rows"
DATASET = "inclusionAI/SWE-CARE"
SEATS = ("deepseek/deepseek-v4-pro", "qwen/qwen3.8-max", "z-ai/glm-5.2")
AUX_MODEL = "deepseek/deepseek-v4-flash"
SKEPTIC_MODEL = JUDGE_MODEL = "deepseek/deepseek-v4-pro"
PROMPTS_DIR = AGENT_ROOT / "skills" / "pr-review" / "prompts"
EVAL_ROOT = AGENT_ROOT / "eval"
DOCS_EVAL = AGENT_ROOT / "docs" / "eval"
LIMITATIONS = (
    "Diff-only: no local checkout per PR, so context, blast-radius, unwired-export and linter passes are skipped.",
    "Ground truth is human review comments; a real defect humans did not comment on counts against precision.",
    "Intent pass sees the PR title only (PullRequest carries no body/commits in the headless path).",
    "SWE-CARE contains no cosmetic/silent PRs, so the silence rate is not measured.",
    "Offline ladder applies citation validation after refutation; production applies it before.",
    ("Reference comments without a line number are excluded from the recall denominator; they still "
     "count for file-level matching."),
)


@dataclass(frozen=True)
class Provider:
    make_reviewer_model_fn: Callable[[str], Callable[[str], str]]
    aux_fn: Callable[[str], str]
    skeptic_fn: Callable[[str], str]
    judge_fn: Callable[[str], str]
    box: MeterBox
    seat_models: tuple[str, ...] = SEATS


def build_provider(client: httpx.Client, box: MeterBox) -> Provider:
    return Provider(
        make_reviewer_model_fn=lambda model: prime_model_fn(client, model, box),
        aux_fn=prime_model_fn(client, AUX_MODEL, box),
        skeptic_fn=prime_model_fn(client, SKEPTIC_MODEL, box),
        judge_fn=prime_model_fn(client, JUDGE_MODEL, box),
        box=box,
    )


def eval_config(base: Config, row: Row) -> Config:
    owner, name = row.repo.split("/", 1)
    return replace(
        base,
        repo=replace(base.repo, owner=owner, name=name, read_only=True),
        review=replace(base.review, dry_run=True, ensemble_size=3, min_agreement=1, judge_merge=True,
                       check_refute=True, validate_citations=True, check_intent=True,
                       repo_root="", graph_path="", bot_login=""),
        sinks=replace(base.sinks, pr_comment=False, webhook=False, local_file=True),
    )


def run_one(row: Row, config: Config, provider: Provider, run_dir: Path, prompts_dir: Path,
            fallback_runner: github.GhRunner = github.default_runner) -> dict:
    inst = run_dir / row.instance_id
    if is_done(inst):
        return {"instance_id": row.instance_id, "skipped": True, "error": None}
    inst.mkdir(parents=True, exist_ok=True)
    recorder = Recorder(inst)
    reviewer = recording_reviewer(provider.seat_models, provider.make_reviewer_model_fn, recorder, prompts_dir)
    enrichment = Enrichment(
        model_fn=recording_model_fn("aux", AUX_MODEL, provider.aux_fn, recorder),
        prompts_dir=prompts_dir,
        skeptic_fn=recording_model_fn("skeptic", SKEPTIC_MODEL, provider.skeptic_fn, recorder),
        judge_fn=recording_model_fn("judge", JUDGE_MODEL, provider.judge_fn, recorder),
    )
    runner = corpus_runner(row, fallback_runner, HeadFileStore(inst / arms_mod.HEAD_FILES))
    started, spent_before = time.monotonic(), provider.box.meter.spent_usd
    report, _ = sweep_lane(config, LANE_OPEN, reviewer, State.empty(), runner=runner,
                           reviews_dir=run_dir / "reviews", enrichment=enrichment)
    outcome = report.outcomes[0] if report.outcomes else None
    result = {
        "instance_id": row.instance_id, "skipped": False,
        "error": outcome.error if outcome else "no outcome",
        "notes": list(outcome.notes) if outcome else [],
        "verdict": _verdict_json(outcome.verdict) if outcome and outcome.verdict else None,
        "seconds": time.monotonic() - started,
        "cost_usd": provider.box.meter.spent_usd - spent_before,
    }
    (inst / "outcome.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    mark_done(inst)
    return result


def _verdict_json(verdict: Verdict) -> dict:
    return json.loads(json.dumps(asdict(verdict), default=str))


def _pr_for(row: Row) -> github.PullRequest:
    return github._parse_pr_list(pr_list_json(row))[0]


def score_run(run_dir: Path, prompts_dir: Path) -> dict:
    rows = {r.instance_id: r for r in parse_rows(json.loads((run_dir / "rows.json").read_text(encoding="utf-8")))}
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    acc = _Accumulator()
    done_dirs = sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name in rows and is_done(p))
    for inst in done_dirs:
        _score_instance_dir(inst, rows[inst.name], prompts_dir, acc)
    aggregates = [aggregate(arm, mode, scores, acc.fabricated.get(arm, 0))
                  for (arm, mode), scores in acc.per_arm.items()]
    summary = {"run_id": config.get("run_id"), "config": config, "instances": len(done_dirs),
               "aggregates": [asdict(a) for a in aggregates], "severity": acc.severity, "drift": acc.drift,
               "replay_misses": acc.misses, "cost_usd": acc.cost, "seconds": acc.seconds,
               "limitations": list(LIMITATIONS)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


class _Accumulator:
    def __init__(self) -> None:
        self.per_arm: dict[tuple[str, str], list] = {}
        self.fabricated: dict[str, int] = {}
        self.severity: dict[str, list[int]] = {}
        self.drift = self.misses = 0
        self.cost = self.seconds = 0.0


def _score_instance_dir(inst: Path, row: Row, prompts_dir: Path, acc: _Accumulator) -> None:
    outcome = json.loads((inst / "outcome.json").read_text(encoding="utf-8"))
    acc.cost += outcome.get("cost_usd", 0.0)
    acc.seconds += outcome.get("seconds", 0.0)
    pr, refs = _pr_for(row), row.reference_comments
    for arm in arms_mod.ARMS:
        built = arms_mod.build_arm(arm, inst, pr, row.patch, LANE_OPEN, prompts_dir)
        acc.misses += built.replay_miss
        if built.verdict is None:
            continue
        acc.per_arm.setdefault((arm, "off"), []).append(score_instance(built.verdict, refs))
        validated, dropped = arms_mod.apply_citations(built.verdict, row.patch, inst, row.repo, row.head_sha)
        acc.fabricated[arm] = acc.fabricated.get(arm, 0) + dropped
        acc.per_arm.setdefault((arm, "on"), []).append(score_instance(validated, refs, dropped))
        if arm == "full":
            acc.drift += _drifted(outcome.get("verdict"), built.verdict)
            for f in validated.introduces:
                if not f.refuted:
                    bucket = acc.severity.setdefault(str(f.severity.value), [0, 0])
                    bucket[0] += 1
                    bucket[1] += any(match(f, r) for r in refs)


def _drifted(live: dict | None, replayed: Verdict) -> int:
    if live is None:
        return 1

    def key(items):
        return sorted((str(i.get("file")), i.get("line"), str(i.get("claim"))) for i in items)

    return int(key(live.get("introduces", [])) != key([asdict(f) for f in replayed.introduces]))


def write_report(run_dir: Path, out_dir: Path) -> Path:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    aggs = [Aggregate(**a) for a in summary["aggregates"]]
    sev = [(k, v[0], v[1]) for k, v in sorted(summary["severity"].items())]
    md = render_markdown(summary["run_id"], summary["config"], aggs, sev, summary["limitations"],
                         summary["cost_usd"], summary["seconds"], summary["drift"], summary["replay_misses"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{summary['run_id']}.md"
    path.write_text(md, encoding="utf-8")
    (out_dir / f"{summary['run_id']}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return path


def _fetch_page(offset: int, length: int) -> dict:
    params = {"dataset": DATASET, "config": "default", "split": "test", "offset": offset, "length": length}
    response = httpx.get(DATASET_URL, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def _row_raw(row: Row) -> dict:
    return {"instance_id": row.instance_id, "repo": row.repo, "language": row.language,
            "pull_number": row.pull_number, "title": row.title, "body": row.body, "base_commit": row.base_commit,
            "commit_to_review": {"head_commit": row.head_sha, "head_commit_message": row.head_commit_message,
                                 "patch_to_review": row.patch},
            "reference_review_comments": [{"path": c.path, "line": c.line, "original_line": c.line,
                                           "start_line": c.start_line, "original_start_line": c.start_line,
                                           "text": c.text, "diff_hunk": ""} for c in row.reference_comments]}


def cmd_run(args: argparse.Namespace) -> int:
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%d-%H%M")
    run_dir = EVAL_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = select(fetch_rows(EVAL_ROOT / "corpus" / "swecare-test.json", _fetch_page), args.count, args.seed)
    (run_dir / "rows.json").write_text(json.dumps({"rows": [{"row": _row_raw(r)} for r in rows]}), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps({"run_id": run_id, "count": args.count, "seed": args.seed,
                                                     "cap_usd": args.cap_usd, "seats": SEATS}), encoding="utf-8")
    client = make_client(resolve_prime_key())
    pricing = fetch_pricing(client, [*SEATS, AUX_MODEL, SKEPTIC_MODEL, JUDGE_MODEL])
    meter_path = run_dir / "meter.json"
    meter = (CostMeter.from_json(meter_path.read_text(encoding="utf-8"), pricing) if meter_path.is_file()
             else CostMeter(cap_usd=args.cap_usd, pricing=pricing))
    provider = build_provider(client, MeterBox(meter))
    base = load_config(AGENT_ROOT / "config.toml")
    for i, row in enumerate(rows, 1):
        try:
            result = run_one(row, eval_config(base, row), provider, run_dir, PROMPTS_DIR)
        except Exception as exc:  # noqa: BLE001 - one PR must never end the run
            result = {"instance_id": row.instance_id, "error": repr(exc), "skipped": False}
        meter_path.write_text(provider.box.meter.to_json(), encoding="utf-8")
        status = "skipped" if result.get("skipped") else (result.get("error") or "ok")
        print(f"[{i}/{len(rows)}] {row.instance_id}: {status} | spent ${provider.box.meter.spent_usd:.2f}")
        try:
            provider.box.meter.check()
        except BudgetExceeded as exc:
            print(f"stopping: {exc}")
            return 3
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="prime-review eval")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--count", type=int, default=200)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--cap-usd", type=float, default=40.0)
    run.add_argument("--run-id", default="")
    run.set_defaults(fn=cmd_run)
    score = sub.add_parser("score")
    score.add_argument("--run-id", required=True)
    score.set_defaults(fn=lambda a: (score_run(EVAL_ROOT / "runs" / a.run_id, PROMPTS_DIR) and 0))
    report = sub.add_parser("report")
    report.add_argument("--run-id", required=True)
    report.set_defaults(fn=lambda a: (print(write_report(EVAL_ROOT / "runs" / a.run_id, DOCS_EVAL)) or 0))
    args = parser.parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
