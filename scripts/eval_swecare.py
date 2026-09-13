"""SWE-CARE ablation harness: `run` reviews sampled PRs live on Prime Inference
and records every model call; `score` rebuilds each ablation arm offline and
scores it; `report` renders docs/eval/<run_id>.md.
Spec: docs/superpowers/specs/2026-09-11-swecare-ablation-harness-design.md."""
from __future__ import annotations

import argparse
import json
import shutil
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
    select_with_counts,
)
from prime_pr_review.evaluation.recording import (
    Recorder,
    is_done,
    mark_done,
    recording_model_fn,
    recording_reviewer,
)
from prime_pr_review.evaluation.repair import repair_run
from prime_pr_review.evaluation.report import render_markdown
from prime_pr_review.evaluation.runner import HeadFileStore, corpus_runner
from prime_pr_review.evaluation.scorer import (
    Accumulator,
    build_aggregates,
    load_pricing,
    score_instance_dir,
)
from prime_pr_review.evaluation.scoring import Aggregate
from prime_pr_review.providers import (
    BudgetExceeded,
    CostMeter,
    MeterBox,
    Usage,
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
SEATS = ("deepseek/deepseek-v4-pro", "openai/gpt-5.4-mini", "z-ai/glm-5.2")
DEEPSEEK_V4_PRO_MAX_TOKENS = 64_000
SEAT_OPTIONS: dict[str, dict] = {
    "openai/gpt-5.4-mini": {"reasoning": {"effort": "medium"}},
    # 40/111 live instances lost their deepseek/deepseek-v4-pro seat to the
    # default 16k MAX_COMPLETION_TOKENS cap (long reasoning, skewed to larger
    # patches; no call came near the 300s timeout) -- TruncatedResponse is
    # deliberately not retried (see providers.TruncatedResponse), so the fix
    # is a bigger per-model cap, not a retry.
    "deepseek/deepseek-v4-pro": {"max_tokens": DEEPSEEK_V4_PRO_MAX_TOKENS},
}
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
    ("The full arm replays production order (citation validation, then refutation); the other arms "
     "are scored raw and again with validation applied afterwards."),
    ("Reference comments without a line number are excluded from the recall denominator; they still "
     "count for file-level matching."),
    ("Head-file line counts were fetched live only for paths the ensemble+judge verdict needed, so "
     "seat arms may keep an out-of-hunk citation as unverified where the full arm would drop it; "
     "seat-arm fabrication rates are therefore lower bounds."),
    ("Evaluation seat 2 is openai/gpt-5.4-mini (reasoning effort medium) instead of the production "
     "lineup's qwen/qwen3.8-max, which reasons without bound (~7 min and $0.10 per PR)."),
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
        make_reviewer_model_fn=lambda model: prime_model_fn(client, model, box, extra=SEAT_OPTIONS.get(model)),
        aux_fn=prime_model_fn(client, AUX_MODEL, box),
        skeptic_fn=prime_model_fn(client, SKEPTIC_MODEL, box, extra=SEAT_OPTIONS.get(SKEPTIC_MODEL)),
        judge_fn=prime_model_fn(client, JUDGE_MODEL, box, extra=SEAT_OPTIONS.get(JUDGE_MODEL)),
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
    if inst.exists():
        # A directory without a `done` marker is a partial/interrupted attempt
        # (or, in a resumed run, stale from a prior interpreter version) --
        # never resume into it, or the new recordings mix with the old ones.
        shutil.rmtree(inst)
    inst.mkdir(parents=True, exist_ok=True)
    recorder = Recorder(inst)
    usage = _usage_source(provider.box)
    reviewer = recording_reviewer(provider.seat_models, provider.make_reviewer_model_fn, recorder,
                                  prompts_dir, usage)
    enrichment = Enrichment(
        model_fn=recording_model_fn("aux", AUX_MODEL, provider.aux_fn, recorder, usage),
        prompts_dir=prompts_dir,
        skeptic_fn=recording_model_fn("skeptic", SKEPTIC_MODEL, provider.skeptic_fn, recorder, usage),
        judge_fn=recording_model_fn("judge", JUDGE_MODEL, provider.judge_fn, recorder, usage),
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
    # Only a clean outcome earns the `done` marker -- an errored attempt must be
    # retried (with its stale recordings cleared, see above), never treated as
    # finished.
    if result["error"] is None:
        mark_done(inst)
    return result


def _usage_source(box: MeterBox) -> Callable[[], tuple[int, int]]:
    """The tokens of the call that just returned. `prime_model_fn` parks each
    response's usage on the box, so the recorder can attribute it without the
    provider and the recorder having to know about each other."""
    def source() -> tuple[int, int]:
        usage: Usage | None = box.last_usage
        return (usage.prompt_tokens, usage.completion_tokens) if usage else (0, 0)
    return source


def _verdict_json(verdict: Verdict) -> dict:
    return json.loads(json.dumps(asdict(verdict), default=str))


def score_run(run_dir: Path, prompts_dir: Path) -> dict:
    rows = {r.instance_id: r for r in parse_rows(json.loads((run_dir / "rows.json").read_text(encoding="utf-8")))}
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    seat_models = tuple(config.get("seats") or SEATS)
    pricing, notes = load_pricing(run_dir)
    acc = Accumulator()
    done_dirs = sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name in rows and is_done(p))
    for inst in done_dirs:
        score_instance_dir(inst, rows[inst.name], prompts_dir, acc, seat_models, pricing)
    summary = {"run_id": config.get("run_id"), "config": config, "instances": acc.instances,
               "aggregates": [asdict(a) for a in build_aggregates(acc)], "severity": acc.severity,
               "drift": acc.drift, "replay_misses": acc.excluded["replay_miss"],
               "cost_usd": acc.cost, "seconds": acc.seconds,
               "pricing": {m: list(r) for m, r in pricing.items()},
               "per_instance": acc.per_instance, "excluded": acc.excluded,
               "arm_notes": acc.arm_notes, "notes": notes,
               "limitations": list(LIMITATIONS)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def write_report(run_dir: Path, out_dir: Path) -> Path:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    aggs = [Aggregate(**a) for a in summary["aggregates"]]
    sev = [(k, v[0], v[1]) for k, v in sorted(summary["severity"].items())]
    md = render_markdown(summary["run_id"], summary["config"], aggs, sev, summary["limitations"],
                         summary["cost_usd"], summary["seconds"], summary["drift"],
                         summary["replay_misses"], summary["instances"],
                         filters=summary["config"].get("filters"), excluded=summary.get("excluded"))
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


def instance_dirs(run_dir: Path) -> tuple[Path, ...]:
    """Directories a previous `run` left behind for this run-id."""
    if not run_dir.is_dir():
        return ()
    return tuple(p for p in sorted(run_dir.iterdir())
                 if p.is_dir() and ((p / "calls").is_dir() or (p / "outcome.json").is_file()))


def _run_seat_models(run_dir: Path) -> tuple[str, ...]:
    """The seat lineup a run was actually started with, so a repair issues
    calls against the same models the run's live recordings used even if
    `SEATS` has moved on since. Falls back to today's `SEATS` when the run
    predates the config, or has none yet (nothing to repair either way)."""
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        return SEATS
    return tuple(json.loads(config_path.read_text(encoding="utf-8")).get("seats") or SEATS)


def _cmd_repair(run_dir: Path, cap_usd: float) -> int:
    """`run --repair`: re-issue only the seat calls missing from this run's
    `done` instances. Never writes rows.json/config.json (the run already
    has them) and never calls `run_one` -- see `repair_run`."""
    client = make_client(resolve_prime_key())
    pricing = fetch_pricing(client, [*SEATS, AUX_MODEL, SKEPTIC_MODEL, JUDGE_MODEL])
    meter_path = run_dir / "meter.json"
    if meter_path.is_file():
        loaded = CostMeter.from_json(meter_path.read_text(encoding="utf-8"), pricing)
        meter = replace(loaded, cap_usd=cap_usd)
    else:
        meter = CostMeter(cap_usd=cap_usd, pricing=pricing)
    provider = build_provider(client, MeterBox(meter))
    provider = replace(provider, seat_models=_run_seat_models(run_dir))
    try:
        counts = repair_run(run_dir, provider)
    except BudgetExceeded as exc:
        print(f"stopping: {exc}")
        return 3
    print(f"repair: {counts['repaired']}/{counts['instances']} instances repaired, "
          f"{counts['calls']} calls, {counts['unrepairable']} unrepairable")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%d-%H%M")
    run_dir = EVAL_ROOT / "runs" / run_id
    existing = instance_dirs(run_dir)
    if existing and not (args.resume or args.repair):
        # Silently continuing would mix a new sample (or new seats) into an
        # existing run's recordings, and the scorer cannot tell them apart.
        print(f"run-id {run_id!r} already has {len(existing)} instance(s); "
              f"pass --resume to continue it, or choose another --run-id")
        return 2
    if args.repair:
        # --repair implies --resume semantics for the check above (an
        # existing run is the whole point) but must never fall through into
        # the normal review loop below.
        return _cmd_repair(run_dir, args.cap_usd)
    run_dir.mkdir(parents=True, exist_ok=True)
    rows, filters = select_with_counts(
        fetch_rows(EVAL_ROOT / "corpus" / "swecare-test.json", _fetch_page), args.count, args.seed)
    (run_dir / "rows.json").write_text(json.dumps({"rows": [{"row": _row_raw(r)} for r in rows]}), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps({"run_id": run_id, "count": args.count, "seed": args.seed,
                                                     "cap_usd": args.cap_usd, "seats": SEATS,
                                                     "seat_options": SEAT_OPTIONS,
                                                     "filters": filters}), encoding="utf-8")
    client = make_client(resolve_prime_key())
    pricing = fetch_pricing(client, [*SEATS, AUX_MODEL, SKEPTIC_MODEL, JUDGE_MODEL])
    # The scorer prices the recorded calls against this snapshot, so it has to
    # be the prices this run actually paid, not whatever /models says later.
    (run_dir / "pricing.json").write_text(
        json.dumps({m: list(r) for m, r in pricing.items()}, indent=1), encoding="utf-8")
    meter_path = run_dir / "meter.json"
    if meter_path.is_file():
        # `--cap-usd` always wins, even on resume: a cap raised (or lowered)
        # between invocations must take effect immediately, not stay pinned to
        # whatever was persisted the first time this run-id was started.
        loaded = CostMeter.from_json(meter_path.read_text(encoding="utf-8"), pricing)
        meter = replace(loaded, cap_usd=args.cap_usd)
    else:
        meter = CostMeter(cap_usd=args.cap_usd, pricing=pricing)
    provider = build_provider(client, MeterBox(meter))
    base = load_config(AGENT_ROOT / "config.toml")
    return _run_review_loop(rows, base, provider, run_dir, meter_path)


def _run_review_loop(rows: list[Row], base: Config, provider: Provider, run_dir: Path,
                     meter_path: Path) -> int:
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
    run.add_argument("--resume", action="store_true",
                     help="continue an existing run-id instead of refusing to reuse it")
    run.add_argument("--repair", action="store_true",
                     help="re-issue only missing seat calls for this run-id's done instances, "
                          "instead of running the normal review loop")
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
