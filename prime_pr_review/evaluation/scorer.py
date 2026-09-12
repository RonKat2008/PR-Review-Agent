"""Score every ablation arm of one recorded run, offline and at zero model cost.

Split out of the CLI because the accounting is the substance: which recorded
calls each arm would have paid for, which instances are scorable at all, and
what the single-pass baseline looks like when it is the mean of the three seats
rather than the luckiest one. Everything here reads the run directory; nothing
here calls a model."""
from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from pathlib import Path

from .. import github
from ..review import Verdict
from ..state import LANE_OPEN
from . import arms as arms_mod
from .corpus import Row
from .recording import Call, Recorder
from .runner import pr_list_json
from .scoring import (
    CITATIONS_OFF,
    CITATIONS_ON,
    Aggregate,
    aggregate,
    match,
    score_instance,
)

SINGLE_ARM = "single"
SEAT_ARMS = tuple(a for a in arms_mod.ARMS if a.startswith("seat-"))
MODES = (CITATIONS_OFF, CITATIONS_ON)
PRICING_FILE = "pricing.json"
NO_PRICING_NOTE = f"no {PRICING_FILE} in the run dir: per-arm cost reported as $0.00"
EXCLUSION_KEYS = ("replay_miss", "error", "no_anchored_refs")

Pricing = dict[str, tuple[float, float]]


class Accumulator:
    """Every running total the scorer builds across one run's instances."""

    def __init__(self) -> None:
        self.per_arm: dict[tuple[str, str], list] = {}
        self.fabricated: dict[tuple[str, str], int] = {}
        self.arm_cost: dict[tuple[str, str], float] = {}
        self.arm_seconds: dict[tuple[str, str], float] = {}
        self.severity: dict[str, list[int]] = {}
        self.arm_notes: dict[str, list[str]] = {}
        self.per_instance: list[dict] = []
        self.excluded: dict[str, int] = dict.fromkeys(EXCLUSION_KEYS, 0)
        self.drift = self.instances = 0
        self.cost = self.seconds = 0.0


def load_pricing(run_dir: Path) -> tuple[Pricing, list[str]]:
    """The pricing snapshot `run` persisted. Absent (an older run), every arm
    costs $0.00 and the summary says so rather than implying the run was free."""
    path = Path(run_dir) / PRICING_FILE
    if not path.is_file():
        return {}, [NO_PRICING_NOTE]
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {model: (float(rates[0]), float(rates[1])) for model, rates in raw.items()}, []


def call_cost(call: Call, pricing: Pricing) -> float:
    rates = pricing.get(call.model)
    if rates is None:
        return 0.0
    return call.prompt_tokens * rates[0] / 1e6 + call.completion_tokens * rates[1] / 1e6


def pr_for_row(row: Row) -> github.PullRequest:
    return github._parse_pr_list(pr_list_json(row))[0]


def score_instance_dir(inst: Path, row: Row, prompts_dir: Path, acc: Accumulator,
                       seat_models: Sequence[str], pricing: Pricing) -> None:
    outcome = json.loads((inst / "outcome.json").read_text(encoding="utf-8"))
    acc.cost += outcome.get("cost_usd", 0.0)
    acc.seconds += outcome.get("seconds", 0.0)
    if not any(c.line is not None for c in row.reference_comments):
        # Nothing here can be matched or missed: `match()` needs a line on both
        # sides. Scoring it would add findings to every precision denominator
        # against ground truth that cannot, even in principle, confirm them.
        acc.excluded["no_anchored_refs"] += 1
        return
    acc.instances += 1
    recorder, pr = Recorder(inst), pr_for_row(row)
    for arm in arms_mod.ARMS:
        _score_arm(arm, inst, row, pr, prompts_dir, acc, seat_models, pricing, recorder, outcome)


def _score_arm(arm, inst, row, pr, prompts_dir, acc, seat_models, pricing, recorder, outcome) -> None:
    built = arms_mod.build_arm(arm, inst, pr, row.patch, LANE_OPEN, prompts_dir, seat_models)
    if built.replay_miss or built.verdict is None:
        key = "replay_miss" if built.replay_miss else "error"
        acc.excluded[key] += 1
        _note(acc, arm, f"{row.instance_id}: {key} ({built.error or 'no verdict'})")
        return
    calls = arms_mod.arm_calls(recorder, arm, seat_models)
    spend = (sum(call_cost(c, pricing) for c in calls), sum(c.seconds for c in calls))
    validated, dropped = arms_mod.apply_citations(built.verdict, row.patch, inst, row.repo, row.head_sha)
    total_dropped = built.dropped + dropped
    # Fabrication is a property of citation validation, so it only ever applies
    # to the "on" row -- the raw "off" row was never validated.
    acc.fabricated[(arm, CITATIONS_ON)] = acc.fabricated.get((arm, CITATIONS_ON), 0) + total_dropped
    _record(acc, arm, CITATIONS_OFF, score_instance(built.verdict, row.reference_comments), spend, row)
    _record(acc, arm, CITATIONS_ON, score_instance(validated, row.reference_comments, total_dropped), spend, row)
    if arm == "full":
        acc.drift += drifted(outcome.get("verdict"), built.verdict)
        _tally_severity(acc, validated, row)


def _record(acc: Accumulator, arm: str, mode: str, score, spend: tuple[float, float], row: Row) -> None:
    cost, seconds = spend
    acc.per_arm.setdefault((arm, mode), []).append(score)
    acc.arm_cost[(arm, mode)] = acc.arm_cost.get((arm, mode), 0.0) + cost
    acc.arm_seconds[(arm, mode)] = acc.arm_seconds.get((arm, mode), 0.0) + seconds
    acc.per_instance.append({"instance_id": row.instance_id, "arm": arm, "mode": mode,
                             "findings": score.findings, "matched_findings": score.matched_findings,
                             "refs": score.refs, "matched_refs": score.matched_refs,
                             "dropped": score.dropped, "cost_usd": cost, "seconds": seconds})


def _note(acc: Accumulator, arm: str, text: str) -> None:
    """An arm that could not be built is missing from both of its rows, so the
    note is filed against both."""
    for mode in MODES:
        acc.arm_notes.setdefault(f"{arm}|{mode}", []).append(text)


def _tally_severity(acc: Accumulator, validated: Verdict, row: Row) -> None:
    for finding in validated.introduces:
        if finding.refuted:
            continue
        bucket = acc.severity.setdefault(str(finding.severity.value), [0, 0])
        bucket[0] += 1
        bucket[1] += any(match(finding, r) for r in row.reference_comments)


def drifted(live: dict | None, replayed: Verdict) -> int:
    if live is None:
        return 1

    def key(items):
        return sorted(
            (str(i.get("file")), i.get("line") if i.get("line") is not None else -1, str(i.get("claim")))
            for i in items
        )

    return int(key(live.get("introduces", [])) != key([asdict(f) for f in replayed.introduces]))


def build_aggregates(acc: Accumulator) -> list[Aggregate]:
    aggregates = [aggregate(arm, mode, scores, acc.fabricated.get((arm, mode), 0),
                            acc.arm_cost.get((arm, mode), 0.0), acc.arm_seconds.get((arm, mode), 0.0))
                  for (arm, mode), scores in acc.per_arm.items()]
    return aggregates + [a for a in (_single(aggregates, mode) for mode in MODES) if a is not None]


def _single(aggregates: Sequence[Aggregate], mode: str) -> Aggregate | None:
    """The single-pass baseline: the mean of the three seat arms, so the ladder
    is read against a typical seat rather than whichever one happened to win."""
    seats = [a for a in aggregates if a.arm in SEAT_ARMS and a.mode == mode]
    if len(seats) != len(SEAT_ARMS):
        return None
    return Aggregate(
        arm=SINGLE_ARM, mode=mode,
        instances=min(a.instances for a in seats),
        precision=_mean(a.precision for a in seats),
        recall=_mean(a.recall for a in seats),
        file_precision=_mean(a.file_precision for a in seats),
        fabrication_rate=(_mean(a.fabrication_rate or 0.0 for a in seats)
                          if mode == CITATIONS_ON else None),
        findings_per_pr=_mean(a.findings_per_pr for a in seats),
        cost_usd_per_pr=_mean(a.cost_usd_per_pr for a in seats),
        seconds_per_pr=_mean(a.seconds_per_pr for a in seats),
    )


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0
