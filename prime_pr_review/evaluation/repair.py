"""Repair missing seat recordings in an already-finished eval run.

A seat has no recording when its live call raised: the harness swallows a
per-seat exception (see `sweep_lane`/the ensemble reviewer), records nothing
for that seat, and the instance can still finish and earn its `done` marker
-- a seat call failing and the instance-level outcome are independent.
`repair_run` re-issues only those missing calls. It is offline-safe: every
seat in an instance was sent the same built prompt (`recording_reviewer`
round-robins seats over one `build_prompt` call per review), so the missing
seat's prompt is read back off any surviving `*-seat.json` rather than
rebuilt, and no sweep, judge, skeptic, or `done` marker is touched.

`repair_passes` cleans up after it. A repaired seat changes what the ensemble
merges, so the judged arms rebuild a DIFFERENT verdict than the live run
produced -- and the judge prompt is built from that verdict, so it no longer
matches anything recorded. It re-issues exactly the judge and skeptic calls
the rebuilt verdict needs, recording them so future replays hit.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ..github import PullRequest
from ..providers import BudgetExceeded, MeterBox, ProviderError, Usage
from ..state import LANE_OPEN
from . import arms
from .recording import Recorder, is_done, recording_model_fn, seat_calls

ModelFn = Callable[[str], str]
# Only the judged arms can miss a judge or skeptic recording; `seat-*` and
# `ensemble` replay off the seat calls alone. `ensemble+judge` comes first so
# the judge call it records is already on disk when `full` replays the same
# judge prompt.
PASS_ARMS = ("ensemble+judge", "full")


class SeatProvider(Protocol):
    """The subset of `eval_swecare.Provider` repair needs, named locally so
    this module never has to import the CLI script that imports it."""

    make_reviewer_model_fn: Callable[[str], ModelFn]
    box: MeterBox
    seat_models: Sequence[str]


def _usage_source(box: MeterBox) -> Callable[[], tuple[int, int]]:
    def source() -> tuple[int, int]:
        usage: Usage | None = box.last_usage
        return (usage.prompt_tokens, usage.completion_tokens) if usage else (0, 0)
    return source


def _missing_seat_models(recorder: Recorder, seat_models: Sequence[str]) -> list[str]:
    """Models whose seat position has no recorded call.

    `seat_calls` matches recordings to seats by *model*, taking the K-th
    recorded call for a model as the K-th seat listing that model (see its
    docstring). A seat repaired below is appended with a higher sequence
    number than anything already on disk, so it always lands as that
    model's next unclaimed occurrence -- appending is safe, it never
    reorders or displaces an existing match."""
    calls = seat_calls(recorder, seat_models)
    return [model for call, model in zip(calls, seat_models) if call is None]


def _shared_seat_prompt(inst: Path) -> str | None:
    """The one prompt every seat in `inst` was sent, read back from any
    surviving seat recording. `None` when no seat call survived at all, so
    there is nothing to rebuild the missing ones from."""
    files = sorted((inst / "calls").glob("*-seat.json"))
    if not files:
        return None
    return json.loads(files[0].read_text(encoding="utf-8"))["prompt"]


def repair_run(run_dir: Path, provider: SeatProvider) -> dict:
    """Re-issue only the seat calls missing from every `done` instance under
    `run_dir`. Never touches outcome.json, judge/skeptic recordings, or the
    `done` marker, and never re-runs the review sweep.

    Persists `run_dir/meter.json` after each repaired instance, then checks
    the meter -- `BudgetExceeded` propagates to the caller once the cap is
    crossed, after that instance's spend is already on disk.
    """
    meter_path = run_dir / "meter.json"
    if not run_dir.is_dir():
        return {"instances": 0, "repaired": 0, "calls": 0, "unrepairable": 0, "failed": 0}
    done = sorted(p for p in run_dir.iterdir() if p.is_dir() and is_done(p))
    repaired = calls = unrepairable = failed = 0
    for i, inst in enumerate(done, 1):
        recorder = Recorder(inst)
        missing = _missing_seat_models(recorder, provider.seat_models)
        if not missing:
            continue
        prompt = _shared_seat_prompt(inst)
        if prompt is None:
            unrepairable += 1
            continue
        usage = _usage_source(provider.box)
        for model in missing:
            wrapped = recording_model_fn("seat", model, provider.make_reviewer_model_fn(model),
                                         recorder, usage)
            try:
                wrapped(prompt)
            except BudgetExceeded:
                raise
            except Exception as exc:  # noqa: BLE001 - a seat call may fail in any manner
                failed += 1
                print(f"[repair {i}/{len(done)}] {inst.name}: seat {model} failed: "
                      f"{type(exc).__name__}: {str(exc)[:100]}")
                continue
            calls += 1
        repaired += 1
        print(f"[repair {i}/{len(done)}] {inst.name}: +{','.join(missing)} "
              f"| spent ${provider.box.meter.spent_usd:.2f}")
        meter_path.write_text(provider.box.meter.to_json(), encoding="utf-8")
        provider.box.meter.check()
    return {"instances": len(done), "repaired": repaired, "calls": calls,
            "unrepairable": unrepairable, "failed": failed}


class PassProvider(Protocol):
    """The subset of `eval_swecare.Provider` `repair_passes` needs, named
    locally so this module never imports the CLI script that imports it."""

    judge_fn: ModelFn
    skeptic_fn: ModelFn
    judge_model: str
    skeptic_model: str
    box: MeterBox


@dataclass
class _Tally:
    """One instance's live calls, tracked outside the arm build because every
    consumer of these fns fails open: `ensemble._judge_merge` and
    `refute_findings` both swallow the exception, so a failure not captured
    here would render as an arm that simply had no judge."""

    calls: dict[str, int] = field(default_factory=lambda: {"judge": 0, "skeptic": 0})
    failures: tuple[str, ...] = ()
    budget: BudgetExceeded | None = None

    @property
    def total(self) -> int:
        return sum(self.calls.values())


def repair_passes(run_dir: Path, provider: PassProvider, seat_models: Sequence[str],
                  prompts_dir: Path | str, diff_for: Callable[[str], str],
                  pr_for: Callable[[str], PullRequest]) -> dict:
    """Re-issue the judge and skeptic calls the REBUILT verdict needs, for
    every `done` instance under `run_dir` whose judged arms replay-miss.

    Recordings only: no sweep, no seat call, no `done` marker, no outcome.json.
    Persists `run_dir/meter.json` after each instance that spent, then checks
    the meter, so `BudgetExceeded` propagates once the cap is crossed with
    that instance's spend already on disk.
    """
    totals = {"instances": 0, "repaired": 0, "judge_calls": 0, "skeptic_calls": 0, "failed": 0}
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return totals
    done = sorted(p for p in run_dir.iterdir() if p.is_dir() and is_done(p))
    totals["instances"] = len(done)
    for i, inst in enumerate(done, 1):
        try:
            pr, diff = pr_for(inst.name), diff_for(inst.name)
        except KeyError:
            continue  # an instance this run's rows.json no longer describes
        tally = _repair_instance(inst, provider, seat_models, prompts_dir, pr, diff)
        if not tally.total and not tally.failures:
            continue
        _tally_into(totals, tally)
        _report(i, len(done), inst.name, tally, provider.box.meter.spent_usd)
        (run_dir / "meter.json").write_text(provider.box.meter.to_json(), encoding="utf-8")
        provider.box.meter.check()
    return totals


def _repair_instance(inst: Path, provider: PassProvider, seat_models: Sequence[str],
                     prompts_dir: Path | str, pr: PullRequest, diff: str) -> _Tally:
    """Rebuild each judged arm and answer its replay misses with live calls.

    The arm is built twice on purpose: the first, fallback-free build is the
    question ("does this still replay?"), and only a `replay_miss` earns the
    second build, the one allowed to spend. A seat-level miss cannot be fixed
    here -- that is `repair_run`'s job -- so the second build simply misses
    again and the instance reports no calls.
    """
    tally = _Tally()
    recorder = Recorder(inst)
    for arm in PASS_ARMS:
        built = arms.build_arm(arm, inst, pr, diff, LANE_OPEN, prompts_dir, seat_models)
        if not built.replay_miss:
            continue
        arms.build_arm(arm, inst, pr, diff, LANE_OPEN, prompts_dir, seat_models,
                       live=_live_fallbacks(recorder, provider, tally))
        if tally.budget is not None:
            raise tally.budget
        if tally.failures:
            break
    return tally


def _live_fallbacks(recorder: Recorder, provider: PassProvider, tally: _Tally) -> arms.LiveFallbacks:
    usage = _usage_source(provider.box)
    return arms.LiveFallbacks(
        judge_fn=_live_call("judge", provider.judge_model, provider.judge_fn, recorder, usage, tally),
        skeptic_fn=_live_call("skeptic", provider.skeptic_model, provider.skeptic_fn, recorder,
                              usage, tally),
    )


def _live_call(role: str, model: str, inner: ModelFn, recorder: Recorder,
               usage: Callable[[], tuple[int, int]], tally: _Tally) -> ModelFn:
    """`recording_model_fn` plus the bookkeeping its fail-open callers erase.

    After a failure the fallback disarms itself: `refute_findings` calls the
    skeptic once per finding, so a provider that is failing would otherwise be
    billed once per remaining finding of an instance already being skipped.
    """
    recording = recording_model_fn(role, model, inner, recorder, usage)

    def model_fn(prompt: str) -> str:
        if tally.failures or tally.budget is not None:
            raise ProviderError(f"{role}: skipped, an earlier live call failed")
        try:
            response = recording(prompt)
        except BudgetExceeded as exc:
            tally.budget = exc
            raise
        except ProviderError as exc:
            tally.failures = (*tally.failures,
                              f"{role} call failed: {type(exc).__name__}: {str(exc)[:100]}")
            raise
        tally.calls[role] += 1
        return response
    return model_fn


def _tally_into(totals: dict, tally: _Tally) -> None:
    """A failed instance still counts the calls it made before failing -- they
    were paid for and recorded -- but counts as failed, never repaired."""
    totals["judge_calls"] += tally.calls["judge"]
    totals["skeptic_calls"] += tally.calls["skeptic"]
    totals["failed" if tally.failures else "repaired"] += 1


def _report(index: int, total: int, name: str, tally: _Tally, spent: float) -> None:
    head = f"[repair-passes {index}/{total}] {name}"
    if tally.failures:
        print(f"{head}: {tally.failures[0]} | spent ${spent:.2f}")
        return
    print(f"{head}: +judge {tally.calls['judge']}, +skeptic {tally.calls['skeptic']} "
          f"| spent ${spent:.2f}")
