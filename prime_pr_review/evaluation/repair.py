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
"""
from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from ..providers import MeterBox, Usage
from .recording import Recorder, is_done, recording_model_fn, seat_calls

ModelFn = Callable[[str], str]


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
        return {"instances": 0, "repaired": 0, "calls": 0, "unrepairable": 0}
    done = sorted(p for p in run_dir.iterdir() if p.is_dir() and is_done(p))
    repaired = calls = unrepairable = 0
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
            wrapped(prompt)
            calls += 1
        repaired += 1
        print(f"[repair {i}/{len(done)}] {inst.name}: +{','.join(missing)} "
              f"| spent ${provider.box.meter.spent_usd:.2f}")
        meter_path.write_text(provider.box.meter.to_json(), encoding="utf-8")
        provider.box.meter.check()
    return {"instances": len(done), "repaired": repaired, "calls": calls, "unrepairable": unrepairable}
