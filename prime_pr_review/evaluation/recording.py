"""Record every model call of one live review to disk, and replay them offline.

Replay is exact or it is an error: a judge/skeptic prompt that differs from the
recorded one means the offline arm is not the live pipeline, and the scorer
must know that rather than silently calling a model."""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from ..github import PullRequest
from ..reviewers import build_prompt

ModelFn = Callable[[str], str]
Reviewer = Callable[[PullRequest, str, str], str]
UsageSource = Callable[[], tuple[int, int]]  # -> (prompt_tokens, completion_tokens) of the last call
CALLS_DIR = "calls"
DONE_MARKER = "done"


class ReplayMiss(RuntimeError):
    """No recorded response for this prompt (or no seats left)."""


@dataclass(frozen=True)
class Call:
    seq: int
    role: str
    model: str
    prompt_sha256: str
    prompt: str
    response: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Recorder:
    def __init__(self, directory: Path) -> None:
        self._dir = Path(directory) / CALLS_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    def record(self, role: str, model: str, prompt: str, response: str, seconds: float,
               usage: tuple[int, int] = (0, 0)) -> Call:
        seq = len(list(self._dir.glob("*.json")))
        call = Call(seq, role, model, sha256(prompt), prompt, response, usage[0], usage[1], seconds)
        (self._dir / f"{seq:03d}-{role}.json").write_text(json.dumps(asdict(call)), encoding="utf-8")
        return call

    def calls(self, role: str | None = None) -> tuple[Call, ...]:
        loaded = (Call(**json.loads(p.read_text(encoding="utf-8"))) for p in sorted(self._dir.glob("*.json")))
        return tuple(c for c in loaded if role is None or c.role == role)


def recording_model_fn(role: str, model: str, inner: ModelFn, recorder: Recorder,
                       usage_source: UsageSource | None = None) -> ModelFn:
    """Wrap `inner` so every call lands on disk. `usage_source` is read *after*
    the call and reports the tokens that call consumed; without it the
    recording carries (0, 0) and every offline cost reads as free."""
    def model_fn(prompt: str) -> str:
        started = time.monotonic()
        response = inner(prompt)
        usage = usage_source() if usage_source is not None else (0, 0)
        recorder.record(role, model, prompt, response, time.monotonic() - started, usage=usage)
        return response
    return model_fn


def recording_reviewer(seat_models: Sequence[str], make_model_fn: Callable[[str], ModelFn],
                       recorder: Recorder, prompts_dir: Path | str,
                       usage_source: UsageSource | None = None) -> Reviewer:
    seats = [recording_model_fn("seat", m, make_model_fn(m), recorder, usage_source) for m in seat_models]
    counter = {"k": 0}

    def reviewer(pr: PullRequest, payload: str, lane: str) -> str:
        template = (Path(prompts_dir) / f"{lane}_pr.md").read_text(encoding="utf-8")
        seat = seats[counter["k"] % len(seats)]
        counter["k"] += 1
        return seat(build_prompt(template, pr, payload))

    return reviewer


def replay_model_fn(recorder: Recorder, role: str) -> ModelFn:
    by_hash: dict[str, list[str]] = {}
    for c in recorder.calls(role):
        by_hash.setdefault(c.prompt_sha256, []).append(c.response)

    def model_fn(prompt: str) -> str:
        pending = by_hash.get(sha256(prompt))
        if not pending:
            raise ReplayMiss(f"no recorded {role} response for this prompt")
        return pending.pop(0)
    return model_fn


def seat_calls(recorder: Recorder, seat_models: Sequence[str]) -> tuple[Call | None, ...]:
    """The recorded call per seat *position*, `None` where that seat has none.

    Recorded position is not seat identity: a seat whose live call failed wrote
    no record, so indexing the sequence would relabel every seat after it. Seats
    are matched by model instead; a model listed twice takes its recordings in
    order."""
    calls = recorder.calls("seat")
    picked: list[Call | None] = []
    for index, model in enumerate(seat_models):
        matching = [c for c in calls if c.model == model]
        occurrence = list(seat_models[:index]).count(model)
        picked.append(matching[occurrence] if occurrence < len(matching) else None)
    return tuple(picked)


def replay_reviewer(recorder: Recorder, seat_models: Sequence[str]) -> Reviewer:
    """Serve the ensemble's k-th call from the k-th *seat*, not the k-th
    recording. A seat with no recording raises rather than letting the next
    seat's verdict stand in for it."""
    seats = seat_calls(recorder, seat_models)
    counter = {"k": 0}

    def reviewer(pr: PullRequest, payload: str, lane: str) -> str:
        index = counter["k"]
        counter["k"] += 1
        if index >= len(seats):
            raise ReplayMiss("more seat calls than seats")
        call = seats[index]
        if call is None:
            raise ReplayMiss(f"no recorded response for seat {index + 1} ({seat_models[index]})")
        return call.response
    return reviewer


def mark_done(directory: Path) -> None:
    (Path(directory) / DONE_MARKER).write_text("ok", encoding="utf-8")


def is_done(directory: Path) -> bool:
    return (Path(directory) / DONE_MARKER).is_file()
