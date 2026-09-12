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


def recording_model_fn(role: str, model: str, inner: ModelFn, recorder: Recorder) -> ModelFn:
    def model_fn(prompt: str) -> str:
        started = time.monotonic()
        response = inner(prompt)
        recorder.record(role, model, prompt, response, time.monotonic() - started)
        return response
    return model_fn


def recording_reviewer(seat_models: Sequence[str], make_model_fn: Callable[[str], ModelFn],
                       recorder: Recorder, prompts_dir: Path | str) -> Reviewer:
    seats = [recording_model_fn("seat", m, make_model_fn(m), recorder) for m in seat_models]
    counter = {"k": 0}

    def reviewer(pr: PullRequest, payload: str, lane: str) -> str:
        template = (Path(prompts_dir) / f"{lane}_pr.md").read_text(encoding="utf-8")
        seat = seats[counter["k"] % len(seats)]
        counter["k"] += 1
        return seat(build_prompt(template, pr, payload))

    return reviewer


def replay_model_fn(recorder: Recorder, role: str) -> ModelFn:
    by_hash = {c.prompt_sha256: c.response for c in recorder.calls(role)}

    def model_fn(prompt: str) -> str:
        try:
            return by_hash[sha256(prompt)]
        except KeyError as exc:
            raise ReplayMiss(f"no recorded {role} response for this prompt") from exc
    return model_fn


def replay_reviewer(recorder: Recorder) -> Reviewer:
    responses = [c.response for c in recorder.calls("seat")]
    counter = {"k": 0}

    def reviewer(pr: PullRequest, payload: str, lane: str) -> str:
        if counter["k"] >= len(responses):
            raise ReplayMiss("more seat calls than recorded")
        response = responses[counter["k"]]
        counter["k"] += 1
        return response
    return reviewer


def mark_done(directory: Path) -> None:
    (Path(directory) / DONE_MARKER).write_text("ok", encoding="utf-8")


def is_done(directory: Path) -> bool:
    return (Path(directory) / DONE_MARKER).is_file()
