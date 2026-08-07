"""Watermarks and idempotency.

Two guarantees:
  1. A PR is never reviewed twice at the same head SHA (push new commits, get a new review).
  2. A comment is never posted twice, even if local state is lost — the marker embedded
     in the comment body is checked against the PR's existing comments.

State transitions are immutable: `mark_reviewed` returns a new State.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

DEFAULT_STATE_PATH = Path("state/watermark.json")

MARKER_PREFIX = "<!-- prime-agent-review:"
MARKER_SUFFIX = " -->"

LANE_OPEN = "open"
LANE_MERGED = "merged"
LANES = (LANE_OPEN, LANE_MERGED)


class StateError(RuntimeError):
    """State file is unreadable or malformed."""


@dataclass(frozen=True)
class State:
    """`reviewed` maps "<lane>:<pr_number>" to the head SHA last reviewed."""

    reviewed: Mapping[str, str]
    merged_cursor: str | None = None

    @staticmethod
    def empty() -> State:
        return State(reviewed=MappingProxyType({}), merged_cursor=None)


def load_state(path: Path | str = DEFAULT_STATE_PATH) -> State:
    """Read state from disk. A missing file is a cold start, not an error."""
    state_path = Path(path)
    if not state_path.is_file():
        return State.empty()

    try:
        raw = json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise StateError(f"Could not read state file {state_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise StateError(f"State file {state_path} must contain a JSON object")

    reviewed = raw.get("reviewed", {})
    if not isinstance(reviewed, dict):
        raise StateError(f"State file {state_path}: 'reviewed' must be an object")

    return State(
        reviewed=MappingProxyType({str(k): str(v) for k, v in reviewed.items()}),
        merged_cursor=raw.get("merged_cursor") or None,
    )


def save_state(state: State, path: Path | str = DEFAULT_STATE_PATH) -> None:
    """Write state atomically, so an interrupted sweep cannot corrupt it."""
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reviewed": dict(state.reviewed),
        "merged_cursor": state.merged_cursor,
    }

    temp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    try:
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(state_path)
    except OSError as exc:
        raise StateError(f"Could not write state file {state_path}: {exc}") from exc


def review_key(lane: str, number: int) -> str:
    if lane not in LANES:
        raise ValueError(f"Unknown lane {lane!r}; expected one of {LANES}")
    return f"{lane}:{number}"


def is_reviewed(state: State, lane: str, number: int, head_sha: str) -> bool:
    """True when this exact head SHA has already been reviewed in this lane."""
    return state.reviewed.get(review_key(lane, number)) == head_sha


def mark_reviewed(state: State, lane: str, number: int, head_sha: str) -> State:
    """Return a new State recording this PR as reviewed at this head SHA."""
    updated = dict(state.reviewed)
    updated[review_key(lane, number)] = head_sha
    return replace(state, reviewed=MappingProxyType(updated))


def set_merged_cursor(state: State, cursor: str | None) -> State:
    return replace(state, merged_cursor=cursor)


def build_marker(head_sha: str) -> str:
    """Hidden HTML marker embedded in every posted comment."""
    return f"{MARKER_PREFIX}{head_sha}{MARKER_SUFFIX}"


def has_marker(comment_bodies: object, head_sha: str) -> bool:
    """True when any existing comment already carries this head SHA's marker."""
    if not isinstance(comment_bodies, (list, tuple)):
        return False
    marker = build_marker(head_sha)
    return any(marker in str(body) for body in comment_bodies)
