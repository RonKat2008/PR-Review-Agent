"""Rebuild each ablation arm from one instance's recordings, at zero model cost.

The ladder: seat-i (single pass) -> ensemble (agreement 1, no judge) ->
ensemble+judge -> full (plus skeptic; refuted findings count as not reported).
The live run *is* `full`; the scorer checks the replayed `full` matches it."""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from ..citations import head_line_counts, paths_needing_head_counts, validate_citations
from ..ensemble import ensemble_review_detailed
from ..github import PullRequest
from ..refute import RefuteError, refute_findings
from ..review import Verdict, VerdictError, parse_verdict
from .recording import (
    Call,
    Recorder,
    ReplayMiss,
    replay_model_fn,
    replay_reviewer,
    seat_calls,
)
from .runner import HeadFileStore, replay_runner

ARMS = ("seat-1", "seat-2", "seat-3", "ensemble", "ensemble+judge", "full")
# Charged to every arm: the intent pass runs once per live review, before any
# arm's own calls, so no arm could have been run without paying for it.
SHARED_ROLES = ("aux",)
HEAD_FILES = "head_files.json"
JUDGED_ARMS = frozenset({"ensemble+judge", "full"})
_PR_URL_RE = re.compile(r"^https://github\.com/([^/]+)/([^/]+)/pull/\d+$")


@dataclass(frozen=True)
class ArmResult:
    arm: str
    verdict: Verdict | None
    notes: tuple[str, ...] = ()
    replay_miss: bool = False
    error: str = ""
    dropped: int = 0


def build_arm(arm: str, instance_dir: Path, pr: PullRequest, diff: str, lane: str,
              prompts_dir: Path | str, seat_models: Sequence[str]) -> ArmResult:
    recorder = Recorder(instance_dir)
    try:
        verdict, notes, dropped = _build(arm, recorder, instance_dir, pr, diff, lane, prompts_dir,
                                         seat_models)
    except ReplayMiss as exc:
        return ArmResult(arm, None, replay_miss=True, error=str(exc))
    except (VerdictError, IndexError, RefuteError, OSError, ValueError) as exc:
        return ArmResult(arm, None, error=str(exc))
    return ArmResult(arm, verdict, notes, dropped=dropped)


def arm_calls(recorder: Recorder, arm: str, seat_models: Sequence[str]) -> tuple[Call, ...]:
    """Exactly the recorded calls this arm consumes, for cost and wall time.

    A seat arm pays for its own seat; the ensemble pays for all three; the
    judge and skeptic passes are added by the arms that run them. The shared
    passes are charged to every arm, since every live review ran them."""
    calls = recorder.calls()
    seats = seat_calls(recorder, seat_models)
    if arm.startswith("seat-"):
        index = int(arm[5:]) - 1
        own = [seats[index]] if 0 <= index < len(seats) else []
    else:
        own = list(seats)
        if arm in JUDGED_ARMS:
            own += [c for c in calls if c.role == "judge"]
        if arm == "full":
            own += [c for c in calls if c.role == "skeptic"]
    shared = [c for c in calls if c.role in SHARED_ROLES]
    return tuple(shared) + tuple(c for c in own if c is not None)


def _build(arm, recorder, instance_dir, pr, diff, lane, prompts_dir,
           seat_models) -> tuple[Verdict, tuple[str, ...], int]:
    if arm.startswith("seat-"):
        call = seat_calls(recorder, seat_models)[int(arm[5:]) - 1]
        if call is None:
            raise ReplayMiss(f"no recorded call for {arm}")
        return parse_verdict(call.response), (), 0
    verdict, notes = _ensemble(arm, recorder, pr, diff, lane, prompts_dir, seat_models)
    if arm != "full":
        return verdict, notes, 0
    # Production order is validate-then-refute (sweep.py runs citation validation
    # before the skeptic pass). Refuting first would send a skeptic prompt for a
    # fabricated finding that citation validation would have dropped -- a prompt
    # that was never recorded live, so replay would raise ReplayMiss and lose the
    # whole instance for exactly the PRs with fabrications.
    validated, dropped = apply_citations(verdict, diff, instance_dir, _slug_from_url(pr.url), pr.head_sha)
    findings, refute_notes = _refute(validated.introduces, diff, recorder, prompts_dir)
    return replace(validated, introduces=findings), notes + refute_notes, dropped


def _ensemble(arm, recorder, pr, diff, lane, prompts_dir, seat_models):
    """`ensemble_review_detailed` records a failed reviewer run and carries on —
    correct for a flaky subagent, wrong for a replay, where a missing recording
    means this arm is not the live pipeline. Capture the misses the ensemble
    swallows and surface them, exactly as `_refute` does for the skeptic."""
    misses: list[ReplayMiss] = []
    replay = replay_reviewer(recorder, seat_models)

    def reviewer(pr_, payload, lane_):
        try:
            return replay(pr_, payload, lane_)
        except ReplayMiss as exc:
            misses.append(exc)
            raise

    judge = replay_model_fn(recorder, "judge") if arm in JUDGED_ARMS else None
    try:
        verdict, notes = ensemble_review_detailed(
            pr, diff, lane, reviewer, size=3, min_agreement=1,
            judge_fn=judge, prompts_dir=prompts_dir,
        )
    except VerdictError:
        if misses:
            raise ReplayMiss(f"{len(misses)} seat call(s) had no recording") from None
        raise
    if misses:
        raise ReplayMiss(f"{len(misses)} seat call(s) had no recording")
    return verdict, notes


def _slug_from_url(url: str) -> str:
    """`https://github.com/{owner}/{name}/pull/{number}` -> `owner/name`."""
    match = _PR_URL_RE.match(url)
    if match is None:
        raise ValueError(f"cannot parse repo slug from PR url: {url!r}")
    return f"{match.group(1)}/{match.group(2)}"


def _refute(introduces, diff, recorder, prompts_dir):
    """`refute_findings` fails open per finding (a broken skeptic costs a note,
    not a finding) — but a *missing recording* is not a broken skeptic, it is
    a gap in the corpus, and must surface as `replay_miss` rather than render
    as a clean, unrefuted finding."""
    misses: list[ReplayMiss] = []
    replay = replay_model_fn(recorder, "skeptic")

    def skeptic(prompt: str) -> str:
        try:
            return replay(prompt)
        except ReplayMiss as exc:
            misses.append(exc)
            raise

    findings, notes = refute_findings(introduces, diff, skeptic, prompts_dir)
    if misses:
        raise ReplayMiss(f"{len(misses)} skeptic prompt(s) had no recording")
    return findings, notes


def apply_citations(verdict: Verdict, diff: str, instance_dir: Path, repo_slug: str,
                    head_sha: str) -> tuple[Verdict, int]:
    store = HeadFileStore(Path(instance_dir) / HEAD_FILES)
    needed = paths_needing_head_counts(verdict.introduces, diff, None)
    counts = head_line_counts(repo_slug, head_sha, needed, replay_runner(store)) if needed else {}
    kept, _ = validate_citations(verdict.introduces, diff, counts, None)
    return replace(verdict, introduces=kept), len(verdict.introduces) - len(kept)
