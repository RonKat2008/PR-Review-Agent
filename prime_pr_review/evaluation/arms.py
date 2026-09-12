"""Rebuild each ablation arm from one instance's recordings, at zero model cost.

The ladder: seat-i (single pass) -> ensemble (agreement 1, no judge) ->
ensemble+judge -> full (plus skeptic; refuted findings count as not reported).
The live run *is* `full`; the scorer checks the replayed `full` matches it."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from ..citations import head_line_counts, paths_needing_head_counts, validate_citations
from ..ensemble import ensemble_review_detailed
from ..github import PullRequest
from ..refute import refute_findings
from ..review import Verdict, VerdictError, parse_verdict
from .recording import Recorder, ReplayMiss, replay_model_fn, replay_reviewer
from .runner import HeadFileStore, replay_runner

ARMS = ("seat-1", "seat-2", "seat-3", "ensemble", "ensemble+judge", "full")
HEAD_FILES = "head_files.json"
JUDGED_ARMS = frozenset({"ensemble+judge", "full"})


@dataclass(frozen=True)
class ArmResult:
    arm: str
    verdict: Verdict | None
    notes: tuple[str, ...] = ()
    replay_miss: bool = False
    error: str = ""


def build_arm(arm: str, instance_dir: Path, pr: PullRequest, diff: str, lane: str,
              prompts_dir: Path | str) -> ArmResult:
    recorder = Recorder(instance_dir)
    try:
        verdict, notes = _build(arm, recorder, pr, diff, lane, prompts_dir)
    except ReplayMiss as exc:
        return ArmResult(arm, None, replay_miss=True, error=str(exc))
    except (VerdictError, IndexError) as exc:
        return ArmResult(arm, None, error=str(exc))
    return ArmResult(arm, verdict, notes)


def _build(arm, recorder, pr, diff, lane, prompts_dir) -> tuple[Verdict, tuple[str, ...]]:
    if arm.startswith("seat-"):
        seats = recorder.calls("seat")
        if not seats:
            raise ReplayMiss("no recorded seats")
        return parse_verdict(seats[int(arm[5:]) - 1].response), ()
    if not recorder.calls("seat"):
        raise ReplayMiss("no recorded seats")
    judge = replay_model_fn(recorder, "judge") if arm in JUDGED_ARMS else None
    verdict, notes = ensemble_review_detailed(
        pr, diff, lane, replay_reviewer(recorder), size=3, min_agreement=1,
        judge_fn=judge, prompts_dir=prompts_dir,
    )
    if arm != "full":
        return verdict, notes
    findings, refute_notes = refute_findings(
        verdict.introduces, diff, replay_model_fn(recorder, "skeptic"), prompts_dir
    )
    return replace(verdict, introduces=findings), notes + refute_notes


def apply_citations(verdict: Verdict, diff: str, instance_dir: Path, repo_slug: str,
                    head_sha: str) -> tuple[Verdict, int]:
    store = HeadFileStore(Path(instance_dir) / HEAD_FILES)
    needed = paths_needing_head_counts(verdict.introduces, diff, None)
    counts = head_line_counts(repo_slug, head_sha, needed, replay_runner(store)) if needed else {}
    kept, _ = validate_citations(verdict.introduces, diff, counts, None)
    return replace(verdict, introduces=kept), len(verdict.introduces) - len(kept)
