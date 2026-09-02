"""Judge-merge (P15): semantic dedup of ensemble findings before agreement is counted.

The ensemble matches findings across runs on `(file, line // LINE_BUCKET,
severity)` — deterministic and cheap, but blind to meaning. When two seats
describe the SAME defect thirty lines apart, or at different severities, the
key match fails: agreement is undercounted (a real 2/3 finding renders as two
1/3 duplicates) and, under a min_agreement of 2, a corroborated finding can be
dropped outright.

This module adds one model call that sees every grouped finding TOGETHER —
safe precisely because it runs after the seats are done, so there is nothing
left to anchor on — and proposes which groups describe the same underlying
defect. The judge's power is deliberately bounded by deterministic validation:

  - it can only MERGE groups, never drop or invent one;
  - a cluster may only span groups in the SAME file — the worst failure mode
    of a chatty judge is collapsing two distinct bugs into one, and the
    same-file rule caps that blast radius;
  - malformed output (bad JSON, unknown ids, overlapping clusters) raises
    `JudgeError`, and the caller falls back to the deterministic grouping.

Standalone on purpose: `ensemble` imports this module, so this module must not
import `ensemble` back. It speaks in `Candidate` rows the caller builds from
its own internal groups.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .review import VerdictError, _extract_json

ModelFn = Callable[[str], str]

PROMPT_FILE = "judge_merge.md"

# The judge sees claims and evidence, not the diff — trim each so a large
# ensemble cannot balloon this one prompt past what a cheap call should cost.
_EVIDENCE_LIMIT = 400


class JudgeError(RuntimeError):
    """The judge returned something that is not a usable cluster proposal."""


@dataclass(frozen=True)
class Candidate:
    """One finding group as the judge sees it: a stable index plus the
    representative's identifying fields. `index` is the caller's group index —
    the judge's output refers to groups only through it."""

    index: int
    file: str
    line: int | None
    severity: str
    claim: str
    evidence: str


def propose_clusters(
    candidates: Sequence[Candidate],
    model_fn: ModelFn,
    prompts_dir: Path | str,
) -> tuple[tuple[int, ...], ...]:
    """Ask the judge which candidates describe the same defect.

    Returns validated clusters of candidate indices, each with at least two
    members, disjoint, and all in the same file. Raises `JudgeError` on any
    response that fails validation — the caller decides what a failure costs
    (it should cost nothing: fall back to the deterministic grouping).
    """
    template = _read_prompt(Path(prompts_dir))
    raw = model_fn(build_judge_prompt(template, candidates))
    return _validate(_parse_clusters(raw), candidates)


def build_judge_prompt(template: str, candidates: Sequence[Candidate]) -> str:
    rows = "\n\n".join(_render_candidate(c) for c in candidates)
    return f"{template}\n\n## Findings\n\n{rows}\n"


def _render_candidate(candidate: Candidate) -> str:
    location = f"{candidate.file}:{candidate.line}" if candidate.line else candidate.file
    lines = [
        f"[{candidate.index}] {candidate.severity} `{location}` — {candidate.claim}"
    ]
    if candidate.evidence:
        lines.append(f"    evidence: {candidate.evidence[:_EVIDENCE_LIMIT]}")
    return "\n".join(lines)


def _read_prompt(directory: Path) -> str:
    path = directory / PROMPT_FILE
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise JudgeError(f"Could not read prompt {path}: {exc}") from exc


def _parse_clusters(raw: str) -> tuple[tuple[int, ...], ...]:
    try:
        payload = _extract_json(raw)
    except VerdictError as exc:
        raise JudgeError(f"Judge response is not usable: {exc}") from exc

    clusters = payload.get("clusters")
    if not isinstance(clusters, list):
        raise JudgeError("Judge response is missing a 'clusters' array")

    parsed: list[tuple[int, ...]] = []
    for cluster in clusters:
        if not isinstance(cluster, list):
            raise JudgeError(f"Every cluster must be an array, got {type(cluster).__name__}")
        try:
            members = tuple(int(member) for member in cluster)
        except (TypeError, ValueError) as exc:
            raise JudgeError(f"Cluster members must be integers: {exc}") from exc
        # A singleton "cluster" merges nothing; a chatty judge that lists every
        # finding on its own is harmless, so drop these instead of rejecting.
        if len(members) >= 2:
            parsed.append(members)
    return tuple(parsed)


def _validate(
    clusters: tuple[tuple[int, ...], ...],
    candidates: Sequence[Candidate],
) -> tuple[tuple[int, ...], ...]:
    by_index = {c.index: c for c in candidates}
    seen: set[int] = set()

    for cluster in clusters:
        for member in cluster:
            if member not in by_index:
                raise JudgeError(f"Cluster names unknown finding index {member}")
            if member in seen:
                raise JudgeError(f"Finding index {member} appears in more than one cluster")
            seen.add(member)

        files = {by_index[m].file for m in cluster}
        if len(files) > 1:
            raise JudgeError(
                "Cluster spans different files "
                f"({', '.join(sorted(files))}) — same-file merges only"
            )

    return clusters
