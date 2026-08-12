"""Ensemble agreement (P2): confidence as an observed quantity, not a self-report.

Live sweeps returned 95% confidence on every PR reviewed, including one where the
model found nothing to say. A single subagent call has no way to know how sure it
should be -- self-reported confidence is decoration, and `min_confidence` was
filtering nothing.

The fix: run the same review `size` times independently, keep only what the runs
agree on, and let `Verdict.confidence` be the fraction of runs that actually
agreed -- an observed quantity instead of a number the model made up.

Standalone module: depends only on `review` (the Verdict schema and the tolerant
parser) and `sweep` (for the `Reviewer` callable type). Wiring this into a sweep is
left to the caller -- nothing here changes how a sweep is assembled.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace

from collections.abc import Callable

from .github import PullRequest
from .review import Finding, FixClaim, Severity, Verdict, VerdictError, parse_verdict

# Structurally identical to sweep.Reviewer, declared locally instead of imported:
# sweep imports this module to wire the ensemble in, so importing sweep back
# from here would be a circular import.
Reviewer = Callable[[PullRequest, str, str], str]

# Independent reviewer runs almost never agree on the exact line of the same
# defect -- one counts from the hunk header, another from the decorator, a third
# rounds to the enclosing statement -- but they land within a few lines of each
# other. `LINE_BUCKET` quantizes `Finding.line` into `line // LINE_BUCKET` so
# nearby reports collapse onto the same key instead of requiring an exact match.
# The width is a direct precision/recall tradeoff:
#   - too wide, and two genuinely different bugs a few lines apart in the same
#     file merge into one finding -- a real, distinct defect silently vanishes
#     into another's corroboration count.
#   - too narrow, and three reports of the identical bug at lines 41, 43 and 44
#     land in three different buckets, agreement never accumulates, and a true
#     positive gets dropped by `min_agreement` as if no one else saw it.
# 5 is the chosen midpoint for a typical single-statement or single-block
# defect. Note this is a fixed-width bucket, not a pairwise distance: two lines
# assigned to the same bucket always match regardless of how close either is to
# the bucket's own edge, so two lines only a couple of lines apart can still
# land in different buckets (9 -> bucket 1, 11 -> bucket 2). That edge effect
# is accepted for the determinism and O(n) grouping a fixed key buys over
# pairwise clustering.
LINE_BUCKET = 5

_CLAIM_PUNCTUATION_RE = re.compile(r"[^\w\s]")


def ensemble_review(
    pr: PullRequest,
    payload: str,
    lane: str,
    reviewer: Reviewer,
    size: int = 3,
    min_agreement: int = 2,
) -> Verdict:
    """Run `reviewer` against the same `(pr, payload, lane)` `size` times and
    merge the results into one Verdict via observed agreement, not any single
    run's self-reported confidence.

    `size=1` is the off switch and the cost lever: exactly one call, and the
    result is `parse_verdict` of its response, completely unchanged -- no
    matching, no corroboration rewriting, nothing an ensemble of one could
    disagree with.

    For `size > 1`:
      1. Each run is parsed independently (`_collect_runs`); a run that raises
         or fails to parse as a Verdict is recorded and skipped. If every run
         fails, raises `VerdictError` naming how many did.
      2. `introduces` findings are matched across runs on `(file, line //
         LINE_BUCKET, severity)` and kept when at least `min_agreement`
         distinct runs reported the same key (`_group_findings`).
      3. `confidence` is the agreement ratio of the most-corroborated
         surviving finding, or -- when nothing survives -- the fraction of
         runs that independently agreed the diff was clean (`_confidence`).
      4. `fixes` are unioned across every run at an effective min_agreement of
         1 (`_merge_fixes`): a fix claim only one run happened to mention is
         still reported, because missing a fix costs nothing -- it is never
         gated, never blocks a merge, never suppresses a real finding --
         while dropping it would understate what the PR does. This is
         deliberately the opposite of `introduces`, where a single-run
         finding with no corroboration is exactly the noise this module
         exists to remove.
      5. `scope`, `blast_radius`, `files`, and `manual_checks` are copied from
         the first run that parsed successfully -- not merged. Each comes
         from a separate deterministic pass (the P8 intent check, the P9
         blast-radius walk) or a whole-diff inventory (the per-file
         walkthrough, manual-check suggestions), not a set of independently
         falsifiable claims. There is no sound way to "agree" on two
         different intent summaries the way there is for two reports of the
         same bug at the same line, so ensembling them would just be
         arbitrarily picking a winner while pretending it was a vote.
    """
    if size == 1:
        return parse_verdict(reviewer(pr, payload, lane))

    runs, failures = _collect_runs(pr, payload, lane, reviewer, size)
    if not runs:
        raise VerdictError(
            f"All {failures}/{size} ensemble reviewer runs failed to produce a usable verdict"
        )

    groups = _group_findings(runs)
    survivors = tuple(group for group in groups if group.matches >= min_agreement)
    first = runs[0]

    return Verdict(
        introduces=tuple(_finalize(group, size) for group in survivors),
        fixes=_merge_fixes(runs),
        confidence=_confidence(survivors, runs, size),
        scope=first.scope,
        blast_radius=first.blast_radius,
        files=first.files,
        manual_checks=first.manual_checks,
    )


def _collect_runs(
    pr: PullRequest, payload: str, lane: str, reviewer: Reviewer, size: int
) -> tuple[tuple[Verdict, ...], int]:
    """Call `reviewer` `size` times and parse each response into a Verdict.

    A run can fail two ways: the call itself raises -- a subagent can fail in
    any manner, a timeout or a transport error included -- or it returns text
    `parse_verdict` cannot use (`VerdictError`). Both are recorded as a
    failure and skipped; one bad call must not sink a review the other runs
    completed successfully.
    """
    runs: list[Verdict] = []
    failures = 0
    for _ in range(size):
        try:
            runs.append(parse_verdict(reviewer(pr, payload, lane)))
        except Exception:  # noqa: BLE001 - a subagent run may fail in any manner
            failures += 1
    return tuple(runs), failures


@dataclass(frozen=True)
class _Group:
    """Every finding across all runs that normalized to the same match key,
    plus how many DISTINCT runs reported it. A run that reports the same key
    twice (two findings that both round into one bucket) still counts once --
    agreement is measured in reviewers, not in raw finding count."""

    candidates: tuple[Finding, ...]
    matches: int


def _finding_key(finding: Finding) -> tuple[str, int | None, Severity]:
    """The normalized match key: file, quantized line, severity. See
    `LINE_BUCKET` for why the line is bucketed rather than compared exactly."""
    bucket = finding.line // LINE_BUCKET if finding.line is not None else None
    return (finding.file, bucket, finding.severity)


def _group_findings(runs: Sequence[Verdict]) -> tuple[_Group, ...]:
    """Every `introduces` finding from every run, grouped by `_finding_key`."""
    candidates: dict[tuple[str, int | None, Severity], list[Finding]] = {}
    run_counts: dict[tuple[str, int | None, Severity], int] = {}

    for run in runs:
        keys_in_this_run: set[tuple[str, int | None, Severity]] = set()
        for finding in run.introduces:
            key = _finding_key(finding)
            candidates.setdefault(key, []).append(finding)
            keys_in_this_run.add(key)
        for key in keys_in_this_run:
            run_counts[key] = run_counts.get(key, 0) + 1

    return tuple(
        _Group(candidates=tuple(items), matches=run_counts[key])
        for key, items in candidates.items()
    )


def _finalize(group: _Group, size: int) -> Finding:
    """The kept representative for one surviving group, with the agreement
    ratio appended to `corroboration` -- appended, never overwritten, so
    linter evidence from the P3 pre-pass (e.g. `bandit:B608`) and the
    ensemble's own agreement ratio both survive on the same finding."""
    representative = _pick_representative(group.candidates)
    tag = f"{group.matches}/{size} reviewers"
    corroboration = (
        f"{representative.corroboration} · {tag}" if representative.corroboration else tag
    )
    return replace(representative, corroboration=corroboration)


def _pick_representative(candidates: Sequence[Finding]) -> Finding:
    """Highest severity, then longest evidence. Every candidate in one group
    already shares the same severity -- it is part of the match key -- so in
    practice this picks the most detailed report of the finding: the richest
    evidence to show the reader, chosen deterministically."""
    return min(candidates, key=lambda f: (f.severity.rank, -len(f.evidence)))


def _confidence(survivors: Sequence[_Group], runs: Sequence[Verdict], size: int) -> float:
    """The agreement ratio of the strongest surviving finding, or -- when
    nothing survived `min_agreement` -- the fraction of runs that
    independently reported a clean diff.

    Both ratios divide by `size` (the requested ensemble size), not the count
    of runs that happened to succeed: a run that never responded is a missing
    opinion, not a vote either way, and confidence should reflect that the
    full ensemble was not actually heard from.
    """
    if survivors:
        return max(group.matches for group in survivors) / size
    clean_runs = sum(1 for run in runs if not run.introduces)
    return clean_runs / size


def _merge_fixes(runs: Sequence[Verdict]) -> tuple[FixClaim, ...]:
    """Union every fix claim across all runs, deduplicated by a loose
    normalized-text key -- min_agreement=1 in effect. Unlike `introduces`, a
    fix claim is not a warning that can mislead someone into an unsafe merge,
    so a claim only one run happened to mention is kept rather than discarded
    for lack of corroboration. The asymmetry is deliberate: under-reporting a
    fix costs a missing line of praise, while under-reporting a bug at the
    same bar this module exists to raise costs a regression that ships.
    """
    seen: dict[str, FixClaim] = {}
    for run in runs:
        for fix in run.fixes:
            key = _normalize_claim(fix.claim)
            seen.setdefault(key, fix)
    return tuple(seen.values())


def _normalize_claim(text: str) -> str:
    """Order- and punctuation-insensitive dedup key for a fix claim.

    Deliberately loose -- "fingerprint-ish", not a hash: lowercase, strip
    punctuation, sort tokens. Compare `feedback.claim_fingerprint`, which
    hashes to a fixed-width digest for persisted, cross-sweep rejection
    matching; that precision is wasted here, where the only consumer is
    `_merge_fixes` within a single call and the worst case of a loose match
    is "two similarly-worded fix claims render as one line" -- never a
    suppressed finding.
    """
    normalized = _CLAIM_PUNCTUATION_RE.sub(" ", text.lower())
    return " ".join(sorted(normalized.split()))
