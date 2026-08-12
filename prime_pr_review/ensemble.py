"""Ensemble agreement in place of self-reported confidence.

A single subagent's `confidence` field is an assertion, not a measurement: every
live sweep so far returned 0.95 on every PR, including one where nothing at all was
found. This module replaces that with an observed quantity — run the same PR through
several independent reviews and treat how many agree on a finding as the confidence
signal, instead of trusting any one run's self-assessment.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from .github import PullRequest
from .review import Finding, FixClaim, Severity, Verdict, VerdictError, parse_verdict

# (pull_request, review_payload, lane) -> raw verdict JSON text. Shaped identically
# to sweep.py's Reviewer alias but redefined locally so this module has no import
# dependency on sweep.py (which will import this module once it is wired in).
Reviewer = Callable[[PullRequest, str, str], str]

# Independent reviewers essentially never agree on the exact line of a defect even
# when they agree the defect itself exists — one counts from the hunk header, one
# from the start of the enclosing function, one is thrown off by a line of diff
# context. Findings in the same file, of the same severity, whose lines fall within
# this many lines of the cluster's first (anchor) member are treated as one finding
# rather than several. Wide enough to absorb that typical drift, narrow enough that
# two genuinely distinct nearby bugs don't collapse into one.
LINE_WINDOW = 3

# A missed fix claim costs nothing — at worst a PR is under-credited for work it
# already did — so fix claims are unioned across runs rather than required to
# corroborate. Only findings, which drive the merge-blocking gate, need agreement.
# This is the asymmetry: min_agreement for findings is caller-configurable and
# defaults to 2; for fixes it is fixed at 1.
_FIX_MIN_AGREEMENT = 1


def ensemble_review(
    pr: PullRequest,
    payload: str,
    lane: str,
    reviewer: Reviewer,
    size: int = 3,
    min_agreement: int = 2,
) -> Verdict:
    """Run `reviewer` `size` times and merge the results by observed agreement.

    `confidence` on the returned Verdict is the agreement ratio (matches / size) of
    the most strongly corroborated surviving finding — not an average across
    findings and not a self-report from any single run. `size=1` is a pass-through:
    the lone run's own Verdict is returned untouched, with no matching or
    agreement filtering applied.

    Raises VerdictError if every run fails or returns an unparseable verdict.
    """
    verdicts, errors = _run_all(pr, payload, lane, reviewer, size)
    if not verdicts:
        raise VerdictError(f"All {size} ensemble runs failed: {'; '.join(errors)}")
    if size == 1:
        return verdicts[0]

    introduces, confidence = _merge_findings(verdicts, size, min_agreement)
    return Verdict(
        introduces=introduces,
        fixes=_merge_fixes(verdicts),
        confidence=confidence,
        scope=verdicts[0].scope,
        blast_radius=verdicts[0].blast_radius,
    )


def _run_all(
    pr: PullRequest, payload: str, lane: str, reviewer: Reviewer, size: int
) -> tuple[list[Verdict], list[str]]:
    """Call `reviewer` `size` times. A failing or unparseable run is recorded, not fatal."""
    verdicts: list[Verdict] = []
    errors: list[str] = []
    for _ in range(size):
        try:
            verdicts.append(parse_verdict(reviewer(pr, payload, lane)))
        except Exception as exc:  # noqa: BLE001 - a reviewer run may fail in any manner
            errors.append(str(exc))
    return verdicts, errors


def _merge_findings(
    verdicts: list[Verdict], size: int, min_agreement: int
) -> tuple[tuple[Finding, ...], float]:
    """Cluster findings across runs and keep those meeting the agreement bar.

    Confidence is set from the single strongest (highest matches/size) surviving
    cluster, never averaged across clusters — a weak, poorly-corroborated finding
    must not dilute the confidence earned by a strong one.
    """
    entries = [
        (run_idx, finding)
        for run_idx, verdict in enumerate(verdicts)
        for finding in verdict.introduces
    ]

    survivors: list[Finding] = []
    strongest_ratio = 0.0
    for cluster in _cluster_findings(entries):
        matches = len({run_idx for run_idx, _ in cluster})
        if matches < min_agreement:
            continue
        survivors.append(_represent(cluster, matches, size))
        strongest_ratio = max(strongest_ratio, matches / size)

    return tuple(survivors), strongest_ratio


def _cluster_findings(
    entries: list[tuple[int, Finding]],
) -> list[list[tuple[int, Finding]]]:
    """Group findings by (file, severity, line proximity within LINE_WINDOW)."""
    groups: dict[tuple[str, Severity], list[list[tuple[int, Finding]]]] = {}
    for entry in entries:
        _, finding = entry
        key = (finding.file, finding.severity)
        clusters = groups.setdefault(key, [])
        target = _matching_cluster(clusters, finding.line)
        if target is None:
            clusters.append([entry])
        else:
            target.append(entry)
    return [cluster for clusters in groups.values() for cluster in clusters]


def _matching_cluster(
    clusters: list[list[tuple[int, Finding]]], line: int | None
) -> list[tuple[int, Finding]] | None:
    """Find a cluster whose anchor (its first member's line) is within LINE_WINDOW."""
    for cluster in clusters:
        anchor = cluster[0][1].line
        if anchor is None and line is None:
            return cluster
        if anchor is not None and line is not None and abs(anchor - line) <= LINE_WINDOW:
            return cluster
    return None


def _represent(cluster: list[tuple[int, Finding]], matches: int, size: int) -> Finding:
    """Pick the best finding to represent a surviving cluster and stamp its agreement.

    Highest severity first, then longest evidence — proxies for which of the
    (near-)duplicate reports is most useful to actually show a reader. Severity is
    already fixed by the cluster's grouping key, so in practice this tie-breaks on
    evidence length; the severity comparison stays in case that ever changes.
    """
    _, best = max(cluster, key=lambda e: (-e[1].severity.rank, len(e[1].evidence)))
    return replace(best, corroboration=f"{matches}/{size} reviewers")


def _merge_fixes(verdicts: list[Verdict]) -> tuple[FixClaim, ...]:
    """Union fix claims across runs, deduplicated by normalized claim text.

    See _FIX_MIN_AGREEMENT above for why the bar here is lower than for findings.
    """
    groups: dict[str, list[FixClaim]] = {}
    for verdict in verdicts:
        for fix in verdict.fixes:
            groups.setdefault(fix.claim.strip().lower(), []).append(fix)

    return tuple(
        max(items, key=lambda f: len(f.evidence))
        for items in groups.values()
        if len(items) >= _FIX_MIN_AGREEMENT
    )
