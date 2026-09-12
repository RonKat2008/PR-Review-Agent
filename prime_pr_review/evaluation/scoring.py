"""Score one arm's verdict against SWE-CARE reference comments.

A finding matches a reference comment on the same path when its line falls in
the comment's range widened by `window` lines. Refuted findings are treated as
not reported. Metrics are micro-averaged across instances. `fabrication_rate`
is dropped / (kept + dropped): the share of raw findings that pointed nowhere.
Line-less (file-level) reference comments can never satisfy `match()`, so they
are excluded from the recall denominator/numerator but still count toward
file-level matching."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..review import Finding, Verdict
from .corpus import ReferenceComment

DEFAULT_WINDOW = 5


@dataclass(frozen=True)
class InstanceScore:
    findings: int
    matched_findings: int
    file_matched_findings: int
    refs: int
    matched_refs: int
    dropped: int


@dataclass(frozen=True)
class Aggregate:
    arm: str
    mode: str
    instances: int
    precision: float
    recall: float
    file_precision: float
    fabrication_rate: float
    findings_per_pr: float


def match(finding: Finding, ref: ReferenceComment, window: int = DEFAULT_WINDOW) -> bool:
    if finding.file != ref.path or finding.line is None or ref.line is None:
        return False
    lo = ref.start_line if ref.start_line is not None else ref.line
    return lo - window <= finding.line <= ref.line + window


def score_instance(verdict: Verdict | None, refs: Sequence[ReferenceComment], dropped: int = 0,
                   window: int = DEFAULT_WINDOW) -> InstanceScore:
    """Score a verdict's findings against reference comments.

    Line-less references are excluded from `refs`/`matched_refs` (recall)
    since `match()` can never satisfy them, but still count toward
    `file_matched_findings`.
    """
    findings = tuple(f for f in (verdict.introduces if verdict else ()) if not f.refuted)
    anchored = tuple(r for r in refs if r.line is not None)
    matched = sum(any(match(f, r, window) for r in refs) for f in findings)
    file_matched = sum(any(f.file == r.path for r in refs) for f in findings)
    matched_refs = sum(any(match(f, r, window) for f in findings) for r in anchored)
    return InstanceScore(len(findings), matched, file_matched, len(anchored), matched_refs, dropped)


def aggregate(arm: str, mode: str, scores: Sequence[InstanceScore], fabricated_total: int) -> Aggregate:
    findings = sum(s.findings for s in scores)
    refs = sum(s.refs for s in scores)
    return Aggregate(
        arm=arm, mode=mode, instances=len(scores),
        precision=_ratio(sum(s.matched_findings for s in scores), findings),
        recall=_ratio(sum(s.matched_refs for s in scores), refs),
        file_precision=_ratio(sum(s.file_matched_findings for s in scores), findings),
        fabrication_rate=_ratio(fabricated_total, findings + fabricated_total),
        findings_per_pr=_ratio(findings, len(scores)),
    )


def _ratio(num: int, den: int) -> float:
    return num / den if den else 0.0
