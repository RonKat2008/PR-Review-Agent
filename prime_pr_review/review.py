"""The review verdict: its schema, tolerant parsing, the confidence gate, and rendering.

Subagents return JSON rather than prose. Prose cannot be thresholded, deduplicated,
or counted against a rate cap — the structure is what makes the safety gates possible.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum

from .github import PullRequest
from .state import LANE_MERGED, build_marker

FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

SEVERITY_ORDER = ("CRITICAL", "HIGH", "MEDIUM", "LOW")
BLOCKING_SEVERITIES = frozenset({"CRITICAL", "HIGH"})


class VerdictError(ValueError):
    """A subagent returned something that is not a usable verdict."""


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

    @property
    def rank(self) -> int:
        return SEVERITY_ORDER.index(self.value)


@dataclass(frozen=True)
class Finding:
    """A bug the PR is claimed to introduce."""

    file: str
    line: int | None
    severity: Severity
    claim: str
    evidence: str


@dataclass(frozen=True)
class FixClaim:
    """A bug the PR is claimed to fix."""

    claim: str
    evidence: str


@dataclass(frozen=True)
class Verdict:
    introduces: tuple[Finding, ...]
    fixes: tuple[FixClaim, ...]
    confidence: float

    @property
    def has_blocking(self) -> bool:
        return any(f.severity.value in BLOCKING_SEVERITIES for f in self.introduces)

    @property
    def is_silent(self) -> bool:
        """Nothing found either way — not worth posting."""
        return not self.introduces and not self.fixes

    @property
    def worst_severity(self) -> Severity | None:
        if not self.introduces:
            return None
        return min((f.severity for f in self.introduces), key=lambda s: s.rank)


def parse_verdict(raw: str) -> Verdict:
    """Parse a subagent response into a Verdict, tolerating code fences and stray prose."""
    payload = _extract_json(raw)

    try:
        introduces = tuple(_parse_finding(item) for item in payload.get("introduces", []))
        fixes = tuple(_parse_fix(item) for item in payload.get("fixes", []))
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError, AttributeError) as exc:
        raise VerdictError(f"Verdict fields are malformed: {exc}") from exc

    if not 0.0 <= confidence <= 1.0:
        raise VerdictError(f"confidence must be between 0.0 and 1.0, got {confidence}")

    return Verdict(introduces=introduces, fixes=fixes, confidence=confidence)


def passes_gate(verdict: Verdict, min_confidence: float) -> bool:
    """Whether this verdict is allowed to reach a public sink."""
    return verdict.confidence >= min_confidence and not verdict.is_silent


def render_markdown(pr: PullRequest, verdict: Verdict, lane: str) -> str:
    """Render a verdict as the comment body, including the idempotency marker."""
    heading = (
        "### Prime Agent — post-merge review"
        if lane == LANE_MERGED
        else "### Prime Agent — automated review"
    )
    lines = [
        build_marker(pr.head_sha),
        heading,
        "",
        f"**PR #{pr.number}** · `{pr.head_sha[:8]}` · confidence {verdict.confidence:.0%}",
        "",
    ]

    if verdict.introduces:
        lines.append(f"#### Potential bugs introduced ({len(verdict.introduces)})")
        lines.append("")
        for finding in _sorted_findings(verdict.introduces):
            location = finding.file + (f":{finding.line}" if finding.line else "")
            lines.append(f"- **{finding.severity.value}** `{location}` — {finding.claim}")
            if finding.evidence:
                lines.append(f"  - {finding.evidence}")
        lines.append("")

    if verdict.fixes:
        lines.append(f"#### Bugs fixed ({len(verdict.fixes)})")
        lines.append("")
        for fix in verdict.fixes:
            lines.append(f"- {fix.claim}")
            if fix.evidence:
                lines.append(f"  - {fix.evidence}")
        lines.append("")

    lines.append("<sub>Automated review. Verify before acting on it.</sub>")
    return "\n".join(lines)


def _sorted_findings(findings: tuple[Finding, ...]) -> tuple[Finding, ...]:
    return tuple(sorted(findings, key=lambda f: (f.severity.rank, f.file)))


def _extract_json(raw: str) -> dict:
    if not raw or not raw.strip():
        raise VerdictError("Subagent returned an empty response")

    candidate = raw.strip()
    fenced = FENCE_PATTERN.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    else:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start != -1 and end > start:
            candidate = candidate[start : end + 1]

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise VerdictError(f"Subagent response is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise VerdictError(f"Verdict must be a JSON object, got {type(payload).__name__}")
    return payload


def _parse_finding(item: dict) -> Finding:
    severity_raw = str(item.get("severity", "")).upper()
    try:
        severity = Severity(severity_raw)
    except ValueError as exc:
        raise VerdictError(
            f"Unknown severity {severity_raw!r}; expected one of {SEVERITY_ORDER}"
        ) from exc

    claim = str(item.get("claim", "")).strip()
    if not claim:
        raise VerdictError("Every finding must carry a non-empty 'claim'")

    line_raw = item.get("line")
    return Finding(
        file=str(item.get("file", "")).strip(),
        line=int(line_raw) if isinstance(line_raw, (int, str)) and str(line_raw).isdigit() else None,
        severity=severity,
        claim=claim,
        evidence=str(item.get("evidence", "")).strip(),
    )


def _parse_fix(item: dict) -> FixClaim:
    claim = str(item.get("claim", "")).strip()
    if not claim:
        raise VerdictError("Every fix must carry a non-empty 'claim'")
    return FixClaim(claim=claim, evidence=str(item.get("evidence", "")).strip())
