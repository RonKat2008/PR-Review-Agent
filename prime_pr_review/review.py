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
    # Replacement code rendered as a GitHub ```suggestion block. Empty when the
    # fix is not a clean drop-in for the flagged lines.
    suggestion: str = ""
    # End of the replaced range for multi-line suggestions; None means one line.
    line_end: int | None = None
    # Corroborating static-analysis rule, e.g. "bandit:B608". Raises trust and
    # lets the reader verify without taking the model's word for it.
    corroboration: str = ""

    @property
    def has_suggestion(self) -> bool:
        return bool(self.suggestion.strip())


@dataclass(frozen=True)
class FixClaim:
    """A bug the PR is claimed to fix."""

    claim: str
    evidence: str


@dataclass(frozen=True)
class ScopeIssue:
    """A change that does not serve the PR's stated intent."""

    file: str
    lines: str
    severity: Severity
    claim: str
    evidence: str


@dataclass(frozen=True)
class Scope:
    """Whether the diff matches what the PR says it does (P8)."""

    intent: str
    aligned: bool
    unrelated: tuple[ScopeIssue, ...] = ()

    @property
    def worst_severity(self) -> Severity | None:
        if not self.unrelated:
            return None
        return min((i.severity for i in self.unrelated), key=lambda s: s.rank)


@dataclass(frozen=True)
class BrokenCaller:
    """A call site the change breaks."""

    file: str
    line: int | None
    severity: Severity
    claim: str


@dataclass(frozen=True)
class BlastRadius:
    """Impact of one changed symbol on the rest of the codebase (P9)."""

    symbol: str
    kind: str
    change: str
    breaks: tuple[BrokenCaller, ...] = ()
    unbroken_callers: int = 0

    @property
    def checked(self) -> int:
        """Total call sites examined. Reporting this proves the check ran."""
        return len(self.breaks) + self.unbroken_callers


@dataclass(frozen=True)
class Verdict:
    introduces: tuple[Finding, ...]
    fixes: tuple[FixClaim, ...]
    confidence: float
    scope: Scope | None = None
    blast_radius: tuple[BlastRadius, ...] = ()

    @property
    def broken_callers(self) -> tuple[BrokenCaller, ...]:
        return tuple(caller for entry in self.blast_radius for caller in entry.breaks)

    @property
    def callers_checked(self) -> int:
        return sum(entry.checked for entry in self.blast_radius)

    @property
    def has_blocking(self) -> bool:
        """Anything severe enough to justify blocking a merge.

        A broken caller is always blocking: the PR demonstrably breaks working code.
        """
        if any(f.severity.value in BLOCKING_SEVERITIES for f in self.introduces):
            return True
        if any(c.severity.value in BLOCKING_SEVERITIES for c in self.broken_callers):
            return True
        scope = self.scope
        return bool(
            scope and any(i.severity is Severity.CRITICAL for i in scope.unrelated)
        )

    @property
    def is_silent(self) -> bool:
        """Nothing found on any axis — not worth posting."""
        return not (
            self.introduces
            or self.fixes
            or self.broken_callers
            or (self.scope and self.scope.unrelated)
        )

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
        scope = _parse_scope(payload.get("scope"))
        blast = tuple(_parse_blast(item) for item in payload.get("blast_radius", []))
    except (TypeError, ValueError, AttributeError) as exc:
        raise VerdictError(f"Verdict fields are malformed: {exc}") from exc

    if not 0.0 <= confidence <= 1.0:
        raise VerdictError(f"confidence must be between 0.0 and 1.0, got {confidence}")

    return Verdict(
        introduces=introduces,
        fixes=fixes,
        confidence=confidence,
        scope=scope,
        blast_radius=blast,
    )


def _parse_scope(raw: object) -> Scope | None:
    """Parse the P8 intent-alignment block. Absent is valid — older prompts omit it."""
    if not isinstance(raw, dict):
        return None

    intent = str(raw.get("intent", "")).strip()
    if not intent:
        raise VerdictError("scope.intent must be a non-empty string when scope is present")

    issues = tuple(_parse_scope_issue(item) for item in raw.get("unrelated", []))
    # Trust the findings over the flag: a model that lists problems then claims
    # alignment is contradicting itself, and the list is the harder evidence.
    aligned = bool(raw.get("aligned", True)) and not issues

    return Scope(intent=intent, aligned=aligned, unrelated=issues)


def _parse_scope_issue(item: dict) -> ScopeIssue:
    claim = str(item.get("claim", "")).strip()
    if not claim:
        raise VerdictError("Every scope issue must carry a non-empty 'claim'")
    return ScopeIssue(
        file=str(item.get("file", "")).strip(),
        lines=str(item.get("lines", "")).strip(),
        severity=_parse_severity(item.get("severity")),
        claim=claim,
        evidence=str(item.get("evidence", "")).strip(),
    )


def _parse_blast(item: dict) -> BlastRadius:
    """Parse one P9 blast-radius entry."""
    symbol = str(item.get("symbol", "")).strip()
    if not symbol:
        raise VerdictError("Every blast_radius entry must name a 'symbol'")

    unbroken = item.get("unbroken_callers", 0)
    return BlastRadius(
        symbol=symbol,
        kind=str(item.get("kind", "")).strip(),
        change=str(item.get("change", "")).strip(),
        breaks=tuple(_parse_broken_caller(b) for b in item.get("breaks", [])),
        unbroken_callers=int(unbroken) if str(unbroken).isdigit() else 0,
    )


def _parse_broken_caller(item: dict) -> BrokenCaller:
    claim = str(item.get("claim", "")).strip()
    if not claim:
        raise VerdictError("Every broken caller must carry a non-empty 'claim'")
    line_raw = item.get("line")
    return BrokenCaller(
        file=str(item.get("file", "")).strip(),
        line=int(line_raw) if _is_int_like(line_raw) else None,
        severity=_parse_severity(item.get("severity")),
        claim=claim,
    )


def _parse_severity(raw: object) -> Severity:
    value = str(raw or "").upper()
    try:
        return Severity(value)
    except ValueError as exc:
        raise VerdictError(
            f"Unknown severity {value!r}; expected one of {SEVERITY_ORDER}"
        ) from exc


def _is_int_like(value: object) -> bool:
    return isinstance(value, int) or (isinstance(value, str) and value.isdigit())


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
    claim = str(item.get("claim", "")).strip()
    if not claim:
        raise VerdictError("Every finding must carry a non-empty 'claim'")

    line_raw = item.get("line")
    end_raw = item.get("line_end")
    return Finding(
        file=str(item.get("file", "")).strip(),
        line=int(line_raw) if _is_int_like(line_raw) else None,
        severity=_parse_severity(item.get("severity")),
        claim=claim,
        evidence=str(item.get("evidence", "")).strip(),
        suggestion=str(item.get("suggestion", "")).rstrip(),
        line_end=int(end_raw) if _is_int_like(end_raw) else None,
        corroboration=str(item.get("corroboration", "")).strip(),
    )


def _parse_fix(item: dict) -> FixClaim:
    claim = str(item.get("claim", "")).strip()
    if not claim:
        raise VerdictError("Every fix must carry a non-empty 'claim'")
    return FixClaim(claim=claim, evidence=str(item.get("evidence", "")).strip())
