"""Renders a `Verdict` into the PR-ready markdown described in
`docs/IMPROVEMENT-PLAN.md` section 5.

`render_review` is the sole public entry point. It is a pure function — no I/O,
no network, no filesystem access — so it can be unit tested against hand-built
`Verdict` instances with no fixtures beyond the ones already in `review.py`.

Every section is omitted when it has no data, except Scope and Blast radius:
those render whenever the reviewer populated the corresponding field at all,
because "the check ran and found nothing" and "the check never ran" must not
look identical to the reader.
"""

from __future__ import annotations

from collections.abc import Iterable

from .analysis import AnalysisResult
from .github import PullRequest
from .review import (
    BlastRadius,
    BrokenCaller,
    Finding,
    FixClaim,
    Scope,
    ScopeIssue,
    Severity,
    Verdict,
)
from .state import LANE_MERGED, build_marker

_BLOCKING_SEVERITIES = frozenset({Severity.CRITICAL, Severity.HIGH})


def render_review(
    pr: PullRequest,
    verdict: Verdict,
    lane: str,
    model: str = "",
    analysis: AnalysisResult | None = None,
) -> str:
    """Render a verdict as a PR comment body.

    The idempotency marker is always the first line, byte-identical to
    `build_marker(pr.head_sha)`. `analysis` is accepted but not yet rendered —
    it is a hook for a future static-analysis-provenance section; every
    `Finding.corroboration` already carries the `tool:rule_id` tag it would
    otherwise supply.
    """
    del analysis

    body_sections = (
        _render_blocking(verdict),
        _render_non_blocking(verdict),
        _render_scope(verdict.scope),
        _render_blast_radius(verdict),
        _render_fixes(verdict.fixes),
    )
    sections = [
        build_marker(pr.head_sha),
        _render_header(pr, verdict, lane, model),
        _render_callout(verdict),
        *(section for section in body_sections if section is not None),
        _render_footer(),
    ]
    return "\n\n".join(sections)


def _render_header(pr: PullRequest, verdict: Verdict, lane: str, model: str) -> str:
    title = (
        "## \U0001f50d Automated Review — post-merge"
        if lane == LANE_MERGED
        else "## \U0001f50d Automated Review"
    )
    stats = [
        f"`{pr.changed_files} files`",
        f"`{pr.additions}+ {pr.deletions}-`",
        f"confidence `{verdict.confidence:.0%}`",
    ]
    if model:
        stats.append(f"`{model}`")
    return f"{title}\n\n{' · '.join(stats)}"


def _render_callout(verdict: Verdict) -> str:
    findings = _blocking_findings(verdict)
    callers = verdict.broken_callers
    total = len(findings) + len(callers)
    if total == 0:
        return "> No blocking issues found."

    lead_file, lead_line, lead_claim = (
        (findings[0].file, findings[0].line, findings[0].claim)
        if findings
        else (callers[0].file, callers[0].line, callers[0].claim)
    )
    noun = "issue" if total == 1 else "issues"
    return f"> **BLOCKING {total} {noun}** — {lead_claim} in `{_location(lead_file, lead_line)}`"


def _render_blocking(verdict: Verdict) -> str | None:
    findings = _blocking_findings(verdict)
    caller_pairs = _broken_caller_pairs(verdict)
    if not findings and not caller_pairs:
        return None

    blocks = [_render_finding_block(f) for f in findings]
    blocks.extend(_render_broken_caller_block(entry, caller) for entry, caller in caller_pairs)
    return "### ⛔ Blocking\n\n" + "\n\n---\n\n".join(blocks)


def _render_non_blocking(verdict: Verdict) -> str | None:
    findings = _non_blocking_findings(verdict)
    if not findings:
        return None

    count = len(findings)
    noun = "suggestion" if count == 1 else "suggestions"
    items = "\n".join(
        f"**{f.severity.value} · `{_location(f.file, f.line, f.line_end)}`** — {f.claim}"
        for f in findings
    )
    return f"### \U0001f4a1 Non-blocking\n\n<details><summary>{count} {noun}</summary>\n\n{items}\n\n</details>"


def _render_scope(scope: Scope | None) -> str | None:
    if scope is None:
        return None

    lines = ["### \U0001f3af Scope", "", f"**Stated intent:** {scope.intent}", ""]
    if not scope.unrelated:
        lines.append("✅ Every change serves the stated intent.")
        return "\n".join(lines)

    count = len(scope.unrelated)
    noun = "change" if count == 1 else "changes"
    lines.append(f"⚠️ **{count} {noun} do not serve that intent**")
    lines.append("")
    lines.extend(_render_scope_issue(issue) for issue in _sorted_scope_issues(scope.unrelated))
    return "\n".join(lines)


def _render_scope_issue(issue: ScopeIssue) -> str:
    location = f"{issue.file}:{issue.lines}" if issue.lines else issue.file
    line = f"- **{issue.severity.value} · `{location}`** — {issue.claim}"
    if issue.evidence:
        line += f"\n  - {issue.evidence}"
    return line


def _render_blast_radius(verdict: Verdict) -> str | None:
    if not verdict.blast_radius:
        return None

    checked = verdict.callers_checked
    broken = len(verdict.broken_callers)
    symbols = len(verdict.blast_radius)
    lines = [
        "### \U0001f4a5 Blast radius",
        "",
        f"Checked **{checked}** call site(s) across **{symbols}** changed symbol(s). "
        f"**{broken}** break(s).",
        "",
        "| Caller | Status |",
        "|---|---|",
    ]
    for entry in verdict.blast_radius:
        lines.extend(_blast_radius_rows(entry))
    return "\n".join(lines)


def _render_fixes(fixes: tuple[FixClaim, ...]) -> str | None:
    if not fixes:
        return None

    lines = ["### ✅ Fixes in this PR", ""]
    for fix in fixes:
        lines.append(f"- {fix.claim}")
        if fix.evidence:
            lines.append(f"  - {fix.evidence}")
    return "\n".join(lines)


def _render_footer() -> str:
    return (
        "<sub>Automated review — verify before acting on it. "
        "React \U0001f44e to suppress a finding · comment `@prime-bot recheck` to re-run.</sub>"
    )


def _render_finding_block(finding: Finding) -> str:
    location = _location(finding.file, finding.line, finding.line_end)
    lines = [f"**{finding.severity.value} · `{location}` · {finding.claim}**"]
    if finding.evidence:
        lines.append("")
        lines.append(finding.evidence)
    if finding.has_suggestion:
        lines.append("")
        lines.append(f"```suggestion\n{finding.suggestion}\n```")
    if finding.corroboration:
        lines.append("")
        lines.append(f"*corroborated by `{finding.corroboration}`*")
    return "\n".join(lines)


def _render_broken_caller_block(entry: BlastRadius, caller: BrokenCaller) -> str:
    location = _location(caller.file, caller.line)
    return (
        f"**{caller.severity.value} · `{location}` · breaks `{entry.symbol}`**"
        f"\n\n{caller.claim}"
    )


def _blast_radius_rows(entry: BlastRadius) -> tuple[str, ...]:
    rows = tuple(f"| `{_location(c.file, c.line)}` | ⛔ {c.claim} |" for c in entry.breaks)
    if entry.unbroken_callers:
        rows += (
            f"| — | ✅ {entry.unbroken_callers} other call site(s) of "
            f"`{entry.symbol}` checked; no issues found |",
        )
    return rows


def _broken_caller_pairs(verdict: Verdict) -> tuple[tuple[BlastRadius, BrokenCaller], ...]:
    return tuple((entry, caller) for entry in verdict.blast_radius for caller in entry.breaks)


def _blocking_findings(verdict: Verdict) -> tuple[Finding, ...]:
    return _sorted_findings(f for f in verdict.introduces if f.severity in _BLOCKING_SEVERITIES)


def _non_blocking_findings(verdict: Verdict) -> tuple[Finding, ...]:
    return _sorted_findings(f for f in verdict.introduces if f.severity not in _BLOCKING_SEVERITIES)


def _sorted_findings(findings: Iterable[Finding]) -> tuple[Finding, ...]:
    return tuple(sorted(findings, key=lambda f: (f.severity.rank, f.file, f.line or 0)))


def _sorted_scope_issues(issues: tuple[ScopeIssue, ...]) -> tuple[ScopeIssue, ...]:
    return tuple(sorted(issues, key=lambda i: (i.severity.rank, i.file)))


def _location(file: str, line: int | None, line_end: int | None = None) -> str:
    if not line:
        return file
    if line_end and line_end != line:
        return f"{file}:{line}-{line_end}"
    return f"{file}:{line}"
