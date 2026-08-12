"""Renders a `Verdict` into the PR-ready markdown described in
`docs/IMPROVEMENT-PLAN.md` section 5, restructured to the owner's fixed
six-section skeleton.

`render_review` is the sole public entry point. It is a pure function — no I/O,
no network, no filesystem access — so it can be unit tested against hand-built
`Verdict` instances with no fixtures beyond the ones already in `review.py`.

Unlike the section this replaces, every numbered section (1-6) ALWAYS renders.
None are ever omitted: a section with nothing to say states its one-line
"none/unavailable" fact instead of disappearing. A fixed skeleton means a reader
(or a downstream parser) can rely on every heading being present every time, and
"the check ran and found nothing" never looks identical to "the check never
ran" — the same principle the old design applied to Scope and Blast radius,
now applied uniformly.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .analysis import AnalysisResult
from .github import PullRequest
from .review import (
    BlastRadius,
    BrokenCaller,
    Finding,
    FileChange,
    ManualCheck,
    Scope,
    Severity,
    Verdict,
)
from .state import LANE_MERGED, build_marker

_BLOCKING_SEVERITIES = frozenset({Severity.CRITICAL, Severity.HIGH})

_FILE_RELATION_LABELS = {
    "serves_intent": "yes",
    "unrelated": "no",
    "mechanical": "mechanical",
}
# On PRs with many files, section 2 must not sprawl: only the most important
# rows (unrelated/flagged first) render directly; the rest fold into <details>.
_FILES_VISIBLE_LIMIT = 15

_SUGGESTION_NOTE = (
    "<sub>These post as one-click commitable comments on the diff lines "
    "when inline delivery is on.</sub>"
)
_MANUAL_CHECKS_NOTE = (
    "<sub>suggested from the features these files belong to — not an "
    "exhaustive QA plan</sub>"
)


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

    sections = [
        build_marker(pr.head_sha),
        _render_header(pr, verdict, lane, model),
        _render_callout(verdict),
        _render_intent(verdict.scope),
        _render_files(verdict),
        _render_issues(verdict),
        _render_proposed_changes(verdict),
        _render_what_to_test(verdict),
        _render_verdict_summary(verdict),
        _render_footer(),
    ]
    return "\n\n".join(sections)


# --- header + callout ----------------------------------------------------------


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
    """The one-line, read-in-3-seconds verdict: blocking count, unrelated-change
    count, or "clean". Deliberately does not repeat a claim or a location —
    that detail lives in sections 3 and 2, this is only the headline."""
    blocking = len(_blocking_findings(verdict)) + len(verdict.broken_callers)
    unrelated = len(verdict.scope.unrelated) if verdict.scope else 0

    if not blocking and not unrelated:
        return "> ✅ **Clean** — no blocking issues, no scope concerns."

    parts = []
    if blocking:
        parts.append(f"{blocking} blocking {_noun(blocking, 'issue')}")
    if unrelated:
        parts.append(f"{unrelated} unrelated {_noun(unrelated, 'change')}")

    icon = "⛔" if blocking else "⚠️"
    return f"> {icon} **{' · '.join(parts)}**"


# --- 1 · Intent ------------------------------------------------------------------


def _render_intent(scope: Scope | None) -> str:
    heading = "### 1 · Intent"
    if scope is None:
        return f"{heading}\n\nIntent check did not run."
    return f"{heading}\n\n> {scope.intent}"


# --- 2 · Changes by file ----------------------------------------------------------


def _render_files(verdict: Verdict) -> str:
    heading = "### 2 · Changes by file"
    if not verdict.files:
        return f"{heading}\n\nPer-file walkthrough unavailable."

    unrelated_files = _scope_unrelated_files(verdict.scope)
    ordered = sorted(verdict.files, key=lambda f: _file_importance_rank(f, unrelated_files))
    visible, folded = ordered[:_FILES_VISIBLE_LIMIT], ordered[_FILES_VISIBLE_LIMIT:]

    lines = [heading, "", *_file_table_rows(visible, unrelated_files)]
    if folded:
        count = len(folded)
        lines += [
            "",
            f"<details><summary>{count} more {_noun(count, 'file')}</summary>",
            "",
            *_file_table_rows(folded, unrelated_files),
            "",
            "</details>",
        ]
    return "\n".join(lines)


def _scope_unrelated_files(scope: Scope | None) -> frozenset[str]:
    return frozenset(issue.file for issue in scope.unrelated) if scope else frozenset()


def _file_importance_rank(change: FileChange, unrelated_files: frozenset[str]) -> int:
    """Unrelated/flagged changes first, then ordinary changes, mechanical
    changes last — so a >15-file PR folds away the least important rows
    rather than whichever ones the model happened to list last."""
    if change.relation == "unrelated" or change.file in unrelated_files:
        return 0
    if change.relation == "mechanical":
        return 2
    return 1


def _file_table_rows(
    files: Sequence[FileChange], unrelated_files: frozenset[str]
) -> tuple[str, ...]:
    rows = ["| File | Change | Serves intent |", "|---|---|---|"]
    rows.extend(
        f"| `{f.file}` | {f.summary} | {_relation_cell(f, unrelated_files)} |" for f in files
    )
    return tuple(rows)


def _relation_cell(change: FileChange, unrelated_files: frozenset[str]) -> str:
    label = _FILE_RELATION_LABELS.get(change.relation, change.relation or "?")
    # Fold the (separately gathered) scope-check verdict onto this row too: a
    # file can be flagged unrelated by the intent check even when the same-pass
    # walkthrough that produced this row called it something else.
    flagged = change.relation == "unrelated" or change.file in unrelated_files
    return f"⚠ {label}" if flagged else label


# --- 3 · Issues --------------------------------------------------------------------


def _render_issues(verdict: Verdict) -> str:
    heading = "### 3 · Issues"
    blocking = _blocking_findings(verdict)
    non_blocking = _non_blocking_findings(verdict)
    if not blocking and not non_blocking:
        return f"{heading}\n\nNo issues found."

    lines = [heading, ""]
    if blocking:
        lines.append("\n\n---\n\n".join(_render_finding_narrative(f) for f in blocking))
    if non_blocking:
        if blocking:
            lines.append("")
        lines.append(_render_non_blocking_details(non_blocking))
    return "\n".join(lines)


def _render_non_blocking_details(findings: tuple[Finding, ...]) -> str:
    count = len(findings)
    items = "\n\n---\n\n".join(_render_finding_narrative(f) for f in findings)
    summary = f"{count} lower-severity {_noun(count, 'issue')}"
    return f"<details><summary>{summary}</summary>\n\n{items}\n\n</details>"


def _render_finding_narrative(finding: Finding) -> str:
    """Severity, location, claim, evidence, corroboration — never the
    suggestion fence; that is section 4's job, not this one's."""
    location = _location(finding.file, finding.line, finding.line_end)
    lines = [f"**{finding.severity.value} · `{location}` · {finding.claim}**"]
    if finding.evidence:
        lines.append("")
        lines.append(finding.evidence)
    if finding.corroboration:
        lines.append("")
        lines.append(f"*corroborated by `{finding.corroboration}`*")
    return "\n".join(lines)


# --- 4 · Proposed changes -----------------------------------------------------------


def _render_proposed_changes(verdict: Verdict) -> str:
    heading = "### 4 · Proposed changes"
    findings = tuple(f for f in _sorted_findings(verdict.introduces) if f.has_suggestion)
    if not findings:
        return f"{heading}\n\nNo committable suggestions for this diff."

    blocks = [_render_suggestion_block(f) for f in findings]
    return "\n\n".join([heading, *blocks, _SUGGESTION_NOTE])


def _render_suggestion_block(finding: Finding) -> str:
    location = _location(finding.file, finding.line, finding.line_end)
    return f"`{location}`\n\n```suggestion\n{finding.suggestion}\n```"


# --- 5 · What to test ---------------------------------------------------------------


def _render_what_to_test(verdict: Verdict) -> str:
    heading = "### 5 · What to test"
    lines = [heading, "", "**Impact analysis**", "", _render_impact_analysis(verdict)]

    manual = _render_manual_checks(verdict.manual_checks)
    if manual is not None:
        lines += ["", "**Manual checks**", "", manual, "", _MANUAL_CHECKS_NOTE]

    return "\n".join(lines)


def _render_impact_analysis(verdict: Verdict) -> str:
    """5a: the deterministic caller list from `verdict.blast_radius`. Never
    invents a test target the data does not support."""
    if not verdict.blast_radius:
        return "No impact analysis available for this diff."

    broken: list[BrokenCaller] = sorted(
        verdict.broken_callers, key=lambda c: (c.severity.rank, c.file, c.line or 0)
    )
    clean: list[BlastRadius] = sorted(
        (e for e in verdict.blast_radius if e.unbroken_callers > 0), key=lambda e: e.symbol
    )

    items = [f"- test `{_location(c.file, c.line)}` — {c.claim}" for c in broken]
    items += [
        f"- {e.unbroken_callers} other call site(s) of `{e.symbol}` checked clean — "
        f"smoke-test the callers of `{e.symbol}`"
        for e in clean
    ]
    if not items:
        return "No call sites were found for the changed symbols."
    return "\n".join(items)


def _render_manual_checks(checks: tuple[ManualCheck, ...]) -> str | None:
    """5b: model-suggested manual smoke tests. `None` means omit the subsection
    entirely — 5a's own honesty line already covers "nothing available" for
    the section as a whole, so an empty 5b does not need a second one."""
    if not checks:
        return None
    return "\n".join(
        f"- **{c.feature}** ({', '.join(f'`{f}`' for f in c.files)}): {c.steps}"
        for c in checks
    )


# --- 6 · Verdict ---------------------------------------------------------------------


def _render_verdict_summary(verdict: Verdict) -> str:
    """A reasoned 2-4 sentence summary assembled from the counts (blocking,
    scope alignment, callers broken, fixes) — template-generated text from
    data, never model prose."""
    heading = "### 6 · Verdict"
    findings_blocking = len(_blocking_findings(verdict))
    broken = len(verdict.broken_callers)
    fixes = len(verdict.fixes)

    sentences = [
        _blocking_sentence(findings_blocking, broken),
        _scope_sentence(verdict.scope),
    ]
    if fixes:
        sentences.append(f"It also fixes {fixes} known {_noun(fixes, 'bug')}.")
    sentences.append(
        "Do not merge until these are resolved."
        if verdict.has_blocking
        else "No changes are required before merging."
    )
    return f"{heading}\n\n" + " ".join(sentences)


def _blocking_sentence(findings_blocking: int, broken: int) -> str:
    if not findings_blocking and not broken:
        return "No blocking issues were found."
    parts = []
    if findings_blocking:
        parts.append(f"{findings_blocking} blocking {_noun(findings_blocking, 'finding')}")
    if broken:
        parts.append(f"{broken} broken {_noun(broken, 'caller')}")
    return "This PR has " + " and ".join(parts) + "."


def _scope_sentence(scope: Scope | None) -> str:
    if scope is None:
        return "No intent/scope check ran for this diff."
    if not scope.unrelated:
        return "Every changed file serves the stated intent."
    count = len(scope.unrelated)
    verb = "falls" if count == 1 else "fall"
    return f"{count} {_noun(count, 'change')} {verb} outside the stated intent."


# --- footer ------------------------------------------------------------------------


def _render_footer() -> str:
    return (
        "<sub>Automated review — verify before acting on it. "
        "React \U0001f44e to suppress a finding · comment `@prime-bot recheck` to re-run.</sub>"
    )


# --- shared helpers ------------------------------------------------------------------


def _blocking_findings(verdict: Verdict) -> tuple[Finding, ...]:
    return _sorted_findings(f for f in verdict.introduces if f.severity in _BLOCKING_SEVERITIES)


def _non_blocking_findings(verdict: Verdict) -> tuple[Finding, ...]:
    return _sorted_findings(f for f in verdict.introduces if f.severity not in _BLOCKING_SEVERITIES)


def _sorted_findings(findings: Iterable[Finding]) -> tuple[Finding, ...]:
    return tuple(sorted(findings, key=lambda f: (f.severity.rank, f.file, f.line or 0)))


def _location(file: str, line: int | None, line_end: int | None = None) -> str:
    if not line:
        return file
    if line_end and line_end != line:
        return f"{file}:{line}-{line_end}"
    return f"{file}:{line}"


def _noun(count: int, singular: str) -> str:
    return singular if count == 1 else f"{singular}s"
