"""`render_review` — the structured PR review template (IMPROVEMENT-PLAN.md §5)."""

from __future__ import annotations

from prime_pr_review.review import (
    BlastRadius,
    BrokenCaller,
    Finding,
    FixClaim,
    Scope,
    ScopeIssue,
    Severity,
    Verdict,
)
from prime_pr_review.state import LANE_MERGED, LANE_OPEN, build_marker
from prime_pr_review.template import render_review

from .conftest import make_pr

EMPTY_VERDICT = Verdict(introduces=(), fixes=(), confidence=0.95)


def _finding(
    *,
    file: str = "shop/customers.py",
    line: int | None = 15,
    severity: Severity = Severity.CRITICAL,
    claim: str = "SQL injection",
    evidence: str = "User input is interpolated into a query string.",
    suggestion: str = "",
    line_end: int | None = None,
    corroboration: str = "",
) -> Finding:
    return Finding(
        file=file,
        line=line,
        severity=severity,
        claim=claim,
        evidence=evidence,
        suggestion=suggestion,
        line_end=line_end,
        corroboration=corroboration,
    )


# --- marker -------------------------------------------------------------------


def test_marker_is_the_first_line_and_byte_identical_to_build_marker():
    pr = make_pr(head_sha="cafebabe0000")

    body = render_review(pr, EMPTY_VERDICT, LANE_OPEN)

    assert body.splitlines()[0] == build_marker("cafebabe0000")


def test_marker_reflects_the_prs_own_head_sha_not_a_hardcoded_value():
    pr = make_pr(head_sha="deadbeef1111")

    body = render_review(pr, EMPTY_VERDICT, LANE_OPEN)

    assert body.splitlines()[0] == build_marker("deadbeef1111")


# --- header ---------------------------------------------------------------------


def test_header_reports_file_count_diff_stat_and_confidence():
    pr = make_pr(changed_files=3, additions=42, deletions=7)
    verdict = Verdict(introduces=(), fixes=(), confidence=0.8)

    body = render_review(pr, verdict, LANE_OPEN)

    assert "`3 files`" in body
    assert "`42+ 7-`" in body
    assert "confidence `80%`" in body


def test_header_includes_model_when_given():
    body = render_review(make_pr(), EMPTY_VERDICT, LANE_OPEN, model="gemini-flash-latest")

    assert "`gemini-flash-latest`" in body


def test_header_omits_model_segment_when_not_given():
    body = render_review(make_pr(), EMPTY_VERDICT, LANE_OPEN)

    assert "``" not in body


def test_merged_lane_heading_differs_from_open_lane():
    open_body = render_review(make_pr(), EMPTY_VERDICT, LANE_OPEN)
    merged_body = render_review(make_pr(), EMPTY_VERDICT, LANE_MERGED)

    assert "post-merge" in merged_body
    assert "post-merge" not in open_body


# --- verdict callout --------------------------------------------------------------


def test_callout_reports_no_blocking_issues_when_verdict_is_silent():
    body = render_review(make_pr(), EMPTY_VERDICT, LANE_OPEN)

    assert "> No blocking issues found." in body


def test_callout_names_the_count_and_top_finding_when_blocking():
    verdict = Verdict(introduces=(_finding(),), fixes=(), confidence=0.9)

    body = render_review(make_pr(), verdict, LANE_OPEN)

    callout_line = next(line for line in body.splitlines() if line.startswith("> **BLOCKING"))
    assert callout_line == "> **BLOCKING 1 issue** — SQL injection in `shop/customers.py:15`"


def test_callout_pluralizes_when_more_than_one_blocking_issue():
    verdict = Verdict(
        introduces=(
            _finding(severity=Severity.CRITICAL, claim="one"),
            _finding(severity=Severity.HIGH, claim="two", file="b.py"),
        ),
        fixes=(),
        confidence=0.9,
    )

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "> **BLOCKING 2 issues**" in body


def test_callout_counts_broken_callers_even_with_no_findings():
    caller = BrokenCaller(file="shop/invoice.py", line=44, severity=Severity.HIGH, claim="breaks")
    blast = BlastRadius(symbol="total_price", kind="signature_change", change="x", breaks=(caller,))
    verdict = Verdict(introduces=(), fixes=(), confidence=0.9, blast_radius=(blast,))

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "> **BLOCKING 1 issue**" in body


# --- blocking vs non-blocking split ------------------------------------------------


def test_blocking_section_contains_only_critical_and_high_findings():
    verdict = Verdict(
        introduces=(
            _finding(severity=Severity.CRITICAL, claim="critical one", file="a.py"),
            _finding(severity=Severity.HIGH, claim="high one", file="b.py"),
            _finding(severity=Severity.MEDIUM, claim="medium one", file="c.py"),
            _finding(severity=Severity.LOW, claim="low one", file="d.py"),
        ),
        fixes=(),
        confidence=0.9,
    )

    body = render_review(make_pr(), verdict, LANE_OPEN)
    blocking, _, non_blocking = body.partition("### \U0001f4a1 Non-blocking")

    assert "critical one" in blocking
    assert "high one" in blocking
    assert "medium one" not in blocking
    assert "low one" not in blocking
    assert "medium one" in non_blocking
    assert "low one" in non_blocking


def test_blocking_findings_are_ordered_by_severity_then_file():
    verdict = Verdict(
        introduces=(
            _finding(severity=Severity.HIGH, claim="high one", file="b.py"),
            _finding(severity=Severity.CRITICAL, claim="critical one", file="a.py"),
        ),
        fixes=(),
        confidence=0.9,
    )

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert body.index("critical one") < body.index("high one")


def test_blocking_section_includes_broken_callers_regardless_of_their_severity():
    caller = BrokenCaller(
        file="shop/invoice.py", line=44, severity=Severity.LOW, claim="calls with one argument"
    )
    blast = BlastRadius(
        symbol="total_price", kind="signature_change", change="added tax_rate", breaks=(caller,)
    )
    verdict = Verdict(introduces=(), fixes=(), confidence=0.9, blast_radius=(blast,))

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "### ⛔ Blocking" in body
    assert "breaks `total_price`" in body
    assert "calls with one argument" in body


# --- non-blocking details wrapper --------------------------------------------------


def test_non_blocking_findings_are_wrapped_in_a_details_element():
    verdict = Verdict(
        introduces=(_finding(severity=Severity.MEDIUM, claim="bare except"),), fixes=(), confidence=0.9
    )

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "<details><summary>1 suggestion</summary>" in body
    assert "bare except" in body
    assert "</details>" in body


def test_non_blocking_summary_pluralizes_the_suggestion_count():
    verdict = Verdict(
        introduces=(
            _finding(severity=Severity.MEDIUM, claim="one", file="a.py"),
            _finding(severity=Severity.LOW, claim="two", file="b.py"),
        ),
        fixes=(),
        confidence=0.9,
    )

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "<details><summary>2 suggestions</summary>" in body


# --- suggestion blocks --------------------------------------------------------------


def test_finding_with_a_suggestion_renders_a_committable_suggestion_block():
    finding = _finding(suggestion='    query = "SELECT id FROM customers WHERE name LIKE ?"')
    verdict = Verdict(introduces=(finding,), fixes=(), confidence=0.9)

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "```suggestion" in body
    assert 'query = "SELECT id FROM customers WHERE name LIKE ?"' in body


def test_finding_without_a_suggestion_renders_no_suggestion_block():
    verdict = Verdict(introduces=(_finding(suggestion=""),), fixes=(), confidence=0.9)

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "```suggestion" not in body


def test_finding_with_corroboration_reports_it():
    verdict = Verdict(introduces=(_finding(corroboration="bandit:B608"),), fixes=(), confidence=0.9)

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "corroborated by `bandit:B608`" in body


def test_finding_without_corroboration_omits_the_corroboration_line():
    verdict = Verdict(introduces=(_finding(corroboration=""),), fixes=(), confidence=0.9)

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "corroborated by" not in body


def test_finding_location_reports_a_multiline_range_when_line_end_differs():
    finding = _finding(line=22, line_end=31, claim="session timeout change")
    verdict = Verdict(introduces=(finding,), fixes=(), confidence=0.9)

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "shop/customers.py:22-31" in body


def test_finding_with_no_line_reports_only_the_file():
    finding = _finding(line=None, claim="module-level issue")
    verdict = Verdict(introduces=(finding,), fixes=(), confidence=0.9)

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "`shop/customers.py`" in body
    assert "shop/customers.py:" not in body


# --- scope ----------------------------------------------------------------------


def test_scope_renders_when_aligned_with_no_unrelated_changes():
    scope = Scope(intent="Fix the off-by-one in total_price", aligned=True, unrelated=())
    verdict = Verdict(introduces=(), fixes=(), confidence=0.9, scope=scope)

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "### \U0001f3af Scope" in body
    assert "Fix the off-by-one in total_price" in body
    assert "Every change serves the stated intent" in body


def test_scope_lists_unrelated_changes_when_present():
    issue = ScopeIssue(
        file="shop/auth.py",
        lines="22-31",
        severity=Severity.HIGH,
        claim="Changes the session timeout.",
        evidence="Title and body mention only total_price.",
    )
    scope = Scope(intent="Fix total_price", aligned=False, unrelated=(issue,))
    verdict = Verdict(introduces=(), fixes=(), confidence=0.9, scope=scope)

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "1 change do not serve that intent" in body or "1 change" in body
    assert "shop/auth.py:22-31" in body
    assert "Changes the session timeout." in body


def test_scope_section_is_omitted_when_scope_is_none():
    verdict = Verdict(introduces=(), fixes=(), confidence=0.9, scope=None)

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "### \U0001f3af Scope" not in body


# --- blast radius -------------------------------------------------------------------


def test_blast_radius_reports_the_total_checked_not_just_the_breaks():
    caller = BrokenCaller(file="shop/invoice.py", line=44, severity=Severity.HIGH, claim="breaks")
    blast = BlastRadius(
        symbol="total_price",
        kind="signature_change",
        change="added required parameter tax_rate",
        breaks=(caller,),
        unbroken_callers=3,
    )
    verdict = Verdict(introduces=(), fixes=(), confidence=0.9, blast_radius=(blast,))

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "Checked **4** call site(s)" in body
    assert "**1** break(s)" in body


def test_blast_radius_table_includes_a_row_per_broken_caller():
    caller = BrokenCaller(
        file="shop/invoice.py", line=44, severity=Severity.HIGH, claim="raises TypeError"
    )
    blast = BlastRadius(
        symbol="total_price", kind="signature_change", change="x", breaks=(caller,)
    )
    verdict = Verdict(introduces=(), fixes=(), confidence=0.9, blast_radius=(blast,))

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "| `shop/invoice.py:44` | ⛔ raises TypeError |" in body


def test_blast_radius_summarizes_unbroken_callers_when_no_breaks():
    blast = BlastRadius(
        symbol="total_price", kind="signature_change", change="x", breaks=(), unbroken_callers=4
    )
    verdict = Verdict(introduces=(), fixes=(), confidence=0.9, blast_radius=(blast,))

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "Checked **4** call site(s)" in body
    assert "**0** break(s)" in body
    assert "4 other call site(s) of `total_price` checked; no issues found" in body


def test_blast_radius_section_is_omitted_when_blast_radius_is_empty():
    verdict = Verdict(introduces=(), fixes=(), confidence=0.9, blast_radius=())

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "### \U0001f4a5 Blast radius" not in body


# --- fixes ------------------------------------------------------------------------


def test_fixes_section_lists_fix_claims_and_evidence():
    fix = FixClaim(claim="Guards order_summary against a missing order", evidence="TypeError on None subscript")
    verdict = Verdict(introduces=(), fixes=(fix,), confidence=0.9)

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "### ✅ Fixes in this PR" in body
    assert "Guards order_summary against a missing order" in body
    assert "TypeError on None subscript" in body


def test_fixes_section_is_omitted_when_there_are_no_fixes():
    body = render_review(make_pr(), EMPTY_VERDICT, LANE_OPEN)

    assert "### ✅ Fixes in this PR" not in body


# --- sections omitted when empty ----------------------------------------------------


def test_blocking_section_is_omitted_when_there_is_nothing_blocking():
    verdict = Verdict(introduces=(_finding(severity=Severity.LOW),), fixes=(), confidence=0.9)

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "### ⛔ Blocking" not in body


def test_non_blocking_section_is_omitted_when_there_is_nothing_non_blocking():
    verdict = Verdict(introduces=(_finding(severity=Severity.CRITICAL),), fixes=(), confidence=0.9)

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "### \U0001f4a1 Non-blocking" not in body


# --- minimal document / footer -----------------------------------------------------


def test_empty_verdict_produces_a_minimal_sane_document():
    body = render_review(make_pr(), EMPTY_VERDICT, LANE_OPEN)

    assert body.startswith(build_marker(make_pr().head_sha))
    assert "> No blocking issues found." in body
    assert "### ⛔ Blocking" not in body
    assert "### \U0001f4a1 Non-blocking" not in body
    assert "### \U0001f3af Scope" not in body
    assert "### \U0001f4a5 Blast radius" not in body
    assert "### ✅ Fixes in this PR" not in body
    assert "<sub>" in body


def test_footer_carries_a_feedback_affordance():
    body = render_review(make_pr(), EMPTY_VERDICT, LANE_OPEN)

    assert "@prime-bot recheck" in body


def test_render_review_is_a_pure_function_of_its_inputs():
    pr = make_pr()
    verdict = Verdict(introduces=(_finding(),), fixes=(), confidence=0.9)

    first = render_review(pr, verdict, LANE_OPEN)
    second = render_review(pr, verdict, LANE_OPEN)

    assert first == second


def test_analysis_argument_is_accepted_but_does_not_break_rendering():
    from prime_pr_review.analysis import AnalysisResult

    body = render_review(make_pr(), EMPTY_VERDICT, LANE_OPEN, analysis=AnalysisResult())

    assert body.splitlines()[0] == build_marker(make_pr().head_sha)
