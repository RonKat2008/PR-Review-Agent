"""`render_review` — the owner's fixed six-section PR review template."""

from __future__ import annotations

from prime_pr_review.review import (
    BlastRadius,
    BrokenCaller,
    Finding,
    FileChange,
    FixClaim,
    ManualCheck,
    Scope,
    ScopeIssue,
    Severity,
    Verdict,
)
from prime_pr_review.state import LANE_MERGED, LANE_OPEN, build_marker
from prime_pr_review.template import render_review

from .conftest import make_pr

EMPTY_VERDICT = Verdict(introduces=(), fixes=(), confidence=0.95)

SECTION_HEADINGS = (
    "### 1 · Intent",
    "### 2 · Changes by file",
    "### 3 · Issues",
    "### 4 · Proposed changes",
    "### 5 · What to test",
    "### 6 · Verdict",
)


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


def _file_change(
    *,
    file: str = "shop/customers.py",
    summary: str = "Adds a helper for computing tax",
    relation: str = "serves_intent",
) -> FileChange:
    return FileChange(file=file, summary=summary, relation=relation)


def _manual_check(
    *,
    feature: str = "Customer search",
    files: tuple[str, ...] = ("shop/customers.py",),
    steps: str = "Open the customer list, search by name, confirm results appear.",
) -> ManualCheck:
    return ManualCheck(feature=feature, files=files, steps=steps)


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


# --- one-line verdict callout ------------------------------------------------------


def test_callout_reports_clean_when_verdict_is_silent():
    body = render_review(make_pr(), EMPTY_VERDICT, LANE_OPEN)

    assert "> ✅ **Clean** — no blocking issues, no scope concerns." in body


def test_callout_states_the_blocking_count():
    verdict = Verdict(introduces=(_finding(),), fixes=(), confidence=0.9)

    body = render_review(make_pr(), verdict, LANE_OPEN)

    callout_line = next(line for line in body.splitlines() if line.startswith("> ⛔"))
    assert callout_line == "> ⛔ **1 blocking issue**"


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

    assert "> ⛔ **2 blocking issues**" in body


def test_callout_counts_broken_callers_even_with_no_findings():
    caller = BrokenCaller(file="shop/invoice.py", line=44, severity=Severity.HIGH, claim="breaks")
    blast = BlastRadius(symbol="total_price", kind="signature_change", change="x", breaks=(caller,))
    verdict = Verdict(introduces=(), fixes=(), confidence=0.9, blast_radius=(blast,))

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "> ⛔ **1 blocking issue**" in body


def test_callout_states_the_unrelated_change_count():
    issue = ScopeIssue(
        file="shop/auth.py", lines="22-31", severity=Severity.HIGH,
        claim="Changes the session timeout.", evidence="Title mentions only total_price.",
    )
    scope = Scope(intent="Fix total_price", aligned=False, unrelated=(issue,))
    verdict = Verdict(introduces=(), fixes=(), confidence=0.9, scope=scope)

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "> ⚠️ **1 unrelated change**" in body


def test_callout_combines_blocking_and_unrelated_counts():
    issue = ScopeIssue(
        file="shop/auth.py", lines="22-31", severity=Severity.HIGH,
        claim="Changes the session timeout.", evidence="Title mentions only total_price.",
    )
    scope = Scope(intent="Fix total_price", aligned=False, unrelated=(issue,))
    verdict = Verdict(introduces=(_finding(),), fixes=(), confidence=0.9, scope=scope)

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "> ⛔ **1 blocking issue · 1 unrelated change**" in body


def test_callout_appears_before_section_1():
    verdict = Verdict(introduces=(_finding(),), fixes=(), confidence=0.9)

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert body.index("> ⛔") < body.index("### 1 · Intent")


# --- section order ------------------------------------------------------------------


def test_sections_always_render_in_the_owners_required_order():
    verdict = Verdict(
        introduces=(_finding(),),
        fixes=(FixClaim(claim="fix", evidence="ev"),),
        confidence=0.9,
        scope=Scope(intent="Fix the thing", aligned=True),
        blast_radius=(BlastRadius(symbol="f", kind="k", change="c"),),
        files=(_file_change(),),
        manual_checks=(_manual_check(),),
    )

    body = render_review(make_pr(), verdict, LANE_OPEN)

    positions = [body.index(heading) for heading in SECTION_HEADINGS]
    assert positions == sorted(positions)


def test_every_numbered_section_is_present_even_on_a_fully_empty_verdict():
    body = render_review(make_pr(), EMPTY_VERDICT, LANE_OPEN)

    for heading in SECTION_HEADINGS:
        assert heading in body


# --- 1 · Intent -----------------------------------------------------------------


def test_intent_renders_the_stated_intent_when_scope_is_present():
    scope = Scope(intent="Fix the off-by-one in total_price", aligned=True, unrelated=())
    verdict = Verdict(introduces=(), fixes=(), confidence=0.9, scope=scope)

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "Fix the off-by-one in total_price" in body


def test_intent_renders_even_when_scope_is_aligned_with_no_unrelated_changes():
    """A clean scope check and a scope check that never ran must not read the same."""
    scope = Scope(intent="Fix the off-by-one in total_price", aligned=True, unrelated=())
    verdict = Verdict(introduces=(), fixes=(), confidence=0.9, scope=scope)

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "Intent check did not run." not in body


def test_intent_states_it_did_not_run_when_scope_is_none_never_fabricating():
    verdict = Verdict(introduces=(), fixes=(), confidence=0.9, scope=None)

    body = render_review(make_pr(), verdict, LANE_OPEN)

    section = body.split("### 1 · Intent")[1].split("### 2")[0]
    assert "Intent check did not run." in section


# --- 2 · Changes by file --------------------------------------------------------


def test_files_section_states_unavailable_when_verdict_files_is_empty():
    verdict = Verdict(introduces=(), fixes=(), confidence=0.9, files=())

    body = render_review(make_pr(), verdict, LANE_OPEN)

    section = body.split("### 2 · Changes by file")[1].split("### 3")[0]
    assert "Per-file walkthrough unavailable." in section


def test_files_section_renders_a_table_with_file_change_and_relation_columns():
    change = _file_change(
        file="shop/customers.py", summary="Adds raw SQL string interpolation", relation="serves_intent"
    )
    verdict = Verdict(introduces=(), fixes=(), confidence=0.9, files=(change,))

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "| File | Change | Serves intent |" in body
    assert "| `shop/customers.py` | Adds raw SQL string interpolation | yes |" in body


def test_files_table_marks_unrelated_relation_as_no_with_a_warning():
    change = _file_change(file="shop/auth.py", summary="Widens the session timeout", relation="unrelated")
    verdict = Verdict(introduces=(), fixes=(), confidence=0.9, files=(change,))

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "| `shop/auth.py` | Widens the session timeout | ⚠ no |" in body


def test_files_table_marks_mechanical_relation_without_a_warning():
    change = _file_change(file="uv.lock", summary="Lockfile bump", relation="mechanical")
    verdict = Verdict(introduces=(), fixes=(), confidence=0.9, files=(change,))

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "| `uv.lock` | Lockfile bump | mechanical |" in body


def test_files_table_folds_scope_unrelated_markers_onto_matching_rows():
    """A file the walkthrough itself called serves_intent still gets ⚠ when the
    (separately gathered) intent check flags it in scope.unrelated."""
    change = _file_change(file="shop/auth.py", summary="Widens the session timeout", relation="serves_intent")
    issue = ScopeIssue(
        file="shop/auth.py", lines="22-31", severity=Severity.HIGH,
        claim="Not part of the stated intent.", evidence="Title mentions only total_price.",
    )
    scope = Scope(intent="Fix total_price", aligned=False, unrelated=(issue,))
    verdict = Verdict(introduces=(), fixes=(), confidence=0.9, files=(change,), scope=scope)

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "| `shop/auth.py` | Widens the session timeout | ⚠ yes |" in body


def test_files_table_folds_rows_beyond_the_most_important_fifteen():
    unrelated = _file_change(file="scope/creep.py", relation="unrelated")
    ordinary = tuple(
        _file_change(file=f"pkg/mod_{i}.py", relation="serves_intent") for i in range(15)
    )
    verdict = Verdict(introduces=(), fixes=(), confidence=0.9, files=(unrelated, *ordinary))

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "<details><summary>1 more file</summary>" in body
    before, _, after = body.partition("<details>")
    assert "scope/creep.py" in before
    assert before.count("| `pkg/mod_") == 14
    assert "pkg/mod_14.py" in after


def test_files_table_does_not_fold_when_fifteen_or_fewer():
    files = tuple(_file_change(file=f"pkg/mod_{i}.py") for i in range(15))
    verdict = Verdict(introduces=(), fixes=(), confidence=0.9, files=files)

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "<details>" not in body


# --- 3 · Issues --------------------------------------------------------------------


def test_issues_section_states_none_found_when_there_are_no_findings():
    body = render_review(make_pr(), EMPTY_VERDICT, LANE_OPEN)

    section = body.split("### 3 · Issues")[1].split("### 4")[0]
    assert "No issues found." in section


def test_issues_section_contains_only_critical_and_high_findings_visibly():
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
    visible, _, hidden = body.partition("<details>")

    assert "critical one" in visible
    assert "high one" in visible
    assert "medium one" not in visible
    assert "low one" not in visible
    assert "medium one" in hidden
    assert "low one" in hidden


def test_issues_are_ordered_by_severity_then_file():
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


def test_issue_narrative_reports_severity_location_claim_evidence_and_corroboration():
    finding = _finding(corroboration="bandit:B608")
    verdict = Verdict(introduces=(finding,), fixes=(), confidence=0.9)

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "**CRITICAL · `shop/customers.py:15` · SQL injection**" in body
    assert "User input is interpolated into a query string." in body
    assert "corroborated by `bandit:B608`" in body


def test_issue_narrative_omits_corroboration_line_when_absent():
    verdict = Verdict(introduces=(_finding(corroboration=""),), fixes=(), confidence=0.9)

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "corroborated by" not in body


def test_issues_section_never_renders_a_suggestion_fence():
    """Suggestion fences are section 4's job now, not section 3's."""
    finding = _finding(suggestion='    query = "SELECT id FROM customers WHERE name LIKE ?"')
    verdict = Verdict(introduces=(finding,), fixes=(), confidence=0.9)

    body = render_review(make_pr(), verdict, LANE_OPEN)
    issues_section = body.split("### 3 · Issues")[1].split("### 4")[0]

    assert "```suggestion" not in issues_section


def test_issue_location_reports_a_multiline_range_when_line_end_differs():
    finding = _finding(line=22, line_end=31, claim="session timeout change")
    verdict = Verdict(introduces=(finding,), fixes=(), confidence=0.9)

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "shop/customers.py:22-31" in body


def test_issue_with_no_line_reports_only_the_file():
    finding = _finding(line=None, claim="module-level issue")
    verdict = Verdict(introduces=(finding,), fixes=(), confidence=0.9)

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "`shop/customers.py`" in body
    assert "shop/customers.py:" not in body


# --- 4 · Proposed changes -----------------------------------------------------------


def test_proposed_changes_states_none_when_no_finding_has_a_suggestion():
    verdict = Verdict(introduces=(_finding(suggestion=""),), fixes=(), confidence=0.9)

    body = render_review(make_pr(), verdict, LANE_OPEN)

    section = body.split("### 4 · Proposed changes")[1].split("### 5")[0]
    assert "No committable suggestions for this diff." in section


def test_proposed_changes_renders_a_committable_suggestion_fence_with_a_location_caption():
    finding = _finding(
        line=15, suggestion='    query = "SELECT id FROM customers WHERE name LIKE ?"'
    )
    verdict = Verdict(introduces=(finding,), fixes=(), confidence=0.9)

    body = render_review(make_pr(), verdict, LANE_OPEN)
    section = body.split("### 4 · Proposed changes")[1].split("### 5")[0]

    assert "`shop/customers.py:15`" in section
    assert "```suggestion" in section
    assert 'query = "SELECT id FROM customers WHERE name LIKE ?"' in section


def test_proposed_changes_notes_that_suggestions_post_as_inline_comments():
    finding = _finding(suggestion="fixed = True")
    verdict = Verdict(introduces=(finding,), fixes=(), confidence=0.9)

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "<sub>These post as one-click commitable comments on the diff lines" in body


def test_proposed_changes_includes_non_blocking_findings_with_suggestions_too():
    finding = _finding(severity=Severity.LOW, claim="tidy this up", suggestion="x = 1")
    verdict = Verdict(introduces=(finding,), fixes=(), confidence=0.9)

    body = render_review(make_pr(), verdict, LANE_OPEN)
    section = body.split("### 4 · Proposed changes")[1].split("### 5")[0]

    assert "x = 1" in section


# --- 5 · What to test — impact analysis (5a) ----------------------------------------


def test_impact_analysis_states_unavailable_when_blast_radius_is_empty():
    verdict = Verdict(introduces=(), fixes=(), confidence=0.9, blast_radius=())

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "**Impact analysis**" in body
    assert "No impact analysis available for this diff." in body


def test_impact_analysis_lists_every_broken_caller_as_a_test_target():
    caller = BrokenCaller(
        file="shop/invoice.py", line=44, severity=Severity.HIGH, claim="raises TypeError"
    )
    blast = BlastRadius(symbol="total_price", kind="signature_change", change="x", breaks=(caller,))
    verdict = Verdict(introduces=(), fixes=(), confidence=0.9, blast_radius=(blast,))

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "- test `shop/invoice.py:44` — raises TypeError" in body


def test_impact_analysis_states_unbroken_caller_counts_per_symbol():
    blast = BlastRadius(
        symbol="total_price", kind="signature_change", change="x", breaks=(), unbroken_callers=4
    )
    verdict = Verdict(introduces=(), fixes=(), confidence=0.9, blast_radius=(blast,))

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "4 other call site(s) of `total_price` checked clean — smoke-test the callers of `total_price`" in body


def test_impact_analysis_never_invents_a_target_when_symbols_had_zero_call_sites():
    blast = BlastRadius(symbol="helper", kind="signature_change", change="x")
    verdict = Verdict(introduces=(), fixes=(), confidence=0.9, blast_radius=(blast,))

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "No call sites were found for the changed symbols." in body
    assert "test `" not in body


# --- 5 · What to test — manual checks (5b) ------------------------------------------


def test_manual_checks_subsection_is_omitted_entirely_when_there_are_none():
    verdict = Verdict(introduces=(), fixes=(), confidence=0.9, manual_checks=())

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "**Manual checks**" not in body


def test_manual_checks_render_with_the_justifying_files_cited():
    check = _manual_check(
        feature="Space Tools toolbar",
        files=("src/components/SpaceTools/Toolbar.tsx", "src/components/SpaceTools/Panel.tsx"),
        steps="Open a Space, run each tool in the toolbar, confirm results render.",
    )
    verdict = Verdict(introduces=(), fixes=(), confidence=0.9, manual_checks=(check,))

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "**Manual checks**" in body
    assert (
        "- **Space Tools toolbar** "
        "(`src/components/SpaceTools/Toolbar.tsx`, `src/components/SpaceTools/Panel.tsx`): "
        "Open a Space, run each tool in the toolbar, confirm results render." in body
    )


def test_manual_checks_carry_a_provenance_label():
    verdict = Verdict(introduces=(), fixes=(), confidence=0.9, manual_checks=(_manual_check(),))

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert (
        "<sub>suggested from the features these files belong to — "
        "not an exhaustive QA plan</sub>" in body
    )


def test_manual_checks_render_even_when_impact_analysis_is_unavailable():
    verdict = Verdict(
        introduces=(), fixes=(), confidence=0.9, blast_radius=(), manual_checks=(_manual_check(),)
    )

    body = render_review(make_pr(), verdict, LANE_OPEN)

    assert "No impact analysis available for this diff." in body
    assert "**Manual checks**" in body


# --- 6 · Verdict ---------------------------------------------------------------------


def test_verdict_summary_states_no_blocking_issues_when_clean():
    body = render_review(make_pr(), EMPTY_VERDICT, LANE_OPEN)

    section = body.split("### 6 · Verdict")[1].split("<sub>")[0]
    assert "No blocking issues were found." in section
    assert "No changes are required before merging." in section


def test_verdict_summary_reflects_blocking_finding_and_broken_caller_counts():
    caller = BrokenCaller(file="shop/invoice.py", line=44, severity=Severity.HIGH, claim="breaks")
    blast = BlastRadius(symbol="total_price", kind="signature_change", change="x", breaks=(caller,))
    verdict = Verdict(
        introduces=(_finding(severity=Severity.CRITICAL),), fixes=(), confidence=0.9, blast_radius=(blast,)
    )

    body = render_review(make_pr(), verdict, LANE_OPEN)
    section = body.split("### 6 · Verdict")[1]

    assert "1 blocking finding" in section
    assert "1 broken caller" in section
    assert "Do not merge until these are resolved." in section


def test_verdict_summary_reflects_scope_alignment():
    issue = ScopeIssue(
        file="shop/auth.py", lines="22-31", severity=Severity.HIGH,
        claim="Not part of the stated intent.", evidence="ev",
    )
    scope = Scope(intent="Fix total_price", aligned=False, unrelated=(issue,))
    verdict = Verdict(introduces=(), fixes=(), confidence=0.9, scope=scope)

    body = render_review(make_pr(), verdict, LANE_OPEN)
    section = body.split("### 6 · Verdict")[1]

    assert "1 change falls outside the stated intent." in section


def test_verdict_summary_reflects_fix_count():
    verdict = Verdict(
        introduces=(), fixes=(FixClaim(claim="Guards against a null user", evidence="ev"),), confidence=0.9
    )

    body = render_review(make_pr(), verdict, LANE_OPEN)
    section = body.split("### 6 · Verdict")[1]

    assert "It also fixes 1 known bug." in section


def test_verdict_summary_recommends_no_changes_required_when_not_blocking():
    verdict = Verdict(introduces=(_finding(severity=Severity.LOW),), fixes=(), confidence=0.9)

    body = render_review(make_pr(), verdict, LANE_OPEN)
    section = body.split("### 6 · Verdict")[1]

    assert "No changes are required before merging." in section


# --- silence property / minimal document --------------------------------------------


def test_empty_verdict_produces_a_short_document_with_every_section_stating_unavailable():
    body = render_review(make_pr(), EMPTY_VERDICT, LANE_OPEN)

    assert body.startswith(build_marker(make_pr().head_sha))
    assert "> ✅ **Clean**" in body
    assert "Intent check did not run." in body
    assert "Per-file walkthrough unavailable." in body
    assert "No issues found." in body
    assert "No committable suggestions for this diff." in body
    assert "No impact analysis available for this diff." in body
    assert "**Manual checks**" not in body
    assert "<details>" not in body
    assert "```suggestion" not in body
    assert "| File | Change | Serves intent |" not in body
    assert "<sub>" in body


def test_footer_carries_a_feedback_affordance():
    body = render_review(make_pr(), EMPTY_VERDICT, LANE_OPEN)

    assert "@prime-bot recheck" in body


# --- purity / hook compatibility ------------------------------------------------------


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
