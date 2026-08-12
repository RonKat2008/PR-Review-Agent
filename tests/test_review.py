"""Verdict parsing, the confidence gate, and comment rendering."""

from __future__ import annotations

import pytest

from prime_pr_review.review import (
    Severity,
    VerdictError,
    parse_verdict,
    passes_gate,
    render_markdown,
)
from prime_pr_review.state import LANE_MERGED, LANE_OPEN, build_marker

from .conftest import VERDICT_EMPTY, VERDICT_LOW_CONFIDENCE, VERDICT_WITH_BUG, make_pr


def test_parses_a_well_formed_verdict():
    verdict = parse_verdict(VERDICT_WITH_BUG)

    assert len(verdict.introduces) == 1
    assert verdict.introduces[0].severity is Severity.HIGH
    assert verdict.introduces[0].line == 10
    assert len(verdict.fixes) == 1
    assert verdict.confidence == 0.9


def test_strips_a_json_code_fence():
    """Models wrap JSON in fences constantly; that must not break the pipeline."""
    verdict = parse_verdict(f"```json\n{VERDICT_EMPTY}\n```")

    assert verdict.confidence == 0.95


def test_strips_surrounding_prose():
    raw = f"Here is my review:\n{VERDICT_EMPTY}\nHope that helps!"

    assert parse_verdict(raw).confidence == 0.95


def test_rejects_empty_response():
    with pytest.raises(VerdictError, match="empty response"):
        parse_verdict("   ")


def test_rejects_non_json():
    with pytest.raises(VerdictError, match="not valid JSON"):
        parse_verdict("the PR looks fine to me")


def test_rejects_json_that_is_not_an_object():
    with pytest.raises(VerdictError, match="must be a JSON object"):
        parse_verdict("[1, 2, 3]")


def test_rejects_out_of_range_confidence():
    with pytest.raises(VerdictError, match="between 0.0 and 1.0"):
        parse_verdict('{"introduces":[],"fixes":[],"confidence":5}')


def test_rejects_unknown_severity():
    raw = '{"introduces":[{"file":"a.py","severity":"APOCALYPTIC","claim":"x"}],"fixes":[],"confidence":0.9}'

    with pytest.raises(VerdictError, match="Unknown severity"):
        parse_verdict(raw)


def test_rejects_finding_without_a_claim():
    raw = '{"introduces":[{"file":"a.py","severity":"HIGH","claim":"  "}],"fixes":[],"confidence":0.9}'

    with pytest.raises(VerdictError, match="non-empty 'claim'"):
        parse_verdict(raw)


def test_rejects_fix_without_a_claim():
    raw = '{"introduces":[],"fixes":[{"claim":""}],"confidence":0.9}'

    with pytest.raises(VerdictError, match="non-empty 'claim'"):
        parse_verdict(raw)


def test_missing_line_becomes_none():
    raw = '{"introduces":[{"file":"a.py","severity":"LOW","claim":"x"}],"fixes":[],"confidence":0.8}'

    assert parse_verdict(raw).introduces[0].line is None


def test_empty_verdict_is_silent():
    assert parse_verdict(VERDICT_EMPTY).is_silent is True


def test_verdict_with_findings_is_not_silent():
    assert parse_verdict(VERDICT_WITH_BUG).is_silent is False


def test_high_severity_counts_as_blocking():
    assert parse_verdict(VERDICT_WITH_BUG).has_blocking is True


def test_medium_severity_is_not_blocking():
    assert parse_verdict(VERDICT_LOW_CONFIDENCE).has_blocking is False


def test_worst_severity_picks_the_most_severe():
    raw = (
        '{"introduces":[{"file":"a","severity":"LOW","claim":"x"},'
        '{"file":"b","severity":"CRITICAL","claim":"y"}],"fixes":[],"confidence":0.9}'
    )

    assert parse_verdict(raw).worst_severity is Severity.CRITICAL


def test_gate_rejects_below_threshold():
    assert passes_gate(parse_verdict(VERDICT_LOW_CONFIDENCE), 0.7) is False


def test_gate_accepts_at_threshold_exactly():
    """The threshold is inclusive — 0.7 with min 0.7 passes."""
    raw = '{"introduces":[{"file":"a","severity":"LOW","claim":"x"}],"fixes":[],"confidence":0.7}'

    assert passes_gate(parse_verdict(raw), 0.7) is True


def test_gate_rejects_a_silent_verdict_even_at_high_confidence():
    """Nothing found is not worth a public comment."""
    assert passes_gate(parse_verdict(VERDICT_EMPTY), 0.7) is False


def test_rendered_comment_carries_the_idempotency_marker():
    pr = make_pr(head_sha="cafebabe0000")

    body = render_markdown(pr, parse_verdict(VERDICT_WITH_BUG), LANE_OPEN)

    assert build_marker("cafebabe0000") in body


def test_rendered_comment_reports_both_sections():
    body = render_markdown(make_pr(), parse_verdict(VERDICT_WITH_BUG), LANE_OPEN)

    assert "Potential bugs introduced (1)" in body
    assert "Bugs fixed (1)" in body
    assert "Off-by-one" in body


def test_merged_lane_uses_a_distinct_heading():
    body = render_markdown(make_pr(), parse_verdict(VERDICT_WITH_BUG), LANE_MERGED)

    assert "post-merge review" in body


def test_findings_are_ordered_by_severity():
    raw = (
        '{"introduces":[{"file":"a","severity":"LOW","claim":"low one"},'
        '{"file":"b","severity":"CRITICAL","claim":"critical one"}],'
        '"fixes":[],"confidence":0.9}'
    )

    body = render_markdown(make_pr(), parse_verdict(raw), LANE_OPEN)

    assert body.index("critical one") < body.index("low one")


# --- FileChange parsing (per-file walkthrough) --------------------------------------


def test_files_parses_when_present():
    raw = (
        '{"introduces":[],"fixes":[],"confidence":0.9,'
        '"files":[{"file":"a.py","summary":"Adds a helper","relation":"serves_intent"}]}'
    )

    verdict = parse_verdict(raw)

    assert len(verdict.files) == 1
    assert verdict.files[0].file == "a.py"
    assert verdict.files[0].summary == "Adds a helper"
    assert verdict.files[0].relation == "serves_intent"


def test_files_key_absent_defaults_to_empty_tuple():
    assert parse_verdict(VERDICT_WITH_BUG).files == ()


def test_files_null_defaults_to_empty_tuple():
    raw = '{"introduces":[],"fixes":[],"confidence":0.9,"files":null}'

    assert parse_verdict(raw).files == ()


def test_files_empty_array_is_empty_tuple():
    raw = '{"introduces":[],"fixes":[],"confidence":0.9,"files":[]}'

    assert parse_verdict(raw).files == ()


def test_files_malformed_entries_are_skipped_without_failing_the_whole_verdict():
    """One bad row costs only that row -- unlike `introduces`/`fixes`, a
    malformed per-file entry must not blow up the entire verdict."""
    raw = (
        '{"introduces":[],"fixes":[],"confidence":0.9,"files":['
        '{"file":"good.py","summary":"ok","relation":"serves_intent"},'
        '{"file":"","summary":"missing file","relation":"serves_intent"},'
        '{"file":"nosum.py","summary":"","relation":"serves_intent"},'
        '{"file":"bad.py","summary":"bad relation","relation":"made_up"},'
        '"not-an-object"'
        "]}"
    )

    verdict = parse_verdict(raw)

    assert len(verdict.files) == 1
    assert verdict.files[0].file == "good.py"


def test_files_value_that_is_not_an_array_raises():
    """The whole array being unusable (not array-shaped at all) is the one
    case that does fail the verdict -- unlike a single malformed entry."""
    raw = '{"introduces":[],"fixes":[],"confidence":0.9,"files":"oops"}'

    with pytest.raises(VerdictError, match="'files' must be a JSON array"):
        parse_verdict(raw)


# --- ManualCheck parsing (manual smoke-test suggestions) ----------------------------


def test_manual_checks_parses_when_present():
    raw = (
        '{"introduces":[],"fixes":[],"confidence":0.9,'
        '"manual_checks":[{"feature":"Customer search","files":["shop/customers.py"],'
        '"steps":"Open the customer list, search by name, confirm results appear."}]}'
    )

    verdict = parse_verdict(raw)

    assert len(verdict.manual_checks) == 1
    check = verdict.manual_checks[0]
    assert check.feature == "Customer search"
    assert check.files == ("shop/customers.py",)
    assert check.steps == "Open the customer list, search by name, confirm results appear."


def test_manual_checks_key_absent_defaults_to_empty_tuple():
    assert parse_verdict(VERDICT_WITH_BUG).manual_checks == ()


def test_manual_checks_null_defaults_to_empty_tuple():
    raw = '{"introduces":[],"fixes":[],"confidence":0.9,"manual_checks":null}'

    assert parse_verdict(raw).manual_checks == ()


def test_manual_check_without_files_is_skipped():
    """A check that cannot cite a changed file must not be emitted."""
    raw = (
        '{"introduces":[],"fixes":[],"confidence":0.9,'
        '"manual_checks":[{"feature":"Ghost feature","files":[],'
        '"steps":"Open it and look."}]}'
    )

    assert parse_verdict(raw).manual_checks == ()


def test_manual_check_with_a_non_list_files_field_is_skipped():
    """`files` on a single entry must itself be an array; a bare string (an
    invalid type, not just an empty one) is malformed and dropped."""
    raw = (
        '{"introduces":[],"fixes":[],"confidence":0.9,'
        '"manual_checks":[{"feature":"Ghost feature","files":"shop/customers.py",'
        '"steps":"Open it and look."}]}'
    )

    assert parse_verdict(raw).manual_checks == ()


def test_manual_checks_malformed_entries_are_skipped_without_failing_the_whole_verdict():
    raw = (
        '{"introduces":[],"fixes":[],"confidence":0.9,"manual_checks":['
        '{"feature":"Good","files":["a.tsx"],"steps":"Open a, confirm b."},'
        '{"feature":"","files":["a.tsx"],"steps":"missing feature"},'
        '{"feature":"No steps","files":["a.tsx"],"steps":""},'
        '{"feature":"No files","files":[],"steps":"Open it."},'
        '"not-an-object"'
        "]}"
    )

    verdict = parse_verdict(raw)

    assert len(verdict.manual_checks) == 1
    assert verdict.manual_checks[0].feature == "Good"


def test_manual_checks_value_that_is_not_an_array_raises():
    raw = '{"introduces":[],"fixes":[],"confidence":0.9,"manual_checks":"oops"}'

    with pytest.raises(VerdictError, match="'manual_checks' must be a JSON array"):
        parse_verdict(raw)


def test_manual_checks_are_capped_at_three_dropping_the_rest():
    entries = ",".join(
        f'{{"feature":"Feature {i}","files":["f{i}.tsx"],"steps":"Open {i}."}}' for i in range(5)
    )
    raw = f'{{"introduces":[],"fixes":[],"confidence":0.9,"manual_checks":[{entries}]}}'

    verdict = parse_verdict(raw)

    assert len(verdict.manual_checks) == 3
    assert [c.feature for c in verdict.manual_checks] == ["Feature 0", "Feature 1", "Feature 2"]


# --- old verdicts keep parsing --------------------------------------------------------


def test_old_style_verdict_json_without_files_or_manual_checks_still_parses():
    verdict = parse_verdict(VERDICT_WITH_BUG)

    assert verdict.files == ()
    assert verdict.manual_checks == ()
