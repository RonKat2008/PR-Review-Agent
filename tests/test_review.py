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
