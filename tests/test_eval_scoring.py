from __future__ import annotations

from prime_pr_review.evaluation.corpus import ReferenceComment
from prime_pr_review.evaluation.scoring import (
    InstanceScore,
    aggregate,
    match,
    score_instance,
)
from prime_pr_review.review import Finding, Severity, Verdict


def f(file="a.py", line=10, refuted=False):
    return Finding(file=file, line=line, severity=Severity.HIGH, claim="c", evidence="e", refuted=refuted)


def ref(path="a.py", line=10, start=None):
    return ReferenceComment(path=path, line=line, start_line=start, text="t")


def test_match_window_edges():
    assert match(f(line=15), ref(line=10)) and not match(f(line=16), ref(line=10))
    assert match(f(line=3), ref(line=10, start=8)) and not match(f(line=2), ref(line=10, start=8))
    assert not match(f(file="b.py"), ref())


def test_line_none_matches_file_level_only():
    assert not match(f(line=None), ref())
    s = score_instance(Verdict(introduces=(f(line=None),), fixes=(), confidence=0.9), (ref(),))
    assert (s.matched_findings, s.file_matched_findings) == (0, 1)


def test_score_instance_excludes_refuted_and_counts_refs():
    v = Verdict(introduces=(f(line=10), f(line=40, refuted=True), f(line=90)), fixes=(), confidence=0.9)
    s = score_instance(v, (ref(line=10), ref(line=200)), dropped=1)
    assert s == InstanceScore(findings=2, matched_findings=1, file_matched_findings=2,
                              refs=2, matched_refs=1, dropped=1)


def test_none_verdict_scores_as_no_findings():
    assert score_instance(None, (ref(),)).findings == 0


def test_aggregate_micro_averages():
    a = aggregate("full", "on", [InstanceScore(2, 1, 2, 2, 1, 0), InstanceScore(2, 2, 2, 1, 1, 0)], fabricated_total=1)
    assert (a.precision, a.recall, a.file_precision) == (0.75, 2 / 3, 1.0)
    assert a.fabrication_rate == 0.2 and a.findings_per_pr == 2.0 and a.instances == 2


def test_aggregate_with_zero_findings_is_defined():
    a = aggregate("x", "off", [InstanceScore(0, 0, 0, 1, 0, 0)], fabricated_total=0)
    assert a.precision == 0.0 and a.recall == 0.0 and a.fabrication_rate == 0.0


def test_line_less_refs_are_excluded_from_recall_but_count_at_file_level():
    refs = (ref(line=10), ReferenceComment(path="a.py", line=None, start_line=None, text="t"))
    v = Verdict(introduces=(f(line=10),), fixes=(), confidence=0.9)
    s = score_instance(v, refs)
    assert s.refs == 1 and s.matched_refs == 1 and s.file_matched_findings == 1
