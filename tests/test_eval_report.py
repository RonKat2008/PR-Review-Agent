from prime_pr_review.evaluation.report import render_markdown
from prime_pr_review.evaluation.scoring import Aggregate


def test_report_has_ladder_and_limitations():
    aggs = [Aggregate("seat-1", "off", 3, 0.5, 0.25, 0.6, 0.1, 2.0),
            Aggregate("full", "on", 3, 0.7, 0.3, 0.8, 0.0, 1.5)]
    md = render_markdown("run1", {"seed": 0, "count": 3}, aggs, severity_rows=[("HIGH", 2, 1)],
                         limitations=["diff-only"], cost_usd=1.23, seconds=100.0, drift=0, misses=0)
    assert "| full | on |" in md and "0.70" in md and "diff-only" in md and "$1.23" in md
    assert "prime-review eval run --count 3 --seed 0" in md
