from prime_pr_review.evaluation.report import render_markdown
from prime_pr_review.evaluation.scoring import Aggregate


def _agg(arm, mode, fabrication=0.1, **kw):
    return Aggregate(arm, mode, 3, 0.5, 0.25, 0.6, fabrication, 2.0, **kw)


def _render(**kw):
    args = {"run_id": "run1", "config": {"seed": 0, "count": 3},
            "aggregates": [_agg("seat-1", "off", None), _agg("full", "on", 0.0),
                           _agg("single", "on", 0.2)],
            "severity_rows": [("HIGH", 2, 1)], "limitations": ["diff-only"],
            "cost_usd": 1.23, "seconds": 100.0, "drift": 0, "misses": 0, "instances": 3}
    return render_markdown(**{**args, **kw})


def test_report_has_ladder_and_limitations():
    md = _render(aggregates=[Aggregate("seat-1", "off", 3, 0.5, 0.25, 0.6, None, 2.0),
                             Aggregate("full", "on", 3, 0.7, 0.3, 0.8, 0.0, 1.5)])
    assert "| full | on |" in md and "0.70" in md and "diff-only" in md and "$1.23" in md
    assert "prime-review eval run --count 3 --seed 0" in md


def test_header_reports_scored_instances_not_the_requested_count():
    md = _render(instances=2, config={"seed": 0, "count": 200})
    assert "Instances: 2" in md
    assert "--count 200" in md  # the Reproduce block still asks for the full run


def test_ladder_carries_cost_and_wall_time_per_pr():
    md = _render(aggregates=[_agg("full", "on", 0.0, cost_usd_per_pr=0.0321, seconds_per_pr=42.0)])
    assert "cost $/PR" in md and "wall s/PR" in md
    assert "0.032" in md and "42.0" in md


def test_unmeasured_fabrication_renders_as_a_dash():
    md = _render(aggregates=[_agg("seat-1", "off", None)])
    row = next(line for line in md.splitlines() if line.startswith("| seat-1 |"))
    assert "| — |" in row


def test_single_arm_sits_between_the_seats_and_the_ensemble():
    md = _render(aggregates=[_agg("ensemble", "on"), _agg("single", "on"), _agg("seat-3", "on")])
    wanted = ("| seat-3 |", "| single |", "| ensemble |")
    arms = [line.split("|")[1].strip() for line in md.splitlines() if line.startswith(wanted)]
    assert arms == ["seat-3", "single", "ensemble"]


def test_corpus_line_reports_the_filter_counts():
    md = _render(filters={"total": 500, "not_python": 300, "oversize": 20,
                          "unscorable": 30, "eligible": 150, "selected": 3})
    assert "Corpus:" in md and "500" in md and "not Python 300" in md and "eligible 150" in md


def test_exclusions_become_a_limitation_bullet():
    md = _render(excluded={"replay_miss": 2, "error": 0, "no_anchored_refs": 5})
    bullet = next(line for line in md.splitlines() if "excluded" in line.lower())
    assert "replay_miss 2" in bullet and "no_anchored_refs 5" in bullet
    assert "error" not in bullet  # zero counts are not worth a line


def test_no_exclusions_adds_no_bullet():
    md = _render(excluded={"replay_miss": 0, "error": 0, "no_anchored_refs": 0})
    assert not any("excluded" in line.lower() for line in md.splitlines())
