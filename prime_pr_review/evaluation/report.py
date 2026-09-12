"""Render the ablation results as markdown for docs/eval/."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from .scoring import Aggregate

LADDER = ("seat-1", "seat-2", "seat-3", "single", "ensemble", "ensemble+judge", "full")
NOT_MEASURED = "—"
COLUMNS = ("arm", "citations", "PRs", "precision", "recall", "file-level precision",
           "fabrication", "findings/PR", "cost $/PR", "wall s/PR")
FILTER_LABELS = (("total", "total"), ("not_python", "not Python"), ("oversize", "oversize patch"),
                 ("unscorable", "no usable comment"), ("eligible", "eligible"), ("selected", "sampled"))


def render_markdown(run_id: str, config: dict, aggregates: Sequence[Aggregate],
                    severity_rows: Sequence[tuple[str, int, int]], limitations: Sequence[str],
                    cost_usd: float, seconds: float, drift: int, misses: int, instances: int,
                    filters: Mapping[str, int] | None = None,
                    excluded: Mapping[str, int] | None = None) -> str:
    lines = _header(run_id, instances, config, cost_usd, seconds, drift, misses, filters)
    lines += _ladder(aggregates)
    lines += ["", "## Findings by severity (full arm, citations on)", "",
              "| severity | findings | matched |", "|---|---|---|"]
    lines += [f"| {sev} | {n} | {m} |" for sev, n, m in severity_rows]
    lines += ["", "## Limitations", ""] + [f"- {item}" for item in [*limitations, *_exclusions(excluded)]]
    lines += ["", "## Reproduce", "", "```",
              f"prime-review eval run --count {config.get('count')} --seed {config.get('seed')}",
              f"prime-review eval score --run-id {run_id}",
              f"prime-review eval report --run-id {run_id}", "```", ""]
    return "\n".join(lines)


def _header(run_id, instances, config, cost_usd, seconds, drift, misses, filters) -> list[str]:
    lines = [f"# SWE-CARE ablation — `{run_id}`", "",
             (f"Instances: {instances} · seed {config.get('seed')} · "
              f"cost ${cost_usd:.2f} · wall {seconds / 60:.0f} min · "
              f"live/replay drift {drift} · replay misses {misses}"), ""]
    if filters:
        rendered = " · ".join(f"{label} {filters[key]}" for key, label in FILTER_LABELS if key in filters)
        lines += [f"Corpus: {rendered}", ""]
    return lines


def _ladder(aggregates: Sequence[Aggregate]) -> list[str]:
    lines = ["## Ablation ladder", "",
             f"| {' | '.join(COLUMNS)} |", f"|{'---|' * len(COLUMNS)}"]
    order = {a: i for i, a in enumerate(LADDER)}
    for a in sorted(aggregates, key=lambda a: (order.get(a.arm, 99), a.mode)):
        fabrication = NOT_MEASURED if a.fabrication_rate is None else f"{a.fabrication_rate:.2f}"
        lines.append(f"| {a.arm} | {a.mode} | {a.instances} | {a.precision:.2f} | {a.recall:.2f} | "
                     f"{a.file_precision:.2f} | {fabrication} | {a.findings_per_pr:.2f} | "
                     f"{a.cost_usd_per_pr:.3f} | {a.seconds_per_pr:.1f} |")
    return lines


def _exclusions(excluded: Mapping[str, int] | None) -> list[str]:
    """Say what the metrics were *not* computed over, in the units each count
    actually is: a PR with no line-anchored ground truth is excluded once,
    from every arm; an arm that could not be rebuilt for one PR excludes only
    that arm-instance pair, and the same PR counts again for a second arm that
    also failed to build. Either sentence is dropped when its counts are zero."""
    excluded = excluded or {}
    no_anchored_refs = excluded.get("no_anchored_refs", 0)
    replay_miss = excluded.get("replay_miss", 0)
    error = excluded.get("error", 0)
    lines = []
    if no_anchored_refs:
        lines.append(f"PRs excluded from every arm: no line-anchored reference comments {no_anchored_refs}.")
    if replay_miss or error:
        lines.append(f"Arm-instance pairs excluded: replay_miss {replay_miss}, error {error} "
                     "(each PR counts once per affected arm).")
    return lines
