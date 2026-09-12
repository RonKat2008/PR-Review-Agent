"""Render the ablation results as markdown for docs/eval/."""
from __future__ import annotations

from collections.abc import Sequence

from .scoring import Aggregate

LADDER = ("seat-1", "seat-2", "seat-3", "ensemble", "ensemble+judge", "full")


def render_markdown(run_id: str, config: dict, aggregates: Sequence[Aggregate],
                    severity_rows: Sequence[tuple[str, int, int]], limitations: Sequence[str],
                    cost_usd: float, seconds: float, drift: int, misses: int) -> str:
    lines = [f"# SWE-CARE ablation — `{run_id}`", "",
             (f"Instances: {config.get('count')} · seed {config.get('seed')} · "
              f"cost ${cost_usd:.2f} · wall {seconds / 60:.0f} min · "
              f"live/replay drift {drift} · replay misses {misses}"), "",
             "## Ablation ladder", "",
             "| arm | citations | PRs | precision | recall | file-level precision | fabrication | findings/PR |",
             "|---|---|---|---|---|---|---|---|"]
    order = {a: i for i, a in enumerate(LADDER)}
    for a in sorted(aggregates, key=lambda a: (order.get(a.arm, 99), a.mode)):
        lines.append(f"| {a.arm} | {a.mode} | {a.instances} | {a.precision:.2f} | {a.recall:.2f} | "
                     f"{a.file_precision:.2f} | {a.fabrication_rate:.2f} | {a.findings_per_pr:.2f} |")
    lines += ["", "## Findings by severity (full arm, citations on)", "",
              "| severity | findings | matched |", "|---|---|---|"]
    lines += [f"| {sev} | {n} | {m} |" for sev, n, m in severity_rows]
    lines += ["", "## Limitations", ""] + [f"- {item}" for item in limitations]
    lines += ["", "## Reproduce", "", "```",
              f"prime-review eval run --count {config.get('count')} --seed {config.get('seed')}",
              f"prime-review eval score --run-id {run_id}",
              f"prime-review eval report --run-id {run_id}", "```", ""]
    return "\n".join(lines)
