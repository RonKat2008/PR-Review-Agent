"""E1 replay harness: replay real merged PRs in dry-run, write one report.

    python scripts/replay_corpus.py --repo example-org/service-b --count 20

This is the go-live gate from docs/FINAL-PLAN.md (Phase E, E1): replay 20+
already-merged PRs from a real repo, read every review, judge precision by
hand. This script automates the mechanics -- fetch, review, collect, report --
so a human only does the judging.

IT MUST NEVER POST. `run_replay` unconditionally locks the resolved config down
(forced `dry_run=True`, forced `sinks.pr_comment=False`) before a single sweep
runs, regardless of what config.toml says -- see `_lock_down`. State is a
throwaway in-memory `State.empty()` per PR, never loaded from or saved to disk,
so a replay can never mark a PR "reviewed" for the real pipeline and never
writes a watermark file anywhere.

Several pieces here duplicate scripts/run_sweep.py (`resolve_api_key`,
`_single_pr_list_json`, the reviewer/enrichment construction in
`build_enrichment`) rather than importing it: scripts/ has no package
`__init__.py`, so one script cannot cleanly import from a sibling script.
Each duplicated piece is called out at its definition.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prime_pr_review import github  # noqa: E402
from prime_pr_review.analysis import run_analysis  # noqa: E402
from prime_pr_review.config import (  # noqa: E402
    Config,
    ConfigError,
    load_config,
    require_repo,
    resolve_active,
)
from prime_pr_review.feedback import FeedbackError, load_rejections  # noqa: E402
from prime_pr_review.github import PullRequest  # noqa: E402
from prime_pr_review.graph import strict_runner  # noqa: E402
from prime_pr_review.review import BLOCKING_SEVERITIES  # noqa: E402
from prime_pr_review.reviewers import (  # noqa: E402
    DEFAULT_GEMINI_MODEL,
    gemini_model_fn,
    gemini_reviewer,
)
from prime_pr_review.state import LANE_OPEN, State  # noqa: E402
from prime_pr_review.sweep import (  # noqa: E402
    Enrichment,
    PullRequestOutcome,
    Reviewer,
    sweep_lane,
)
from prime_pr_review.template import render_review  # noqa: E402

# Duplicated from scripts/run_sweep.py (see module docstring).
AUTH_FILE = Path.home() / ".prime" / "agent" / "auth.json"

# Every agent-owned path is resolved against the agent's own directory, never the
# process working directory -- same reasoning as scripts/run_sweep.py: call-site
# discovery requires running from inside the reviewed repo's checkout, so cwd
# belongs to the repo under review, not the agent.
AGENT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = AGENT_ROOT / "skills" / "pr-review" / "prompts"
REVIEWS_DIR = AGENT_ROOT / "reviews"
REJECTIONS_FILE = AGENT_ROOT / "state" / "rejections.json"
DEFAULT_REPORTS_DIR = AGENT_ROOT / "reports"

DEFAULT_COUNT = 20
# Generous lookback for "the last --count merged PRs": wide enough that a
# slow-moving repo still yields `count` results, trimmed client-side afterward.
SINCE_DAYS = 365
# Markdown table cell budget for the joined notes column.
NOTES_TRUNCATE_CHARS = 160


def resolve_api_key() -> str:
    """Prefer the environment; fall back to the prime-agent auth file.

    Duplicated verbatim from scripts/run_sweep.py (see module docstring).
    """
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key

    try:
        entry = json.loads(AUTH_FILE.read_text(encoding="utf-8")).get("google", {})
    except (OSError, json.JSONDecodeError):
        entry = {}

    key = str(entry.get("key", "")).strip()
    # The auth file may store an env var *name* rather than a literal key.
    if key and not key.startswith("AQ.") and not key.startswith("AIza"):
        key = os.environ.get(key, "").strip()

    if not key:
        raise SystemExit(
            "No Gemini key found. Set GEMINI_API_KEY, or put a literal key in "
            f"{AUTH_FILE} under \"google\"."
        )
    return key


def _single_pr_list_json(pr: github.PullRequest) -> str:
    """Serialize an already-fetched PullRequest into the `gh pr list --json
    <PR_FIELDS>` shape.

    Duplicated from scripts/run_sweep.py's `_single_pr_list_json` (see module
    docstring): this is what feeds `github.single_pr_runner`, letting one
    replayed PR reuse `sweep_lane`'s per-PR machinery -- dedup, diff fetch,
    rendering, the local-file sink -- instead of a second, parallel
    implementation of any of it.
    """
    return json.dumps(
        [
            {
                "number": pr.number,
                "title": pr.title,
                "author": {"login": pr.author},
                "headRefOid": pr.head_sha,
                "baseRefName": pr.base_ref,
                "url": pr.url,
                "additions": pr.additions,
                "deletions": pr.deletions,
                "changedFiles": pr.changed_files,
                "mergedAt": pr.merged_at,
            }
        ]
    )


def _lock_down(config: Config) -> Config:
    """HARD SAFETY GATE: force dry-run semantics regardless of what config.toml
    says. This harness evaluates; it must never talk to GitHub beyond reads.

    Forces `review.dry_run = True` and `sinks.pr_comment = False` via
    `dataclasses.replace`, producing a new Config rather than mutating the one
    passed in. Called unconditionally at the top of `run_replay`, so this
    guarantee holds for every caller -- CLI and tests alike -- not only one
    that remembered to call it first.
    """
    return replace(
        config,
        review=replace(config.review, dry_run=True),
        sinks=replace(config.sinks, pr_comment=False),
    )


def build_enrichment(config: Config, api_key: str, model: str) -> Enrichment:
    """Build the Enrichment the same way scripts/run_sweep.py does.

    Duplicated from scripts/run_sweep.py's `main` (see module docstring):
    same reviewer/model_fn wiring, same `strict_runner` gate on `repo_root`,
    same rejection-store load. `REJECTIONS_FILE` is referenced as a bare
    module global (not a default parameter) so it can be overridden per-call
    by tests via `monkeypatch.setattr`.
    """
    root_path = Path(config.review.repo_root or ".")
    try:
        rejections = load_rejections(REJECTIONS_FILE)
    except FeedbackError as exc:
        print(f"Warning: rejection store unreadable, continuing without it: {exc}")
        rejections = ()

    return Enrichment(
        model_fn=gemini_model_fn(api_key, model=model),
        repo_root=root_path,
        prompts_dir=PROMPTS_DIR,
        # Strict on purpose: for `merge-base --is-ancestor`, exit 1 means "stale
        # graph, refuse" -- the lenient grep runner would swallow it.
        git_runner=strict_runner(root_path) if config.review.repo_root else None,
        analysis_fn=run_analysis,
        rejections=rejections,
    )


@dataclass(frozen=True)
class ReplayResult:
    """One replayed PR's outcome.

    `error` mirrors `PullRequestOutcome.error` when the sweep produced an
    outcome, and carries its own message when the replay of this one PR
    crashed before `sweep_lane` could return an outcome at all (e.g. an
    unexpected exception, not one of the failure modes `sweep_lane` already
    converts into a recorded error).
    """

    pr: PullRequest
    outcome: PullRequestOutcome | None = None
    error: str | None = None


@dataclass(frozen=True)
class ReplayRun:
    """Everything one `replay_corpus` invocation produced, ready to render."""

    repo_slug: str
    model: str
    count_requested: int
    now: datetime
    config: Config  # the LOCKED-DOWN config every PR in this run actually used
    results: tuple[ReplayResult, ...]
    # "merged" replays history (the E1 gate); "open" previews live candidates.
    pr_state: str = "merged"

    @property
    def errors(self) -> tuple[ReplayResult, ...]:
        return tuple(r for r in self.results if r.error)


def _select_prs(
    repo_slug: str,
    runner: github.GhRunner,
    count: int,
    now: datetime,
    pr_state: str = "merged",
) -> tuple[PullRequest, ...]:
    """The last `count` PRs on `repo_slug` in the requested state, newest first.

    merged: `since` is deliberately generous (`SINCE_DAYS`) rather than derived
    from `count` — reach far enough back that a slow-moving repo still yields
    `count` results, then trim client-side. Both listings return newest first;
    trimming never reorders, so a short result is always the newest
    `len(result)` PRs, never an arbitrary subset.
    """
    if pr_state == "open":
        return github.list_open_prs(repo_slug, runner)[:count]
    since = now - timedelta(days=SINCE_DAYS)
    merged = github.list_merged_prs(repo_slug, since, runner)
    return merged[:count]


def _replay_one(
    config: Config,
    pr: PullRequest,
    reviewer: Reviewer,
    enrichment: Enrichment | None,
    runner: github.GhRunner,
    reviews_dir: Path | str,
) -> ReplayResult:
    """Review exactly one already-merged PR via `sweep_lane`.

    Always uses `State.empty()` -- fresh, never loaded, never saved -- and lane
    `"open"` rather than `"merged"`. The lane choice mirrors run_sweep.py's own
    `--pr` entry point: `LANE_MERGED` candidate selection re-applies
    `review.merged_lookback_days` against `pr.merged_at`, which would silently
    drop this very PR whenever it merged further back than that window -- and
    a corpus replay walking a full year of history hits that constantly.
    `LANE_OPEN` applies no such filter, so it is always safe here; the lane
    label only ever changed which watermark key a real sweep would record
    under, and this harness never writes a watermark at all.
    """
    single_runner = github.single_pr_runner(runner, _single_pr_list_json(pr))
    try:
        report, _state = sweep_lane(
            config,
            LANE_OPEN,
            reviewer,
            State.empty(),
            runner=single_runner,
            enrichment=enrichment,
            reviews_dir=reviews_dir,
        )
    except Exception as exc:  # noqa: BLE001 - one PR's crash must not sink the replay
        return ReplayResult(pr=pr, outcome=None, error=f"replay crashed: {exc}")

    if not report.outcomes:
        return ReplayResult(
            pr=pr, outcome=None, error="sweep produced no outcome for this PR"
        )

    outcome = report.outcomes[0]
    return ReplayResult(pr=pr, outcome=outcome, error=outcome.error)


def run_replay(
    config: Config,
    repo_slug: str,
    count: int,
    model: str,
    reviewer: Reviewer,
    enrichment: Enrichment | None,
    runner: github.GhRunner = github.default_runner,
    reviews_dir: Path | str = REVIEWS_DIR,
    now: datetime | None = None,
    pr_state: str = "merged",
) -> ReplayRun:
    """The testable core: list the corpus, replay every PR in it, return the run.

    `config` is locked down here (see `_lock_down`) before candidate listing
    or any sweep runs, so "this harness never posts" holds for every caller,
    including a test that passes a config with `dry_run=False`.
    """
    locked = _lock_down(config)
    moment = now or datetime.now(timezone.utc)
    candidates = _select_prs(repo_slug, runner, count, moment, pr_state)
    results = tuple(
        _replay_one(locked, pr, reviewer, enrichment, runner, reviews_dir)
        for pr in candidates
    )
    return ReplayRun(
        repo_slug=repo_slug,
        model=model,
        count_requested=count,
        now=moment,
        config=locked,
        results=results,
        pr_state=pr_state,
    )


# --- reporting ---------------------------------------------------------------

_READING_GUIDE = """## Reading guide -- go-live criteria (FINAL-PLAN.md, Phase E)

This is the number that decides go-live, not the planted demo. Fill this in by
hand after reading every review above. Miss the bar on any line -> tune, then
re-run E1. Never go live on a missed bar.

- [ ] Zero fabricated findings -- no finding cites a file/line that does not
      exist in the diff. A single one blocks go-live.
- [ ] At least 60% of findings are ones a maintainer would actually act on.
- [ ] The bot stayed silent on cosmetic / low-risk PRs (PR-#6-style silence),
      rather than manufacturing a finding to justify a comment.
"""


def _report_filename(repo_slug: str, now: datetime, pr_state: str = "merged") -> str:
    owner, name = repo_slug.split("/", 1)
    # "open" gets a suffix so a same-day merged replay is never overwritten.
    suffix = "-open" if pr_state == "open" else ""
    return f"replay-{owner}-{name}{suffix}-{now:%Y%m%d}.md"


def _graph_status(run: ReplayRun) -> str:
    """"Graph used or not", derived from notes -- there is no positive "the
    graph loaded fine" note (silence on success), only degradation notes, so
    that is the signal this reads."""
    graph_path = run.config.review.graph_path
    if not graph_path:
        return "not configured"

    degraded = sorted(
        {
            note
            for result in run.results
            if result.outcome is not None
            for note in result.outcome.notes
            if note.lower().startswith("graph")
        }
    )
    if not degraded:
        return f"configured (`{graph_path}`), no degradation noted"
    return f"configured (`{graph_path}`), degraded: " + "; ".join(degraded)


def _escape_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _truncate(text: str, limit: int = NOTES_TRUNCATE_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "\u2026"


def _row_notes(result: ReplayResult) -> str:
    parts: list[str] = []
    outcome = result.outcome
    if result.error:
        parts.append(f"ERROR: {result.error}")
    if outcome is not None:
        if outcome.reason and outcome.verdict is None and not result.error:
            parts.append(f"skipped: {outcome.reason}")
        parts.extend(outcome.notes)
    joined = "; ".join(parts) if parts else "-"
    return _truncate(_escape_cell(joined))


def _scorecard_row(result: ReplayResult) -> str:
    pr = result.pr
    outcome = result.outcome
    verdict = outcome.verdict if outcome is not None else None

    if verdict is None:
        findings = blocking = scope_unrelated = callers_broken = "-"
    else:
        findings = str(len(verdict.introduces))
        blocking = str(
            sum(1 for f in verdict.introduces if f.severity.value in BLOCKING_SEVERITIES)
        )
        scope_unrelated = str(len(verdict.scope.unrelated)) if verdict.scope else "0"
        callers_broken = str(len(verdict.broken_callers))

    notes = _row_notes(result)
    return (
        f"| #{pr.number} | {pr.changed_files} | {findings} | {blocking} | "
        f"{scope_unrelated} | {callers_broken} | {notes} |"
    )


def _render_scorecard(results: Sequence[ReplayResult]) -> str:
    header = "| PR# | files | findings | blocking | scope-unrelated | callers broken | notes |"
    separator = "|---|---|---|---|---|---|---|"
    rows = [header, separator]
    rows.extend(_scorecard_row(result) for result in results)
    return "\n".join(rows)


def _pr_body(result: ReplayResult) -> str:
    """The full rendered review body for one PR.

    Prefers reading the local file the local-file sink already wrote
    (`outcome.local_path`), since that is the byte-for-byte record of what
    the reviewed. Falls back to re-rendering from the verdict when there is
    no file (e.g. `sinks.local_file` disabled), and to a plain note when
    there is no verdict at all (an error or a legitimate skip).
    """
    outcome = result.outcome
    if outcome is None:
        return f"_Replay error: {result.error}_"

    local_path = outcome.local_path
    if local_path is not None:
        path = Path(local_path)
        if path.is_file():
            return path.read_text(encoding="utf-8")

    if outcome.verdict is not None:
        return render_review(result.pr, outcome.verdict, outcome.lane)

    reason = outcome.reason or result.error or "no reason recorded"
    return f"_Skipped: {reason}_"


def _render_pr_section(result: ReplayResult) -> str:
    pr = result.pr
    heading = f"## PR #{pr.number} \u2014 {pr.title}\n\n{pr.url}"
    return f"{heading}\n\n{_pr_body(result)}"


def render_report(run: ReplayRun) -> str:
    """The one consolidated markdown report for `run`: header, scorecard,
    every per-PR review body, then the go-live reading guide."""
    reviewed = len(run.results)
    lines = [
        f"# Replay report \u2014 {run.repo_slug}",
        "",
        "## Summary",
        "",
        f"- **repo**: `{run.repo_slug}`",
        f"- **model**: `{run.model}`",
        f"- **mode**: DRY RUN (forced -- this harness never posts)",
        f"- **PR state**: {run.pr_state}",
        f"- **count requested**: {run.count_requested}",
        f"- **count reviewed**: {reviewed}",
        f"- **timestamp**: {run.now.isoformat()}",
        f"- **knowledge graph**: {_graph_status(run)}",
        f"- **total errors**: {len(run.errors)}",
    ]
    if reviewed < run.count_requested:
        lines.append(
            f"- **shortfall**: requested {run.count_requested}, only {reviewed} "
            f"merged PR(s) were available on {run.repo_slug}"
        )

    lines += ["", "## Scorecard", "", _render_scorecard(run.results), ""]
    lines += ["## Per-PR reviews", ""]
    for result in run.results:
        lines.append(_render_pr_section(result))
        lines.append("")

    lines.append(_READING_GUIDE)
    return "\n".join(lines)


def write_report(run: ReplayRun, out_dir: Path | str) -> Path:
    """Render and write the single consolidated report. Returns its path."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / _report_filename(run.repo_slug, run.now, run.pr_state)
    path.write_text(render_report(run), encoding="utf-8")
    return path


# --- CLI -----------------------------------------------------------------------


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: github.GhRunner = github.default_runner,
    reviewer: Reviewer | None = None,
    enrichment: Enrichment | None = None,
    api_key_resolver: Callable[[], str] = resolve_api_key,
    reviews_dir: Path | str = REVIEWS_DIR,
) -> int:
    """CLI entry point. `runner`/`reviewer`/`enrichment`/`api_key_resolver`/
    `reviews_dir` are injectable seams for tests; production callers (the
    `__main__` block below) use every default, which talks to real `gh` and
    a real Gemini key.

    The API key is resolved -- and the real reviewer/enrichment built -- only
    when the caller did not already supply both, so a fully-stubbed test never
    needs a real key, an auth file, or a network.
    """
    parser = argparse.ArgumentParser(
        description=(
            "E1 replay harness (docs/FINAL-PLAN.md, Phase E): replay merged "
            "PRs in dry-run and write one consolidated markdown report."
        )
    )
    parser.add_argument(
        "--repo",
        default="",
        help=(
            "Select one config.toml [[repos]] entry by 'owner/name' or bare "
            "name (case-insensitive). Required when more than one is "
            "configured; unused when config.toml has none."
        ),
    )
    parser.add_argument("--config", default="config.toml")
    parser.add_argument(
        "--count", type=int, default=DEFAULT_COUNT,
        help=f"How many of the most recently merged PRs to replay (default {DEFAULT_COUNT}).",
    )
    parser.add_argument("--model", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument(
        "--state", choices=("merged", "open"), default="merged",
        help="Replay merged history (the E1 gate) or preview currently open PRs.",
    )
    parser.add_argument(
        "--out", default=str(DEFAULT_REPORTS_DIR),
        help="Directory the consolidated report is written into.",
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        config = resolve_active(config, args.repo)
        repo = require_repo(config)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Replaying {repo.slug} | count={args.count} | model={args.model} | "
        "DRY RUN (forced -- never posts)\n"
    )

    if reviewer is None or enrichment is None:
        api_key = api_key_resolver()
        if reviewer is None:
            reviewer = gemini_reviewer(api_key, model=args.model, prompts_dir=PROMPTS_DIR)
        if enrichment is None:
            enrichment = build_enrichment(config, api_key, args.model)

    try:
        run = run_replay(
            config,
            repo.slug,
            args.count,
            args.model,
            reviewer,
            enrichment,
            runner=runner,
            reviews_dir=reviews_dir,
            pr_state=args.state,
        )
    except github.GitHubError as exc:
        print(f"Could not list {args.state} PRs for {repo.slug}: {exc}", file=sys.stderr)
        return 1

    out_path = write_report(run, args.out)

    reviewed = len(run.results)
    if reviewed < args.count:
        print(
            f"Note: only {reviewed} merged PR(s) available on {repo.slug} "
            f"(requested {args.count})."
        )

    errors = run.errors
    for result in errors:
        print(f"  PR #{result.pr.number}: {result.error}", file=sys.stderr)

    print(f"\n{reviewed}/{args.count} PR(s) reviewed | {len(errors)} error(s)")
    print(f"Report written to {out_path}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
