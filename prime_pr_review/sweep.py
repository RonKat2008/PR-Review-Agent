"""Sweep orchestration: fetch → dedupe → review → sink → advance watermark.

The `reviewer` is injected. Inside prime-agent it is backed by an `rlm(...)` subagent
call; in tests it is a stub. That boundary is what lets the whole pipeline be tested
without a model, a token, or a network.

One failing PR never aborts a sweep. Failures are recorded on the report and surfaced
in the digest rather than swallowed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import github
from .config import Config, Secrets
from .diffs import filter_diff
from .github import PullRequest
from .review import Verdict, VerdictError, parse_verdict, render_markdown
from .sinks import CommentBudget, DEFAULT_REVIEWS_DIR, post_pr_comment, write_local
from .state import (
    LANE_MERGED,
    LANE_OPEN,
    State,
    is_reviewed,
    mark_reviewed,
    set_merged_cursor,
)

# (pull_request, filtered_diff, lane) -> raw verdict JSON text
Reviewer = Callable[[PullRequest, str, str], str]


@dataclass(frozen=True)
class PullRequestOutcome:
    pr: PullRequest
    lane: str
    verdict: Verdict | None = None
    posted: bool = False
    reason: str = ""
    local_path: Path | None = None
    error: str | None = None

    def summary_line(self) -> str:
        if self.error:
            return f"PR #{self.pr.number} — error: {self.error}"
        if self.verdict is None:
            return f"PR #{self.pr.number} — skipped ({self.reason})"

        bugs = len(self.verdict.introduces)
        fixes = len(self.verdict.fixes)
        worst = self.verdict.worst_severity
        severity = f" [{worst.value}]" if worst else ""
        status = "posted" if self.posted else f"held ({self.reason})"
        return (
            f"PR #{self.pr.number}{severity} — {bugs} introduced, {fixes} fixed, "
            f"{self.verdict.confidence:.0%} confidence — {status}"
        )


@dataclass(frozen=True)
class SweepReport:
    lane: str
    considered: int = 0
    reviewed: int = 0
    posted: int = 0
    skipped: int = 0
    errors: int = 0
    outcomes: tuple[PullRequestOutcome, ...] = field(default_factory=tuple)

    def summaries(self) -> tuple[str, ...]:
        return tuple(o.summary_line() for o in self.outcomes)


def sweep_lane(
    config: Config,
    lane: str,
    reviewer: Reviewer,
    state: State,
    runner: github.GhRunner = github.default_runner,
    reviews_dir: Path | str = DEFAULT_REVIEWS_DIR,
    now: datetime | None = None,
) -> tuple[SweepReport, State]:
    """Review every eligible PR in one lane. Returns the report and the advanced state."""
    repo_slug = config.repo.slug
    candidates = _select_candidates(config, lane, runner, now)
    budget = CommentBudget(limit=config.review.max_comments_per_sweep)

    outcomes: list[PullRequestOutcome] = []
    working_state = state

    for pr in candidates:
        skip_reason = _skip_reason(config, working_state, lane, pr)
        if skip_reason:
            outcomes.append(PullRequestOutcome(pr=pr, lane=lane, reason=skip_reason))
            continue

        outcome, budget = _review_one(
            config, lane, pr, reviewer, budget, runner, repo_slug, reviews_dir
        )
        outcomes.append(outcome)
        if outcome.error is None:
            working_state = mark_reviewed(working_state, lane, pr.number, pr.head_sha)

    if lane == LANE_MERGED:
        working_state = set_merged_cursor(
            working_state, (now or datetime.now(timezone.utc)).isoformat()
        )

    return _build_report(lane, len(candidates), tuple(outcomes)), working_state


def _review_one(
    config: Config,
    lane: str,
    pr: PullRequest,
    reviewer: Reviewer,
    budget: CommentBudget,
    runner: github.GhRunner,
    repo_slug: str,
    reviews_dir: Path | str,
) -> tuple[PullRequestOutcome, CommentBudget]:
    """Review a single PR, converting any failure into a recorded outcome."""
    try:
        raw_diff = github.fetch_diff(repo_slug, pr.number, runner)
    except github.GitHubError as exc:
        return PullRequestOutcome(pr=pr, lane=lane, error=f"diff fetch failed: {exc}"), budget

    filtered = filter_diff(raw_diff, config.review.ignore_paths, config.review.max_diff_bytes)
    if filtered.is_empty:
        return (
            PullRequestOutcome(pr=pr, lane=lane, reason="diff empty after filtering"),
            budget,
        )

    try:
        verdict = parse_verdict(reviewer(pr, filtered.text, lane))
    except VerdictError as exc:
        return PullRequestOutcome(pr=pr, lane=lane, error=f"unusable verdict: {exc}"), budget
    except Exception as exc:  # noqa: BLE001 - a subagent may fail in any manner
        return PullRequestOutcome(pr=pr, lane=lane, error=f"reviewer failed: {exc}"), budget

    body = render_markdown(pr, verdict, lane)
    local_path = write_local(pr, verdict, body, lane, reviews_dir) if config.sinks.local_file else None
    outcome = post_pr_comment(config, pr, verdict, body, budget, runner)

    return (
        PullRequestOutcome(
            pr=pr,
            lane=lane,
            verdict=verdict,
            posted=outcome.posted,
            reason=outcome.reason,
            local_path=local_path,
        ),
        outcome.budget,
    )


def _select_candidates(
    config: Config,
    lane: str,
    runner: github.GhRunner,
    now: datetime | None,
) -> tuple[PullRequest, ...]:
    repo_slug = config.repo.slug
    if lane == LANE_OPEN:
        return github.list_open_prs(repo_slug, runner)
    if lane == LANE_MERGED:
        cutoff = github.lookback_cutoff(config.review.merged_lookback_days, now)
        return github.list_merged_prs(repo_slug, cutoff, runner)
    raise ValueError(f"Unknown lane {lane!r}")


def _skip_reason(config: Config, state: State, lane: str, pr: PullRequest) -> str:
    """Reasons to skip a PR before spending any model tokens on it."""
    if is_reviewed(state, lane, pr.number, pr.head_sha):
        return f"already reviewed at {pr.head_sha[:8]}"
    bot_login = config.review.bot_login
    if bot_login and pr.author == bot_login:
        return f"authored by bot account {bot_login}"
    return ""


def _build_report(
    lane: str,
    considered: int,
    outcomes: tuple[PullRequestOutcome, ...],
) -> SweepReport:
    return SweepReport(
        lane=lane,
        considered=considered,
        reviewed=sum(1 for o in outcomes if o.verdict is not None),
        posted=sum(1 for o in outcomes if o.posted),
        skipped=sum(1 for o in outcomes if o.verdict is None and o.error is None),
        errors=sum(1 for o in outcomes if o.error is not None),
        outcomes=outcomes,
    )
