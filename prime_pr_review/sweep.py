"""Sweep orchestration: fetch → dedupe → review → sink → advance watermark.

The `reviewer` is injected. Inside prime-agent it is backed by an `rlm(...)` subagent
call; in tests it is a stub. That boundary is what lets the whole pipeline be tested
without a model, a token, or a network.

One failing PR never aborts a sweep. Failures are recorded on the report and surfaced
in the digest rather than swallowed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from . import context as context_mod
from . import github
from .blast import analyze_blast_radius
from .config import Config, Secrets
from .diffs import filter_diff
from .github import PullRequest
from .intent import IntentError, run_intent_check
from .review import Verdict, VerdictError, parse_verdict
from .template import render_review
from .sinks import CommentBudget, DEFAULT_REVIEWS_DIR, post_pr_comment, write_local
from .state import (
    LANE_MERGED,
    LANE_OPEN,
    State,
    is_reviewed,
    mark_reviewed,
    set_merged_cursor,
)

# (pull_request, review_payload, lane) -> raw verdict JSON text.
# The payload is the filtered diff plus any gathered repository context, so the
# reviewer contract never had to change to accommodate enrichment.
Reviewer = Callable[[PullRequest, str, str], str]

# (prompt) -> raw model response. Used for analysis passes that are not the review
# itself, such as the intent check.
ModelFn = Callable[[str], str]

DEFAULT_PROMPTS_DIR = Path("skills/pr-review/prompts")


@dataclass(frozen=True)
class Enrichment:
    """Optional analysis layered on top of the diff before the review runs.

    Absent (the default) means the reviewer sees only the diff, which is the
    behavior every existing test asserts. Enrichment never fails a review: if
    context gathering or the intent check errors, the review proceeds without it
    and the reason is recorded on the outcome.
    """

    model_fn: ModelFn | None = None
    repo_root: Path = Path(".")
    prompts_dir: Path = DEFAULT_PROMPTS_DIR


@dataclass(frozen=True)
class PullRequestOutcome:
    pr: PullRequest
    lane: str
    verdict: Verdict | None = None
    posted: bool = False
    reason: str = ""
    local_path: Path | None = None
    error: str | None = None
    # Enrichment that was attempted and failed. Non-fatal: the review still ran,
    # just with less context than intended. Surfaced so a silently degraded
    # review is distinguishable from a fully informed one.
    notes: tuple[str, ...] = ()

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
        line = (
            f"PR #{self.pr.number}{severity} — {bugs} introduced, {fixes} fixed, "
            f"{self.verdict.confidence:.0%} confidence — {status}"
        )
        # Degraded enrichment must be visible. Otherwise "the intent check found
        # nothing" and "the intent check silently failed" produce identical output.
        if self.notes:
            line += f"  [{'; '.join(self.notes)}]"
        return line


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
    enrichment: Enrichment | None = None,
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
            config, lane, pr, reviewer, budget, runner, repo_slug, reviews_dir, enrichment
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
    enrichment: Enrichment | None = None,
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

    payload, notes = _build_payload(config, pr, filtered.text, repo_slug, runner, enrichment)

    try:
        verdict = parse_verdict(reviewer(pr, payload, lane))
    except VerdictError as exc:
        return PullRequestOutcome(pr=pr, lane=lane, error=f"unusable verdict: {exc}"), budget
    except Exception as exc:  # noqa: BLE001 - a subagent may fail in any manner
        return PullRequestOutcome(pr=pr, lane=lane, error=f"reviewer failed: {exc}"), budget

    verdict, scope_notes = _attach_scope(config, pr, filtered.text, verdict, enrichment)
    notes += scope_notes

    verdict, blast_notes = _attach_blast(config, pr, filtered.text, verdict, enrichment)
    notes += blast_notes

    body = render_review(pr, verdict, lane)
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
            notes=notes,
        ),
        outcome.budget,
    )


def _build_payload(
    config: Config,
    pr: PullRequest,
    diff: str,
    repo_slug: str,
    runner: github.GhRunner,
    enrichment: Enrichment | None,
) -> tuple[str, tuple[str, ...]]:
    """Diff plus repository context, when enabled and available.

    Enrichment is best-effort by design. A repo we cannot read, a git binary that
    is missing, an API that rate-limits — none of those should cost us the review
    entirely. We degrade to the bare diff and say so.
    """
    if enrichment is None or not config.review.gather_context:
        return diff, ()

    # Call-site discovery shells out to `git grep`, which runs in the process's
    # working directory. Without a configured checkout of the reviewed repo we
    # would be grepping whatever repo the sweep happens to run from and reporting
    # its matches as call sites of the PR's symbols. Confidently wrong beats
    # nothing only in the wrong direction, so skip instead.
    if not config.review.repo_root:
        return diff, ("context skipped: review.repo_root is not set",)

    try:
        gathered = context_mod.gather_context(
            repo_slug=repo_slug,
            head_sha=pr.head_sha,
            diff=diff,
            repo_root=enrichment.repo_root,
            gh_runner=runner,
            max_bytes=config.review.max_context_bytes,
        )
    except Exception as exc:  # noqa: BLE001 - degrade, never fail the review
        return diff, (f"context unavailable: {exc}",)

    return f"{diff}\n\n{gathered.render()}", gathered.dropped


def _attach_scope(
    config: Config,
    pr: PullRequest,
    diff: str,
    verdict: Verdict,
    enrichment: Enrichment | None,
) -> tuple[Verdict, tuple[str, ...]]:
    """Run the two-pass intent check and attach its Scope to the verdict."""
    if enrichment is None or enrichment.model_fn is None or not config.review.check_intent:
        return verdict, ()

    try:
        scope = run_intent_check(pr, diff, enrichment.model_fn, enrichment.prompts_dir)
    except IntentError as exc:
        return verdict, (f"intent check unusable: {exc}",)
    except Exception as exc:  # noqa: BLE001 - degrade, never fail the review
        return verdict, (f"intent check failed: {exc}",)

    return replace(verdict, scope=scope), ()


def _attach_blast(
    config: Config,
    pr: PullRequest,
    diff: str,
    verdict: Verdict,
    enrichment: Enrichment | None,
) -> tuple[Verdict, tuple[str, ...]]:
    """Run blast-radius analysis and attach its entries to the verdict."""
    if enrichment is None or enrichment.model_fn is None or not config.review.check_blast:
        return verdict, ()

    # Same constraint as context gathering: call-site discovery runs `git grep` in
    # the process working directory. Without the reviewed repo checked out we would
    # report matches from an unrelated codebase as callers of this PR's symbols.
    if not config.review.repo_root:
        return verdict, ("blast radius skipped: review.repo_root is not set",)

    try:
        entries = analyze_blast_radius(
            pr,
            diff,
            enrichment.model_fn,
            context_mod.default_git_runner,
            enrichment.repo_root,
            enrichment.prompts_dir,
        )
    except Exception as exc:  # noqa: BLE001 - degrade, never fail the review
        return verdict, (f"blast radius failed: {exc}",)

    return replace(verdict, blast_radius=entries), ()


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
