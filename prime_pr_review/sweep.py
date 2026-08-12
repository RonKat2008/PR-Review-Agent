"""Sweep orchestration: fetch → dedupe → review → sink → advance watermark.

The `reviewer` is injected. Inside prime-agent it is backed by an `rlm(...)` subagent
call; in tests it is a stub. That boundary is what lets the whole pipeline be tested
without a model, a token, or a network.

One failing PR never aborts a sweep. Failures are recorded on the report and surfaced
in the digest rather than swallowed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from . import context as context_mod
from . import github
from . import graph as graph_mod
from .analysis import AnalysisResult
from .blast import analyze_blast_radius, extract_changed_symbols
from .context import GitRunner
from .diffs import split_by_file
from .feedback import Rejection, filter_rejected, render_rejection_guidance
from .reviews_api import commentable_lines
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

# (paths, diff_lines) -> AnalysisResult. The static-analysis pre-pass (P3);
# injected so tests never shell out to ruff/bandit/mypy.
AnalysisFn = Callable[[Sequence[str], frozenset[tuple[str, int]]], AnalysisResult]

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
    # Strict git runner for the graph freshness check (graph.strict_runner).
    # Deliberately NOT the lenient grep runner: exit 1 must mean "refuse".
    git_runner: GitRunner | None = None
    # Static-analysis pre-pass; None disables it.
    analysis_fn: AnalysisFn | None = None
    # Maintainer feedback (P6): findings previously rejected with a thumbs-down
    # or dismissal reply. Injected as prompt guidance AND enforced as a
    # post-verdict filter; suppressions are recorded on the outcome, never silent.
    rejections: tuple[Rejection, ...] = ()


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

    verdict, suppression_notes = _apply_feedback(verdict, enrichment)
    notes += suppression_notes

    body = render_review(pr, verdict, lane)
    local_path = write_local(pr, verdict, body, lane, reviews_dir) if config.sinks.local_file else None
    outcome = post_pr_comment(config, pr, verdict, body, budget, runner, diff=filtered.text)

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
    if enrichment is None:
        return diff, ()

    parts: list[str] = [diff]
    notes: list[str] = []
    has_root = bool(config.review.repo_root)

    if config.review.gather_context:
        if not has_root:
            # Call-site discovery shells out to `git grep`, which runs in the
            # process's working directory. Without a configured checkout of the
            # reviewed repo we would grep whatever repo the sweep runs from and
            # report its matches as call sites of the PR's symbols. Confidently
            # wrong beats nothing only in the wrong direction, so skip instead.
            notes.append("context skipped: review.repo_root is not set")
        else:
            try:
                gathered = context_mod.gather_context(
                    repo_slug=repo_slug,
                    head_sha=pr.head_sha,
                    diff=diff,
                    repo_root=enrichment.repo_root,
                    gh_runner=runner,
                    max_bytes=config.review.max_context_bytes,
                )
                parts.append(gathered.render())
                notes.extend(gathered.dropped)
            except Exception as exc:  # noqa: BLE001 - degrade, never fail the review
                notes.append(f"context unavailable: {exc}")

    graph_part, graph_notes = _graph_section(config, pr, diff, enrichment)
    if graph_part:
        parts.append(graph_part)
    notes.extend(graph_notes)

    analysis_part, analysis_notes = _analysis_section(config, diff, enrichment)
    if analysis_part:
        parts.append(analysis_part)
    notes.extend(analysis_notes)

    guidance = render_rejection_guidance(enrichment.rejections)
    if guidance:
        parts.append(guidance)

    return "\n\n".join(parts), tuple(notes)


def _graph_section(
    config: Config,
    pr: PullRequest,
    diff: str,
    enrichment: Enrichment,
) -> tuple[str, tuple[str, ...]]:
    """Knowledge-graph evidence for the prompt: co-change warnings and callers.

    The freshness check compares the graph's commit against the FETCHED base
    (`origin/<base_ref>`), not the local branch — local base branches on dev
    machines routinely lag by dozens of commits, which would wrongly refuse a
    perfectly current graph.
    """
    if not config.review.graph_path:
        return "", ()
    if enrichment.git_runner is None:
        return "", ("graph skipped: no git runner configured for the ancestry check",)

    graph, reason = graph_mod.load_for_review(
        config.review.graph_path,
        f"origin/{pr.base_ref}",
        enrichment.git_runner,
        enrichment.repo_root,
    )
    if graph is None:
        return "", (reason,)

    diff_files = tuple(f.path for f in split_by_file(diff))
    symbol_ids = tuple(f"{s.file}::{s.name}" for s in extract_changed_symbols(diff))
    return graph_mod.render(graph, diff_files, symbol_ids), ()


def _apply_feedback(
    verdict: Verdict,
    enrichment: Enrichment | None,
) -> tuple[Verdict, tuple[str, ...]]:
    """Suppress findings maintainers already rejected (P6).

    Suppression is auditable, never silent: every dropped finding is named in the
    outcome notes. The prompt guidance usually prevents these findings from being
    produced at all; this filter is the hard guarantee when it does not.
    """
    if enrichment is None or not enrichment.rejections or not verdict.introduces:
        return verdict, ()

    kept, suppressed = filter_rejected(verdict.introduces, enrichment.rejections)
    if not suppressed:
        return verdict, ()

    notes = tuple(
        f"suppressed by maintainer feedback: {f.file} — {f.claim[:80]}"
        for f in suppressed
    )
    return replace(verdict, introduces=kept), notes


def _analysis_section(
    config: Config,
    diff: str,
    enrichment: Enrichment,
) -> tuple[str, tuple[str, ...]]:
    """Static-analysis pre-pass (P3): deterministic findings as grounding evidence.

    Runs only with a repo checkout present — the analyzers read files on disk,
    and the default runner assumes the process cwd is the reviewed repo.
    """
    if enrichment.analysis_fn is None or not config.review.repo_root:
        return "", ()

    diff_files = tuple(f.path for f in split_by_file(diff))
    try:
        result = enrichment.analysis_fn(diff_files, commentable_lines(diff))
    except Exception as exc:  # noqa: BLE001 - degrade, never fail the review
        return "", (f"static analysis failed: {exc}",)

    if result.is_empty and not result.errors:
        return "", ()
    return result.render(), ()


def _is_docs_only(diff: str, globs: Sequence[str]) -> bool:
    """True when every file in the diff matches a docs glob.

    Docs-only diffs skip the intent and blast passes (C4): prose cannot break
    callers, and each skipped pass is a model call saved at scale.
    """
    import fnmatch

    paths = [f.path for f in split_by_file(diff)]
    if not paths:
        return False

    def matches(path: str) -> bool:
        for pattern in globs:
            if fnmatch.fnmatch(path, pattern):
                return True
            if pattern.startswith("**/") and fnmatch.fnmatch(path, pattern[3:]):
                return True
        return False

    return all(matches(p) for p in paths)


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

    changed = len(split_by_file(diff))
    if config.review.intent_min_files and changed < config.review.intent_min_files:
        return verdict, (
            f"intent check skipped: {changed} file(s) below intent_min_files",
        )
    if _is_docs_only(diff, config.review.docs_globs):
        return verdict, ("intent check skipped: docs-only diff",)

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
    if (
        enrichment is not None
        and config.review.check_blast
        and _is_docs_only(diff, config.review.docs_globs)
    ):
        return verdict, ("blast radius skipped: docs-only diff",)
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
