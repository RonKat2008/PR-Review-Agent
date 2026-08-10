"""Output sinks and the gates guarding the public one.

The local file sink always runs and is the audit trail: every verdict is written to
disk whether or not it was allowed to reach GitHub. The gate decision is a pure
function (`evaluate_comment_gates`) kept separate from the side effect, so the rules
protecting your teammates from a miscalibrated run are directly testable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import httpx

from . import github, reviews_api
from .config import Config, Secrets
from .github import PullRequest
from .review import Verdict, passes_gate
from .state import has_marker

DEFAULT_REVIEWS_DIR = Path("reviews")
WEBHOOK_TIMEOUT_SECONDS = 15
WEBHOOK_PREVIEW_LIMIT = 10


class SinkError(RuntimeError):
    """A sink failed in a way the sweep should surface rather than swallow."""


@dataclass(frozen=True)
class CommentBudget:
    """Immutable spend tracker for the per-sweep comment cap."""

    limit: int
    used: int = 0

    @property
    def exhausted(self) -> bool:
        return self.used >= self.limit

    def spend(self) -> CommentBudget:
        return replace(self, used=self.used + 1)


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    reason: str


@dataclass(frozen=True)
class CommentOutcome:
    posted: bool
    reason: str
    budget: CommentBudget


def evaluate_comment_gates(
    config: Config,
    pr: PullRequest,
    verdict: Verdict,
    budget: CommentBudget,
    existing_comments: Sequence[str] = (),
) -> GateDecision:
    """Decide whether this verdict may be posted publicly. Pure — no side effects.

    Ordered cheapest-check-first so the reason returned is the most fundamental one.
    """
    if not config.sinks.pr_comment:
        return GateDecision(False, "pr_comment sink disabled")

    if config.review.dry_run:
        return GateDecision(False, "dry_run enabled")

    if verdict.is_silent:
        return GateDecision(False, "verdict is empty")

    if not passes_gate(verdict, config.review.min_confidence):
        return GateDecision(
            False,
            f"confidence {verdict.confidence:.2f} below threshold "
            f"{config.review.min_confidence:.2f}",
        )

    bot_login = config.review.bot_login
    if bot_login and pr.author == bot_login:
        return GateDecision(False, f"PR authored by bot account {bot_login}")

    if budget.exhausted:
        return GateDecision(False, f"comment budget exhausted ({budget.limit} per sweep)")

    if has_marker(list(existing_comments), pr.head_sha):
        return GateDecision(False, f"already commented on head {pr.head_sha[:8]}")

    return GateDecision(True, "ok")


def post_pr_comment(
    config: Config,
    pr: PullRequest,
    verdict: Verdict,
    body: str,
    budget: CommentBudget,
    runner: github.GhRunner = github.default_runner,
    diff: str | None = None,
) -> CommentOutcome:
    """Post to GitHub if every gate allows it. Returns the outcome and updated budget.

    With `diff` supplied and inline comments enabled, findings are delivered as
    line-anchored review comments carrying committable suggestions, and the summary
    body absorbs whatever could not be anchored. Without it, a single summary
    comment is posted — the original behavior.
    """
    repo_slug = config.repo.slug

    existing: Sequence[str] = ()
    if config.sinks.pr_comment and not config.review.dry_run:
        try:
            existing = github.list_comments(repo_slug, pr.number, runner)
        except github.GitHubError as exc:
            # Cannot verify idempotency, so refuse rather than risk a duplicate.
            return CommentOutcome(False, f"could not read existing comments: {exc}", budget)

    decision = evaluate_comment_gates(config, pr, verdict, budget, existing)
    if not decision.allowed:
        return CommentOutcome(False, decision.reason, budget)

    try:
        if diff is not None and config.sinks.inline_comments:
            _post_inline_review(config, pr, verdict, body, diff, runner)
        else:
            github.post_comment(repo_slug, pr.number, body, runner)
    except github.GitHubError as exc:
        return CommentOutcome(False, f"post failed: {exc}", budget)

    return CommentOutcome(True, "posted", budget.spend())


def _post_inline_review(
    config: Config,
    pr: PullRequest,
    verdict: Verdict,
    body: str,
    diff: str,
    runner: github.GhRunner,
) -> None:
    """Deliver findings as line-anchored review comments.

    Findings whose line GitHub would reject are not dropped: they stay in the
    summary body, which is the whole rendered review. Anchoring is an enhancement
    to delivery, never a filter on content.
    """
    commentable = reviews_api.commentable_lines(diff)
    comments, unanchored = reviews_api.build_review_comments(verdict.introduces, commentable)

    summary = body
    if unanchored:
        summary += (
            f"\n\n<sub>{len(unanchored)} finding(s) could not be anchored to a "
            f"changed line and appear above in full.</sub>"
        )

    event = reviews_api.review_event_for(verdict, config.review.allow_request_changes)
    reviews_api.post_review(config.repo.slug, pr.number, summary, comments, event, runner)


def write_local(
    pr: PullRequest,
    verdict: Verdict,
    body: str,
    lane: str,
    reviews_dir: Path | str = DEFAULT_REVIEWS_DIR,
) -> Path:
    """Write the review to disk. Always runs — this is the audit trail."""
    directory = Path(reviews_dir)
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / f"PR-{pr.number}-{pr.head_sha[:8]}.md"
    front_matter = json.dumps(
        {
            "pr": pr.number,
            "lane": lane,
            "head_sha": pr.head_sha,
            "author": pr.author,
            "url": pr.url,
            "confidence": verdict.confidence,
            "introduces": len(verdict.introduces),
            "fixes": len(verdict.fixes),
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        },
        indent=2,
    )

    try:
        path.write_text(f"<!--\n{front_matter}\n-->\n\n{body}\n", encoding="utf-8")
    except OSError as exc:
        raise SinkError(f"Could not write review to {path}: {exc}") from exc
    return path


def send_webhook(
    config: Config,
    secrets: Secrets,
    summaries: Sequence[str],
    client: httpx.Client | None = None,
) -> bool:
    """Push the sweep digest. Returns False when there is nothing or nowhere to send."""
    if not config.sinks.webhook or not summaries:
        return False
    if not secrets.webhook_url:
        raise SinkError("sinks.webhook is enabled but no webhook URL is configured")

    payload = _webhook_payload(config.sinks.webhook_kind, summaries)
    owns_client = client is None
    http = client or httpx.Client(timeout=WEBHOOK_TIMEOUT_SECONDS)

    try:
        response = http.post(secrets.webhook_url, json=payload)
        if response.status_code >= 400:
            raise SinkError(
                f"Webhook returned {response.status_code}: {response.text[:200]}"
            )
    except httpx.HTTPError as exc:
        raise SinkError(f"Webhook delivery failed: {exc}") from exc
    finally:
        if owns_client:
            http.close()

    return True


def _webhook_payload(kind: str, summaries: Sequence[str]) -> dict:
    shown = list(summaries[:WEBHOOK_PREVIEW_LIMIT])
    overflow = len(summaries) - len(shown)
    if overflow > 0:
        shown.append(f"_…and {overflow} more_")

    text = "*Prime Agent PR review sweep*\n" + "\n".join(f"• {line}" for line in shown)

    if kind == "slack":
        return {"text": text}
    if kind == "discord":
        return {"content": text}
    return {"text": text, "count": len(summaries), "items": list(summaries)}
