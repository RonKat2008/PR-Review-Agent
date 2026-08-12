"""GitHub access via the `gh` CLI.

All subprocess use funnels through a single injectable runner, so every function
here is testable against recorded fixtures without a network or a token.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

GhRunner = Callable[[Sequence[str], str | None], str]

PR_FIELDS = (
    "number,title,author,headRefOid,baseRefName,url,additions,deletions,changedFiles,mergedAt"
)
DEFAULT_PAGE_LIMIT = 200
# Upper bound on how many PRs a single sweep will ever request while escalating
# --limit below. Bounds one sweep's listing work; if a repo still doesn't fit,
# the head-SHA watermark makes re-listing on the next sweep cheap, so we return
# what we have (with a warning) rather than requesting an unbounded number of PRs.
HARD_CAP = 2000
GH_TIMEOUT_SECONDS = 120


class GitHubError(RuntimeError):
    """A `gh` invocation failed or returned something unusable."""


@dataclass(frozen=True)
class PullRequest:
    number: int
    title: str
    author: str
    head_sha: str
    base_ref: str
    url: str
    additions: int
    deletions: int
    changed_files: int
    merged_at: str | None

    @property
    def size(self) -> int:
        return self.additions + self.deletions


def default_runner(args: Sequence[str], stdin: str | None = None) -> str:
    """Run `gh` with the given arguments and return stdout."""
    if shutil.which("gh") is None:
        raise GitHubError(
            "The `gh` CLI is not installed or not on PATH. "
            "Install it from https://cli.github.com/ and run `gh auth login`."
        )
    try:
        result = subprocess.run(
            ["gh", *args],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=GH_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise GitHubError(f"`gh {' '.join(args)}` timed out after {GH_TIMEOUT_SECONDS}s") from exc

    if result.returncode != 0:
        raise GitHubError(
            f"`gh {' '.join(args)}` failed with exit code {result.returncode}: "
            f"{result.stderr.strip()}"
        )
    return result.stdout


def list_open_prs(
    repo_slug: str,
    runner: GhRunner = default_runner,
    limit: int = DEFAULT_PAGE_LIMIT,
) -> tuple[PullRequest, ...]:
    """Open PRs, newest first.

    Escalates --limit and re-requests whenever a page comes back exactly as full
    as requested, since a full page is the signature of possible truncation. See
    `_list_prs_with_escalation`.
    """

    def build_args(page_limit: int) -> list[str]:
        return [
            "pr", "list",
            "--repo", repo_slug,
            "--state", "open",
            "--json", PR_FIELDS,
            "--limit", str(page_limit),
        ]

    return _list_prs_with_escalation(repo_slug, build_args, runner, limit)


def list_merged_prs(
    repo_slug: str,
    since: datetime,
    runner: GhRunner = default_runner,
    limit: int = DEFAULT_PAGE_LIMIT,
) -> tuple[PullRequest, ...]:
    """PRs merged on or after `since`, newest first.

    Deliberately uses `--state merged` and filters client-side rather than passing
    `--search merged:>=...`. The search flag hits GitHub's search index, which is
    eventually consistent and lags merges by up to a minute — a PR merged moments
    before a sweep would be invisible to it and, because the next sweep advances the
    watermark past it, could be missed permanently. The list API is immediately
    consistent.

    The `since` filter is applied after limit escalation (see
    `_list_prs_with_escalation`), so a truncated first page can never hide an
    eligible PR from it.
    """

    def build_args(page_limit: int) -> list[str]:
        return [
            "pr", "list",
            "--repo", repo_slug,
            "--state", "merged",
            "--json", PR_FIELDS,
            "--limit", str(page_limit),
        ]

    prs = _list_prs_with_escalation(repo_slug, build_args, runner, limit)
    return tuple(pr for pr in prs if _merged_since(pr, since))


def _list_prs_with_escalation(
    repo_slug: str,
    build_args: Callable[[int], list[str]],
    runner: GhRunner,
    limit: int,
) -> tuple[PullRequest, ...]:
    """Request a page of PRs, doubling `--limit` and re-requesting whenever the
    page comes back exactly as full as requested — the signature of possible
    truncation, since `gh pr list` never returns more than `--limit` asks for.

    `gh pr list` paginates GraphQL internally up to whatever `--limit` it is
    given, so escalating the limit is sufficient on its own; no manual pagination
    is needed here. Escalation stops once a page comes back short (nothing left
    to find) or `HARD_CAP` is reached. A page that is still full at `HARD_CAP` is
    returned anyway — the head-SHA watermark makes re-listing on the next sweep
    cheap — but is logged as a warning so a persistently truncated repo is visible
    instead of silently dropping PRs.
    """
    requested = min(limit, HARD_CAP)
    while True:
        raw = runner(build_args(requested), None)
        prs = _parse_pr_list(raw)
        if len(prs) < requested:
            return prs
        if requested >= HARD_CAP:
            warnings.warn(
                f"gh pr list on {repo_slug!r} returned a full page of "
                f"{len(prs)} at HARD_CAP ({HARD_CAP}); results may still be "
                "truncated.",
                stacklevel=2,
            )
            return prs
        requested = min(requested * 2, HARD_CAP)


def _merged_since(pr: PullRequest, since: datetime) -> bool:
    """Whether a PR merged at or after `since`. Unparseable timestamps are kept —
    reviewing one PR twice is cheap; dropping one silently is not."""
    if not pr.merged_at:
        return False
    try:
        merged = datetime.fromisoformat(pr.merged_at.replace("Z", "+00:00"))
    except ValueError:
        return True
    if merged.tzinfo is None:
        merged = merged.replace(tzinfo=timezone.utc)
    return merged >= since


def fetch_diff(repo_slug: str, number: int, runner: GhRunner = default_runner) -> str:
    """Unified diff for a single PR."""
    return runner(["pr", "diff", str(number), "--repo", repo_slug], None)


def post_comment(
    repo_slug: str,
    number: int,
    body: str,
    runner: GhRunner = default_runner,
) -> None:
    """Post a comment on a PR. Body is passed via stdin to survive any content."""
    runner(
        ["pr", "comment", str(number), "--repo", repo_slug, "--body-file", "-"],
        body,
    )


def list_comments(
    repo_slug: str,
    number: int,
    runner: GhRunner = default_runner,
) -> tuple[str, ...]:
    """Existing comment bodies on a PR, used to enforce idempotency."""
    raw = runner(
        ["api", f"repos/{repo_slug}/issues/{number}/comments", "--jq", ".[].body"],
        None,
    )
    return tuple(line for line in raw.splitlines() if line.strip())


def authenticated_login(runner: GhRunner = default_runner) -> str:
    """The login `gh` is authenticated as. Used for self-exclusion."""
    return runner(["api", "user", "--jq", ".login"], None).strip()


def lookback_cutoff(days: int, now: datetime | None = None) -> datetime:
    """UTC timestamp `days` before now."""
    reference = now or datetime.now(timezone.utc)
    return reference - timedelta(days=days)


def _parse_pr_list(raw: str) -> tuple[PullRequest, ...]:
    if not raw.strip():
        return ()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GitHubError(f"`gh pr list` returned non-JSON output: {exc}") from exc

    if not isinstance(payload, list):
        raise GitHubError(f"Expected a JSON array from `gh pr list`, got {type(payload).__name__}")

    return tuple(_parse_pr(item) for item in payload)


def _parse_pr(item: dict) -> PullRequest:
    try:
        return PullRequest(
            number=int(item["number"]),
            title=str(item.get("title", "")),
            author=str((item.get("author") or {}).get("login", "")),
            head_sha=str(item.get("headRefOid", "")),
            base_ref=str(item.get("baseRefName", "")),
            url=str(item.get("url", "")),
            additions=int(item.get("additions", 0)),
            deletions=int(item.get("deletions", 0)),
            changed_files=int(item.get("changedFiles", 0)),
            merged_at=item.get("mergedAt") or None,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise GitHubError(f"Malformed pull request entry from gh: {item!r}") from exc
