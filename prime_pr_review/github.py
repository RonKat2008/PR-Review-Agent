"""GitHub access via the `gh` CLI.

All subprocess use funnels through a single injectable runner, so every function
here is testable against recorded fixtures without a network or a token.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

GhRunner = Callable[[Sequence[str], str | None], str]

PR_FIELDS = (
    "number,title,author,headRefOid,baseRefName,url,additions,deletions,changedFiles,mergedAt"
)
DEFAULT_PAGE_LIMIT = 50
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
    """Open PRs, newest first."""
    raw = runner(
        [
            "pr", "list",
            "--repo", repo_slug,
            "--state", "open",
            "--json", PR_FIELDS,
            "--limit", str(limit),
        ],
        None,
    )
    return _parse_pr_list(raw)


def list_merged_prs(
    repo_slug: str,
    since: datetime,
    runner: GhRunner = default_runner,
    limit: int = DEFAULT_PAGE_LIMIT,
) -> tuple[PullRequest, ...]:
    """PRs merged on or after `since`, newest first."""
    raw = runner(
        [
            "pr", "list",
            "--repo", repo_slug,
            "--state", "merged",
            "--search", f"merged:>={since.date().isoformat()}",
            "--json", PR_FIELDS,
            "--limit", str(limit),
        ],
        None,
    )
    return _parse_pr_list(raw)


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
