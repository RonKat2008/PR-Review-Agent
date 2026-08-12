"""Shared fixtures and builders.

Everything the system touches externally — gh, the filesystem, the webhook, the model —
is injectable, so no test needs a token, a network, or a repository.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import pytest

from prime_pr_review.config import (
    Config,
    RepoConfig,
    ReviewConfig,
    ScheduleConfig,
    Secrets,
    SinkConfig,
)
from prime_pr_review.github import GitHubError, PullRequest

VERDICT_WITH_BUG = (
    '{"introduces":[{"file":"src/app.py","line":10,"severity":"HIGH",'
    '"claim":"Off-by-one in the loop bound",'
    '"evidence":"range(n) skips the final element"}],'
    '"fixes":[{"claim":"Guards against a null user","evidence":"adds an is-None check"}],'
    '"confidence":0.9}'
)

VERDICT_EMPTY = '{"introduces":[],"fixes":[],"confidence":0.95}'

VERDICT_LOW_CONFIDENCE = (
    '{"introduces":[{"file":"src/app.py","line":3,"severity":"MEDIUM",'
    '"claim":"Possible race","evidence":"shared counter"}],'
    '"fixes":[],"confidence":0.2}'
)

SAMPLE_DIFF = """diff --git a/src/app.py b/src/app.py
index 1111111..2222222 100644
--- a/src/app.py
+++ b/src/app.py
@@ -1,3 +1,3 @@
-total = sum(values)
+total = sum(values[:-1])
diff --git a/uv.lock b/uv.lock
index 3333333..4444444 100644
--- a/uv.lock
+++ b/uv.lock
@@ -1 +1 @@
-version = 1
+version = 2
"""


def make_config(
    *,
    owner: str = "acme",
    name: str = "widget",
    read_only: bool = False,
    dry_run: bool = False,
    min_confidence: float = 0.7,
    max_comments: int = 5,
    merged_lookback_days: int = 7,
    bot_login: str = "prime-bot",
    ignore_paths: tuple[str, ...] = ("**/*.lock",),
    max_diff_bytes: int = 200_000,
    graph_path: str = "",
    pr_comment: bool = True,
    webhook: bool = True,
    local_file: bool = True,
    webhook_kind: str = "slack",
    inline_comments: bool = True,
    allow_request_changes: bool = False,
) -> Config:
    return Config(
        repo=RepoConfig(owner=owner, name=name, read_only=read_only),
        schedule=ScheduleConfig(open_prs="0 */4 * * *", merged_prs="0 9 * * 1-5"),
        review=ReviewConfig(
            dry_run=dry_run,
            min_confidence=min_confidence,
            max_comments_per_sweep=max_comments,
            merged_lookback_days=merged_lookback_days,
            bot_login=bot_login,
            ignore_paths=ignore_paths,
            max_diff_bytes=max_diff_bytes,
            allow_request_changes=allow_request_changes,
            graph_path=graph_path,
        ),
        sinks=SinkConfig(
            pr_comment=pr_comment,
            webhook=webhook,
            local_file=local_file,
            webhook_kind=webhook_kind,
            inline_comments=inline_comments,
        ),
    )


def make_pr(
    *,
    number: int = 101,
    title: str = "Fix the widget",
    author: str = "alice",
    head_sha: str = "abcdef1234567890",
    base_ref: str = "main",
    additions: int = 10,
    deletions: int = 2,
    changed_files: int = 1,
    merged_at: str | None = None,
) -> PullRequest:
    return PullRequest(
        number=number,
        title=title,
        author=author,
        head_sha=head_sha,
        base_ref=base_ref,
        url=f"https://github.com/acme/widget/pull/{number}",
        additions=additions,
        deletions=deletions,
        changed_files=changed_files,
        merged_at=merged_at,
    )


def pr_list_json(*prs: PullRequest) -> str:
    import json

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
            for pr in prs
        ]
    )


@dataclass
class FakeGh:
    """A gh runner backed by predicate/response pairs. Records every call."""

    handlers: list[tuple[Callable[[Sequence[str]], bool], str]] = field(default_factory=list)
    calls: list[tuple[list[str], str | None]] = field(default_factory=list)

    def on(self, predicate: Callable[[Sequence[str]], bool], response: str) -> FakeGh:
        self.handlers.append((predicate, response))
        return self

    def __call__(self, args: Sequence[str], stdin: str | None = None) -> str:
        self.calls.append((list(args), stdin))
        for predicate, response in self.handlers:
            if predicate(args):
                return response
        raise GitHubError(f"unexpected gh call: {' '.join(args)}")

    def calls_matching(self, *fragments: str) -> list[tuple[list[str], str | None]]:
        return [c for c in self.calls if all(f in " ".join(c[0]) for f in fragments)]


def is_pr_list(args: Sequence[str]) -> bool:
    return args[0] == "pr" and args[1] == "list"


def is_pr_diff(args: Sequence[str]) -> bool:
    return args[0] == "pr" and args[1] == "diff"


def is_pr_comment(args: Sequence[str]) -> bool:
    return args[0] == "pr" and args[1] == "comment"


def is_list_comments(args: Sequence[str]) -> bool:
    return args[0] == "api" and "issues" in " ".join(args)


def is_post_review(args: Sequence[str]) -> bool:
    """The reviews API call used for line-anchored comments with suggestions."""
    return args[0] == "api" and "/reviews" in " ".join(args)


@pytest.fixture
def config() -> Config:
    return make_config()


@pytest.fixture
def secrets() -> Secrets:
    return Secrets(github_token="ghp_test", webhook_url="https://hooks.example/abc")


@pytest.fixture
def pr() -> PullRequest:
    return make_pr()
