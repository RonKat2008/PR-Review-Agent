"""Line-anchored PR review comments with committable suggestions (P4).

A summary comment tells the reader "line 15 has SQL injection" and leaves them to
go find line 15. GitHub review comments anchor directly to a line, and a
```suggestion``` fenced block renders as a one-click "Commit suggestion" button.
This module turns `Finding`s into that shape and posts them as a single review.

GitHub rejects a review comment anchored to a line that is not part of the diff,
and a single rejected comment fails the *whole* `POST .../reviews` call — so
`commentable_lines` is the load-bearing filter that keeps every other comment in
the batch from being collateral damage.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from .diffs import split_by_file
from .github import GhRunner, default_runner
from .review import Finding, Severity, Verdict

EVENT_COMMENT = "COMMENT"
EVENT_REQUEST_CHANGES = "REQUEST_CHANGES"
EVENT_APPROVE = "APPROVE"

_SIDE_RIGHT = "RIGHT"

# `@@ -a[,b] +c[,d] @@ optional section heading`. The counts are optional (a bare
# `-1 +1` means "one line"); only the right-side start (`c`) is needed here.
_HUNK_HEADER_PATTERN = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


@dataclass(frozen=True)
class ReviewComment:
    """One line-anchored comment, ready to serialize into a GitHub review payload."""

    path: str
    body: str
    line: int
    side: str = _SIDE_RIGHT
    start_line: int | None = None
    start_side: str | None = None


def commentable_lines(diff: str) -> frozenset[tuple[str, int]]:
    """Every (path, line) on the right side of `diff` that GitHub accepts a comment on.

    "Right side" means anything visible in the new file version: added lines and
    unchanged context lines both carry a right-side line number. Removed-only
    lines do not and are excluded.
    """
    commentable: set[tuple[str, int]] = set()
    for file_diff in split_by_file(diff):
        commentable.update(_right_side_lines(file_diff.path, file_diff.body))
    return frozenset(commentable)


def _right_side_lines(path: str, body: str) -> Iterator[tuple[str, int]]:
    right_line = 0
    in_hunk = False
    for line in body.splitlines():
        header = _HUNK_HEADER_PATTERN.match(line)
        if header:
            right_line = int(header.group(1))
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if line.startswith("-") or line.startswith("\\"):
            continue  # left-only line, or "\ No newline at end of file"
        if line.startswith("+") or line.startswith(" ") or line == "":
            yield path, right_line
            right_line += 1


def build_review_comments(
    findings: Sequence[Finding],
    commentable: frozenset[tuple[str, int]],
) -> tuple[tuple[ReviewComment, ...], tuple[Finding, ...]]:
    """Split findings into ones that can be anchored to a diff line and ones that can't.

    Unanchorable findings (no line, or a line GitHub would reject) are returned
    rather than dropped, so the caller can fold them into the summary body instead.
    """
    anchored: list[ReviewComment] = []
    unanchored: list[Finding] = []
    for finding in findings:
        comment = _anchor(finding, commentable)
        if comment is None:
            unanchored.append(finding)
        else:
            anchored.append(comment)
    return tuple(anchored), tuple(unanchored)


def _anchor(finding: Finding, commentable: frozenset[tuple[str, int]]) -> ReviewComment | None:
    if finding.line is None:
        return None
    if (finding.file, finding.line) not in commentable:
        return None

    body = _render_comment_body(finding)

    # A multi-line comment anchors on line_end, so that line must be commentable
    # too. GitHub rejects the ENTIRE review when any single comment is invalid, so
    # an unverified model-supplied line_end could cost every other finding in the
    # sweep. Degrade to a single-line comment instead of gambling the whole call.
    if (
        finding.line_end is not None
        and finding.line_end > finding.line
        and (finding.file, finding.line_end) in commentable
    ):
        return ReviewComment(
            path=finding.file,
            body=body,
            line=finding.line_end,
            side=_SIDE_RIGHT,
            start_line=finding.line,
            start_side=_SIDE_RIGHT,
        )
    return ReviewComment(path=finding.file, body=body, line=finding.line, side=_SIDE_RIGHT)


def _render_comment_body(finding: Finding) -> str:
    parts = [f"**{finding.severity.value}** — {finding.claim}"]
    if finding.evidence:
        parts.append(finding.evidence)
    if finding.corroboration:
        parts.append(f"_Corroborated by `{finding.corroboration}`._")
    if finding.has_suggestion:
        parts.append(f"```suggestion\n{finding.suggestion}\n```")
    return "\n\n".join(parts)


def post_review(
    repo_slug: str,
    pr_number: int,
    body: str,
    comments: Sequence[ReviewComment],
    event: str,
    runner: GhRunner = default_runner,
) -> None:
    """POST a single review — body, event, and every anchored comment — in one call."""
    payload = {
        "body": body,
        "event": event,
        "comments": [_comment_payload(c) for c in comments],
    }
    runner(
        [
            "api",
            f"repos/{repo_slug}/pulls/{pr_number}/reviews",
            "--method", "POST",
            "--input", "-",
        ],
        json.dumps(payload),
    )


def _comment_payload(comment: ReviewComment) -> dict:
    payload: dict = {
        "path": comment.path,
        "body": comment.body,
        "line": comment.line,
        "side": comment.side,
    }
    if comment.start_line is not None:
        payload["start_line"] = comment.start_line
    if comment.start_side is not None:
        payload["start_side"] = comment.start_side
    return payload


def review_event_for(verdict: Verdict, allow_request_changes: bool) -> str:
    """COMMENT unless the verdict is severe enough to justify blocking the merge.

    CRITICAL severity or any broken caller both qualify — but only ever turn into
    REQUEST_CHANGES when the caller has explicitly opted in. Blocking someone's
    merge automatically is a strong action and must not be the default.
    """
    # A refuted CRITICAL cannot block: the skeptic challenged its premise, and
    # REQUEST_CHANGES on a disputed claim would gate a colleague's merge on it.
    is_critical = any(
        finding.severity is Severity.CRITICAL and not finding.refuted
        for finding in verdict.introduces
    )
    has_broken_caller = bool(verdict.broken_callers)
    if (is_critical or has_broken_caller) and allow_request_changes:
        return EVENT_REQUEST_CHANGES
    return EVENT_COMMENT
