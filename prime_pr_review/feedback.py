"""The false-positive feedback loop (P6).

Reads reactions and dismissal replies on the bot's own prior PR comments, turns
them into persisted `Rejection`s, and spends them two ways: `filter_rejected`
drops findings that were already rejected before they reach a sink, and
`render_rejection_guidance` renders a summary to inject into the next prompt so
the model is told not to re-report them.

Persistence mirrors `state.py` deliberately: atomic tmp-write-then-replace, a
frozen dataclass, a missing file is a cold start, and a malformed file raises a
clear error rather than silently resetting.

Standalone module. Everything GitHub-facing goes through the injectable
`GhRunner` from `github.py`, so nothing here needs a network or a token to test,
and `fetch_rejections` is built to fail closed: any `gh` trouble at all yields an
empty tuple rather than raising, because reading feedback must never break a
sweep.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .github import GhRunner, GitHubError, default_runner
from .review import Finding

DEFAULT_REJECTIONS_PATH = Path("state/rejections.json")

# Case-insensitive substrings. A maintainer reply containing any of these is
# read as "this finding was wrong" even without an explicit reaction.
DEFAULT_DISMISSAL_PHRASES: tuple[str, ...] = (
    "not a bug",
    "false positive",
    "intentional",
    "wontfix",
)

# Small on purpose -- see `claim_fingerprint`. Every word removed from this set
# is a word that can no longer distinguish two otherwise-similar claims.
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "this", "that", "these", "those", "it", "its", "to", "of", "in", "on",
        "at", "for", "and", "or", "but", "with", "as", "by", "from", "not",
    }
)
_PUNCTUATION_RE = re.compile(r"[^\w\s]")
_FINGERPRINT_HEX_LENGTH = 16

# Every finding line our templates render (`template.render_review` and the
# legacy `review.render_markdown`) opens with a bold severity marker, optionally
# behind a `- ` bullet:
#   - **HIGH · `file:line` · claim**            (template.py, blocking)
#   **MEDIUM · `file:line`** -- claim            (template.py, non-blocking / scope)
#   - **HIGH** `file:line` -- claim              (review.py, legacy)
_FINDING_LINE_RE = re.compile(r"^-?\s*\*\*(CRITICAL|HIGH|MEDIUM|LOW)\b(?P<rest>.*)$")
_LOCATION_RE = re.compile(r"`([^`]+)`")
_LEADING_SEPARATORS = " \t·:–—-"  # space/tab, middot, colon, en/em dash, hyphen


class FeedbackError(RuntimeError):
    """The rejections file is unreadable or malformed."""


@dataclass(frozen=True)
class Rejection:
    """A finding a maintainer has already told the bot was wrong."""

    file: str
    claim_fingerprint: str
    reason: str
    pr_number: int
    rejected_at: str


def claim_fingerprint(text: str) -> str:
    """Stable, order-insensitive fingerprint of a finding's claim text.

    Two claims describing the same complaint in slightly different words should
    collapse to the same fingerprint, so a rejected finding doesn't resurface
    just because the model rephrased it on the next sweep. Algorithm: lowercase,
    strip punctuation, drop a small stopword set, sort the remaining tokens,
    sha256 and take the first 16 hex chars. Sorting makes it insensitive to word
    order; lowercasing and punctuation-stripping make it insensitive to
    formatting.

    Tradeoff, chosen deliberately tight: this is exact bag-of-words matching,
    not semantic matching. There is no stemming, no synonym handling, and no
    fuzzy distance. A rejected claim reworded with materially different
    vocabulary (e.g. "off-by-one error" vs. "loop runs one iteration too many")
    will NOT match and can resurface -- the bot may repeat itself. That failure
    mode is accepted on purpose: the alternative, a looser match, risks the
    opposite and much worse failure -- two genuinely different claims collapsing
    onto one fingerprint and a real, new bug being silently suppressed because it
    happens to share vocabulary with something rejected once before. Wrongly
    suppressing a real bug is worse than repeating a rejected finding, so every
    choice here (small stopword list, no stemming, exact token match) errs
    toward under-matching rather than over-matching.
    """
    normalized = _PUNCTUATION_RE.sub(" ", text.lower())
    tokens = [token for token in normalized.split() if token not in _STOPWORDS]
    canonical = " ".join(sorted(tokens))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest[:_FINGERPRINT_HEX_LENGTH]


@dataclass(frozen=True)
class _Comment:
    id: int
    login: str
    body: str
    created_at: str


def fetch_rejections(
    repo_slug: str,
    pr_number: int,
    bot_login: str,
    runner: GhRunner = default_runner,
    dismissal_phrases: Sequence[str] = DEFAULT_DISMISSAL_PHRASES,
    now: datetime | None = None,
) -> tuple[Rejection, ...]:
    """Read the bot's own comments on a PR and turn rejection signals into `Rejection`s.

    A bot comment counts as rejected when either:
      - it carries a "-1" reaction, or
      - a LATER comment from someone other than `bot_login` contains one of
        `dismissal_phrases` (case-insensitive substring match).

    The bot's own later replies are never read as a dismissal, and reactions on
    non-bot comments are never inspected -- only the bot's own comments are ever
    candidates for either check. Findings are extracted from a rejected
    comment's body by `_extract_finding_lines`, which is tolerant of anything
    that doesn't look like a finding line rather than raising.

    ANY `gh` trouble anywhere in this function -- a failed call, unparseable
    JSON, an unexpected response shape -- returns `()`. Reading feedback is an
    enrichment, never a hard dependency of a sweep, and a partial read is more
    dangerous than no read: it could look like a maintainer's rejection was
    honored when only some of it was.
    """
    try:
        comments = _fetch_comments(repo_slug, pr_number, runner)
        bot_comments = tuple(c for c in comments if c.login == bot_login)
        other_comments = tuple(c for c in comments if c.login != bot_login)

        rejected_at = (now or datetime.now(timezone.utc)).isoformat()
        collected: dict[tuple[str, str], Rejection] = {}
        for comment in bot_comments:
            reason = _rejection_reason(repo_slug, comment, other_comments, dismissal_phrases, runner)
            if reason is None:
                continue
            for file, claim in _extract_finding_lines(comment.body):
                fingerprint = claim_fingerprint(claim)
                collected[(file, fingerprint)] = Rejection(
                    file=file,
                    claim_fingerprint=fingerprint,
                    reason=reason,
                    pr_number=pr_number,
                    rejected_at=rejected_at,
                )
        return tuple(collected.values())
    except (
        GitHubError,
        json.JSONDecodeError,
        TypeError,
        KeyError,
        ValueError,
        AttributeError,
        IndexError,
    ):
        return ()


def _fetch_comments(repo_slug: str, pr_number: int, runner: GhRunner) -> tuple[_Comment, ...]:
    raw = runner(["api", f"repos/{repo_slug}/issues/{pr_number}/comments"], None)
    if not raw.strip():
        return ()
    payload = json.loads(raw)
    if not isinstance(payload, list):
        return ()
    return tuple(_parse_comment(item) for item in payload if isinstance(item, dict))


def _parse_comment(item: dict) -> _Comment:
    user = item.get("user")
    login = str(user.get("login", "")) if isinstance(user, dict) else ""
    return _Comment(
        id=int(item.get("id", 0)),
        login=login,
        body=str(item.get("body", "")),
        created_at=str(item.get("created_at", "")),
    )


def _rejection_reason(
    repo_slug: str,
    comment: _Comment,
    other_comments: tuple[_Comment, ...],
    dismissal_phrases: Sequence[str],
    runner: GhRunner,
) -> str | None:
    """Why `comment` counts as rejected, or None if it doesn't."""
    if _has_thumbs_down(repo_slug, comment.id, runner):
        return "thumbs-down reaction"

    for other in other_comments:
        if other.created_at <= comment.created_at:
            continue  # at or before the finding -- can't be a reply dismissing it
        phrase = _matched_dismissal_phrase(other.body, dismissal_phrases)
        if phrase:
            return f"dismissed: {phrase!r}"
    return None


def _has_thumbs_down(repo_slug: str, comment_id: int, runner: GhRunner) -> bool:
    raw = runner(["api", f"repos/{repo_slug}/issues/comments/{comment_id}/reactions"], None)
    if not raw.strip():
        return False
    reactions = json.loads(raw)
    if not isinstance(reactions, list):
        return False
    return any(isinstance(r, dict) and r.get("content") == "-1" for r in reactions)


def _matched_dismissal_phrase(body: str, phrases: Sequence[str]) -> str | None:
    lowered = body.lower()
    for phrase in phrases:
        if phrase.lower() in lowered:
            return phrase
    return None


def _extract_finding_lines(body: str) -> tuple[tuple[str, str], ...]:
    """`(file, claim)` for every finding line in a bot comment body.

    Tolerant by construction -- see `_parse_finding_line`. Comment bodies are
    untrusted external content (anyone who can comment on the PR can shape one),
    so a single malformed or adversarial line is skipped rather than allowed to
    raise and abort the whole read.
    """
    pairs: list[tuple[str, str]] = []
    for raw_line in body.splitlines():
        try:
            pair = _parse_finding_line(raw_line.strip())
        except Exception:
            pair = None
        if pair is not None:
            pairs.append(pair)
    return tuple(dict.fromkeys(pairs))


def _parse_finding_line(line: str) -> tuple[str, str] | None:
    """Pull `(file, claim)` from one rendered finding line, or None.

    Anything that isn't a recognized finding-line shape -- prose, a broken-caller
    block whose claim lives on the next line, a line with no `file` token in
    backticks -- yields None. This function never guesses.
    """
    match = _FINDING_LINE_RE.match(line)
    if not match:
        return None

    rest = match.group("rest")
    location_match = _LOCATION_RE.search(rest)
    if not location_match:
        return None

    file = location_match.group(1).split(":", 1)[0].strip()
    if not file:
        return None

    claim = rest[location_match.end() :].replace("**", " ").strip(_LEADING_SEPARATORS)
    if not claim:
        return None
    return (file, claim)


def load_rejections(path: Path | str = DEFAULT_REJECTIONS_PATH) -> tuple[Rejection, ...]:
    """Read rejections from disk. A missing file is a cold start, not an error."""
    rejections_path = Path(path)
    if not rejections_path.is_file():
        return ()

    try:
        raw = json.loads(rejections_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise FeedbackError(f"Could not read rejections file {rejections_path}: {exc}") from exc

    if not isinstance(raw, list):
        raise FeedbackError(f"Rejections file {rejections_path} must contain a JSON array")

    try:
        return tuple(_parse_rejection(item) for item in raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise FeedbackError(
            f"Rejections file {rejections_path} contains a malformed entry: {exc}"
        ) from exc


def _parse_rejection(item: dict) -> Rejection:
    return Rejection(
        file=str(item["file"]),
        claim_fingerprint=str(item["claim_fingerprint"]),
        reason=str(item["reason"]),
        pr_number=int(item["pr_number"]),
        rejected_at=str(item["rejected_at"]),
    )


def save_rejections(
    rejections: Sequence[Rejection], path: Path | str = DEFAULT_REJECTIONS_PATH
) -> None:
    """Write atomically, merged with whatever is already on disk.

    Merge key is `(file, claim_fingerprint)`; the newest `rejected_at` wins on
    conflict. This makes it safe to call every sweep without losing a rejection
    an earlier sweep recorded but this one didn't happen to re-observe (e.g. a
    dismissal comment that has since scrolled past the API's default page size).
    A malformed existing file raises rather than being silently overwritten --
    see `load_rejections`.
    """
    rejections_path = Path(path)
    existing = load_rejections(rejections_path)
    merged = _merge_rejections(existing, rejections)

    rejections_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [_rejection_to_dict(r) for r in merged]

    temp_path = rejections_path.with_suffix(rejections_path.suffix + ".tmp")
    try:
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(rejections_path)
    except OSError as exc:
        raise FeedbackError(f"Could not write rejections file {rejections_path}: {exc}") from exc


def _merge_rejections(
    existing: Sequence[Rejection], incoming: Sequence[Rejection]
) -> tuple[Rejection, ...]:
    """Union by `(file, claim_fingerprint)`; on conflict the newest `rejected_at` wins.

    Timestamps are compared lexicographically, which is chronologically correct
    for the ISO-8601 UTC strings this module produces (`datetime.isoformat()` on
    a timezone-aware `datetime`) but is not a general-purpose datetime compare.
    """
    merged: dict[tuple[str, str], Rejection] = {}
    for rejection in (*existing, *incoming):
        key = (rejection.file, rejection.claim_fingerprint)
        current = merged.get(key)
        if current is None or rejection.rejected_at >= current.rejected_at:
            merged[key] = rejection
    return tuple(merged.values())


def _rejection_to_dict(rejection: Rejection) -> dict:
    return {
        "file": rejection.file,
        "claim_fingerprint": rejection.claim_fingerprint,
        "reason": rejection.reason,
        "pr_number": rejection.pr_number,
        "rejected_at": rejection.rejected_at,
    }


def filter_rejected(
    findings: Sequence[Finding], rejections: Sequence[Rejection]
) -> tuple[tuple[Finding, ...], tuple[Finding, ...]]:
    """Split `findings` into `(kept, suppressed)` by `(file, claim_fingerprint)`.

    Suppressed findings are returned, never dropped silently -- callers are
    expected to record them (e.g. in the local audit sink) rather than let them
    vanish with no trace that a finding was held back.
    """
    rejected_keys = {(r.file, r.claim_fingerprint) for r in rejections}
    kept: list[Finding] = []
    suppressed: list[Finding] = []
    for finding in findings:
        key = (finding.file, claim_fingerprint(finding.claim))
        (suppressed if key in rejected_keys else kept).append(finding)
    return tuple(kept), tuple(suppressed)


def render_rejection_guidance(rejections: Sequence[Rejection], limit: int = 20) -> str:
    """Markdown block for the next review prompt: don't re-report these.

    Newest first, capped at `limit`, empty string when there is nothing to say
    so callers can splice this in unconditionally. `Rejection` does not retain
    the original claim text (only its fingerprint, which is one-way), so this
    can point at the file and the reason a prior finding there was rejected but
    cannot quote the finding itself -- the exact-match suppression in
    `filter_rejected` is the hard guarantee; this is a softer hint to reduce how
    often the model produces the same finding in the first place.
    """
    if not rejections:
        return ""

    newest_first = sorted(rejections, key=lambda r: r.rejected_at, reverse=True)
    shown = newest_first[:limit]
    lines = [
        "Maintainers previously rejected these findings. "
        "Do not re-report unless materially different.",
        "",
    ]
    lines.extend(
        f"- `{r.file}` -- {r.reason} (PR #{r.pr_number}, fingerprint `{r.claim_fingerprint}`)"
        for r in shown
    )
    return "\n".join(lines)
