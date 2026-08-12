"""Score the demo repo's reviews against a known answer key.

    python scripts/score_demo.py [--reviews-dir reviews]

The demo repo (RonKat2008/prime-agent-review-demo) plants specific, known defects
across a handful of PRs so the review agent's output can be checked against ground
truth instead of read by eye. This turns a quality regression into a failing
command rather than a vibe.

Each PR in ``ANSWER_KEY`` encodes its expectation as data -- count thresholds on
findings, plus an optional word that must appear in the rendered body -- so a new
PR can be added to the key without touching the evaluation logic in
``_failures``. PR 6 is the deliberate "silence" case: a clean PR that must draw
*no* findings at all. A reviewer that flags everything scores no better here than
one that flags nothing, which is why it gets its own callout in the report.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

DEFAULT_REVIEWS_DIR = Path("reviews")

_FRONT_MATTER_START = "<!--"
_FRONT_MATTER_END = "-->"


@dataclass(frozen=True)
class KeyEntry:
    """One row of the demo's known answer key.

    Every field beyond ``pr``/``title`` is an optional constraint: leaving it
    ``None`` means that axis is not checked for this PR. Extend the key by adding
    entries or fields, not branches -- ``_failures`` interprets whatever
    constraints are set on a given entry.
    """

    pr: int
    title: str
    min_introduces: int | None = None
    exact_introduces: int | None = None
    min_fixes: int | None = None
    exact_fixes: int | None = None
    body_contains: str | None = None
    is_silence_case: bool = False


ANSWER_KEY: tuple[KeyEntry, ...] = (
    KeyEntry(
        pr=3,
        title="Speed up total_price",
        min_introduces=2,
        body_contains="HIGH",
    ),
    KeyEntry(
        pr=4,
        title="Guard against a missing order",
        min_fixes=1,
        exact_introduces=0,
    ),
    KeyEntry(
        pr=5,
        title="Add customer search by name",
        min_introduces=1,
        body_contains="CRITICAL",
    ),
    KeyEntry(
        pr=6,
        title="Clarify parameter naming",
        exact_introduces=0,
        exact_fixes=0,
        is_silence_case=True,
    ),
)


@dataclass(frozen=True)
class ReviewRecord:
    """One successfully parsed review file."""

    pr: int
    reviewed_at: datetime
    introduces: int
    fixes: int
    body: str
    path: Path


@dataclass(frozen=True)
class RowResult:
    """The scored outcome for one answer-key entry, ready to print."""

    pr: int
    title: str
    expectation: str
    actual: str
    passed: bool
    reason: str
    is_silence_case: bool = False


def parse_review_file(path: Path) -> ReviewRecord | None:
    """Parse one review file's front matter. Returns None when it is malformed."""
    text = path.read_text(encoding="utf-8")
    start = text.find(_FRONT_MATTER_START)
    end = text.find(_FRONT_MATTER_END, start + len(_FRONT_MATTER_START))
    if start == -1 or end == -1:
        return None

    try:
        payload = json.loads(text[start + len(_FRONT_MATTER_START) : end])
        pr = int(payload["pr"])
        reviewed_at = datetime.fromisoformat(str(payload["reviewed_at"]))
        introduces = int(payload["introduces"])
        fixes = int(payload["fixes"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None

    body = text[end + len(_FRONT_MATTER_END) :].lstrip("\n")
    return ReviewRecord(
        pr=pr, reviewed_at=reviewed_at, introduces=introduces, fixes=fixes,
        body=body, path=path,
    )


def scan_reviews(reviews_dir: Path) -> tuple[dict[int, list[ReviewRecord]], int]:
    """Parse every review file, grouped by PR number. Returns (by_pr, malformed_count).

    A malformed file is skipped, not fatal: it is counted so the caller can report
    it, and the PR it belongs to simply falls back to whichever other valid files
    exist for it.
    """
    by_pr: dict[int, list[ReviewRecord]] = {}
    malformed = 0
    if not reviews_dir.is_dir():
        return by_pr, malformed

    for path in sorted(reviews_dir.glob("*.md")):
        record = parse_review_file(path)
        if record is None:
            malformed += 1
            continue
        by_pr.setdefault(record.pr, []).append(record)
    return by_pr, malformed


def _latest(records: Sequence[ReviewRecord]) -> ReviewRecord:
    """The same PR is re-reviewed at new SHAs; only the newest verdict counts."""
    return max(records, key=lambda r: r.reviewed_at)


def _expectation_text(entry: KeyEntry) -> str:
    """Render an entry's constraints as one human-readable line for the table."""
    parts: list[str] = []
    if entry.min_introduces is not None:
        parts.append(f"introduces>={entry.min_introduces}")
    if entry.exact_introduces is not None:
        parts.append(f"introduces=={entry.exact_introduces}")
    if entry.min_fixes is not None:
        parts.append(f"fixes>={entry.min_fixes}")
    if entry.exact_fixes is not None:
        parts.append(f"fixes=={entry.exact_fixes}")
    if entry.body_contains is not None:
        parts.append(f"body mentions {entry.body_contains!r}")
    return " AND ".join(parts) if parts else "no constraints"


def _failures(entry: KeyEntry, record: ReviewRecord) -> list[str]:
    """Every constraint on ``entry`` that ``record`` violates, if any."""
    failures: list[str] = []
    if entry.min_introduces is not None and record.introduces < entry.min_introduces:
        failures.append(f"introduces {record.introduces} < {entry.min_introduces}")
    if entry.exact_introduces is not None and record.introduces != entry.exact_introduces:
        failures.append(f"introduces {record.introduces} != {entry.exact_introduces}")
    if entry.min_fixes is not None and record.fixes < entry.min_fixes:
        failures.append(f"fixes {record.fixes} < {entry.min_fixes}")
    if entry.exact_fixes is not None and record.fixes != entry.exact_fixes:
        failures.append(f"fixes {record.fixes} != {entry.exact_fixes}")
    if entry.body_contains is not None and entry.body_contains not in record.body:
        failures.append(f"body does not mention {entry.body_contains!r}")
    return failures


def evaluate_entry(
    entry: KeyEntry, records_by_pr: dict[int, list[ReviewRecord]]
) -> RowResult:
    """Score one answer-key entry against the newest matching review, if any."""
    expectation = _expectation_text(entry)
    records = records_by_pr.get(entry.pr, [])
    if not records:
        return RowResult(
            pr=entry.pr,
            title=entry.title,
            expectation=expectation,
            actual="no review found",
            passed=False,
            reason="no review found",
            is_silence_case=entry.is_silence_case,
        )

    record = _latest(records)
    failures = _failures(entry, record)
    return RowResult(
        pr=entry.pr,
        title=entry.title,
        expectation=expectation,
        actual=f"introduces={record.introduces}, fixes={record.fixes}",
        passed=not failures,
        reason="ok" if not failures else "; ".join(failures),
        is_silence_case=entry.is_silence_case,
    )


def _format_row(row: RowResult) -> str:
    result = "PASS" if row.passed else "FAIL"
    line = f"PR#{row.pr:<3} | {row.expectation:<46} | {row.actual:<28} | {result}"
    if not row.passed:
        line += f"\n         reason: {row.reason}"
    return line


def render_table(rows: Sequence[RowResult]) -> str:
    """A compact PR / expectation / actual / PASS-FAIL table."""
    header = f"{'PR':<6} | {'EXPECTATION':<46} | {'ACTUAL':<28} | RESULT"
    lines = [header, "-" * len(header)]
    lines.extend(_format_row(row) for row in rows)
    return "\n".join(lines)


def render_silence_note(rows: Sequence[RowResult]) -> str | None:
    """Call out the silence case(s) explicitly: over-flagging is the worse failure."""
    silence_rows = [row for row in rows if row.is_silence_case]
    if not silence_rows:
        return None

    labels = ", ".join(f"PR#{row.pr}" for row in silence_rows)
    if all(row.passed for row in silence_rows):
        return f"SILENCE CHECK ({labels}): PASS -- stayed quiet on a clean PR."
    return (
        f"SILENCE CHECK ({labels}): FAIL -- flagged a clean PR. A reviewer that "
        "flags everything is worse than one that flags nothing."
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score the demo repo's reviews against the known answer key."
    )
    parser.add_argument(
        "--reviews-dir",
        default=str(DEFAULT_REVIEWS_DIR),
        help="Directory containing PR-{number}-{sha8}.md review files.",
    )
    args = parser.parse_args(argv)

    records_by_pr, malformed = scan_reviews(Path(args.reviews_dir))
    rows = [evaluate_entry(entry, records_by_pr) for entry in ANSWER_KEY]

    print(render_table(rows))

    if malformed:
        print(f"\n{malformed} review file(s) skipped: malformed front matter.")

    note = render_silence_note(rows)
    if note:
        print(f"\n{note}")

    passed = sum(1 for row in rows if row.passed)
    print(f"\n{passed}/{len(rows)} passed")

    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
