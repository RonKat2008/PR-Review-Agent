"""Tests for the scored demo harness (Phase A4).

The demo repo's answer key is fixed: PRs 3-6 each plant a specific, known defect
shape. These tests write synthetic review files that copy the on-disk shape
`sinks.write_local` produces (front-matter JSON inside an HTML comment, followed
by the rendered markdown body) so the harness can be exercised without a live
review run. The shape is copied by hand here rather than by calling into
`prime_pr_review.sinks`, so each test keeps full control over `reviewed_at` --
the field the "newest wins" behavior depends on.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts import score_demo

# --------------------------------------------------------------------------
# Synthetic review bodies -- short, realistic fragments of rendered markdown.
# --------------------------------------------------------------------------

BODY_WITH_HIGH = (
    "#### Potential bugs introduced (2)\n\n"
    "- **HIGH** `shop/orders.py:5` -- slicing drops the last item.\n"
    "- **MEDIUM** `shop/orders.py:12` -- broad except swallows errors.\n"
)
BODY_WITH_CRITICAL = (
    "#### Potential bugs introduced (1)\n\n"
    "- **CRITICAL** `shop/customers.py:15` -- unescaped input reaches SQL.\n"
)
BODY_FIX_ONLY = (
    "#### Bugs fixed (1)\n\n- Prevents a TypeError on a missing order.\n"
)
BODY_SILENT = (
    "#### Blast radius\n\nChecked **0** call site(s). **0** break.\n"
)
BODY_NO_SEVERITY_WORD = (
    "#### Potential bugs introduced (2)\n\n- issue one\n- issue two\n"
)

_BASELINE_AT = "2026-08-10T01:00:00+00:00"
_LATER_AT = "2026-08-10T02:00:00+00:00"
_EARLIER_AT = "2026-08-09T00:00:00+00:00"


def _write_review(
    reviews_dir: Path,
    *,
    pr: int,
    sha: str,
    reviewed_at: str,
    introduces: int,
    fixes: int,
    body: str,
) -> Path:
    """Write one synthetic review file matching write_local's on-disk shape."""
    reviews_dir.mkdir(parents=True, exist_ok=True)
    front_matter = json.dumps(
        {
            "pr": pr,
            "lane": "open",
            "head_sha": sha,
            "author": "someone",
            "url": f"https://github.com/acme/demo/pull/{pr}",
            "confidence": 0.95,
            "introduces": introduces,
            "fixes": fixes,
            "reviewed_at": reviewed_at,
        },
        indent=2,
    )
    path = reviews_dir / f"PR-{pr}-{sha}.md"
    path.write_text(f"<!--\n{front_matter}\n-->\n\n{body}\n", encoding="utf-8")
    return path


def _write_passing_set(reviews_dir: Path) -> None:
    """Write one review per answer-key PR, each satisfying its expectation."""
    _write_review(
        reviews_dir, pr=3, sha="aaaaaaaa", reviewed_at=_BASELINE_AT,
        introduces=2, fixes=0, body=BODY_WITH_HIGH,
    )
    _write_review(
        reviews_dir, pr=4, sha="bbbbbbbb", reviewed_at=_BASELINE_AT,
        introduces=0, fixes=1, body=BODY_FIX_ONLY,
    )
    _write_review(
        reviews_dir, pr=5, sha="cccccccc", reviewed_at=_BASELINE_AT,
        introduces=1, fixes=0, body=BODY_WITH_CRITICAL,
    )
    _write_review(
        reviews_dir, pr=6, sha="dddddddd", reviewed_at=_BASELINE_AT,
        introduces=0, fixes=0, body=BODY_SILENT,
    )


def _row_for(output: str, pr: int) -> str:
    """The single table line for a PR. Fails loudly if the row is missing."""
    marker = f"PR#{pr} "
    matches = [line for line in output.splitlines() if line.startswith(marker)]
    assert matches, f"no table row for PR#{pr} in output:\n{output}"
    return matches[0]


def _silence_line(output: str) -> str:
    matches = [line for line in output.splitlines() if line.startswith("SILENCE CHECK")]
    assert matches, f"no SILENCE CHECK line in output:\n{output}"
    return matches[0]


# --------------------------------------------------------------------------
# Core behaviors required by the spec
# --------------------------------------------------------------------------


def test_all_matching_expectations_exit_zero(tmp_path, capsys):
    # Arrange
    _write_passing_set(tmp_path)

    # Act
    exit_code = score_demo.main(["--reviews-dir", str(tmp_path)])
    output = capsys.readouterr().out

    # Assert
    assert exit_code == 0
    assert "4/4 passed" in output
    for pr in (3, 4, 5, 6):
        assert "PASS" in _row_for(output, pr)


def test_one_failing_expectation_returns_exit_one_and_names_the_pr(tmp_path, capsys):
    # Arrange: PR 3 only introduces 1 bug, but the key requires at least 2
    _write_passing_set(tmp_path)
    _write_review(
        tmp_path, pr=3, sha="ffffffff", reviewed_at=_LATER_AT,
        introduces=1, fixes=0, body=BODY_WITH_HIGH,
    )

    # Act
    exit_code = score_demo.main(["--reviews-dir", str(tmp_path)])
    output = capsys.readouterr().out

    # Assert
    assert exit_code == 1
    assert "FAIL" in _row_for(output, 3)
    assert "PASS" in _row_for(output, 4)
    assert "PASS" in _row_for(output, 5)
    assert "PASS" in _row_for(output, 6)


def test_missing_review_file_fails_with_reason(tmp_path, capsys):
    # Arrange: PR 6 never got a review at all
    _write_review(
        tmp_path, pr=3, sha="aaaaaaaa", reviewed_at=_BASELINE_AT,
        introduces=2, fixes=0, body=BODY_WITH_HIGH,
    )
    _write_review(
        tmp_path, pr=4, sha="bbbbbbbb", reviewed_at=_BASELINE_AT,
        introduces=0, fixes=1, body=BODY_FIX_ONLY,
    )
    _write_review(
        tmp_path, pr=5, sha="cccccccc", reviewed_at=_BASELINE_AT,
        introduces=1, fixes=0, body=BODY_WITH_CRITICAL,
    )

    # Act
    exit_code = score_demo.main(["--reviews-dir", str(tmp_path)])
    output = capsys.readouterr().out

    # Assert
    assert exit_code == 1
    assert "FAIL" in _row_for(output, 6)
    assert "no review found" in output


def test_newest_review_by_reviewed_at_wins(tmp_path, capsys):
    # Arrange: PR 4 re-reviewed at a new SHA; the older file would fail on its own
    _write_passing_set(tmp_path)
    _write_review(
        tmp_path, pr=4, sha="oldsha01", reviewed_at=_EARLIER_AT,
        introduces=3, fixes=0, body=BODY_NO_SEVERITY_WORD,
    )

    # Act
    exit_code = score_demo.main(["--reviews-dir", str(tmp_path)])
    output = capsys.readouterr().out

    # Assert: the newer (baseline) review wins, not the older failing one
    assert exit_code == 0
    row = _row_for(output, 4)
    assert "PASS" in row
    assert "introduces=0, fixes=1" in row


def test_malformed_front_matter_falls_back_to_older_valid_file(tmp_path, capsys):
    # Arrange: PR 5 has a valid older review and an unparseable newer file
    _write_passing_set(tmp_path)
    broken = tmp_path / "PR-5-newbadsh.md"
    broken.write_text("<!--\n{not valid json,,,\n-->\n\nbroken\n", encoding="utf-8")

    # Act
    exit_code = score_demo.main(["--reviews-dir", str(tmp_path)])
    output = capsys.readouterr().out

    # Assert: falls back to the still-valid file, and the skip is counted
    assert exit_code == 0
    assert "PASS" in _row_for(output, 5)
    assert "1 review file(s) skipped" in output


def test_missing_front_matter_markers_is_treated_as_malformed(tmp_path, capsys):
    # Arrange: PR 5 has a valid older review; the newer file has no HTML comment at all
    _write_passing_set(tmp_path)
    no_markers = tmp_path / "PR-5-nomarkers.md"
    no_markers.write_text("just some text with no front matter\n", encoding="utf-8")

    # Act
    exit_code = score_demo.main(["--reviews-dir", str(tmp_path)])
    output = capsys.readouterr().out

    # Assert: falls back to the still-valid file, and the skip is counted
    assert exit_code == 0
    assert "PASS" in _row_for(output, 5)
    assert "1 review file(s) skipped" in output


def test_insufficient_fixes_fails_even_when_introduces_is_exactly_right(tmp_path, capsys):
    # Arrange: PR 4 has zero introduces (correct) but zero fixes (below the minimum)
    _write_passing_set(tmp_path)
    _write_review(
        tmp_path, pr=4, sha="newsha05", reviewed_at=_LATER_AT,
        introduces=0, fixes=0, body=BODY_SILENT,
    )

    # Act
    exit_code = score_demo.main(["--reviews-dir", str(tmp_path)])
    output = capsys.readouterr().out

    # Assert
    assert exit_code == 1
    assert "FAIL" in _row_for(output, 4)


def test_silence_case_passes_when_both_introduces_and_fixes_are_zero(tmp_path, capsys):
    # Arrange
    _write_passing_set(tmp_path)

    # Act
    exit_code = score_demo.main(["--reviews-dir", str(tmp_path)])
    output = capsys.readouterr().out

    # Assert
    assert exit_code == 0
    assert "PASS" in _row_for(output, 6)
    assert "PASS" in _silence_line(output)


def test_silence_case_fails_when_introduces_nonzero_even_if_fixes_zero(tmp_path, capsys):
    # Arrange: PR 6 is supposed to be untouched, but this reviewer flagged a bug
    _write_passing_set(tmp_path)
    _write_review(
        tmp_path, pr=6, sha="newsha02", reviewed_at=_LATER_AT,
        introduces=2, fixes=0, body=BODY_WITH_HIGH,
    )

    # Act
    exit_code = score_demo.main(["--reviews-dir", str(tmp_path)])
    output = capsys.readouterr().out

    # Assert
    assert exit_code == 1
    assert "FAIL" in _row_for(output, 6)
    assert "FAIL" in _silence_line(output)


def test_silence_case_fails_when_fixes_nonzero_even_if_introduces_zero(tmp_path, capsys):
    # Arrange: a claimed fix is still a claim on a PR that should draw nothing
    _write_passing_set(tmp_path)
    _write_review(
        tmp_path, pr=6, sha="newsha03", reviewed_at=_LATER_AT,
        introduces=0, fixes=3, body=BODY_FIX_ONLY,
    )

    # Act
    exit_code = score_demo.main(["--reviews-dir", str(tmp_path)])
    output = capsys.readouterr().out

    # Assert
    assert exit_code == 1
    assert "FAIL" in _row_for(output, 6)
    assert "FAIL" in _silence_line(output)


# --------------------------------------------------------------------------
# Edge cases: empty/missing input, unrelated data, boundary conditions
# --------------------------------------------------------------------------


def test_empty_reviews_directory_fails_every_pr(tmp_path, capsys):
    # Arrange: tmp_path exists but has no review files at all

    # Act
    exit_code = score_demo.main(["--reviews-dir", str(tmp_path)])
    output = capsys.readouterr().out

    # Assert
    assert exit_code == 1
    assert "0/4 passed" in output
    for pr in (3, 4, 5, 6):
        assert "FAIL" in _row_for(output, pr)


def test_nonexistent_reviews_directory_does_not_crash(tmp_path, capsys):
    # Arrange
    missing = tmp_path / "does-not-exist"

    # Act
    exit_code = score_demo.main(["--reviews-dir", str(missing)])
    output = capsys.readouterr().out

    # Assert
    assert exit_code == 1
    assert "no review found" in output


def test_unrelated_review_files_do_not_affect_scoring(tmp_path, capsys):
    # Arrange: an extra review for a PR that is not part of the answer key
    _write_passing_set(tmp_path)
    _write_review(
        tmp_path, pr=999, sha="zzzzzzzz", reviewed_at=_BASELINE_AT,
        introduces=5, fixes=5, body=BODY_WITH_HIGH,
    )

    # Act
    exit_code = score_demo.main(["--reviews-dir", str(tmp_path)])
    output = capsys.readouterr().out

    # Assert
    assert exit_code == 0
    assert "4/4 passed" in output
    assert "skipped" not in output


def test_body_missing_required_word_fails_even_with_enough_findings(tmp_path, capsys):
    # Arrange: PR 3 has enough findings by count, but none say HIGH
    _write_passing_set(tmp_path)
    _write_review(
        tmp_path, pr=3, sha="newsha04", reviewed_at=_LATER_AT,
        introduces=2, fixes=0, body=BODY_NO_SEVERITY_WORD,
    )

    # Act
    exit_code = score_demo.main(["--reviews-dir", str(tmp_path)])
    output = capsys.readouterr().out

    # Assert
    assert exit_code == 1
    assert "FAIL" in _row_for(output, 3)


def test_scoring_does_not_modify_the_reviews_directory(tmp_path):
    # Arrange
    _write_passing_set(tmp_path)
    before = {p.name: p.read_text(encoding="utf-8") for p in tmp_path.iterdir()}

    # Act
    score_demo.main(["--reviews-dir", str(tmp_path)])

    # Assert: no side effects -- same files, same contents
    after = {p.name: p.read_text(encoding="utf-8") for p in tmp_path.iterdir()}
    assert after == before


def test_default_reviews_dir_constant_is_reviews():
    # Arrange / Act / Assert
    assert score_demo.DEFAULT_REVIEWS_DIR == Path("reviews")
