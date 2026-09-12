"""Citation validation: findings that cite a file/line which does not exist in
the diff (after path normalization), or whose line is confirmed beyond the
file's exact length at the PR head, are dropped deterministically, with an
auditable note per drop. Zero fabricated citations is a go-live gate — this is
the only place that enforces it."""

from __future__ import annotations

import base64
from collections.abc import Callable, Sequence

from prime_pr_review.citations import (
    head_line_counts,
    paths_needing_head_counts,
    validate_citations,
)
from prime_pr_review.github import GitHubError
from prime_pr_review.review import Finding, Severity

from .conftest import FakeGh

DIFF = (
    "diff --git a/src/app.py b/src/app.py\n"
    "index 1111111..2222222 100644\n"
    "--- a/src/app.py\n"
    "+++ b/src/app.py\n"
    "@@ -8,5 +8,5 @@\n"
    " context\n"
    " context\n"
    "-old\n"
    "+new\n"
    " context\n"
)

DELETED_DIFF = (
    "diff --git a/src/old.py b/src/old.py\n"
    "deleted file mode 100644\n"
    "index 1111111..0000000\n"
    "--- a/src/old.py\n"
    "+++ /dev/null\n"
    "@@ -1,3 +0,0 @@\n"
    "-a\n"
    "-b\n"
    "-c\n"
)

REPO_SLUG = "acme/widget"
HEAD_SHA = "deadbeef00"


def _finding(
    *,
    file: str = "src/app.py",
    line: int | None = 10,
    line_end: int | None = None,
    claim: str = "Off-by-one in the loop bound",
) -> Finding:
    return Finding(
        file=file,
        line=line,
        line_end=line_end,
        severity=Severity.HIGH,
        claim=claim,
        evidence="the loop runs one extra time",
    )


def _gh_content(text: str) -> str:
    """Encode file text the way `gh api ... --jq .content` would return it."""
    return base64.b64encode(text.encode()).decode() + "\n"


def _is_contents_call(path: str) -> Callable[[Sequence[str]], bool]:
    def predicate(args: Sequence[str]) -> bool:
        return args[0] == "api" and f"contents/{path}" in " ".join(args)

    return predicate


def _aggregate_note(notes: tuple[str, ...]) -> str:
    matches = [n for n in notes if n.startswith("citations: ") and "checked" in n]
    assert len(matches) == 1, notes
    return matches[0]


class TestFileNotInDiff:
    def test_drops_finding_citing_a_file_absent_from_the_diff(self) -> None:
        finding = _finding(file="src/other.py", line=3, claim="A fabricated bug")

        kept, notes = validate_citations((finding,), DIFF)

        assert kept == ()
        assert any("file not in diff" in n for n in notes)
        assert any("src/other.py:3" in n for n in notes)
        assert any("A fabricated bug" in n for n in notes)
        assert "citations: 1 checked, 1 dropped, 0 unverified" in notes


class TestPathNormalization:
    def test_strips_leading_dot_slash(self) -> None:
        finding = _finding(file="./src/app.py", line=10)

        kept, notes = validate_citations((finding,), DIFF)

        assert len(kept) == 1
        assert kept[0].file == "src/app.py"
        assert not any("file not in diff" in n for n in notes)

    def test_strips_leading_a_prefix(self) -> None:
        finding = _finding(file="a/src/app.py", line=10)

        kept, _notes = validate_citations((finding,), DIFF)

        assert len(kept) == 1
        assert kept[0].file == "src/app.py"

    def test_strips_leading_b_prefix(self) -> None:
        finding = _finding(file="b/src/app.py", line=10)

        kept, _notes = validate_citations((finding,), DIFF)

        assert len(kept) == 1
        assert kept[0].file == "src/app.py"

    def test_resolves_absolute_path_against_repo_root(self) -> None:
        finding = _finding(file="/repo/src/app.py", line=10)

        kept, _notes = validate_citations((finding,), DIFF, repo_root="/repo")

        assert len(kept) == 1
        assert kept[0].file == "src/app.py"

    def test_absolute_path_without_repo_root_is_dropped(self) -> None:
        finding = _finding(file="/repo/src/app.py", line=10)

        kept, notes = validate_citations((finding,), DIFF)

        assert kept == ()
        assert any("file not in diff" in n for n in notes)

    def test_absolute_path_outside_repo_root_is_dropped(self) -> None:
        finding = _finding(file="/other/src/app.py", line=10)

        kept, notes = validate_citations((finding,), DIFF, repo_root="/repo")

        assert kept == ()
        assert any("file not in diff" in n for n in notes)

    def test_prefix_that_still_does_not_match_is_dropped(self) -> None:
        finding = _finding(file="a/src/nonexistent.py", line=10)

        kept, notes = validate_citations((finding,), DIFF)

        assert kept == ()
        assert any("file not in diff" in n for n in notes)


class TestLineNone:
    def test_finding_with_no_line_is_kept_unchanged(self) -> None:
        finding = _finding(line=None)

        kept, notes = validate_citations((finding,), DIFF)

        assert kept == (finding,)
        assert notes == ("citations: 1 checked, 0 dropped, 0 unverified",)

    def test_finding_with_no_line_but_a_line_end_is_cleared_with_a_note(self) -> None:
        finding = _finding(line=None, line_end=5)

        kept, notes = validate_citations((finding,), DIFF)

        assert len(kept) == 1
        assert kept[0].line_end is None
        assert any("cleared line_end" in n for n in notes)


class TestNonPositiveLine:
    def test_zero_line_is_dropped(self) -> None:
        finding = _finding(line=0)

        kept, notes = validate_citations((finding,), DIFF)

        assert kept == ()
        assert any("line number must be positive" in n for n in notes)
        assert any("src/app.py:0" in n for n in notes)

    def test_negative_line_is_dropped(self) -> None:
        finding = _finding(line=-3)

        kept, notes = validate_citations((finding,), DIFF)

        assert kept == ()
        assert any("line number must be positive" in n for n in notes)


class TestDeletedFile:
    def test_finding_on_a_deleted_file_is_dropped(self) -> None:
        finding = _finding(file="src/old.py", line=2)

        kept, notes = validate_citations((finding,), DELETED_DIFF)

        assert kept == ()
        assert any("file deleted by this PR" in n for n in notes)
        assert any("src/old.py:2" in n for n in notes)


class TestOnAddedOrContextLine:
    def test_line_on_an_added_line_is_kept_with_no_per_finding_note(self) -> None:
        finding = _finding(line=10)  # "+new"

        kept, notes = validate_citations((finding,), DIFF)

        assert kept == (finding,)
        assert notes == ("citations: 1 checked, 0 dropped, 0 unverified",)

    def test_line_on_a_context_line_is_kept_with_no_per_finding_note(self) -> None:
        finding = _finding(line=8)  # first " context" line

        kept, notes = validate_citations((finding,), DIFF)

        assert kept == (finding,)
        assert notes == ("citations: 1 checked, 0 dropped, 0 unverified",)


class TestOutsideHunks:
    def test_outside_hunks_with_no_head_counts_is_kept_unverified(self) -> None:
        finding = _finding(line=14)

        kept, notes = validate_citations((finding,), DIFF, None)

        assert kept == (finding,)
        assert notes == ("citations: 1 checked, 0 dropped, 1 unverified",)

    def test_outside_hunks_within_head_count_is_kept_unverified(self) -> None:
        finding = _finding(line=14)

        kept, notes = validate_citations((finding,), DIFF, {"src/app.py": 16})

        assert kept == (finding,)
        assert notes == ("citations: 1 checked, 0 dropped, 1 unverified",)

    def test_outside_hunks_beyond_head_count_is_dropped(self) -> None:
        finding = _finding(line=20)

        kept, notes = validate_citations((finding,), DIFF, {"src/app.py": 16})

        assert kept == ()
        assert any("beyond file length" in n for n in notes)
        assert any("16 lines at head" in n for n in notes)
        assert "citations: 1 checked, 1 dropped, 0 unverified" in notes

    def test_outside_hunks_when_head_counts_lacks_the_file_is_kept_unverified(self) -> None:
        finding = _finding(line=14)

        kept, notes = validate_citations((finding,), DIFF, {"other/file.py": 100})

        assert kept == (finding,)
        assert notes == ("citations: 1 checked, 0 dropped, 1 unverified",)


class TestLineEnd:
    def test_line_end_before_line_is_cleared_with_a_note(self) -> None:
        finding = _finding(line=10, line_end=9)

        kept, notes = validate_citations((finding,), DIFF)

        assert len(kept) == 1
        assert kept[0].line_end is None
        assert any("cleared line_end" in n for n in notes)
        assert any("src/app.py:10" in n for n in notes)

    def test_line_end_at_or_after_line_is_untouched(self) -> None:
        finding = _finding(line=10, line_end=11)

        kept, notes = validate_citations((finding,), DIFF)

        assert kept == (finding,)
        assert notes == ("citations: 1 checked, 0 dropped, 0 unverified",)


class TestOrderPreserved:
    def test_order_preserved_when_a_middle_finding_is_dropped(self) -> None:
        first = _finding(line=8, claim="first")
        dropped = _finding(file="not/in/diff.py", line=1, claim="dropped")
        third = _finding(line=9, claim="third")

        kept, notes = validate_citations((first, dropped, third), DIFF)

        assert kept == (first, third)
        assert _aggregate_note(notes) == "citations: 3 checked, 1 dropped, 0 unverified"


class TestEmpty:
    def test_empty_findings_returns_empty_tuples_with_no_note(self) -> None:
        assert validate_citations((), DIFF) == ((), ())


class TestAggregateNote:
    def test_aggregate_note_appears_exactly_once_and_reflects_all_outcomes(self) -> None:
        verified = _finding(line=8, claim="verified")
        unverified = _finding(line=14, claim="unverified")
        beyond = _finding(line=20, claim="beyond")
        fabricated = _finding(file="nope.py", line=1, claim="fabricated")

        kept, notes = validate_citations(
            (verified, unverified, beyond, fabricated), DIFF, {"src/app.py": 16}
        )

        assert kept == (verified, unverified)
        assert _aggregate_note(notes) == "citations: 4 checked, 2 dropped, 1 unverified"


class TestPathsNeedingHeadCounts:
    def test_common_path_needs_no_head_counts(self) -> None:
        finding = _finding(line=10)  # on an added line, verifiable from the diff

        needed = paths_needing_head_counts((finding,), DIFF)

        assert needed == frozenset()

    def test_line_outside_hunks_needs_a_head_count(self) -> None:
        finding = _finding(line=14)

        needed = paths_needing_head_counts((finding,), DIFF)

        assert needed == frozenset({"src/app.py"})

    def test_fabricated_file_needs_no_head_count(self) -> None:
        finding = _finding(file="nope.py", line=1)

        needed = paths_needing_head_counts((finding,), DIFF)

        assert needed == frozenset()

    def test_non_positive_line_needs_no_head_count(self) -> None:
        finding = _finding(line=0)

        needed = paths_needing_head_counts((finding,), DIFF)

        assert needed == frozenset()

    def test_needed_path_is_the_canonical_form(self) -> None:
        finding = _finding(file="a/src/app.py", line=14)

        needed = paths_needing_head_counts((finding,), DIFF)

        assert needed == frozenset({"src/app.py"})


class TestHeadLineCounts:
    def test_fetches_and_counts_lines_with_trailing_newline(self) -> None:
        gh = FakeGh().on(_is_contents_call("src/app.py"), _gh_content("a\nb\nc\n"))

        counts = head_line_counts(REPO_SLUG, HEAD_SHA, ["src/app.py"], gh)

        assert counts == {"src/app.py": 3}

    def test_counts_lines_without_a_trailing_newline(self) -> None:
        gh = FakeGh().on(_is_contents_call("src/app.py"), _gh_content("a\nb\nc"))

        counts = head_line_counts(REPO_SLUG, HEAD_SHA, ["src/app.py"], gh)

        assert counts == {"src/app.py": 3}

    def test_single_blank_line_counts_as_one_line(self) -> None:
        gh = FakeGh().on(_is_contents_call("src/blank.py"), _gh_content("\n"))

        counts = head_line_counts(REPO_SLUG, HEAD_SHA, ["src/blank.py"], gh)

        assert counts == {"src/blank.py": 1}

    def test_omits_a_path_the_api_call_fails_for(self) -> None:
        def raise_error(args, stdin=None):
            raise GitHubError("boom")

        counts = head_line_counts(REPO_SLUG, HEAD_SHA, ["src/app.py"], raise_error)

        assert counts == {}

    def test_an_unexpected_exception_from_the_runner_is_also_omitted(self) -> None:
        def raise_unexpected(args: Sequence[str], stdin: str | None = None) -> str:
            raise RuntimeError("something else entirely")

        counts = head_line_counts(REPO_SLUG, HEAD_SHA, ["src/app.py"], raise_unexpected)

        assert counts == {}

    def test_counts_multiple_paths_independently_and_skips_failures(self) -> None:
        gh = (
            FakeGh()
            .on(_is_contents_call("a.py"), _gh_content("x\ny\n"))
            .on(_is_contents_call("b.py"), _gh_content("x\n"))
        )

        counts = head_line_counts(REPO_SLUG, HEAD_SHA, ["a.py", "b.py", "missing.py"], gh)

        assert counts == {"a.py": 2, "b.py": 1}
