"""Diff splitting, noise filtering, and the size guard."""

from __future__ import annotations

from prime_pr_review.diffs import filter_diff, split_by_file

from .conftest import SAMPLE_DIFF


def test_splits_diff_into_one_chunk_per_file():
    files = split_by_file(SAMPLE_DIFF)

    assert [f.path for f in files] == ["src/app.py", "uv.lock"]


def test_split_returns_empty_for_blank_diff():
    assert split_by_file("") == ()
    assert split_by_file("   \n  ") == ()


def test_each_chunk_retains_its_own_header():
    files = split_by_file(SAMPLE_DIFF)

    assert files[0].body.startswith("diff --git a/src/app.py")
    assert "uv.lock" not in files[0].body


def test_filters_out_ignored_paths():
    result = filter_diff(SAMPLE_DIFF, ignore_paths=("**/*.lock",))

    assert result.included == ("src/app.py",)
    assert result.ignored == ("uv.lock",)
    assert "uv.lock" not in result.text


def test_double_star_pattern_matches_top_level_file():
    """`**/*.lock` must catch a root-level uv.lock, which bare fnmatch does not."""
    result = filter_diff(SAMPLE_DIFF, ignore_paths=("**/*.lock",))

    assert "uv.lock" in result.ignored


def test_keeps_everything_when_no_ignore_patterns():
    result = filter_diff(SAMPLE_DIFF, ignore_paths=())

    assert set(result.included) == {"src/app.py", "uv.lock"}
    assert result.ignored == ()


def test_truncates_on_a_file_boundary_when_over_budget():
    result = filter_diff(SAMPLE_DIFF, ignore_paths=(), max_bytes=10)

    assert result.truncated is True
    assert "truncated" in result.text


def test_truncation_never_splits_a_file():
    """A partial diff hunk would produce garbage findings, so files are all-or-nothing."""
    files = split_by_file(SAMPLE_DIFF)
    budget = files[0].size + 5  # room for the first file but not the second

    result = filter_diff(SAMPLE_DIFF, ignore_paths=(), max_bytes=budget)

    assert result.included == ("src/app.py",)
    assert result.truncated is True


def test_not_truncated_when_within_budget():
    result = filter_diff(SAMPLE_DIFF, ignore_paths=(), max_bytes=1_000_000)

    assert result.truncated is False
    assert "truncated" not in result.text


def test_is_empty_when_every_file_is_ignored():
    result = filter_diff(SAMPLE_DIFF, ignore_paths=("**/*.py", "**/*.lock"))

    assert result.is_empty is True


def test_handles_windows_style_paths_in_patterns():
    """Backslash separators are normalized, so patterns and reported paths agree."""
    diff = "diff --git a/dist\\bundle.js b/dist\\bundle.js\n+x\n"

    result = filter_diff(diff, ignore_paths=("**/dist/**",))

    assert result.ignored == ("dist/bundle.js",)


def test_malformed_header_falls_back_to_the_raw_remainder():
    diff = "diff --git nonsense-header\n+x\n"

    files = split_by_file(diff)

    assert files[0].path == "nonsense-header"
