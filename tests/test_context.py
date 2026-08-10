"""Repository context gathering: full files, call sites, sibling tests, conventions.

Every external call is injected (`gh` for contents, `git` for grep), so nothing here
touches the network, a token, or a real git repository. Sibling tests and convention
files are read straight off disk under a `tmp_path` repo root.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pytest

from prime_pr_review.context import (
    CallSite,
    ChangedFile,
    Convention,
    GitError,
    ReviewContext,
    SiblingTest,
    gather_context,
)
from prime_pr_review.github import GitHubError

from .conftest import FakeGh

REPO_SLUG = "acme/widget"
HEAD_SHA = "deadbeef00"

SIMPLE_DIFF = """diff --git a/src/app.py b/src/app.py
index 1111111..2222222 100644
--- a/src/app.py
+++ b/src/app.py
@@ -1,3 +1,4 @@
-def old_helper():
+def new_helper():
+class NewThing:
     pass
"""


def gh_content(text: str) -> str:
    """Encode file text the way `gh api ... --jq .content` would return it."""
    return base64.b64encode(text.encode()).decode() + "\n"


def is_contents_call(path: str) -> Callable[[Sequence[str]], bool]:
    def predicate(args: Sequence[str]) -> bool:
        return args[0] == "api" and f"contents/{path}" in " ".join(args)

    return predicate


@dataclass
class FakeGit:
    """A git runner keyed on the searched symbol. Records every call."""

    responses: dict[str, str] = field(default_factory=dict)
    raise_for: frozenset[str] = frozenset()
    calls: list[list[str]] = field(default_factory=list)

    def __call__(self, args: Sequence[str]) -> str:
        self.calls.append(list(args))
        symbol = args[-1]
        if symbol in self.raise_for:
            raise GitError(f"git grep failed for {symbol}")
        return self.responses.get(symbol, "")


def empty_gh() -> FakeGh:
    """A gh fake that answers every contents lookup with an empty (missing) file."""
    return FakeGh().on(lambda args: args[0] == "api", "")


# --- symbol extraction / call sites -----------------------------------------


def test_call_sites_are_searched_for_every_def_and_class_on_a_changed_line():
    gh = empty_gh()
    git = FakeGit()

    gather_context(REPO_SLUG, HEAD_SHA, SIMPLE_DIFF, Path("."), gh, git)

    searched = {call[-1] for call in git.calls}
    assert searched == {"old_helper", "new_helper", "NewThing"}


def test_context_lines_and_diff_headers_are_not_treated_as_symbols():
    diff = (
        "diff --git a/src/app.py b/src/app.py\n"
        "--- a/src/app.py\n"
        "+++ b/src/app.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def unrelated_context_line():\n"
        "-    return 1\n"
        "+    return 2\n"
    )
    gh = empty_gh()
    git = FakeGit()

    gather_context(REPO_SLUG, HEAD_SHA, diff, Path("."), gh, git)

    assert git.calls == []


def test_call_sites_exclude_hits_inside_files_already_in_the_diff():
    gh = empty_gh()
    git = FakeGit(
        responses={
            "new_helper": (
                "src/app.py:12:new_helper()\n"  # in-diff file — excluded
                "src/other.py:5:result = new_helper()\n"  # outside the diff — kept
            ),
            "old_helper": "",
            "NewThing": "",
        }
    )

    context = gather_context(REPO_SLUG, HEAD_SHA, SIMPLE_DIFF, Path("."), gh, git)

    assert [site.file for site in context.call_sites] == ["src/other.py"]
    assert context.call_sites[0].line == 5
    assert context.call_sites[0].symbol == "new_helper"


def test_grep_lines_that_do_not_parse_are_skipped_without_raising():
    gh = empty_gh()
    git = FakeGit(
        responses={
            "new_helper": "Binary file src/blob.bin matches\nsrc/other.py:7:new_helper()\n",
            "old_helper": "",
            "NewThing": "",
        }
    )

    context = gather_context(REPO_SLUG, HEAD_SHA, SIMPLE_DIFF, Path("."), gh, git)

    assert [s.file for s in context.call_sites] == ["src/other.py"]


def test_a_failing_grep_for_one_symbol_does_not_block_the_others():
    gh = empty_gh()
    git = FakeGit(
        responses={"new_helper": "src/other.py:3:new_helper()\n"},
        raise_for=frozenset({"old_helper"}),
    )

    context = gather_context(REPO_SLUG, HEAD_SHA, SIMPLE_DIFF, Path("."), gh, git)

    assert [s.symbol for s in context.call_sites] == ["new_helper"]


# --- full changed files -------------------------------------------------------


def test_full_content_of_changed_files_is_fetched_at_the_head_sha():
    gh = FakeGh().on(is_contents_call("src/app.py"), gh_content("print('hello')\n"))
    git = FakeGit()

    context = gather_context(REPO_SLUG, HEAD_SHA, SIMPLE_DIFF, Path("."), gh, git)

    assert context.changed_files == (
        ChangedFile(path="src/app.py", content="print('hello')\n"),
    )
    call_args = gh.calls[0][0]
    assert f"contents/src/app.py?ref={HEAD_SHA}" in " ".join(call_args)


def test_missing_full_file_content_is_handled_gracefully_not_as_an_error():
    def raising_runner(args, stdin=None):
        raise GitHubError("404 Not Found")

    git = FakeGit()

    context = gather_context(REPO_SLUG, HEAD_SHA, SIMPLE_DIFF, Path("."), raising_runner, git)

    assert context.changed_files == ()


def test_blank_content_response_is_treated_as_missing():
    gh = FakeGh().on(is_contents_call("src/app.py"), "\n")
    git = FakeGit()

    context = gather_context(REPO_SLUG, HEAD_SHA, SIMPLE_DIFF, Path("."), gh, git)

    assert context.changed_files == ()


def test_invalid_base64_content_is_treated_as_missing_not_a_crash():
    gh = FakeGh().on(is_contents_call("src/app.py"), "not-valid-base64!!!\n")
    git = FakeGit()

    context = gather_context(REPO_SLUG, HEAD_SHA, SIMPLE_DIFF, Path("."), gh, git)

    assert context.changed_files == ()


def test_changed_files_budget_truncates_on_a_file_boundary():
    diff = (
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -0,0 +1 @@\n+x\n"
        "diff --git a/b.py b/b.py\n--- a/b.py\n+++ b/b.py\n@@ -0,0 +1 @@\n+y\n"
    )
    gh = (
        FakeGh()
        .on(is_contents_call("a.py"), gh_content("a" * 30))
        .on(is_contents_call("b.py"), gh_content("b" * 30))
    )
    git = FakeGit()
    # 40% of max_bytes must fit the first file (30 bytes) but not both (60 bytes).
    max_bytes = int(35 / 0.40)

    context = gather_context(REPO_SLUG, HEAD_SHA, diff, Path("."), gh, git, max_bytes=max_bytes)

    assert [f.path for f in context.changed_files] == ["a.py"]
    assert any("changed files" in note for note in context.dropped)


# --- sibling tests -------------------------------------------------------------


def test_sibling_tests_are_discovered_by_naming_convention(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "widget.py").write_text("class Widget: ...\n")
    (tmp_path / "test_widget.py").write_text("def test_a(): ...\n")
    (tmp_path / "widget_test.py").write_text("def test_b(): ...\n")
    tests_dir = tmp_path / "tests" / "nested"
    tests_dir.mkdir(parents=True)
    (tests_dir / "check_widget_behaviour.py").write_text("def test_c(): ...\n")
    (tmp_path / "unrelated.py").write_text("# not a sibling\n")

    diff = (
        "diff --git a/src/widget.py b/src/widget.py\n"
        "--- a/src/widget.py\n+++ b/src/widget.py\n@@ -0,0 +1 @@\n+class Widget: ...\n"
    )
    gh = empty_gh()
    git = FakeGit()

    context = gather_context(REPO_SLUG, HEAD_SHA, diff, tmp_path, gh, git)

    found_paths = {t.path for t in context.sibling_tests}
    assert found_paths == {
        "test_widget.py",
        "widget_test.py",
        "tests/nested/check_widget_behaviour.py",
    }


def test_sibling_test_search_excludes_files_already_in_the_diff(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_widget.py").write_text("def test_a(): ...\n")

    diff = (
        "diff --git a/tests/test_widget.py b/tests/test_widget.py\n"
        "--- a/tests/test_widget.py\n+++ b/tests/test_widget.py\n@@ -0,0 +1 @@\n+def test_a(): ...\n"
    )
    gh = empty_gh()
    git = FakeGit()

    context = gather_context(REPO_SLUG, HEAD_SHA, diff, tmp_path, gh, git)

    assert context.sibling_tests == ()


def test_no_sibling_tests_found_is_not_an_error(tmp_path):
    diff = (
        "diff --git a/src/lonely.py b/src/lonely.py\n"
        "--- a/src/lonely.py\n+++ b/src/lonely.py\n@@ -0,0 +1 @@\n+x = 1\n"
    )
    gh = empty_gh()
    git = FakeGit()

    context = gather_context(REPO_SLUG, HEAD_SHA, diff, tmp_path, gh, git)

    assert context.sibling_tests == ()


def test_sibling_tests_budget_truncates_on_a_file_boundary(tmp_path):
    (tmp_path / "test_widget_one.py").write_text("A" * 30)
    (tmp_path / "test_widget_two.py").write_text("B" * 30)
    diff = (
        "diff --git a/widget.py b/widget.py\n"
        "--- a/widget.py\n+++ b/widget.py\n@@ -0,0 +1 @@\n+x = 1\n"
    )
    gh = empty_gh()
    git = FakeGit()
    max_bytes = int(35 / 0.20)  # sibling tests get a 20% share

    context = gather_context(REPO_SLUG, HEAD_SHA, diff, tmp_path, gh, git, max_bytes=max_bytes)

    assert len(context.sibling_tests) == 1
    assert any("sibling tests" in note for note in context.dropped)


# --- conventions ---------------------------------------------------------------


def test_reads_contributing_claude_and_agents_files_when_present(tmp_path):
    (tmp_path / "CONTRIBUTING.md").write_text("Run tests first.\n")
    (tmp_path / "CLAUDE.md").write_text("Follow the house style.\n")
    (tmp_path / "AGENTS.md").write_text("Agents behave.\n")
    gh = empty_gh()
    git = FakeGit()

    context = gather_context(REPO_SLUG, HEAD_SHA, "", tmp_path, gh, git)

    assert {c.path for c in context.conventions} == {
        "CONTRIBUTING.md",
        "CLAUDE.md",
        "AGENTS.md",
    }


def test_missing_convention_files_are_not_an_error(tmp_path):
    context = gather_context(REPO_SLUG, HEAD_SHA, "", tmp_path, empty_gh(), FakeGit())

    assert context.conventions == ()


def test_reads_standalone_ruff_config_when_present(tmp_path):
    (tmp_path / "ruff.toml").write_text("line-length = 100\n")

    context = gather_context(REPO_SLUG, HEAD_SHA, "", tmp_path, empty_gh(), FakeGit())

    assert context.conventions == (
        Convention(path="ruff.toml", content="line-length = 100\n"),
    )


def test_pyproject_is_included_only_when_it_configures_ruff(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "widget"\n')

    context = gather_context(REPO_SLUG, HEAD_SHA, "", tmp_path, empty_gh(), FakeGit())

    assert context.conventions == ()


def test_pyproject_is_included_when_it_has_a_ruff_section(tmp_path):
    content = '[project]\nname = "widget"\n\n[tool.ruff]\nline-length = 100\n'
    (tmp_path / "pyproject.toml").write_text(content)

    context = gather_context(REPO_SLUG, HEAD_SHA, "", tmp_path, empty_gh(), FakeGit())

    assert context.conventions == (Convention(path="pyproject.toml", content=content),)


def test_conventions_budget_truncates_on_a_file_boundary(tmp_path):
    (tmp_path / "AGENTS.md").write_text("A" * 30)
    (tmp_path / "CLAUDE.md").write_text("C" * 30)
    max_bytes = int(35 / 0.15)  # conventions get a 15% share

    context = gather_context(
        REPO_SLUG, HEAD_SHA, "", tmp_path, empty_gh(), FakeGit(), max_bytes=max_bytes
    )

    assert len(context.conventions) == 1
    assert any("conventions" in note for note in context.dropped)


# --- overall behaviour / render -------------------------------------------------


def test_gather_context_on_a_bare_repo_with_nothing_is_not_an_error(tmp_path):
    context = gather_context(REPO_SLUG, HEAD_SHA, "", tmp_path, empty_gh(), FakeGit())

    assert context == ReviewContext()
    assert context.is_empty is True


def test_is_empty_is_false_once_any_section_has_content(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("style guide\n")

    context = gather_context(REPO_SLUG, HEAD_SHA, "", tmp_path, empty_gh(), FakeGit())

    assert context.is_empty is False


def test_render_includes_a_header_for_every_section():
    context = ReviewContext()

    markdown = context.render()

    assert "## Repository context" in markdown
    assert "### Full changed files" in markdown
    assert "### Call sites outside the diff" in markdown
    assert "### Sibling tests" in markdown
    assert "### Conventions" in markdown


def test_render_reports_none_found_for_empty_sections():
    markdown = ReviewContext().render()

    assert markdown.count("_None found._") == 4


def test_render_shows_full_file_content_and_call_site_detail():
    context = ReviewContext(
        changed_files=(ChangedFile(path="src/app.py", content="print(1)\n"),),
        call_sites=(CallSite(symbol="run", file="src/other.py", line=9, text="run()"),),
    )

    markdown = context.render()

    assert "`src/app.py`" in markdown
    assert "print(1)" in markdown
    assert "`src/other.py:9`" in markdown
    assert "(`run`)" in markdown


def test_render_lists_dropped_sections_when_budget_forced_omissions():
    context = ReviewContext(dropped=("call sites: 3 of 5 omitted to fit the context budget",))

    markdown = context.render()

    assert "### Context omitted" in markdown
    assert "call sites: 3 of 5 omitted to fit the context budget" in markdown


def test_render_omits_the_dropped_section_entirely_when_nothing_was_dropped():
    markdown = ReviewContext().render()

    assert "Context omitted" not in markdown


def test_sibling_test_and_convention_dataclasses_are_frozen():
    convention = Convention(path="CLAUDE.md", content="x")
    with pytest.raises(AttributeError):
        convention.path = "other.md"  # type: ignore[misc]

    sibling = SiblingTest(path="test_x.py", content="x")
    with pytest.raises(AttributeError):
        sibling.content = "y"  # type: ignore[misc]
