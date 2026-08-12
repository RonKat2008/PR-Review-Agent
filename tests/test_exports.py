"""Unwired-export detector (P13): deterministic added-symbol extraction, git-grep
call-site partitioning, and the two render helpers, kept model-free throughout.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from prime_pr_review.context import GitError
from prime_pr_review.exports import (
    UnwiredExport,
    activity_note,
    find_unwired_exports,
    render,
)

REPO_ROOT = Path(".")

ADDED_FUNCTION_DIFF = """diff --git a/state.py b/state.py
index 1111111..2222222 100644
--- a/state.py
+++ b/state.py
@@ -594,3 +594,6 @@ def existing_helper():
     return True


+def write_capability_state(payload):
+    return payload
+
 def other_existing():
     pass
"""

MODIFIED_ONLY_DIFF = """diff --git a/state.py b/state.py
--- a/state.py
+++ b/state.py
@@ -1,3 +1,3 @@
-def total_price(items):
+def total_price(items, tax_rate):
     return sum(items)
"""

PRIVATE_ADDED_DIFF = """diff --git a/state.py b/state.py
--- a/state.py
+++ b/state.py
@@ -1,2 +1,4 @@
 def existing():
     pass
+def _internal_helper():
+    pass
"""

JS_ADDED_DIFF = """diff --git a/src/pricing.js b/src/pricing.js
--- a/src/pricing.js
+++ b/src/pricing.js
@@ -1,2 +1,5 @@
 function existingThing() {
   return 1;
 }
+function newHelper(x) {
+  return x;
+}
"""

CLASS_ADDED_DIFF = """diff --git a/models.py b/models.py
--- a/models.py
+++ b/models.py
@@ -1,2 +1,4 @@
 def existing():
     pass
+class NewWidget:
+    pass
"""

MULTI_FILE_DIFF = """diff --git a/b_module.py b/b_module.py
--- a/b_module.py
+++ b/b_module.py
@@ -1,2 +1,4 @@
 def existing_b():
     pass
+def zeta_new():
+    pass
diff --git a/a_module.py b/a_module.py
--- a/a_module.py
+++ b/a_module.py
@@ -1,2 +1,4 @@
 def existing_a():
     pass
+def alpha_new():
+    pass
"""


@dataclass
class FakeGit:
    """A git runner keyed on the searched symbol (the last arg). Records every call."""

    responses: dict[str, str] = field(default_factory=dict)
    raise_for: frozenset[str] = frozenset()
    calls: list[list[str]] = field(default_factory=list)

    def __call__(self, args: Sequence[str]) -> str:
        self.calls.append(list(args))
        symbol = args[-1]
        if symbol in self.raise_for:
            raise GitError(f"git grep failed for {symbol}")
        return self.responses.get(symbol, "")


# --- added-public detection ---------------------------------------------------


def test_added_public_function_with_no_callers_is_flagged():
    git = FakeGit()

    result = find_unwired_exports(ADDED_FUNCTION_DIFF, git, REPO_ROOT)

    assert len(result) == 1
    entry = result[0]
    assert entry.symbol == "write_capability_state"
    assert entry.file == "state.py"
    assert entry.definition_kind == "function"
    assert entry.test_references == 0


def test_modified_only_symbol_is_not_flagged():
    git = FakeGit()

    result = find_unwired_exports(MODIFIED_ONLY_DIFF, git, REPO_ROOT)

    assert result == ()
    assert git.calls == []  # no added symbols -> git is never even queried


def test_private_symbol_is_not_flagged_even_with_zero_callers():
    git = FakeGit()

    result = find_unwired_exports(PRIVATE_ADDED_DIFF, git, REPO_ROOT)

    assert result == ()
    assert git.calls == []


def test_added_class_is_reported_with_definition_kind_class():
    git = FakeGit()

    result = find_unwired_exports(CLASS_ADDED_DIFF, git, REPO_ROOT)

    assert len(result) == 1
    assert result[0].symbol == "NewWidget"
    assert result[0].definition_kind == "class"


def test_non_python_added_function_is_detected_via_the_fallback_path():
    git = FakeGit()

    result = find_unwired_exports(JS_ADDED_DIFF, git, REPO_ROOT)

    assert len(result) == 1
    entry = result[0]
    assert entry.symbol == "newHelper"
    assert entry.file == "src/pricing.js"
    assert entry.definition_kind == "function"


def test_zero_added_symbols_returns_empty_tuple_without_touching_git():
    git = FakeGit()

    result = find_unwired_exports("", git, REPO_ROOT)

    assert result == ()
    assert git.calls == []


# --- git-grep partitioning ------------------------------------------------------


def test_production_caller_anywhere_excludes_the_symbol():
    git = FakeGit(
        responses={"write_capability_state": "shop/wiring.py:10:write_capability_state(x)\n"}
    )

    result = find_unwired_exports(ADDED_FUNCTION_DIFF, git, REPO_ROOT)

    assert result == ()


def test_test_only_references_are_counted_and_the_symbol_is_still_flagged():
    git = FakeGit(
        responses={
            "write_capability_state": (
                "tests/test_state.py:5:write_capability_state(1)\n"
                "tests/test_state.py:9:write_capability_state(2)\n"
            )
        }
    )

    result = find_unwired_exports(ADDED_FUNCTION_DIFF, git, REPO_ROOT)

    assert len(result) == 1
    assert result[0].test_references == 2


def test_hits_inside_the_diffs_own_files_are_ignored_for_the_caller_count():
    git = FakeGit(
        responses={
            "write_capability_state": "state.py:596:def write_capability_state(payload):\n"
        }
    )

    result = find_unwired_exports(ADDED_FUNCTION_DIFF, git, REPO_ROOT)

    assert len(result) == 1
    assert result[0].test_references == 0
    assert result[0].line == 596


def test_zero_grep_hits_at_all_is_flagged_with_an_unknown_line():
    git = FakeGit()  # no responses configured -> empty grep output

    result = find_unwired_exports(ADDED_FUNCTION_DIFF, git, REPO_ROOT)

    assert len(result) == 1
    assert result[0].line is None
    assert result[0].test_references == 0


def test_malformed_grep_lines_are_skipped_without_raising():
    git = FakeGit(
        responses={
            "write_capability_state": (
                "Binary file state.bin matches\n"
                "tests/test_state.py:3:write_capability_state()\n"
            )
        }
    )

    result = find_unwired_exports(ADDED_FUNCTION_DIFF, git, REPO_ROOT)

    assert len(result) == 1
    assert result[0].test_references == 1


def test_custom_test_globs_override_the_default_partition():
    git = FakeGit(
        responses={"write_capability_state": "spec/state_spec.py:4:write_capability_state()\n"}
    )

    default_result = find_unwired_exports(ADDED_FUNCTION_DIFF, git, REPO_ROOT)
    custom_result = find_unwired_exports(
        ADDED_FUNCTION_DIFF, git, REPO_ROOT, test_globs=("spec/**",)
    )

    assert default_result == ()  # "spec/**" isn't a default test glob -> looks like production
    assert len(custom_result) == 1
    assert custom_result[0].test_references == 1


# --- git failure -----------------------------------------------------------------


def test_a_failing_git_grep_returns_an_empty_tuple_not_partial_results():
    git = FakeGit(raise_for=frozenset({"write_capability_state"}))

    result = find_unwired_exports(ADDED_FUNCTION_DIFF, git, REPO_ROOT)

    assert result == ()


# --- ordering ----------------------------------------------------------------------


def test_multiple_unwired_exports_are_sorted_by_file_then_symbol():
    git = FakeGit()

    result = find_unwired_exports(MULTI_FILE_DIFF, git, REPO_ROOT)

    assert [(u.file, u.symbol) for u in result] == [
        ("a_module.py", "alpha_new"),
        ("b_module.py", "zeta_new"),
    ]


# --- render --------------------------------------------------------------------------


def test_render_of_empty_tuple_is_empty_string():
    assert render(()) == ""


def test_render_includes_file_line_symbol_kind_test_count_and_guidance():
    item = UnwiredExport(
        symbol="write_capability_state",
        file="state.py",
        line=596,
        test_references=2,
        definition_kind="function",
    )

    text = render((item,))

    assert "## Newly added public symbols with no production callers" in text
    assert "state.py:596" in text
    assert "write_capability_state" in text
    assert "function" in text
    assert "2 test reference(s)" in text
    assert "PR" in text  # cross-check-against-intent guidance for the model


def test_render_uses_bare_file_when_the_line_is_unknown():
    item = UnwiredExport(
        symbol="foo", file="bar.py", line=None, test_references=0, definition_kind="function"
    )

    text = render((item,))

    assert "`bar.py`" in text
    assert "bar.py:" not in text


# --- activity_note -----------------------------------------------------------------


def test_activity_note_of_empty_tuple_is_empty_string():
    assert activity_note(()) == ""


def test_activity_note_counts_unwired_exports():
    items = (
        UnwiredExport(symbol="a", file="a.py", line=1, test_references=0, definition_kind="function"),
        UnwiredExport(symbol="b", file="b.py", line=2, test_references=1, definition_kind="class"),
    )

    assert activity_note(items) == "exports: 2 added public symbol(s) with zero production callers"


# --- dataclass is frozen -------------------------------------------------------------


def test_unwired_export_is_frozen():
    item = UnwiredExport(
        symbol="x", file="a.py", line=1, test_references=0, definition_kind="function"
    )
    with pytest.raises(AttributeError):
        item.symbol = "y"  # type: ignore[misc]
