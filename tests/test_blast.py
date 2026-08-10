"""Blast radius (P9): deterministic symbol extraction, git-grep call-site
discovery, and the model-judgment contract, kept strictly separate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from prime_pr_review.blast import (
    BlastRadiusError,
    CallSite,
    ChangedSymbol,
    analyze_blast_radius,
    build_blast_prompt,
    extract_changed_symbols,
)
from prime_pr_review.context import GitError
from prime_pr_review.review import BlastRadius, VerdictError

from .conftest import make_pr

REPO_ROOT = Path(".")
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "pr-review" / "prompts"

SIGNATURE_DIFF = """diff --git a/src/pricing.py b/src/pricing.py
index 1111111..2222222 100644
--- a/src/pricing.py
+++ b/src/pricing.py
@@ -1,3 +1,3 @@
-def total_price(items):
+def total_price(items, tax_rate):
     return sum(items)
"""

REMOVAL_DIFF = """diff --git a/src/pricing.py b/src/pricing.py
index 1111111..2222222 100644
--- a/src/pricing.py
+++ b/src/pricing.py
@@ -1,3 +1,1 @@
-def legacy_helper(x):
-    return x
+pass
"""

NO_REAL_CHANGE_DIFF = """diff --git a/src/pricing.py b/src/pricing.py
index 1111111..2222222 100644
--- a/src/pricing.py
+++ b/src/pricing.py
@@ -1,2 +1,2 @@
-def total_price(items):
+def total_price(items):
     return sum(items)
"""

JS_DIFF = """diff --git a/src/pricing.js b/src/pricing.js
index 1111111..2222222 100644
--- a/src/pricing.js
+++ b/src/pricing.js
@@ -1,3 +1,3 @@
-function totalPrice(items) {
+function totalPrice(items, taxRate) {
   return items;
"""

BLAST_RESPONSE_ONE_BREAK = (
    '{"symbol": "total_price", "kind": "signature_change", '
    '"change": "added required parameter tax_rate", '
    '"breaks": [{"file": "shop/invoice.py", "line": 44, "severity": "HIGH", '
    '"claim": "Calls total_price(items) with one argument; TypeError."}], '
    '"unbroken_callers": 2}'
)


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


class RecordingModel:
    """A model_fn stub that returns canned responses in order and records every prompt."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._responses.pop(0)


def failing_model(_: str) -> str:
    raise AssertionError("model_fn must not be called when there are no call sites")


# --- Step 1: ast detection on Python diffs ------------------------------------


def test_ast_detects_an_added_required_parameter():
    symbols = extract_changed_symbols(SIGNATURE_DIFF)

    assert len(symbols) == 1
    symbol = symbols[0]
    assert symbol.name == "total_price"
    assert symbol.file == "src/pricing.py"
    assert symbol.language == "python"
    assert "added required parameter(s) tax_rate" in symbol.change
    assert symbol.kind_hint == "signature_change"


def test_ast_detects_a_removed_required_parameter():
    diff = SIGNATURE_DIFF.replace(
        "-def total_price(items):\n+def total_price(items, tax_rate):",
        "-def total_price(items, tax_rate):\n+def total_price(items):",
    )

    symbols = extract_changed_symbols(diff)

    assert len(symbols) == 1
    assert "removed parameter(s) tax_rate" in symbols[0].change


def test_ast_detects_reordered_positional_parameters():
    diff = SIGNATURE_DIFF.replace(
        "-def total_price(items):\n+def total_price(items, tax_rate):",
        "-def total_price(items, tax_rate):\n+def total_price(tax_rate, items):",
    )

    symbols = extract_changed_symbols(diff)

    assert len(symbols) == 1
    assert "reordered positional parameters" in symbols[0].change


def test_ast_detects_a_changed_return_annotation():
    diff = """diff --git a/src/pricing.py b/src/pricing.py
--- a/src/pricing.py
+++ b/src/pricing.py
@@ -1,2 +1,2 @@
-def total_price(items) -> int:
+def total_price(items) -> float:
     return sum(items)
"""
    symbols = extract_changed_symbols(diff)

    assert len(symbols) == 1
    symbol = symbols[0]
    assert "return annotation changed from 'int' to 'float'" in symbol.change
    assert symbol.kind_hint == "return_shape_change"


def test_ast_detects_a_removed_function():
    symbols = extract_changed_symbols(REMOVAL_DIFF)

    assert len(symbols) == 1
    symbol = symbols[0]
    assert symbol.name == "legacy_helper"
    assert symbol.new_signature == ""
    assert symbol.kind_hint == "removal_or_rename"
    assert "removed entirely" in symbol.change


def test_multiline_signature_is_reconstructed_before_ast_parsing():
    diff = """diff --git a/src/pricing.py b/src/pricing.py
--- a/src/pricing.py
+++ b/src/pricing.py
@@ -1,4 +1,5 @@
 def total_price(
     items,
+    tax_rate,
 ):
     return sum(items)
"""
    symbols = extract_changed_symbols(diff)

    assert len(symbols) == 1
    assert "added required parameter(s) tax_rate" in symbols[0].change


def test_pure_reformatting_with_no_signature_change_is_not_reported():
    assert extract_changed_symbols(NO_REAL_CHANGE_DIFF) == ()


def test_constant_value_change_is_detected():
    diff = """diff --git a/src/config.py b/src/config.py
--- a/src/config.py
+++ b/src/config.py
@@ -1 +1 @@
-MAX_RETRIES = 3
+MAX_RETRIES = 5
"""
    symbols = extract_changed_symbols(diff)

    assert len(symbols) == 1
    symbol = symbols[0]
    assert symbol.name == "MAX_RETRIES"
    assert symbol.kind_hint == "constant_change"
    assert "changed from '3' to '5'" in symbol.change


def test_a_newly_added_function_with_no_prior_existence_is_not_reported():
    diff = """diff --git a/src/pricing.py b/src/pricing.py
--- a/src/pricing.py
+++ b/src/pricing.py
@@ -1,1 +1,3 @@
 def existing():
     pass
+def brand_new():
+    pass
"""
    symbols = extract_changed_symbols(diff)

    assert symbols == ()


def test_unparseable_signature_fragments_are_still_reported_not_silently_dropped():
    """Over-matching is intentional: a false positive costs a glance, a missed break costs an incident."""
    diff = """diff --git a/src/pricing.py b/src/pricing.py
--- a/src/pricing.py
+++ b/src/pricing.py
@@ -1,2 +1,2 @@
-def total_price(items *** not python:
+def total_price(items, tax_rate *** still broken:
     return items
"""
    symbols = extract_changed_symbols(diff)

    assert len(symbols) == 1
    assert "could not fully parse" in symbols[0].change


# --- Step 1: non-Python regex fallback ----------------------------------------


def test_non_python_files_use_the_regex_fallback_not_ast():
    symbols = extract_changed_symbols(JS_DIFF)

    assert len(symbols) == 1
    symbol = symbols[0]
    assert symbol.name == "totalPrice"
    assert symbol.language == "other"
    assert symbol.file == "src/pricing.js"
    assert "non-Python" in symbol.change


def test_fallback_captures_both_before_and_after_lines_when_both_present():
    symbols = extract_changed_symbols(JS_DIFF)

    symbol = symbols[0]
    assert "totalPrice(items)" in symbol.old_signature
    assert "totalPrice(items, taxRate)" in symbol.new_signature


# --- Step 2: git grep call-site discovery -------------------------------------


def test_call_site_discovery_excludes_files_already_in_the_diff():
    git = FakeGit(
        responses={
            "total_price": (
                "src/pricing.py:1:def total_price(items, tax_rate):\n"  # in-diff — excluded
                "shop/invoice.py:44:total = total_price(items)\n"  # outside — kept
            )
        }
    )
    model = RecordingModel([BLAST_RESPONSE_ONE_BREAK])

    analyze_blast_radius(make_pr(), SIGNATURE_DIFF, model, git, REPO_ROOT)

    assert len(model.prompts) == 1
    assert "shop/invoice.py:44" in model.prompts[0]
    assert "src/pricing.py:1" not in model.prompts[0]


def test_git_grep_is_scoped_to_the_given_repo_root():
    git = FakeGit()

    analyze_blast_radius(make_pr(), SIGNATURE_DIFF, failing_model, git, Path("/repo/checkout"))

    assert git.calls[0][:2] == ["-C", str(Path("/repo/checkout"))]


# --- unbroken_callers / checked accounting -------------------------------------


def test_unbroken_callers_are_counted_alongside_breaks():
    git = FakeGit(
        responses={
            "total_price": (
                "shop/invoice.py:44:total_price(items)\n"
                "shop/cart.py:10:total_price(items, 0.1)\n"
                "shop/refund.py:5:total_price(items, 0.1)\n"
            )
        }
    )
    model = RecordingModel([BLAST_RESPONSE_ONE_BREAK])

    result = analyze_blast_radius(make_pr(), SIGNATURE_DIFF, model, git, REPO_ROOT)

    assert len(result) == 1
    entry = result[0]
    assert isinstance(entry, BlastRadius)
    assert len(entry.breaks) == 1
    assert entry.unbroken_callers == 2
    assert entry.checked == 3


# --- zero call sites -----------------------------------------------------------


def test_zero_call_sites_is_a_valid_result_and_skips_the_model():
    git = FakeGit()  # no responses configured -> empty grep output for every symbol

    result = analyze_blast_radius(make_pr(), SIGNATURE_DIFF, failing_model, git, REPO_ROOT)

    assert len(result) == 1
    entry = result[0]
    assert entry.symbol == "total_price"
    assert entry.breaks == ()
    assert entry.unbroken_callers == 0
    assert entry.checked == 0


def test_no_changed_symbols_returns_an_empty_tuple_without_touching_git_or_model():
    git = FakeGit()

    result = analyze_blast_radius(make_pr(), NO_REAL_CHANGE_DIFF, failing_model, git, REPO_ROOT)

    assert result == ()
    assert git.calls == []


# --- git grep failure handling --------------------------------------------------


def test_a_failing_git_grep_is_handled_as_zero_call_sites_not_raised():
    git = FakeGit(raise_for=frozenset({"total_price"}))

    result = analyze_blast_radius(make_pr(), SIGNATURE_DIFF, failing_model, git, REPO_ROOT)

    assert len(result) == 1
    assert result[0].checked == 0


def test_a_failing_grep_for_one_symbol_does_not_block_another_symbol_in_the_same_diff():
    diff = SIGNATURE_DIFF + REMOVAL_DIFF.replace("diff --git a/src/pricing.py", "diff --git a/src/other.py").replace(
        "b/src/pricing.py", "b/src/other.py"
    )
    git = FakeGit(
        responses={"legacy_helper": "shop/caller.py:9:legacy_helper(1)\n"},
        raise_for=frozenset({"total_price"}),
    )
    model = RecordingModel(
        ['{"symbol": "legacy_helper", "kind": "removal_or_rename", "change": "removed", '
         '"breaks": [{"file": "shop/caller.py", "line": 9, "severity": "HIGH", "claim": "no longer exists"}], '
         '"unbroken_callers": 0}']
    )

    result = analyze_blast_radius(make_pr(), diff, model, git, REPO_ROOT)

    names = {entry.symbol for entry in result}
    assert names == {"total_price", "legacy_helper"}
    checked = {entry.symbol: entry.checked for entry in result}
    assert checked["total_price"] == 0
    assert checked["legacy_helper"] == 1


# --- prompt assembly -----------------------------------------------------------


def test_build_blast_prompt_includes_symbol_and_every_call_site():
    symbol = ChangedSymbol(
        name="total_price",
        file="src/pricing.py",
        language="python",
        old_signature="def total_price(items):",
        new_signature="def total_price(items, tax_rate):",
        change="added required parameter(s) tax_rate",
    )
    call_sites = (
        CallSite(file="shop/invoice.py", line=44, text="total_price(items)"),
        CallSite(file="shop/cart.py", line=10, text="total_price(items, 0.1)"),
    )

    prompt = build_blast_prompt("TEMPLATE", make_pr(), symbol, call_sites)

    assert "TEMPLATE" in prompt
    assert "total_price" in prompt
    assert "shop/invoice.py:44" in prompt
    assert "shop/cart.py:10" in prompt
    assert "2 total" in prompt


def test_build_blast_prompt_labels_a_removed_symbol_clearly():
    symbol = ChangedSymbol(
        name="legacy_helper",
        file="src/pricing.py",
        language="python",
        old_signature="def legacy_helper(x):",
        new_signature="",
        change="'legacy_helper' was removed entirely",
        kind_hint="removal_or_rename",
    )

    prompt = build_blast_prompt("TEMPLATE", make_pr(), symbol, ())

    assert "(removed by this diff)" in prompt


# --- end-to-end wiring / error handling -----------------------------------------


def test_analyze_blast_radius_reads_the_real_prompt_file_without_raising():
    git = FakeGit(responses={"total_price": "shop/invoice.py:44:total_price(items)\n"})
    model = RecordingModel([BLAST_RESPONSE_ONE_BREAK])

    result = analyze_blast_radius(make_pr(), SIGNATURE_DIFF, model, git, REPO_ROOT, PROMPTS_DIR)

    assert len(result) == 1
    assert result[0].symbol == "total_price"


def test_missing_prompt_file_raises_blast_radius_error(tmp_path):
    git = FakeGit(responses={"total_price": "shop/invoice.py:44:total_price(items)\n"})
    model = RecordingModel([BLAST_RESPONSE_ONE_BREAK])

    with pytest.raises(BlastRadiusError, match="Could not read prompt"):
        analyze_blast_radius(make_pr(), SIGNATURE_DIFF, model, git, REPO_ROOT, tmp_path)


def test_malformed_model_response_raises_verdict_error():
    git = FakeGit(responses={"total_price": "shop/invoice.py:44:total_price(items)\n"})
    model = RecordingModel(["not json at all"])

    with pytest.raises(VerdictError):
        analyze_blast_radius(make_pr(), SIGNATURE_DIFF, model, git, REPO_ROOT)


# --- dataclasses are frozen ------------------------------------------------------


def test_changed_symbol_and_call_site_are_frozen():
    symbol = ChangedSymbol(
        name="x", file="a.py", language="python", old_signature="", new_signature="", change=""
    )
    with pytest.raises(AttributeError):
        symbol.name = "y"  # type: ignore[misc]

    site = CallSite(file="a.py", line=1, text="x()")
    with pytest.raises(AttributeError):
        site.line = 2  # type: ignore[misc]
