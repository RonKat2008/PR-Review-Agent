"""Tests for the P10 knowledge-graph builder (`scripts/build_graph.py`).

All git access is injected — `FakeGit` below — so nothing here shells out to
a real repository. Python source for the ast extractor is written into
`tmp_path` for each test that needs it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pytest

import json

from scripts.build_graph import (
    BULK_COMMIT_FILE_THRESHOLD,
    MIN_SAMPLES,
    Edge,
    GitError,
    Graph,
    Node,
    _pick_test_target,
    _resolve_repo_slug,
    build_cochange_edges,
    build_graph,
    extract_ast_graph,
    main,
)

FROZEN_NOW = datetime(2026, 8, 11, 4, 0, 0, tzinfo=timezone.utc)


# --- test doubles ----------------------------------------------------------------


@dataclass
class FakeGit:
    """A git runner keyed on predicate/response pairs, mirroring the `FakeGh`
    pattern in `tests/conftest.py`. Records every call."""

    handlers: list[tuple[Callable[[Sequence[str]], bool], str]] = field(default_factory=list)
    calls: list[list[str]] = field(default_factory=list)

    def on(self, predicate: Callable[[Sequence[str]], bool], response: str) -> "FakeGit":
        self.handlers.append((predicate, response))
        return self

    def __call__(self, args: Sequence[str]) -> str:
        self.calls.append(list(args))
        for predicate, response in self.handlers:
            if predicate(args):
                return response
        raise AssertionError(f"unexpected git call: {args}")


def is_log(args: Sequence[str]) -> bool:
    return "log" in args


def is_rev_parse(args: Sequence[str]) -> bool:
    return "rev-parse" in args


def is_remote(args: Sequence[str]) -> bool:
    return "remote" in args


def make_git(log_text: str = "", head: str = "a" * 40, remote: str = "https://github.com/acme/widget.git") -> FakeGit:
    return FakeGit().on(is_log, log_text).on(is_rev_parse, head + "\n").on(is_remote, remote + "\n")


def commit_block(sha: str, files: Sequence[str]) -> str:
    return "\n".join((sha, *files))


def log_text(*blocks: str) -> str:
    return "\n\n".join(blocks)


# --- co-change: weight arithmetic -------------------------------------------------


def test_cochange_weight_is_commits_touching_both_over_commits_touching_src():
    # Arrange: A appears in 6 commits, B alongside it in 5 of those.
    blocks = [commit_block(f"c{i}", ["a.py", "b.py"]) for i in range(5)]
    blocks.append(commit_block("c5", ["a.py"]))
    git = make_git(log_text(*blocks))

    # Act
    edges = build_cochange_edges(Path("."), git, since_commits=500, min_samples=MIN_SAMPLES)

    # Assert
    a_to_b = next(e for e in edges if e.src == "a.py" and e.dst == "b.py")
    assert a_to_b.samples == 6
    assert a_to_b.weight == pytest.approx(5 / 6, abs=1e-4)


def test_cochange_directionality_a_to_b_differs_from_b_to_a():
    # Arrange: A touched 6 times, B touched 5 times, always together when B appears.
    blocks = [commit_block(f"c{i}", ["a.py", "b.py"]) for i in range(5)]
    blocks.append(commit_block("c5", ["a.py"]))  # A alone
    git = make_git(log_text(*blocks))

    # Act
    edges = build_cochange_edges(Path("."), git, since_commits=500, min_samples=MIN_SAMPLES)

    # Assert
    a_to_b = next(e for e in edges if e.src == "a.py" and e.dst == "b.py")
    b_to_a = next(e for e in edges if e.src == "b.py" and e.dst == "a.py")
    assert a_to_b.samples == 6
    assert b_to_a.samples == 5
    assert a_to_b.weight != b_to_a.weight
    assert a_to_b.weight == pytest.approx(5 / 6, abs=1e-4)
    assert b_to_a.weight == pytest.approx(1.0, abs=1e-4)


def test_both_directions_are_emitted_as_separate_edges():
    blocks = [commit_block(f"c{i}", ["a.py", "b.py"]) for i in range(MIN_SAMPLES)]
    git = make_git(log_text(*blocks))

    edges = build_cochange_edges(Path("."), git, since_commits=500, min_samples=MIN_SAMPLES)

    kinds = {(e.src, e.dst) for e in edges}
    assert ("a.py", "b.py") in kinds
    assert ("b.py", "a.py") in kinds


# --- co-change: bulk-commit exclusion ---------------------------------------------


def test_a_commit_touching_more_than_the_bulk_threshold_is_excluded_entirely():
    # Arrange: 5 normal commits establish a real a.py<->b.py coupling, plus one
    # bulk commit (a mass reformat) that also happens to touch both files.
    normal = [commit_block(f"c{i}", ["a.py", "b.py"]) for i in range(MIN_SAMPLES)]
    bulk_files = [f"file{i}.py" for i in range(BULK_COMMIT_FILE_THRESHOLD + 1)] + ["a.py", "b.py"]
    bulk = commit_block("bulk", bulk_files)
    git = make_git(log_text(*normal, bulk))

    # Act
    edges = build_cochange_edges(Path("."), git, since_commits=500, min_samples=MIN_SAMPLES)

    # Assert: as if the bulk commit never happened at all.
    a_to_b = next(e for e in edges if e.src == "a.py" and e.dst == "b.py")
    assert a_to_b.samples == MIN_SAMPLES
    assert a_to_b.weight == pytest.approx(1.0, abs=1e-6)
    assert not any(e.src == "file0.py" for e in edges)


def test_a_commit_at_exactly_the_bulk_threshold_is_still_included():
    files = [f"file{i}.py" for i in range(BULK_COMMIT_FILE_THRESHOLD)]
    blocks = [commit_block(f"c{i}", files[:2]) for i in range(MIN_SAMPLES)]
    blocks.append(commit_block("boundary", files))  # exactly at the threshold
    git = make_git(log_text(*blocks))

    edges = build_cochange_edges(Path("."), git, since_commits=500, min_samples=MIN_SAMPLES)

    edge = next(e for e in edges if e.src == "file0.py" and e.dst == "file1.py")
    assert edge.samples == MIN_SAMPLES + 1


# --- co-change: minimum-samples floor ---------------------------------------------


def test_edges_below_the_minimum_samples_floor_are_dropped():
    # Arrange: a.py and b.py always change together, but only 4 times — one
    # short of the default floor of 5.
    blocks = [commit_block(f"c{i}", ["a.py", "b.py"]) for i in range(MIN_SAMPLES - 1)]
    git = make_git(log_text(*blocks))

    edges = build_cochange_edges(Path("."), git, since_commits=500, min_samples=MIN_SAMPLES)

    assert edges == ()


def test_edges_at_exactly_the_minimum_samples_floor_are_kept():
    blocks = [commit_block(f"c{i}", ["a.py", "b.py"]) for i in range(MIN_SAMPLES)]
    git = make_git(log_text(*blocks))

    edges = build_cochange_edges(Path("."), git, since_commits=500, min_samples=MIN_SAMPLES)

    assert any(e.src == "a.py" and e.dst == "b.py" for e in edges)


def test_a_custom_min_samples_floor_is_respected():
    blocks = [commit_block(f"c{i}", ["a.py", "b.py"]) for i in range(3)]
    git = make_git(log_text(*blocks))

    edges = build_cochange_edges(Path("."), git, since_commits=500, min_samples=2)

    assert any(e.src == "a.py" and e.dst == "b.py" for e in edges)


def test_a_file_that_never_co_occurs_with_another_produces_no_edge():
    blocks = [commit_block(f"c{i}", ["solo.py"]) for i in range(MIN_SAMPLES)]
    git = make_git(log_text(*blocks))

    edges = build_cochange_edges(Path("."), git, since_commits=500, min_samples=MIN_SAMPLES)

    assert edges == ()


def test_empty_git_log_produces_no_cochange_edges():
    git = make_git("")

    edges = build_cochange_edges(Path("."), git, since_commits=500, min_samples=MIN_SAMPLES)

    assert edges == ()


# --- ast: defines ------------------------------------------------------------------


def test_ast_defines_edges_are_emitted_for_top_level_functions_and_classes(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text(
        "def foo():\n    def _inner():\n        pass\n    return _inner\n\n\nclass Bar:\n    def method(self):\n        pass\n",
        encoding="utf-8",
    )

    result = extract_ast_graph(tmp_path)

    defines = {(e.src, e.dst) for e in result.edges if e.kind == "defines"}
    assert defines == {
        ("pkg/mod.py", "pkg/mod.py::foo"),
        ("pkg/mod.py", "pkg/mod.py::Bar"),
    }
    foo_node = next(n for n in result.nodes if n.id == "pkg/mod.py::foo")
    assert foo_node.kind == "function"
    assert foo_node.file == "pkg/mod.py"
    assert foo_node.line == 1
    bar_node = next(n for n in result.nodes if n.id == "pkg/mod.py::Bar")
    assert bar_node.kind == "class"


def test_nested_functions_and_methods_do_not_get_their_own_defines_edge(tmp_path):
    (tmp_path / "mod.py").write_text(
        "def outer():\n    def inner():\n        pass\n\n\nclass C:\n    def method(self):\n        pass\n",
        encoding="utf-8",
    )

    result = extract_ast_graph(tmp_path)

    dsts = {e.dst for e in result.edges if e.kind == "defines"}
    assert "mod.py::inner" not in dsts
    assert "mod.py::method" not in dsts
    assert "mod.py::C::method" not in dsts


# --- ast: imports ------------------------------------------------------------------


def test_ast_import_edge_resolves_a_repo_internal_absolute_import(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("import pkg.b\n", encoding="utf-8")
    (tmp_path / "pkg" / "b.py").write_text("", encoding="utf-8")

    result = extract_ast_graph(tmp_path)

    imports = {(e.src, e.dst) for e in result.edges if e.kind == "imports"}
    assert ("pkg/a.py", "pkg/b.py") in imports


def test_ast_import_edge_resolves_a_from_import_of_a_repo_internal_module(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("from pkg.b import helper\n", encoding="utf-8")
    (tmp_path / "pkg" / "b.py").write_text("def helper():\n    pass\n", encoding="utf-8")

    result = extract_ast_graph(tmp_path)

    imports = {(e.src, e.dst) for e in result.edges if e.kind == "imports"}
    assert ("pkg/a.py", "pkg/b.py") in imports


def test_ast_import_edge_resolves_a_relative_import(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "a.py").write_text("from . import b\n", encoding="utf-8")
    (tmp_path / "pkg" / "b.py").write_text("", encoding="utf-8")

    result = extract_ast_graph(tmp_path)

    imports = {(e.src, e.dst) for e in result.edges if e.kind == "imports"}
    assert ("pkg/a.py", "pkg/b.py") in imports


def test_ast_import_edge_resolves_a_relative_import_going_up_a_package_level(tmp_path):
    (tmp_path / "pkg" / "sub").mkdir(parents=True)
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "sub" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "sub" / "a.py").write_text("from ..other import helper\n", encoding="utf-8")
    (tmp_path / "pkg" / "other.py").write_text("def helper():\n    pass\n", encoding="utf-8")

    result = extract_ast_graph(tmp_path)

    imports = {(e.src, e.dst) for e in result.edges if e.kind == "imports"}
    assert ("pkg/sub/a.py", "pkg/other.py") in imports


def test_ast_import_edge_resolves_to_a_package_init_file(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "a.py").write_text("import pkg\n", encoding="utf-8")

    result = extract_ast_graph(tmp_path)

    imports = {(e.src, e.dst) for e in result.edges if e.kind == "imports"}
    assert ("a.py", "pkg/__init__.py") in imports


def test_stdlib_import_is_ignored_and_produces_no_edge(tmp_path):
    (tmp_path / "a.py").write_text("import os\nimport json\nfrom collections import defaultdict\n", encoding="utf-8")

    result = extract_ast_graph(tmp_path)

    imports = [e for e in result.edges if e.kind == "imports"]
    assert imports == []


def test_third_party_import_is_ignored_and_produces_no_edge(tmp_path):
    (tmp_path / "a.py").write_text("import httpx\nfrom pandas import DataFrame\n", encoding="utf-8")

    result = extract_ast_graph(tmp_path)

    imports = [e for e in result.edges if e.kind == "imports"]
    assert imports == []


def test_a_file_never_imports_itself(tmp_path):
    (tmp_path / "a.py").write_text("import a\n", encoding="utf-8")

    result = extract_ast_graph(tmp_path)

    imports = [e for e in result.edges if e.kind == "imports"]
    assert imports == []


# --- ast: calls ----------------------------------------------------------------------


def test_a_call_to_a_same_file_top_level_function_is_resolved(tmp_path):
    (tmp_path / "mod.py").write_text(
        "def callee():\n    pass\n\n\ndef caller():\n    callee()\n", encoding="utf-8"
    )

    result = extract_ast_graph(tmp_path)

    calls = {(e.src, e.dst) for e in result.edges if e.kind == "calls"}
    assert ("mod.py::caller", "mod.py::callee") in calls


def test_a_call_to_a_uniquely_named_cross_file_function_is_resolved(tmp_path):
    (tmp_path / "a.py").write_text("def caller():\n    unique_util()\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def unique_util():\n    pass\n", encoding="utf-8")

    result = extract_ast_graph(tmp_path)

    calls = {(e.src, e.dst) for e in result.edges if e.kind == "calls"}
    assert ("a.py::caller", "b.py::unique_util") in calls


def test_a_call_to_an_ambiguously_named_function_defined_in_two_files_is_dropped(tmp_path):
    (tmp_path / "caller.py").write_text("def run():\n    helper()\n", encoding="utf-8")
    (tmp_path / "x.py").write_text("def helper():\n    pass\n", encoding="utf-8")
    (tmp_path / "y.py").write_text("def helper():\n    pass\n", encoding="utf-8")

    result = extract_ast_graph(tmp_path)

    calls = [e for e in result.edges if e.kind == "calls"]
    assert calls == []


def test_a_call_to_an_unknown_symbol_produces_no_edge(tmp_path):
    (tmp_path / "a.py").write_text("def caller():\n    print('hi')\n", encoding="utf-8")

    result = extract_ast_graph(tmp_path)

    calls = [e for e in result.edges if e.kind == "calls"]
    assert calls == []


def test_a_method_call_via_attribute_access_is_conservatively_dropped(tmp_path):
    (tmp_path / "a.py").write_text(
        "class Foo:\n    def method(self):\n        pass\n\n\ndef caller(foo):\n    foo.method()\n",
        encoding="utf-8",
    )

    result = extract_ast_graph(tmp_path)

    calls = [e for e in result.edges if e.kind == "calls"]
    assert calls == []


def test_instantiating_a_top_level_class_directly_is_resolved_as_a_call(tmp_path):
    (tmp_path / "a.py").write_text(
        "class Foo:\n    pass\n\n\ndef caller():\n    Foo()\n", encoding="utf-8"
    )

    result = extract_ast_graph(tmp_path)

    calls = {(e.src, e.dst) for e in result.edges if e.kind == "calls"}
    assert ("a.py::caller", "a.py::Foo") in calls


def test_a_call_made_inside_a_nested_function_is_attributed_to_the_enclosing_top_level_function(tmp_path):
    (tmp_path / "mod.py").write_text(
        "def target():\n    pass\n\n\ndef outer():\n    def inner():\n        target()\n    inner()\n",
        encoding="utf-8",
    )

    result = extract_ast_graph(tmp_path)

    calls = {(e.src, e.dst) for e in result.edges if e.kind == "calls"}
    assert ("mod.py::outer", "mod.py::target") in calls


def test_a_recursive_self_call_does_not_produce_a_self_loop_edge(tmp_path):
    (tmp_path / "mod.py").write_text("def recurse(n):\n    return recurse(n - 1)\n", encoding="utf-8")

    result = extract_ast_graph(tmp_path)

    calls = [e for e in result.edges if e.kind == "calls"]
    assert calls == []


# --- ast: unparseable files ----------------------------------------------------------


def test_a_file_that_fails_to_parse_is_skipped_not_fatal(tmp_path):
    (tmp_path / "broken.py").write_text("def broken(:\n    this is not python\n", encoding="utf-8")
    (tmp_path / "good.py").write_text("def fine():\n    pass\n", encoding="utf-8")

    result = extract_ast_graph(tmp_path)

    assert "broken.py" in result.skipped
    assert not any(e.src == "broken.py" for e in result.edges if e.kind == "defines")
    assert any(e.dst == "good.py::fine" for e in result.edges if e.kind == "defines")


def test_an_unreadable_file_is_skipped_not_fatal(tmp_path):
    (tmp_path / "weird.py").write_bytes(b"\xff\xfe\x00\x01not valid utf8 \x80\x81")

    result = extract_ast_graph(tmp_path)

    assert "weird.py" in result.skipped


def test_vendored_and_venv_directories_are_not_walked(tmp_path):
    (tmp_path / ".venv" / "lib").mkdir(parents=True)
    (tmp_path / ".venv" / "lib" / "vendored.py").write_text("def vendored():\n    pass\n", encoding="utf-8")
    (tmp_path / "real.py").write_text("def real():\n    pass\n", encoding="utf-8")

    result = extract_ast_graph(tmp_path)

    assert not any("vendored" in n.id for n in result.nodes)
    assert any(n.id == "real.py" for n in result.nodes)


# --- ast: tests ------------------------------------------------------------------------


def test_tests_edge_resolves_a_same_directory_module_by_stem(tmp_path):
    (tmp_path / "test_orders.py").write_text("def test_it():\n    pass\n", encoding="utf-8")
    (tmp_path / "orders.py").write_text("def total():\n    pass\n", encoding="utf-8")

    result = extract_ast_graph(tmp_path)

    tests = {(e.src, e.dst) for e in result.edges if e.kind == "tests"}
    assert ("test_orders.py", "orders.py") in tests


def test_tests_edge_resolves_the_suffix_naming_convention(tmp_path):
    (tmp_path / "orders_test.py").write_text("def test_it():\n    pass\n", encoding="utf-8")
    (tmp_path / "orders.py").write_text("def total():\n    pass\n", encoding="utf-8")

    result = extract_ast_graph(tmp_path)

    tests = {(e.src, e.dst) for e in result.edges if e.kind == "tests"}
    assert ("orders_test.py", "orders.py") in tests


def test_tests_edge_prefers_same_directory_match_over_a_distant_one(tmp_path):
    (tmp_path / "shop").mkdir()
    (tmp_path / "shop" / "test_orders.py").write_text("", encoding="utf-8")
    (tmp_path / "shop" / "orders.py").write_text("", encoding="utf-8")
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "orders.py").write_text("", encoding="utf-8")

    result = extract_ast_graph(tmp_path)

    tests = {(e.src, e.dst) for e in result.edges if e.kind == "tests"}
    assert ("shop/test_orders.py", "shop/orders.py") in tests
    assert not any(dst == "other/orders.py" for _src, dst in tests)


def test_tests_edge_is_dropped_when_no_module_matches(tmp_path):
    (tmp_path / "test_ghost.py").write_text("", encoding="utf-8")

    result = extract_ast_graph(tmp_path)

    assert [e for e in result.edges if e.kind == "tests"] == []


def test_pick_test_target_drops_an_ambiguous_match_across_multiple_directories():
    matches = ["a/orders.py", "b/orders.py"]

    target = _pick_test_target("c/test_orders.py", matches)

    assert target is None


def test_file_node_for_a_test_file_uses_the_test_kind(tmp_path):
    (tmp_path / "test_orders.py").write_text("", encoding="utf-8")
    (tmp_path / "orders.py").write_text("", encoding="utf-8")

    result = extract_ast_graph(tmp_path)

    test_node = next(n for n in result.nodes if n.id == "test_orders.py")
    plain_node = next(n for n in result.nodes if n.id == "orders.py")
    assert test_node.kind == "test"
    assert plain_node.kind == "file"


# --- repo slug resolution --------------------------------------------------------------


def test_repo_slug_is_parsed_from_an_https_origin_remote():
    git = FakeGit().on(is_remote, "https://github.com/acme/widget.git\n")

    slug = _resolve_repo_slug(Path("/repo"), git)

    assert slug == "acme/widget"


def test_repo_slug_is_parsed_from_an_ssh_origin_remote():
    git = FakeGit().on(is_remote, "git@github.com:acme/widget.git\n")

    slug = _resolve_repo_slug(Path("/repo"), git)

    assert slug == "acme/widget"


def test_repo_slug_falls_back_to_the_directory_name_when_the_remote_url_is_unparseable():
    git = FakeGit().on(is_remote, "not-a-url\n")

    slug = _resolve_repo_slug(Path("/some/checkout"), git)

    assert slug == "checkout"


def test_repo_slug_falls_back_to_the_directory_name_when_there_is_no_remote():
    class RaisingGit:
        def __call__(self, args: Sequence[str]) -> str:
            raise GitError("No such remote 'origin'")

    slug = _resolve_repo_slug(Path("/some/checkout"), RaisingGit())

    assert slug == "checkout"


# --- full build_graph wiring / schema shape ---------------------------------------------


def _assert_matches_documented_schema(payload: dict) -> None:
    assert set(payload.keys()) == {"version", "repo", "commit", "built_at", "nodes", "edges"}
    assert payload["version"] == 1
    assert isinstance(payload["repo"], str) and payload["repo"]
    assert isinstance(payload["commit"], str) and payload["commit"]
    assert isinstance(payload["built_at"], str) and payload["built_at"].endswith("Z")
    assert isinstance(payload["nodes"], list)
    assert isinstance(payload["edges"], list)

    node_ids = set()
    for node in payload["nodes"]:
        assert set(node.keys()) <= {"id", "kind", "file", "line"}
        assert {"id", "kind"} <= node.keys()
        assert isinstance(node["id"], str)
        assert node["kind"] in {"file", "test", "function", "class", "method"}
        if "line" in node:
            assert isinstance(node["line"], int)
        node_ids.add(node["id"])

    known_kinds = {"imports", "calls", "inherits", "defines", "tests", "co_changes_with"}
    for edge in payload["edges"]:
        assert set(edge.keys()) <= {"src", "dst", "kind", "weight", "samples"}
        assert {"src", "dst", "kind"} <= edge.keys()
        assert edge["kind"] in known_kinds
        if edge["kind"] == "co_changes_with":
            assert "weight" in edge and 0.0 <= edge["weight"] <= 1.0
            assert "samples" in edge and edge["samples"] >= MIN_SAMPLES


def test_build_graph_output_matches_the_documented_schema(tmp_path):
    (tmp_path / "a.py").write_text("def caller():\n    callee()\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def callee():\n    pass\n", encoding="utf-8")
    blocks = [commit_block(f"c{i}", ["a.py", "b.py"]) for i in range(MIN_SAMPLES)]
    git = make_git(log_text(*blocks))

    graph, _skipped = build_graph(tmp_path, git, since_commits=500, now=FROZEN_NOW)

    _assert_matches_documented_schema(graph.to_dict())


def test_build_graph_sets_commit_to_head_sha_from_the_injected_runner():
    git = make_git("", head="deadbeef" * 5)

    graph, _skipped = build_graph(Path("."), git, since_commits=500, now=FROZEN_NOW)

    assert graph.commit == "deadbeef" * 5


def test_build_graph_no_ast_disables_python_extraction(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")
    git = make_git("")

    graph, skipped = build_graph(tmp_path, git, since_commits=500, use_ast=False, now=FROZEN_NOW)

    assert graph.edges == ()
    assert graph.nodes == ()
    assert skipped == 0


def test_build_graph_reports_the_number_of_skipped_unparseable_files(tmp_path):
    (tmp_path / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    git = make_git("")

    _graph, skipped = build_graph(tmp_path, git, since_commits=500, now=FROZEN_NOW)

    assert skipped == 1


def test_build_graph_merges_cochange_file_nodes_with_ast_file_nodes(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("", encoding="utf-8")
    blocks = [commit_block(f"c{i}", ["a.py", "b.py"]) for i in range(MIN_SAMPLES)]
    git = make_git(log_text(*blocks))

    graph, _skipped = build_graph(tmp_path, git, since_commits=500, now=FROZEN_NOW)

    ids = [n.id for n in graph.nodes]
    assert ids.count("a.py") == 1  # not duplicated across the two sources
    assert "b.py" in ids


# --- determinism -------------------------------------------------------------------------


def test_build_graph_output_is_byte_identical_across_repeated_builds(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("import pkg.b\n\n\ndef caller():\n    helper()\n", encoding="utf-8")
    (tmp_path / "pkg" / "b.py").write_text("def helper():\n    pass\n", encoding="utf-8")
    (tmp_path / "test_pkg.py").write_text("", encoding="utf-8")
    blocks = [commit_block(f"c{i}", ["pkg/a.py", "pkg/b.py", "z.py"]) for i in range(MIN_SAMPLES)]
    git_one = make_git(log_text(*blocks))
    git_two = make_git(log_text(*blocks))

    graph_one, _ = build_graph(tmp_path, git_one, since_commits=500, now=FROZEN_NOW)
    graph_two, _ = build_graph(tmp_path, git_two, since_commits=500, now=FROZEN_NOW)

    assert graph_one.to_dict() == graph_two.to_dict()


def test_nodes_are_sorted_by_id():
    git = make_git(log_text(commit_block("c0", ["z.py", "a.py", "m.py"]), commit_block("c1", ["z.py", "a.py", "m.py"]), commit_block("c2", ["z.py", "a.py", "m.py"]), commit_block("c3", ["z.py", "a.py", "m.py"]), commit_block("c4", ["z.py", "a.py", "m.py"])))

    graph, _skipped = build_graph(Path("."), git, since_commits=500, now=FROZEN_NOW)

    ids = [n.id for n in graph.nodes]
    assert ids == sorted(ids)


def test_edges_are_sorted_by_kind_then_src_then_dst():
    blocks = [commit_block(f"c{i}", ["z.py", "a.py"]) for i in range(MIN_SAMPLES)]
    git = make_git(log_text(*blocks))

    graph, _skipped = build_graph(Path("."), git, since_commits=500, now=FROZEN_NOW)

    keys = [(e.kind, e.src, e.dst) for e in graph.edges]
    assert keys == sorted(keys)


# --- since-commits is respected ----------------------------------------------------------


def test_since_commits_is_passed_through_to_the_git_log_call():
    git = make_git("")

    build_cochange_edges(Path("/repo"), git, since_commits=42, min_samples=MIN_SAMPLES)

    log_call = next(c for c in git.calls if "log" in c)
    assert "-n" in log_call
    assert log_call[log_call.index("-n") + 1] == "42"


def test_git_log_is_scoped_to_the_given_repo_root():
    git = make_git("")

    build_cochange_edges(Path("/some/repo"), git, since_commits=10, min_samples=MIN_SAMPLES)

    log_call = next(c for c in git.calls if "log" in c)
    assert log_call[:2] == ["-C", str(Path("/some/repo"))]


# --- dataclasses are frozen ----------------------------------------------------------------


def test_node_and_edge_and_graph_are_frozen():
    node = Node(id="a.py", kind="file")
    with pytest.raises(AttributeError):
        node.id = "b.py"  # type: ignore[misc]

    edge = Edge(src="a.py", dst="b.py", kind="imports")
    with pytest.raises(AttributeError):
        edge.src = "c.py"  # type: ignore[misc]

    graph = Graph(version=1, repo="a/b", commit="x", built_at="now", nodes=(), edges=())
    with pytest.raises(AttributeError):
        graph.repo = "c/d"  # type: ignore[misc]


def test_node_to_dict_omits_file_and_line_for_a_plain_file_node():
    node = Node(id="a.py", kind="file")

    assert node.to_dict() == {"id": "a.py", "kind": "file"}


def test_node_to_dict_includes_file_and_line_for_a_symbol_node():
    node = Node(id="a.py::foo", kind="function", file="a.py", line=3)

    assert node.to_dict() == {"id": "a.py::foo", "kind": "function", "file": "a.py", "line": 3}


def test_edge_to_dict_omits_weight_and_samples_for_a_non_cochange_edge():
    edge = Edge(src="a.py", dst="a.py::foo", kind="defines")

    assert edge.to_dict() == {"src": "a.py", "dst": "a.py::foo", "kind": "defines"}


def test_edge_to_dict_includes_weight_and_samples_for_a_cochange_edge():
    edge = Edge(src="a.py", dst="b.py", kind="co_changes_with", weight=0.875, samples=16)

    assert edge.to_dict() == {
        "src": "a.py",
        "dst": "b.py",
        "kind": "co_changes_with",
        "weight": 0.875,
        "samples": 16,
    }


# --- CLI wiring (main) --------------------------------------------------------------------


def test_main_writes_a_graph_json_file_and_returns_zero(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")
    git = make_git("")
    out_path = tmp_path / "graph.json"

    exit_code = main(
        ["--repo", str(tmp_path), "--out", str(out_path), "--since-commits", "10"],
        git_runner=git,
    )

    assert exit_code == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    _assert_matches_documented_schema(payload)
    assert any(n["id"] == "a.py::foo" for n in payload["nodes"])


def test_main_no_ast_flag_disables_python_extraction(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")
    git = make_git("")
    out_path = tmp_path / "graph.json"

    main(["--repo", str(tmp_path), "--out", str(out_path), "--no-ast"], git_runner=git)

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["nodes"] == []
    assert payload["edges"] == []


def test_main_returns_one_and_prints_to_stderr_on_a_git_error(tmp_path, capsys):
    class RaisingGit:
        def __call__(self, args: Sequence[str]) -> str:
            raise GitError("fatal: not a git repository")

    out_path = tmp_path / "graph.json"

    exit_code = main(
        ["--repo", str(tmp_path), "--out", str(out_path)], git_runner=RaisingGit()
    )

    assert exit_code == 1
    assert not out_path.exists()
    assert "git error" in capsys.readouterr().err


def test_main_reports_skipped_files_on_stdout(tmp_path, capsys):
    (tmp_path / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    git = make_git("")
    out_path = tmp_path / "graph.json"

    main(["--repo", str(tmp_path), "--out", str(out_path)], git_runner=git)

    assert "1 file(s) skipped" in capsys.readouterr().out
