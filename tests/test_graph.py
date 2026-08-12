"""Knowledge-graph consumer: schema validation, ancestry-based freshness, the
query surface, and the compact markdown render.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from prime_pr_review.context import GitError
from prime_pr_review.graph import (
    CoChangeWarning,
    Edge,
    GraphError,
    KnowledgeGraph,
    Node,
    callers_of,
    co_change_partners,
    co_change_warnings,
    importers_of,
    load_for_review,
    load_graph,
    neighborhood,
    render,
    verify_fresh,
)


def write_graph(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def base_payload(**overrides: object) -> dict:
    payload = {
        "version": 1,
        "repo": "acme/widget",
        "commit": "deadbeef00",
        "built_at": "2026-08-11T04:00:00Z",
        "nodes": [
            {"id": "shop/orders.py", "kind": "file"},
            {
                "id": "shop/orders.py::total_price",
                "kind": "function",
                "file": "shop/orders.py",
                "line": 5,
            },
        ],
        "edges": [
            {
                "src": "shop/invoice.py::build",
                "dst": "shop/orders.py::total_price",
                "kind": "calls",
            },
        ],
    }
    payload.update(overrides)
    return payload


def make_graph(edges: Sequence[Edge] = (), nodes: Sequence[Node] = (), **overrides: object) -> KnowledgeGraph:
    fields = dict(
        version=1,
        repo="acme/widget",
        commit="c0ffee",
        built_at="",
        nodes=tuple(nodes),
        edges=tuple(edges),
        skipped_edges=0,
    )
    fields.update(overrides)
    return KnowledgeGraph(**fields)


@dataclass
class FakeGitRunner:
    """A git runner for the ancestry check. Raises `GitError` when configured to."""

    error: str | None = None
    calls: list[list[str]] = field(default_factory=list)

    def __call__(self, args: Sequence[str]) -> str:
        self.calls.append(list(args))
        if self.error is not None:
            raise GitError(self.error)
        return ""


# --- load_graph: schema validation ----------------------------------------------


def test_loads_a_well_formed_graph(tmp_path):
    path = write_graph(tmp_path, base_payload())

    graph = load_graph(path)

    assert graph.version == 1
    assert graph.commit == "deadbeef00"
    assert graph.repo == "acme/widget"
    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1
    assert graph.skipped_edges == 0


def test_missing_file_raises_graph_error_naming_the_path(tmp_path):
    missing = tmp_path / "absent.json"

    with pytest.raises(GraphError, match="not found"):
        load_graph(missing)


def test_malformed_json_raises_graph_error(tmp_path):
    path = tmp_path / "graph.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(GraphError, match="not valid JSON"):
        load_graph(path)


def test_top_level_non_object_raises_graph_error(tmp_path):
    path = tmp_path / "graph.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(GraphError, match="JSON object"):
        load_graph(path)


def test_wrong_version_names_the_version_field(tmp_path):
    path = write_graph(tmp_path, base_payload(version=2))

    with pytest.raises(GraphError, match="version"):
        load_graph(path)


def test_missing_version_names_the_version_field(tmp_path):
    payload = base_payload()
    del payload["version"]
    path = write_graph(tmp_path, payload)

    with pytest.raises(GraphError, match="version"):
        load_graph(path)


def test_missing_commit_names_the_commit_field(tmp_path):
    payload = base_payload()
    del payload["commit"]
    path = write_graph(tmp_path, payload)

    with pytest.raises(GraphError, match="commit"):
        load_graph(path)


def test_blank_commit_names_the_commit_field(tmp_path):
    path = write_graph(tmp_path, base_payload(commit="   "))

    with pytest.raises(GraphError, match="commit"):
        load_graph(path)


def test_node_missing_id_names_the_id_field(tmp_path):
    path = write_graph(tmp_path, base_payload(nodes=[{"kind": "file"}]))

    with pytest.raises(GraphError, match=r"nodes\[0\]\.id"):
        load_graph(path)


def test_node_missing_kind_names_the_kind_field(tmp_path):
    path = write_graph(tmp_path, base_payload(nodes=[{"id": "a.py"}]))

    with pytest.raises(GraphError, match=r"nodes\[0\]\.kind"):
        load_graph(path)


def test_edge_missing_src_names_the_src_field(tmp_path):
    path = write_graph(tmp_path, base_payload(edges=[{"dst": "a.py", "kind": "calls"}]))

    with pytest.raises(GraphError, match=r"edges\[0\]\.src"):
        load_graph(path)


def test_edge_missing_dst_names_the_dst_field(tmp_path):
    path = write_graph(tmp_path, base_payload(edges=[{"src": "a.py", "kind": "calls"}]))

    with pytest.raises(GraphError, match=r"edges\[0\]\.dst"):
        load_graph(path)


def test_edge_missing_kind_names_the_kind_field(tmp_path):
    path = write_graph(tmp_path, base_payload(edges=[{"src": "a.py", "dst": "b.py"}]))

    with pytest.raises(GraphError, match=r"edges\[0\]\.kind"):
        load_graph(path)


def test_missing_nodes_and_edges_default_to_empty(tmp_path):
    payload = base_payload()
    del payload["nodes"]
    del payload["edges"]
    path = write_graph(tmp_path, payload)

    graph = load_graph(path)

    assert graph.nodes == ()
    assert graph.edges == ()


def test_nodes_of_the_wrong_type_raises_graph_error(tmp_path):
    path = write_graph(tmp_path, base_payload(nodes="not-a-list"))

    with pytest.raises(GraphError, match="'nodes' must be a list"):
        load_graph(path)


def test_edges_of_the_wrong_type_raises_graph_error(tmp_path):
    path = write_graph(tmp_path, base_payload(edges="not-a-list"))

    with pytest.raises(GraphError, match="'edges' must be a list"):
        load_graph(path)


def test_a_node_entry_that_is_not_an_object_raises_graph_error(tmp_path):
    path = write_graph(tmp_path, base_payload(nodes=["not-an-object"]))

    with pytest.raises(GraphError, match=r"nodes\[0\] must be an object"):
        load_graph(path)


def test_an_edge_entry_that_is_not_an_object_raises_graph_error(tmp_path):
    path = write_graph(tmp_path, base_payload(edges=["not-an-object"]))

    with pytest.raises(GraphError, match=r"edges\[0\] must be an object"):
        load_graph(path)


# --- load_graph: unknown edge kinds ----------------------------------------------


def test_unknown_edge_kind_is_accepted_not_rejected(tmp_path):
    payload = base_payload(edges=[{"src": "a.py", "dst": "b.py", "kind": "shares_wire_format"}])
    path = write_graph(tmp_path, payload)

    graph = load_graph(path)

    assert graph.edges == (Edge(src="a.py", dst="b.py", kind="shares_wire_format"),)
    assert graph.skipped_edges == 0


# --- load_graph: co_changes_with skip-not-fatal ----------------------------------


def test_co_change_edge_missing_weight_is_skipped_not_fatal(tmp_path):
    payload = base_payload(
        edges=[{"src": "a.py", "dst": "b.py", "kind": "co_changes_with", "samples": 10}]
    )
    path = write_graph(tmp_path, payload)

    graph = load_graph(path)

    assert graph.edges == ()
    assert graph.skipped_edges == 1


def test_co_change_edge_missing_samples_is_skipped_not_fatal(tmp_path):
    payload = base_payload(
        edges=[{"src": "a.py", "dst": "b.py", "kind": "co_changes_with", "weight": 0.9}]
    )
    path = write_graph(tmp_path, payload)

    graph = load_graph(path)

    assert graph.edges == ()
    assert graph.skipped_edges == 1


def test_co_change_edge_with_non_numeric_weight_is_skipped_not_fatal(tmp_path):
    payload = base_payload(
        edges=[
            {
                "src": "a.py",
                "dst": "b.py",
                "kind": "co_changes_with",
                "weight": "high",
                "samples": 10,
            }
        ]
    )
    path = write_graph(tmp_path, payload)

    graph = load_graph(path)

    assert graph.edges == ()
    assert graph.skipped_edges == 1


def test_a_skipped_co_change_edge_does_not_block_other_edges_in_the_same_file(tmp_path):
    payload = base_payload(
        edges=[
            {"src": "a.py", "dst": "b.py", "kind": "co_changes_with", "samples": 10},
            {"src": "c.py::f", "dst": "d.py::g", "kind": "calls"},
        ]
    )
    path = write_graph(tmp_path, payload)

    graph = load_graph(path)

    assert graph.skipped_edges == 1
    assert graph.edges == (Edge(src="c.py::f", dst="d.py::g", kind="calls"),)


def test_a_well_formed_co_change_edge_is_kept_with_its_weight_and_samples(tmp_path):
    payload = base_payload(
        edges=[
            {"src": "a.py", "dst": "b.py", "kind": "co_changes_with", "weight": 0.875, "samples": 16}
        ]
    )
    path = write_graph(tmp_path, payload)

    graph = load_graph(path)

    assert graph.edges == (
        Edge(src="a.py", dst="b.py", kind="co_changes_with", weight=0.875, samples=16),
    )
    assert graph.skipped_edges == 0


# --- verify_fresh ------------------------------------------------------------------


def test_verify_fresh_returns_none_when_the_commit_is_an_ancestor():
    git = FakeGitRunner()

    result = verify_fresh(make_graph(commit="c0ffee"), "base-sha", git, Path("/repo"))

    assert result is None


def test_verify_fresh_runs_merge_base_is_ancestor_with_the_right_arguments():
    git = FakeGitRunner()

    verify_fresh(make_graph(commit="c0ffee"), "base-sha", git, Path("/repo"))

    assert git.calls == [
        ["-C", str(Path("/repo")), "merge-base", "--is-ancestor", "c0ffee", "base-sha"]
    ]


def test_verify_fresh_refuses_when_the_commit_is_not_an_ancestor():
    git = FakeGitRunner(error="exit status 1")

    result = verify_fresh(make_graph(), "base-sha", git, Path("/repo"))

    assert result is not None
    assert "not a verified ancestor" in result


def test_verify_fresh_refuses_on_an_unknown_sha():
    git = FakeGitRunner(error="fatal: Not a valid object name badsha")

    result = verify_fresh(make_graph(), "badsha", git, Path("/repo"))

    assert result is not None
    assert "badsha" in result


# --- load_for_review: never raises ------------------------------------------------


def test_load_for_review_returns_the_graph_and_empty_reason_on_success(tmp_path):
    path = write_graph(tmp_path, base_payload())

    graph, reason = load_for_review(path, "base-sha", FakeGitRunner(), tmp_path)

    assert graph is not None
    assert graph.commit == "deadbeef00"
    assert reason == ""


def test_load_for_review_degrades_on_a_missing_file(tmp_path):
    graph, reason = load_for_review(
        tmp_path / "absent.json", "base-sha", FakeGitRunner(), tmp_path
    )

    assert graph is None
    assert reason != ""


def test_load_for_review_degrades_on_a_malformed_file(tmp_path):
    path = tmp_path / "graph.json"
    path.write_text("{not json", encoding="utf-8")

    graph, reason = load_for_review(path, "base-sha", FakeGitRunner(), tmp_path)

    assert graph is None
    assert "unusable" in reason


def test_load_for_review_degrades_on_a_stale_graph(tmp_path):
    path = write_graph(tmp_path, base_payload())
    git = FakeGitRunner(error="exit status 1")

    graph, reason = load_for_review(path, "base-sha", git, tmp_path)

    assert graph is None
    assert "stale" in reason


def test_load_for_review_never_raises_even_when_the_runner_misbehaves(tmp_path):
    """A runner raising something other than `GitError` must not crash the one
    call site the sweep depends on never raising."""
    path = write_graph(tmp_path, base_payload())

    def broken_runner(args):
        raise RuntimeError("unexpected failure")

    graph, reason = load_for_review(path, "base-sha", broken_runner, tmp_path)

    assert graph is None
    assert reason != ""


# --- callers_of / importers_of ----------------------------------------------------


def test_callers_of_returns_sources_of_calls_edges_into_the_target():
    graph = make_graph(
        edges=[
            Edge(src="z.py::z", dst="target.py::t", kind="calls"),
            Edge(src="a.py::a", dst="target.py::t", kind="calls"),
            Edge(src="d.py::i", dst="target.py::t", kind="imports"),
            Edge(src="e.py::j", dst="other.py::o", kind="calls"),
        ]
    )

    result = callers_of(graph, "target.py::t")

    assert result == ("a.py::a", "z.py::z")


def test_callers_of_deduplicates_repeated_call_edges_from_the_same_source():
    graph = make_graph(
        edges=[
            Edge(src="a.py::f", dst="b.py::g", kind="calls"),
            Edge(src="a.py::f", dst="b.py::g", kind="calls"),
        ]
    )

    assert callers_of(graph, "b.py::g") == ("a.py::f",)


def test_callers_of_returns_empty_tuple_when_nothing_calls_the_target():
    assert callers_of(make_graph(), "b.py::g") == ()


def test_importers_of_returns_sources_of_imports_edges_into_the_target_file():
    graph = make_graph(
        edges=[
            Edge(src="b.py", dst="shared.py", kind="imports"),
            Edge(src="a.py", dst="shared.py", kind="imports"),
            Edge(src="c.py", dst="shared.py", kind="calls"),
        ]
    )

    assert importers_of(graph, "shared.py") == ("a.py", "b.py")


def test_importers_of_returns_empty_tuple_when_nothing_imports_the_target():
    assert importers_of(make_graph(), "shared.py") == ()


# --- neighborhood: BFS depth and cap ----------------------------------------------


def _chain_graph() -> KnowledgeGraph:
    """A -calls-> B -imports-> C -co_changes_with-> D -custom_kind-> E, plus an
    unrelated component F <-> G, to prove neighborhood stays within reach."""
    return make_graph(
        edges=[
            Edge(src="A", dst="B", kind="calls"),
            Edge(src="B", dst="C", kind="imports"),
            Edge(src="C", dst="D", kind="co_changes_with", weight=0.9, samples=10),
            Edge(src="D", dst="E", kind="custom_kind"),
            Edge(src="F", dst="G", kind="calls"),
        ]
    )


def test_neighborhood_at_k_zero_returns_only_the_seeds():
    graph = _chain_graph()

    assert neighborhood(graph, ["C"], k=0) == ("C",)


def test_neighborhood_deduplicates_repeated_seeds():
    graph = _chain_graph()

    assert neighborhood(graph, ["C", "C"], k=0) == ("C",)


def test_neighborhood_traverses_edges_bidirectionally():
    """C has no outgoing edge to B, only B -> C; neighborhood must still reach B."""
    graph = _chain_graph()

    assert neighborhood(graph, ["C"], k=1) == ("B", "C", "D")


def test_neighborhood_expands_across_multiple_hops_and_edge_kinds():
    graph = _chain_graph()

    assert neighborhood(graph, ["C"], k=2) == ("A", "B", "C", "D", "E")


def test_neighborhood_does_not_reach_an_unrelated_component():
    graph = _chain_graph()

    result = neighborhood(graph, ["C"], k=10)

    assert "F" not in result
    assert "G" not in result


def test_neighborhood_supports_multiple_seeds():
    graph = _chain_graph()

    assert neighborhood(graph, ["A", "F"], k=0) == ("A", "F")


def test_neighborhood_cap_stops_once_the_total_node_budget_is_spent():
    graph = _chain_graph()

    result = neighborhood(graph, ["C"], k=2, cap=3)

    assert result == ("B", "C", "D")
    assert len(result) == 3


def test_neighborhood_cap_selection_is_deterministic_across_tied_candidates():
    """Four neighbors of C at once; cap=2 must keep the seed plus the
    alphabetically-first neighbor, not an arbitrary one."""
    graph = make_graph(
        edges=[
            Edge(src="C", dst="Y", kind="calls"),
            Edge(src="C", dst="X", kind="calls"),
            Edge(src="C", dst="D", kind="calls"),
            Edge(src="C", dst="B", kind="calls"),
        ]
    )

    assert neighborhood(graph, ["C"], k=2, cap=2) == ("B", "C")


def test_neighborhood_with_no_seeds_returns_empty_tuple():
    assert neighborhood(_chain_graph(), [], k=2) == ()


# --- co_change_partners: thresholds -----------------------------------------------


def test_co_change_partners_returns_partner_weight_and_samples():
    graph = make_graph(
        edges=[Edge(src="a.py", dst="b.py", kind="co_changes_with", weight=0.875, samples=16)]
    )

    assert co_change_partners(graph, "a.py") == (("b.py", 0.875, 16),)


def test_co_change_partners_considers_edges_in_either_direction():
    graph = make_graph(
        edges=[Edge(src="b.py", dst="a.py", kind="co_changes_with", weight=0.8, samples=10)]
    )

    assert co_change_partners(graph, "a.py") == (("b.py", 0.8, 10),)


def test_co_change_partners_excludes_partners_below_min_weight():
    graph = make_graph(
        edges=[Edge(src="a.py", dst="b.py", kind="co_changes_with", weight=0.4, samples=100)]
    )

    assert co_change_partners(graph, "a.py") == ()


def test_co_change_partners_excludes_partners_below_min_samples():
    graph = make_graph(
        edges=[Edge(src="a.py", dst="b.py", kind="co_changes_with", weight=0.99, samples=2)]
    )

    assert co_change_partners(graph, "a.py") == ()


def test_co_change_partners_respects_custom_thresholds():
    graph = make_graph(
        edges=[Edge(src="a.py", dst="b.py", kind="co_changes_with", weight=0.3, samples=3)]
    )

    assert co_change_partners(graph, "a.py") == ()  # below the 0.5/5 defaults
    assert co_change_partners(graph, "a.py", min_weight=0.2, min_samples=2) == (("b.py", 0.3, 3),)


def test_co_change_partners_ignores_edges_of_other_kinds_even_with_weight_set():
    graph = make_graph(edges=[Edge(src="a.py", dst="b.py", kind="calls", weight=0.9, samples=10)])

    assert co_change_partners(graph, "a.py") == ()


def test_co_change_partners_sorted_by_weight_descending():
    graph = make_graph(
        edges=[
            Edge(src="a.py", dst="low.py", kind="co_changes_with", weight=0.6, samples=10),
            Edge(src="a.py", dst="high.py", kind="co_changes_with", weight=0.95, samples=10),
        ]
    )

    result = co_change_partners(graph, "a.py")

    assert [p[0] for p in result] == ["high.py", "low.py"]


def test_co_change_partners_breaks_weight_ties_by_partner_name():
    graph = make_graph(
        edges=[
            Edge(src="a.py", dst="z.py", kind="co_changes_with", weight=0.9, samples=10),
            Edge(src="a.py", dst="b.py", kind="co_changes_with", weight=0.9, samples=10),
        ]
    )

    assert co_change_partners(graph, "a.py") == (("b.py", 0.9, 10), ("z.py", 0.9, 10))


# --- co_change_warnings: diff exclusion and symmetric dedup -----------------------


def test_warns_about_a_partner_not_touched_by_the_diff():
    graph = make_graph(
        edges=[Edge(src="parser.py", dst="grammar.toml", kind="co_changes_with", weight=0.875, samples=16)]
    )

    warnings = co_change_warnings(graph, diff_files=["parser.py"])

    assert warnings == (
        CoChangeWarning(file="parser.py", partner="grammar.toml", weight=0.875, samples=16),
    )


def test_excludes_partners_that_are_also_in_the_diff():
    graph = make_graph(
        edges=[Edge(src="parser.py", dst="grammar.toml", kind="co_changes_with", weight=0.875, samples=16)]
    )

    warnings = co_change_warnings(graph, diff_files=["parser.py", "grammar.toml"])

    assert warnings == ()


def test_partners_below_default_thresholds_do_not_warn():
    graph = make_graph(
        edges=[Edge(src="parser.py", dst="grammar.toml", kind="co_changes_with", weight=0.5, samples=16)]
    )

    assert co_change_warnings(graph, diff_files=["parser.py"]) == ()


def test_custom_thresholds_override_the_defaults_for_warnings():
    graph = make_graph(
        edges=[Edge(src="a.py", dst="b.py", kind="co_changes_with", weight=0.55, samples=6)]
    )

    assert co_change_warnings(graph, diff_files=["a.py"]) == ()
    warnings = co_change_warnings(graph, diff_files=["a.py"], min_weight=0.5, min_samples=5)
    assert len(warnings) == 1


def test_deduplicates_symmetric_pairs_when_both_directions_qualify():
    graph = make_graph(
        edges=[
            Edge(src="a.py", dst="b.py", kind="co_changes_with", weight=0.9, samples=10),
            Edge(src="b.py", dst="a.py", kind="co_changes_with", weight=0.7, samples=12),
        ]
    )

    warnings = co_change_warnings(graph, diff_files=["a.py"])

    assert warnings == (CoChangeWarning(file="a.py", partner="b.py", weight=0.9, samples=10),)


def test_two_different_diff_files_each_warn_about_the_same_external_partner():
    """Not a symmetric pair — two distinct diff files sharing one partner is
    two real findings, and dedup must not collapse them into one."""
    graph = make_graph(
        edges=[
            Edge(src="a.py", dst="shared.py", kind="co_changes_with", weight=0.9, samples=10),
            Edge(src="c.py", dst="shared.py", kind="co_changes_with", weight=0.8, samples=9),
        ]
    )

    warnings = co_change_warnings(graph, diff_files=["a.py", "c.py"])

    assert {(w.file, w.partner) for w in warnings} == {("a.py", "shared.py"), ("c.py", "shared.py")}


def test_warnings_are_sorted_by_weight_descending():
    graph = make_graph(
        edges=[
            Edge(src="a.py", dst="low.py", kind="co_changes_with", weight=0.65, samples=9),
            Edge(src="a.py", dst="high.py", kind="co_changes_with", weight=0.95, samples=20),
        ]
    )

    warnings = co_change_warnings(graph, diff_files=["a.py"])

    assert [w.partner for w in warnings] == ["high.py", "low.py"]


def test_no_diff_files_produces_no_warnings():
    graph = make_graph(
        edges=[Edge(src="a.py", dst="b.py", kind="co_changes_with", weight=0.9, samples=10)]
    )

    assert co_change_warnings(graph, diff_files=[]) == ()


# --- render: counts present even when empty ---------------------------------------


def test_render_returns_empty_string_when_there_is_nothing_to_check():
    assert render(make_graph(), diff_files=[], changed_symbols=[]) == ""


def test_render_omits_the_co_change_section_when_no_diff_files_given():
    markdown = render(make_graph(), diff_files=[], changed_symbols=["a.py::f"])

    assert "Co-change warnings" not in markdown


def test_render_omits_callers_and_neighborhood_sections_when_no_symbols_given():
    markdown = render(make_graph(), diff_files=["a.py"], changed_symbols=[])

    assert "Callers of changed symbols" not in markdown
    assert "Neighborhood" not in markdown


def test_render_states_zero_callers_explicitly_rather_than_omitting_the_count():
    markdown = render(make_graph(), diff_files=[], changed_symbols=["a.py::f"])

    assert "0 caller(s) found for 1 symbol(s) checked." in markdown


def test_render_states_zero_co_change_warnings_explicitly():
    markdown = render(make_graph(), diff_files=["a.py"], changed_symbols=[])

    assert "0 warning(s) found for 1 file(s) checked." in markdown


def test_render_states_zero_neighborhood_nodes_explicitly_when_seed_is_isolated():
    markdown = render(make_graph(), diff_files=[], changed_symbols=["isolated::sym"])

    assert "seed(s) checked" in markdown
    assert "1 node(s) found within 2 hops of 1 seed(s) checked." in markdown


def test_render_includes_caller_detail_when_present():
    graph = make_graph(edges=[Edge(src="caller.py::c", dst="target.py::t", kind="calls")])

    markdown = render(graph, diff_files=[], changed_symbols=["target.py::t"])

    assert "1 caller(s) found for 1 symbol(s) checked." in markdown
    assert "`caller.py::c` calls `target.py::t`" in markdown


def test_render_includes_co_change_warning_detail_when_present():
    graph = make_graph(
        edges=[Edge(src="parser.py", dst="grammar.toml", kind="co_changes_with", weight=0.875, samples=16)]
    )

    markdown = render(graph, diff_files=["parser.py"], changed_symbols=[])

    assert "1 warning(s) found for 1 file(s) checked." in markdown
    assert "`parser.py` and `grammar.toml`" in markdown
    assert "14 of the last 16" in markdown


def test_render_includes_a_neighborhood_node_listing_when_reachable_nodes_exist():
    graph = make_graph(edges=[Edge(src="a.py::f", dst="b.py::g", kind="calls")])

    markdown = render(graph, diff_files=[], changed_symbols=["a.py::f"])

    assert "`b.py::g`" in markdown


def test_render_includes_a_top_level_header_when_any_section_renders():
    markdown = render(make_graph(), diff_files=["a.py"], changed_symbols=[])

    assert markdown.startswith("## Knowledge graph")


# --- frozen dataclasses -------------------------------------------------------------


def test_node_edge_and_warning_dataclasses_are_frozen():
    node = Node(id="a.py", kind="file")
    with pytest.raises(AttributeError):
        node.id = "b.py"  # type: ignore[misc]

    edge = Edge(src="a.py", dst="b.py", kind="calls")
    with pytest.raises(AttributeError):
        edge.kind = "imports"  # type: ignore[misc]

    warning = CoChangeWarning(file="a.py", partner="b.py", weight=0.9, samples=10)
    with pytest.raises(AttributeError):
        warning.weight = 0.1  # type: ignore[misc]

    graph = make_graph()
    with pytest.raises(AttributeError):
        graph.commit = "other"  # type: ignore[misc]
