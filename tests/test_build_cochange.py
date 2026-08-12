"""Tests for the Tier 0 knowledge-graph co-change miner (`scripts/build_cochange.py`).

All git access is injected -- `FakeGit` below -- so nothing here shells out to a real
repository, mirroring `tests/test_github.py`'s `FakeGh` pattern from conftest.py (not
reused directly: that fixture's runner takes `(args, stdin)`, ours takes just `args`).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.build_cochange import (
    BULK_COMMIT_FILE_THRESHOLD,
    DEFAULT_MIN_SAMPLES,
    DEFAULT_REF,
    DEFAULT_SINCE_COMMITS,
    Edge,
    Graph,
    GitError,
    Node,
    _parse_log,
    _parse_remote_slug,
    _resolve_repo_slug,
    build_cochange_edges,
    build_graph,
    default_runner,
    main,
    write_graph_atomic,
)

FROZEN_NOW = datetime(2026, 8, 11, 4, 0, 0, tzinfo=timezone.utc)


# --- test doubles ------------------------------------------------------------------


@dataclass
class FakeGit:
    """A git runner keyed on predicate/response pairs. Records every call."""

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


def _is_log(args: Sequence[str]) -> bool:
    return "log" in args


def _is_rev_parse(args: Sequence[str]) -> bool:
    return "rev-parse" in args


def _is_remote(args: Sequence[str]) -> bool:
    return "config" in args and "remote.origin.url" in args


def make_git(
    log_output: str = "",
    head: str = "a" * 40,
    remote: str = "https://github.com/acme/widget.git",
) -> FakeGit:
    return FakeGit().on(_is_log, log_output).on(_is_rev_parse, head + "\n").on(_is_remote, remote + "\n")


def commit_block(sha: str, files: Sequence[str]) -> str:
    return "\n".join((sha, *files))


def log_text(*blocks: str) -> str:
    return "\n\n".join(blocks)


def _assert_matches_schema(payload: dict) -> None:
    """docs/KNOWLEDGE-GRAPH.md's envelope, scoped to what this module emits:
    file nodes and co_changes_with edges only."""
    assert set(payload.keys()) == {"version", "repo", "commit", "built_at", "nodes", "edges"}
    assert payload["version"] == 1
    assert isinstance(payload["repo"], str) and payload["repo"]
    assert isinstance(payload["commit"], str) and payload["commit"]
    assert isinstance(payload["built_at"], str) and payload["built_at"].endswith("Z")

    node_ids = set()
    for node in payload["nodes"]:
        assert set(node.keys()) == {"id", "kind"}
        assert node["kind"] == "file"
        node_ids.add(node["id"])

    for edge in payload["edges"]:
        assert set(edge.keys()) == {"src", "dst", "kind", "weight", "samples"}
        assert edge["kind"] == "co_changes_with"
        assert 0.0 <= edge["weight"] <= 1.0
        assert isinstance(edge["samples"], int) and edge["samples"] >= 1
        assert edge["src"] in node_ids
        assert edge["dst"] in node_ids


# --- contract constants --------------------------------------------------------------


def test_default_constants_match_the_contract():
    assert DEFAULT_REF == "HEAD"
    assert DEFAULT_SINCE_COMMITS == 500
    assert DEFAULT_MIN_SAMPLES == 5
    assert BULK_COMMIT_FILE_THRESHOLD == 50


# --- co-change: weight arithmetic -----------------------------------------------------


def test_cochange_weight_is_commits_touching_both_over_commits_touching_src():
    # Arrange: A appears in 6 commits, B alongside it in 5 of those.
    blocks = [commit_block(f"c{i}", ["a.py", "b.py"]) for i in range(5)]
    blocks.append(commit_block("c5", ["a.py"]))
    git = make_git(log_text(*blocks))

    # Act
    edges = build_cochange_edges(git, ref="HEAD", since_commits=500, min_samples=DEFAULT_MIN_SAMPLES)

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
    edges = build_cochange_edges(git, ref="HEAD", since_commits=500, min_samples=DEFAULT_MIN_SAMPLES)

    # Assert
    a_to_b = next(e for e in edges if e.src == "a.py" and e.dst == "b.py")
    b_to_a = next(e for e in edges if e.src == "b.py" and e.dst == "a.py")
    assert a_to_b.samples == 6
    assert b_to_a.samples == 5
    assert a_to_b.weight != b_to_a.weight
    assert a_to_b.weight == pytest.approx(5 / 6, abs=1e-4)
    assert b_to_a.weight == pytest.approx(1.0, abs=1e-4)


def test_both_directions_are_emitted_as_separate_edges_when_each_clears_the_floor():
    blocks = [commit_block(f"c{i}", ["a.py", "b.py"]) for i in range(DEFAULT_MIN_SAMPLES)]
    git = make_git(log_text(*blocks))

    edges = build_cochange_edges(git, ref="HEAD", since_commits=500, min_samples=DEFAULT_MIN_SAMPLES)

    kinds = {(e.src, e.dst) for e in edges}
    assert ("a.py", "b.py") in kinds
    assert ("b.py", "a.py") in kinds


# --- co-change: bulk-commit exclusion --------------------------------------------------


def test_a_commit_touching_more_than_the_bulk_threshold_is_excluded_entirely():
    # Arrange: 5 normal commits establish a real a.py<->b.py coupling, plus one
    # bulk commit (a mass reformat) that also happens to touch both files.
    normal = [commit_block(f"c{i}", ["a.py", "b.py"]) for i in range(DEFAULT_MIN_SAMPLES)]
    bulk_files = [f"file{i}.py" for i in range(BULK_COMMIT_FILE_THRESHOLD + 1)] + ["a.py", "b.py"]
    bulk = commit_block("bulk", bulk_files)
    git = make_git(log_text(*normal, bulk))

    # Act
    edges = build_cochange_edges(git, ref="HEAD", since_commits=500, min_samples=DEFAULT_MIN_SAMPLES)

    # Assert: as if the bulk commit never happened at all.
    a_to_b = next(e for e in edges if e.src == "a.py" and e.dst == "b.py")
    assert a_to_b.samples == DEFAULT_MIN_SAMPLES
    assert a_to_b.weight == pytest.approx(1.0, abs=1e-6)
    assert not any(e.src == "file0.py" for e in edges)


def test_a_commit_at_exactly_the_bulk_threshold_is_still_included():
    files = [f"file{i}.py" for i in range(BULK_COMMIT_FILE_THRESHOLD)]
    blocks = [commit_block(f"c{i}", files[:2]) for i in range(DEFAULT_MIN_SAMPLES)]
    blocks.append(commit_block("boundary", files))  # exactly at the threshold
    git = make_git(log_text(*blocks))

    edges = build_cochange_edges(git, ref="HEAD", since_commits=500, min_samples=DEFAULT_MIN_SAMPLES)

    edge = next(e for e in edges if e.src == "file0.py" and e.dst == "file1.py")
    assert edge.samples == DEFAULT_MIN_SAMPLES + 1


# --- co-change: minimum-samples floor --------------------------------------------------


def test_edges_below_the_minimum_samples_floor_are_dropped():
    # Arrange: a.py and b.py always change together, but only 4 times -- one
    # short of the default floor of 5.
    blocks = [commit_block(f"c{i}", ["a.py", "b.py"]) for i in range(DEFAULT_MIN_SAMPLES - 1)]
    git = make_git(log_text(*blocks))

    edges = build_cochange_edges(git, ref="HEAD", since_commits=500, min_samples=DEFAULT_MIN_SAMPLES)

    assert edges == ()


def test_edges_at_exactly_the_minimum_samples_floor_are_kept():
    blocks = [commit_block(f"c{i}", ["a.py", "b.py"]) for i in range(DEFAULT_MIN_SAMPLES)]
    git = make_git(log_text(*blocks))

    edges = build_cochange_edges(git, ref="HEAD", since_commits=500, min_samples=DEFAULT_MIN_SAMPLES)

    assert any(e.src == "a.py" and e.dst == "b.py" for e in edges)


def test_a_custom_min_samples_floor_is_respected():
    blocks = [commit_block(f"c{i}", ["a.py", "b.py"]) for i in range(3)]
    git = make_git(log_text(*blocks))

    edges = build_cochange_edges(git, ref="HEAD", since_commits=500, min_samples=2)

    assert any(e.src == "a.py" and e.dst == "b.py" for e in edges)


def test_a_file_that_never_co_occurs_with_another_produces_no_edge():
    blocks = [commit_block(f"c{i}", ["solo.py"]) for i in range(DEFAULT_MIN_SAMPLES)]
    git = make_git(log_text(*blocks))

    edges = build_cochange_edges(git, ref="HEAD", since_commits=500, min_samples=DEFAULT_MIN_SAMPLES)

    assert edges == ()


def test_empty_git_log_produces_no_cochange_edges():
    git = make_git("")

    edges = build_cochange_edges(git, ref="HEAD", since_commits=500, min_samples=DEFAULT_MIN_SAMPLES)

    assert edges == ()


# --- git log parsing: malformed lines ---------------------------------------------------


def test_a_commit_header_line_containing_whitespace_is_skipped_and_counted_as_malformed():
    good_blocks = [commit_block(f"c{i}", ["a.py", "b.py"]) for i in range(DEFAULT_MIN_SAMPLES)]
    corrupt_block = "not a real sha\nfile.py"
    raw = log_text(*good_blocks, corrupt_block)

    commits, malformed = _parse_log(raw)

    assert malformed == 1
    assert len(commits) == DEFAULT_MIN_SAMPLES
    assert not any(c.sha == "not a real sha" for c in commits)


def test_malformed_commits_do_not_contribute_cochange_edges():
    good_blocks = [commit_block(f"c{i}", ["a.py", "b.py"]) for i in range(DEFAULT_MIN_SAMPLES)]
    corrupt_block = "not a real sha\nx.py\ny.py"
    git = make_git(log_text(*good_blocks, corrupt_block))

    edges = build_cochange_edges(git, ref="HEAD", since_commits=500, min_samples=DEFAULT_MIN_SAMPLES)

    assert not any(e.src in {"x.py", "y.py"} for e in edges)


def test_empty_raw_log_parses_to_no_commits_and_no_malformed_count():
    commits, malformed = _parse_log("")

    assert commits == ()
    assert malformed == 0


# --- path normalization -----------------------------------------------------------------


def test_backslash_paths_in_git_output_are_normalized_to_forward_slashes():
    blocks = [commit_block(f"c{i}", ["pkg\\a.py", "pkg\\b.py"]) for i in range(DEFAULT_MIN_SAMPLES)]
    git = make_git(log_text(*blocks))

    edges = build_cochange_edges(git, ref="HEAD", since_commits=500, min_samples=DEFAULT_MIN_SAMPLES)

    assert any(e.src == "pkg/a.py" and e.dst == "pkg/b.py" for e in edges)
    assert not any("\\" in e.src or "\\" in e.dst for e in edges)


# --- ref / since-commits passed through to git --------------------------------------------


def test_git_log_command_matches_the_documented_shape():
    git = make_git("")

    build_cochange_edges(git, ref="HEAD", since_commits=500, min_samples=DEFAULT_MIN_SAMPLES)

    log_call = next(c for c in git.calls if "log" in c)
    assert log_call == ["log", "HEAD", "--name-only", "--pretty=format:%H", "-n", "500"]


def test_since_commits_is_passed_through_to_the_git_log_call():
    git = make_git("")

    build_cochange_edges(git, ref="HEAD", since_commits=42, min_samples=DEFAULT_MIN_SAMPLES)

    log_call = next(c for c in git.calls if "log" in c)
    assert log_call[log_call.index("-n") + 1] == "42"


def test_ref_is_passed_through_to_the_git_log_call():
    git = make_git("")

    build_cochange_edges(git, ref="origin/main", since_commits=10, min_samples=DEFAULT_MIN_SAMPLES)

    log_call = next(c for c in git.calls if "log" in c)
    assert log_call[1] == "origin/main"


def test_ref_is_passed_through_to_the_rev_parse_call():
    git = make_git("", head="feedface" * 5)

    graph = build_graph(Path("."), git, ref="origin/main", since_commits=10, now=FROZEN_NOW)

    rev_parse_call = next(c for c in git.calls if "rev-parse" in c)
    assert rev_parse_call == ["rev-parse", "origin/main"]
    assert graph.commit == "feedface" * 5


# --- origin-URL repo naming, folder fallback --------------------------------------------


def test_repo_slug_is_parsed_from_an_https_origin_remote():
    git = make_git("", remote="https://github.com/acme/widget.git")

    slug = _resolve_repo_slug(Path("/repo"), git)

    assert slug == "acme/widget"


def test_repo_slug_is_parsed_from_an_ssh_origin_remote():
    git = make_git("", remote="git@github.com:acme/widget.git")

    slug = _resolve_repo_slug(Path("/repo"), git)

    assert slug == "acme/widget"


def test_repo_slug_falls_back_to_the_directory_name_when_the_remote_url_is_unparseable():
    git = make_git("", remote="not-a-url")

    slug = _resolve_repo_slug(Path("/some/checkout"), git)

    assert slug == "checkout"


def test_repo_slug_falls_back_to_the_directory_name_when_there_is_no_remote():
    class RaisingGit:
        def __call__(self, args: Sequence[str]) -> str:
            raise GitError("No such remote 'origin'")

    slug = _resolve_repo_slug(Path("/some/checkout"), RaisingGit())

    assert slug == "checkout"


def test_parse_remote_slug_handles_an_ssh_protocol_url_with_a_double_slash():
    assert _parse_remote_slug("ssh://git@example.com/acme/widget.git") == "acme/widget"


def test_parse_remote_slug_returns_none_for_a_url_with_too_few_path_segments():
    assert _parse_remote_slug("https://github.com/acme") is None


# --- full build_graph wiring / schema shape -----------------------------------------------


def test_build_graph_output_matches_the_documented_schema():
    blocks = [commit_block(f"c{i}", ["a.py", "b.py"]) for i in range(DEFAULT_MIN_SAMPLES)]
    git = make_git(log_text(*blocks))

    graph = build_graph(Path("."), git, now=FROZEN_NOW)

    _assert_matches_schema(graph.to_dict())


def test_build_graph_emits_a_file_node_for_every_path_referenced_by_a_kept_edge():
    blocks = [commit_block(f"c{i}", ["a.py", "b.py"]) for i in range(DEFAULT_MIN_SAMPLES)]
    git = make_git(log_text(*blocks))

    graph = build_graph(Path("."), git, now=FROZEN_NOW)

    ids = {n.id for n in graph.nodes}
    assert ids == {"a.py", "b.py"}


def test_build_graph_omits_nodes_for_files_that_never_clear_the_min_samples_floor():
    blocks = [commit_block(f"c{i}", ["a.py", "b.py"]) for i in range(DEFAULT_MIN_SAMPLES - 1)]
    git = make_git(log_text(*blocks))

    graph = build_graph(Path("."), git, now=FROZEN_NOW)

    assert graph.nodes == ()


def test_build_graph_on_short_history_produces_a_valid_empty_edge_graph():
    blocks = [commit_block(f"c{i}", ["a.py", "b.py"]) for i in range(DEFAULT_MIN_SAMPLES - 1)]
    git = make_git(log_text(*blocks))

    graph = build_graph(Path("."), git, now=FROZEN_NOW)

    assert graph.edges == ()
    assert graph.nodes == ()
    assert graph.commit
    assert graph.repo


def test_build_graph_sets_commit_to_the_rev_parse_result():
    git = make_git("", head="deadbeef" * 5)

    graph = build_graph(Path("."), git, now=FROZEN_NOW)

    assert graph.commit == "deadbeef" * 5


def test_build_graph_built_at_uses_the_injected_now_as_utc_iso():
    git = make_git("")

    graph = build_graph(Path("."), git, now=FROZEN_NOW)

    assert graph.built_at == "2026-08-11T04:00:00Z"


# --- determinism -----------------------------------------------------------------------------


def test_build_graph_output_is_identical_across_repeated_builds_given_the_same_now():
    blocks = [commit_block(f"c{i}", ["z.py", "a.py", "m.py"]) for i in range(DEFAULT_MIN_SAMPLES)]
    git_one = make_git(log_text(*blocks))
    git_two = make_git(log_text(*blocks))

    graph_one = build_graph(Path("."), git_one, now=FROZEN_NOW)
    graph_two = build_graph(Path("."), git_two, now=FROZEN_NOW)

    assert graph_one.to_dict() == graph_two.to_dict()


def test_written_graph_bytes_are_identical_across_repeated_builds_given_the_same_now(tmp_path):
    blocks = [commit_block(f"c{i}", ["z.py", "a.py", "m.py"]) for i in range(DEFAULT_MIN_SAMPLES)]
    git_one = make_git(log_text(*blocks))
    git_two = make_git(log_text(*blocks))
    graph_one = build_graph(Path("."), git_one, now=FROZEN_NOW)
    graph_two = build_graph(Path("."), git_two, now=FROZEN_NOW)
    path_one, path_two = tmp_path / "one.json", tmp_path / "two.json"

    write_graph_atomic(path_one, graph_one)
    write_graph_atomic(path_two, graph_two)

    assert path_one.read_bytes() == path_two.read_bytes()


def test_nodes_are_sorted_by_id():
    blocks = [commit_block(f"c{i}", ["z.py", "a.py", "m.py"]) for i in range(DEFAULT_MIN_SAMPLES)]
    git = make_git(log_text(*blocks))

    graph = build_graph(Path("."), git, now=FROZEN_NOW)

    ids = [n.id for n in graph.nodes]
    assert ids == ["a.py", "m.py", "z.py"]


def test_edges_are_sorted_by_src_then_dst():
    blocks = [commit_block(f"c{i}", ["z.py", "a.py"]) for i in range(DEFAULT_MIN_SAMPLES)]
    git = make_git(log_text(*blocks))

    graph = build_graph(Path("."), git, now=FROZEN_NOW)

    keys = [(e.src, e.dst) for e in graph.edges]
    assert keys == sorted(keys)


# --- dataclasses are frozen and match the documented dict shape ---------------------------


def test_node_edge_and_graph_are_frozen():
    node = Node(id="a.py")
    with pytest.raises(AttributeError):
        node.id = "b.py"  # type: ignore[misc]

    edge = Edge(src="a.py", dst="b.py", weight=1.0, samples=5)
    with pytest.raises(AttributeError):
        edge.src = "c.py"  # type: ignore[misc]

    graph = Graph(version=1, repo="a/b", commit="x", built_at="now", nodes=(), edges=())
    with pytest.raises(AttributeError):
        graph.repo = "c/d"  # type: ignore[misc]


def test_node_to_dict_matches_the_documented_file_node_shape():
    assert Node(id="a.py").to_dict() == {"id": "a.py", "kind": "file"}


def test_edge_to_dict_includes_weight_samples_and_the_co_changes_with_kind():
    edge = Edge(src="a.py", dst="b.py", weight=0.875, samples=16)

    assert edge.to_dict() == {
        "src": "a.py",
        "dst": "b.py",
        "kind": "co_changes_with",
        "weight": 0.875,
        "samples": 16,
    }


def test_graph_to_dict_matches_the_documented_envelope():
    graph = Graph(
        version=1,
        repo="acme/widget",
        commit="a" * 40,
        built_at="2026-08-11T04:00:00Z",
        nodes=(Node(id="a.py"),),
        edges=(Edge(src="a.py", dst="b.py", weight=1.0, samples=5),),
    )

    payload = graph.to_dict()

    assert payload["version"] == 1
    assert payload["nodes"] == [{"id": "a.py", "kind": "file"}]
    assert payload["edges"] == [
        {"src": "a.py", "dst": "b.py", "kind": "co_changes_with", "weight": 1.0, "samples": 5}
    ]


# --- atomic write --------------------------------------------------------------------------


def test_write_graph_atomic_creates_parent_directories(tmp_path):
    graph = Graph(version=1, repo="a/b", commit="x", built_at="now", nodes=(), edges=())
    path = tmp_path / "nested" / "dir" / "graph.json"

    write_graph_atomic(path, graph)

    assert path.is_file()


def test_write_graph_atomic_leaves_no_tmp_file_behind(tmp_path):
    graph = Graph(version=1, repo="a/b", commit="x", built_at="now", nodes=(), edges=())
    path = tmp_path / "graph.json"

    write_graph_atomic(path, graph)

    assert list(tmp_path.iterdir()) == [path]


def test_write_graph_atomic_round_trips_through_json(tmp_path):
    graph = Graph(
        version=1,
        repo="acme/widget",
        commit="a" * 40,
        built_at="2026-08-11T04:00:00Z",
        nodes=(Node(id="a.py"),),
        edges=(Edge(src="a.py", dst="b.py", weight=0.5, samples=8),),
    )
    path = tmp_path / "graph.json"

    write_graph_atomic(path, graph)

    assert json.loads(path.read_text(encoding="utf-8")) == graph.to_dict()


# --- default_runner (subprocess boundary, always monkeypatched) ---------------------------


def test_default_runner_invokes_git_with_cwd_set_to_the_repo(monkeypatch, tmp_path):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = kwargs.get("cwd")
        return subprocess.CompletedProcess(cmd, 0, stdout="deadbeef\n", stderr="")

    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/git")
    monkeypatch.setattr(subprocess, "run", fake_run)

    runner = default_runner(tmp_path)
    output = runner(["rev-parse", "HEAD"])

    assert output == "deadbeef\n"
    assert captured["cmd"] == ["git", "rev-parse", "HEAD"]
    assert captured["cwd"] == tmp_path


def test_default_runner_raises_git_error_on_nonzero_exit(monkeypatch, tmp_path):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 128, stdout="", stderr="fatal: bad revision\n")

    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/git")
    monkeypatch.setattr(subprocess, "run", fake_run)

    runner = default_runner(tmp_path)

    with pytest.raises(GitError, match="fatal: bad revision"):
        runner(["rev-parse", "bogus"])


def test_default_runner_raises_git_error_on_timeout(monkeypatch, tmp_path):
    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=60)

    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/git")
    monkeypatch.setattr(subprocess, "run", fake_run)

    runner = default_runner(tmp_path)

    with pytest.raises(GitError, match="timed out"):
        runner(["log", "HEAD"])


def test_default_runner_raises_git_error_when_git_is_missing_from_path(monkeypatch, tmp_path):
    monkeypatch.setattr(shutil, "which", lambda _: None)

    runner = default_runner(tmp_path)

    with pytest.raises(GitError, match="not installed"):
        runner(["log", "HEAD"])


# --- CLI wiring (main) -----------------------------------------------------------------------


def test_main_writes_a_graph_json_file_and_returns_zero(tmp_path):
    blocks = [commit_block(f"c{i}", ["a.py", "b.py"]) for i in range(DEFAULT_MIN_SAMPLES)]
    git = make_git(log_text(*blocks))
    out_path = tmp_path / "graph.json"

    exit_code = main(["--repo", str(tmp_path), "--out", str(out_path)], git_runner=git, now=FROZEN_NOW)

    assert exit_code == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    _assert_matches_schema(payload)
    assert any(e["src"] == "a.py" and e["dst"] == "b.py" for e in payload["edges"])


def test_main_prints_the_one_line_summary_with_file_and_edge_counts_and_commit(tmp_path, capsys):
    blocks = [commit_block(f"c{i}", ["a.py", "b.py"]) for i in range(DEFAULT_MIN_SAMPLES)]
    git = make_git(log_text(*blocks), head="c" * 40)
    out_path = tmp_path / "graph.json"

    main(["--repo", str(tmp_path), "--out", str(out_path)], git_runner=git, now=FROZEN_NOW)

    out = capsys.readouterr().out
    assert "2 files" in out
    assert "2 edges" in out
    assert "c" * 40 in out


def test_main_returns_one_and_prints_to_stderr_on_a_git_error_and_writes_nothing(tmp_path, capsys):
    class RaisingGit:
        def __call__(self, args: Sequence[str]) -> str:
            raise GitError("fatal: not a git repository")

    out_path = tmp_path / "graph.json"

    exit_code = main(["--repo", str(tmp_path), "--out", str(out_path)], git_runner=RaisingGit())

    assert exit_code == 1
    assert not out_path.exists()
    assert "git error" in capsys.readouterr().err


def test_main_default_ref_is_head(tmp_path):
    git = make_git("")
    out_path = tmp_path / "graph.json"

    main(["--repo", str(tmp_path), "--out", str(out_path)], git_runner=git, now=FROZEN_NOW)

    log_call = next(c for c in git.calls if "log" in c)
    assert log_call[1] == DEFAULT_REF


def test_main_passes_a_custom_ref_through_to_every_git_call(tmp_path):
    git = make_git("")
    out_path = tmp_path / "graph.json"

    main(
        ["--repo", str(tmp_path), "--out", str(out_path), "--ref", "origin/main"],
        git_runner=git,
        now=FROZEN_NOW,
    )

    log_call = next(c for c in git.calls if "log" in c)
    rev_parse_call = next(c for c in git.calls if "rev-parse" in c)
    assert log_call[1] == "origin/main"
    assert rev_parse_call == ["rev-parse", "origin/main"]


def test_main_default_since_commits_matches_the_contract(tmp_path):
    git = make_git("")
    out_path = tmp_path / "graph.json"

    main(["--repo", str(tmp_path), "--out", str(out_path)], git_runner=git, now=FROZEN_NOW)

    log_call = next(c for c in git.calls if "log" in c)
    assert log_call[log_call.index("-n") + 1] == str(DEFAULT_SINCE_COMMITS)


def test_main_custom_min_samples_flag_lowers_the_floor(tmp_path):
    blocks = [commit_block(f"c{i}", ["a.py", "b.py"]) for i in range(2)]
    git = make_git(log_text(*blocks))
    out_path = tmp_path / "graph.json"

    main(
        ["--repo", str(tmp_path), "--out", str(out_path), "--min-samples", "2"],
        git_runner=git,
        now=FROZEN_NOW,
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert any(e["src"] == "a.py" and e["dst"] == "b.py" for e in payload["edges"])


def test_main_requires_repo_and_out_arguments():
    with pytest.raises(SystemExit):
        main([])


def test_main_falls_back_to_the_resolved_folder_name_for_a_relative_repo_path(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    git = make_git("", remote="not-a-url")
    out_path = tmp_path / "graph.json"

    main(["--repo", ".", "--out", str(out_path)], git_runner=git, now=FROZEN_NOW)

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["repo"] == tmp_path.name


def test_main_uses_the_default_runner_when_none_is_injected(monkeypatch, tmp_path):
    def fake_run(cmd, **kwargs):
        if "rev-parse" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="cafebabe" * 5 + "\n", stderr="")
        if "config" in cmd:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="error: no such remote\n")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/git")
    monkeypatch.setattr(subprocess, "run", fake_run)
    out_path = tmp_path / "graph.json"

    exit_code = main(["--repo", str(tmp_path), "--out", str(out_path)], now=FROZEN_NOW)

    assert exit_code == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["commit"] == "cafebabe" * 5
    assert payload["repo"] == tmp_path.name
