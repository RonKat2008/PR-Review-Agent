"""Knowledge-graph consumer: load a prebuilt repo graph and answer the queries
the reviewer needs from it.

This module never builds a graph — see docs/KNOWLEDGE-GRAPH.md for the schema
and how one gets produced. It only reads one JSON file in that documented
shape and answers the questions the reviewer actually asks: who calls this,
who imports this, what's nearby (`neighborhood`, a capped k-hop walk over
every edge kind), and what tends to change alongside this file but didn't in
this diff (`co_change_warnings` — the one finding class no static analysis can
reach, because it lives in git history rather than in any single file).

Every failure mode here degrades rather than crashes, with one deliberate
exception. A missing or malformed graph file degrades to "no graph" — review
proceeds exactly as if `graph_path` was never configured. A *stale* graph does
not degrade: `verify_fresh` refuses it outright. A stale graph does not fail
loudly, it succeeds convincingly — it reports call sites and co-change
partners with total confidence, about a commit that is no longer the one
under review, which is worse than reporting nothing. `load_for_review` is the
one call site the sweep needs and folds every failure mode — unreadable file,
malformed file, stale file — into the same `(None, reason)` shape, so a caller
never has to tell "never configured" apart from "refused" to stay safe.

Every query is a pure function over an already-parsed `KnowledgeGraph` and
returns sorted, deterministic output: same graph and inputs, same answer,
every time.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .context import GitError, GitRunner


def strict_runner(repo_root: Path | str) -> GitRunner:
    """A git runner for the ancestry check that raises on ANY nonzero exit.

    Do not reuse `context.default_git_runner` here: it deliberately treats exit 1
    as success, because for `git grep` exit 1 means "no matches". For
    `merge-base --is-ancestor`, exit 1 means "NOT an ancestor" and must refuse
    the graph — sharing the lenient runner would let a stale graph sail through
    the one check built to stop it.
    """
    root = Path(repo_root)

    def run(args: Sequence[str]) -> str:
        result = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=False,
        )
        if result.returncode != 0:
            raise GitError(
                f"git {' '.join(args)} exited {result.returncode}: "
                f"{result.stderr.strip()[:200]}"
            )
        return result.stdout

    return run

SUPPORTED_VERSION = 1

CALLS_EDGE = "calls"
IMPORTS_EDGE = "imports"
CO_CHANGE_EDGE = "co_changes_with"

DEFAULT_NEIGHBORHOOD_K = 2
DEFAULT_NEIGHBORHOOD_CAP = 200
DEFAULT_PARTNER_MIN_WEIGHT = 0.5
DEFAULT_PARTNER_MIN_SAMPLES = 5
WARNING_MIN_WEIGHT = 0.6
WARNING_MIN_SAMPLES = 8


class GraphError(RuntimeError):
    """A graph file is missing, unreadable, or fails schema validation."""


@dataclass(frozen=True)
class Node:
    id: str
    kind: str
    file: str = ""
    line: int | None = None


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    kind: str
    weight: float | None = None
    samples: int | None = None


@dataclass(frozen=True)
class KnowledgeGraph:
    version: int
    repo: str
    commit: str
    built_at: str
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    # co_changes_with edges dropped for missing/invalid weight or samples.
    # Counted rather than silently absorbed, so a suspiciously thin co-change
    # section is distinguishable from a graph that never had one.
    skipped_edges: int = 0


@dataclass(frozen=True)
class CoChangeWarning:
    """`file` and `partner` change together historically; this diff touches only `file`."""

    file: str
    partner: str
    weight: float
    samples: int


# --- load_graph ----------------------------------------------------------------


def load_graph(path: Path | str) -> KnowledgeGraph:
    """Read and validate one graph file. Raises `GraphError` naming what is wrong."""
    graph_path = Path(path)
    if not graph_path.is_file():
        raise GraphError(f"Graph file not found: {graph_path}")

    try:
        raw_text = graph_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise GraphError(f"Could not read graph file {graph_path}: {exc}") from exc

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise GraphError(f"Graph file {graph_path} is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise GraphError(
            f"Graph file {graph_path} must contain a JSON object, got {type(payload).__name__}"
        )

    return _build_graph(payload)


def _build_graph(payload: dict) -> KnowledgeGraph:
    version = payload.get("version")
    if version != SUPPORTED_VERSION:
        raise GraphError(f"Graph 'version' must be {SUPPORTED_VERSION}, got {version!r}")

    commit = payload.get("commit")
    if not isinstance(commit, str) or not commit.strip():
        raise GraphError("Graph 'commit' must be a non-empty string")

    raw_nodes = payload.get("nodes", [])
    if not isinstance(raw_nodes, list):
        raise GraphError("Graph 'nodes' must be a list")
    nodes = tuple(_parse_node(item, i) for i, item in enumerate(raw_nodes))

    raw_edges = payload.get("edges", [])
    if not isinstance(raw_edges, list):
        raise GraphError("Graph 'edges' must be a list")
    edges, skipped = _parse_edges(raw_edges)

    return KnowledgeGraph(
        version=version,
        repo=str(payload.get("repo", "")),
        commit=commit,
        built_at=str(payload.get("built_at", "")),
        nodes=nodes,
        edges=edges,
        skipped_edges=skipped,
    )


def _parse_node(raw: object, index: int) -> Node:
    if not isinstance(raw, dict):
        raise GraphError(f"nodes[{index}] must be an object, got {type(raw).__name__}")

    node_id = raw.get("id")
    if not isinstance(node_id, str) or not node_id.strip():
        raise GraphError(f"nodes[{index}].id must be a non-empty string")

    kind = raw.get("kind")
    if not isinstance(kind, str) or not kind.strip():
        raise GraphError(f"nodes[{index}].kind must be a non-empty string")

    file = raw.get("file")
    line = raw.get("line")
    return Node(
        id=node_id,
        kind=kind,
        file=file if isinstance(file, str) else "",
        line=line if _is_count(line) else None,
    )


def _parse_edges(raw_edges: list) -> tuple[tuple[Edge, ...], int]:
    edges: list[Edge] = []
    skipped = 0
    for index, item in enumerate(raw_edges):
        edge, was_skipped = _parse_edge(item, index)
        if was_skipped:
            skipped += 1
        elif edge is not None:
            edges.append(edge)
    return tuple(edges), skipped


def _parse_edge(raw: object, index: int) -> tuple[Edge | None, bool]:
    """Parse one edge. `(None, True)` means skipped, not fatal.

    Only a `co_changes_with` edge missing or misshapen weight/samples is
    skipped rather than fatal — src/dst/kind are required for any edge, known
    or unknown, since without them the edge cannot be joined to anything.
    """
    if not isinstance(raw, dict):
        raise GraphError(f"edges[{index}] must be an object, got {type(raw).__name__}")

    src = raw.get("src")
    if not isinstance(src, str) or not src.strip():
        raise GraphError(f"edges[{index}].src must be a non-empty string")

    dst = raw.get("dst")
    if not isinstance(dst, str) or not dst.strip():
        raise GraphError(f"edges[{index}].dst must be a non-empty string")

    kind = raw.get("kind")
    if not isinstance(kind, str) or not kind.strip():
        raise GraphError(f"edges[{index}].kind must be a non-empty string")

    if kind != CO_CHANGE_EDGE:
        # Unknown kinds are accepted, not specially interpreted — they still
        # participate in `neighborhood`, just not in any kind-specific query.
        return Edge(src=src, dst=dst, kind=kind), False

    weight, samples = raw.get("weight"), raw.get("samples")
    if not _is_number(weight) or not _is_count(samples):
        return None, True
    return Edge(src=src, dst=dst, kind=kind, weight=float(weight), samples=int(samples)), False


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


# --- verify_fresh / load_for_review ---------------------------------------------


def verify_fresh(
    graph: KnowledgeGraph,
    base_sha: str,
    git_runner: GitRunner,
    repo_root: Path | str,
) -> str | None:
    """`None` when `graph` is safe to use against a PR based on `base_sha`.

    Otherwise a human-readable refusal reason. Runs
    `git merge-base --is-ancestor <graph.commit> <base_sha>` through the
    injected runner, which is expected to raise `GitError` on any nonzero
    exit — both "not an ancestor" and "unknown sha" collapse to the same
    refusal here, since either way the graph cannot be trusted for this PR.
    """
    try:
        git_runner(
            ["-C", str(repo_root), "merge-base", "--is-ancestor", graph.commit, base_sha]
        )
    except GitError as exc:
        return (
            f"graph commit {graph.commit[:12]} is not a verified ancestor of "
            f"base {base_sha[:12]}: {exc}"
        )
    return None


def load_for_review(
    path: Path | str,
    base_sha: str,
    git_runner: GitRunner,
    repo_root: Path | str,
) -> tuple[KnowledgeGraph | None, str]:
    """The one call site the sweep uses to obtain a usable graph.

    Never raises. A missing file, a malformed file, and a stale graph all
    degrade to `(None, reason)` — the same shape a caller sees when no graph
    is configured at all. Success returns `(graph, "")`.
    """
    try:
        graph = load_graph(path)
    except GraphError as exc:
        return None, f"graph unusable: {exc}"

    try:
        refusal = verify_fresh(graph, base_sha, git_runner, repo_root)
    except Exception as exc:  # noqa: BLE001 - this call site must never raise
        return None, f"graph freshness check failed: {exc}"

    if refusal is not None:
        return None, f"graph stale: {refusal}"
    return graph, ""


# --- queries ---------------------------------------------------------------------


def activity_counts(
    graph: KnowledgeGraph,
    diff_files: Sequence[str],
    changed_symbols: Sequence[str],
) -> tuple[int, int]:
    """(co-change warnings, caller edges) this graph contributes for one diff.

    Observability for the sweep: 'the graph loaded' and 'the graph actually said
    something' are different facts, and only the second proves the feature is
    earning its place on a given PR. Counts mirror what `render` would inject.
    """
    warnings = len(co_change_warnings(graph, diff_files))
    callers = sum(len(callers_of(graph, symbol)) for symbol in changed_symbols)
    return warnings, callers


def callers_of(graph: KnowledgeGraph, node_id: str) -> tuple[str, ...]:
    """Sources of every `calls` edge into `node_id`, sorted."""
    return tuple(
        sorted({e.src for e in graph.edges if e.kind == CALLS_EDGE and e.dst == node_id})
    )


def importers_of(graph: KnowledgeGraph, file: str) -> tuple[str, ...]:
    """Sources of every `imports` edge into `file`, sorted."""
    return tuple(
        sorted({e.src for e in graph.edges if e.kind == IMPORTS_EDGE and e.dst == file})
    )


def neighborhood(
    graph: KnowledgeGraph,
    seed_ids: Sequence[str],
    k: int = DEFAULT_NEIGHBORHOOD_K,
    cap: int = DEFAULT_NEIGHBORHOOD_CAP,
) -> tuple[str, ...]:
    """Node ids reachable from any seed within `k` hops, over edges of every
    kind and traversed in both directions, capped at `cap` total nodes
    (including the seeds themselves).

    Traversal proceeds in sorted order at every hop, so which nodes survive a
    cap cut is deterministic rather than dependent on input or dict ordering.
    """
    adjacency = _build_adjacency(graph)
    seen: set[str] = set()
    visited: list[str] = []
    frontier = sorted(set(seed_ids))
    _extend(frontier, seen, visited, cap)

    for _ in range(k):
        if len(visited) >= cap:
            break
        candidates = {node for current in frontier for node in adjacency.get(current, ())}
        frontier = sorted(candidates - seen)
        _extend(frontier, seen, visited, cap)

    return tuple(sorted(visited))


def _extend(frontier: list[str], seen: set[str], visited: list[str], cap: int) -> None:
    """Append `frontier` nodes to `visited`/`seen`, in order, until `cap` is hit."""
    for node in frontier:
        if len(visited) >= cap:
            return
        if node not in seen:
            seen.add(node)
            visited.append(node)


def _build_adjacency(graph: KnowledgeGraph) -> dict[str, tuple[str, ...]]:
    """Undirected adjacency over every edge kind, for neighborhood BFS."""
    raw: dict[str, set[str]] = {}
    for edge in graph.edges:
        raw.setdefault(edge.src, set()).add(edge.dst)
        raw.setdefault(edge.dst, set()).add(edge.src)
    return {node: tuple(sorted(neighbors)) for node, neighbors in raw.items()}


def co_change_partners(
    graph: KnowledgeGraph,
    file: str,
    min_weight: float = DEFAULT_PARTNER_MIN_WEIGHT,
    min_samples: int = DEFAULT_PARTNER_MIN_SAMPLES,
) -> tuple[tuple[str, float, int], ...]:
    """Files historically changed alongside `file`, strong enough to matter.

    `co_changes_with` edges are checked in either direction: weight is the
    fraction of commits touching `src` that also touched `dst`, which is not
    symmetric in general, so the edge naming `file` as `src` carries the
    fraction that matters when `file` is the one being changed; an edge
    naming it `dst` is used as the next-best signal when that is all a
    producer recorded.
    """
    partners: list[tuple[str, float, int]] = []
    for edge in graph.edges:
        if edge.kind != CO_CHANGE_EDGE or edge.weight is None or edge.samples is None:
            continue
        if edge.src == file:
            partner = edge.dst
        elif edge.dst == file:
            partner = edge.src
        else:
            continue
        if edge.weight >= min_weight and edge.samples >= min_samples:
            partners.append((partner, edge.weight, edge.samples))

    return tuple(sorted(partners, key=lambda p: (-p[1], p[0])))


def co_change_warnings(
    graph: KnowledgeGraph,
    diff_files: Sequence[str],
    min_weight: float = WARNING_MIN_WEIGHT,
    min_samples: int = WARNING_MIN_SAMPLES,
) -> tuple[CoChangeWarning, ...]:
    """Co-change partners of each diff file that this diff did NOT touch.

    "`parser.py` and `grammar.toml` changed together in 14 of the last 16
    commits. This PR changes `parser.py` alone." `(a, b)` and `(b, a)` collapse
    to a single warning when both directions independently qualify.
    """
    in_diff = frozenset(diff_files)
    seen_pairs: set[frozenset[str]] = set()
    warnings: list[CoChangeWarning] = []

    for file in sorted(in_diff):
        for partner, weight, samples in co_change_partners(graph, file, min_weight, min_samples):
            if partner in in_diff:
                continue
            pair = frozenset({file, partner})
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            warnings.append(CoChangeWarning(file=file, partner=partner, weight=weight, samples=samples))

    return tuple(sorted(warnings, key=lambda w: (-w.weight, w.file, w.partner)))


# --- render ------------------------------------------------------------------------


def render(
    graph: KnowledgeGraph,
    diff_files: Sequence[str],
    changed_symbols: Sequence[str],
) -> str:
    """Compact markdown summary of the graph's relevance to this diff.

    Every section that had something to check states its count explicitly,
    even when the count is zero — "0 caller(s) found for 3 symbol(s) checked"
    is what a check that ran and found nothing looks like. A section with
    nothing to check at all (no diff files, no changed symbols) is omitted
    outright, so the two are never visually identical.
    """
    sections = [
        _render_co_change_section(graph, diff_files),
        _render_callers_section(graph, changed_symbols),
        _render_neighborhood_section(graph, changed_symbols),
    ]
    rendered = [s for s in sections if s is not None]
    if not rendered:
        return ""
    return "## Knowledge graph\n\n" + "\n".join(rendered)


def _render_co_change_section(graph: KnowledgeGraph, diff_files: Sequence[str]) -> str | None:
    files = sorted(set(diff_files))
    if not files:
        return None

    warnings = co_change_warnings(graph, files)
    lines = [
        "### Co-change warnings",
        "",
        f"{len(warnings)} warning(s) found for {len(files)} file(s) checked.",
    ]
    if warnings:
        lines.append("")
        for w in warnings:
            commits = round(w.weight * w.samples)
            lines.append(
                f"- ⚠️ `{w.file}` and `{w.partner}` changed together in "
                f"**{commits} of the last {w.samples}** commits ({w.weight:.0%}); "
                f"this PR touches only `{w.file}`."
            )
    lines.append("")
    return "\n".join(lines)


def _render_callers_section(graph: KnowledgeGraph, changed_symbols: Sequence[str]) -> str | None:
    symbols = sorted(set(changed_symbols))
    if not symbols:
        return None

    results = {symbol: callers_of(graph, symbol) for symbol in symbols}
    total = sum(len(callers) for callers in results.values())
    lines = [
        "### Callers of changed symbols",
        "",
        f"{total} caller(s) found for {len(symbols)} symbol(s) checked.",
    ]
    if total:
        lines.append("")
        for symbol in symbols:
            for caller in results[symbol]:
                lines.append(f"- `{caller}` calls `{symbol}`")
    lines.append("")
    return "\n".join(lines)


def _render_neighborhood_section(graph: KnowledgeGraph, changed_symbols: Sequence[str]) -> str | None:
    seeds = sorted(set(changed_symbols))
    if not seeds:
        return None

    nodes = neighborhood(graph, seeds)
    lines = [
        "### Neighborhood",
        "",
        f"{len(nodes)} node(s) found within {DEFAULT_NEIGHBORHOOD_K} hops of "
        f"{len(seeds)} seed(s) checked.",
    ]
    if nodes:
        lines.append("")
        lines.append(", ".join(f"`{n}`" for n in nodes))
    lines.append("")
    return "\n".join(lines)
