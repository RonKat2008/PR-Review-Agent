"""Co-change miner -- Tier 0 of the knowledge graph (docs/KNOWLEDGE-GRAPH.md).

Mines `git log --name-only` to find files that change together across commits
and emits ONLY `co_changes_with` edges: the temporal-coupling half of the
graph. Weight is directional -- commits touching both src and dst, divided by
commits touching src -- and every edge carries the sample size (commits
touching src) it was computed from, per the schema's "required" note on
`samples`.

Static structure (`imports`, `calls`, `defines`, `tests`, `inherits`) is
deliberately NOT this module's job. That half comes from CodeGraph
(tree-sitter + SQLite) or, once tree-sitter's syntactic resolution starts
producing false call edges, an SCIP indexer -- see docs/KNOWLEDGE-GRAPH.md's
four-tier architecture. Do not add an AST walker to this file.

All git access is funneled through a single injectable runner (`GitRunner`),
mirroring `prime_pr_review/github.py`'s `GhRunner` pattern, so tests never
shell out to a real repository.

    python scripts/build_cochange.py --repo <path> --out <file>
        [--ref REF] [--since-commits N] [--min-samples N]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import permutations
from pathlib import Path

GitRunner = Callable[[Sequence[str]], str]

GRAPH_VERSION = 1

# THIS MATTERS: target repos are mid-work on feature branches, so mining must
# be able to target the fetched origin default branch (e.g. "origin/main")
# instead of a dirty working tree's HEAD without touching that working tree.
DEFAULT_REF = "HEAD"

DEFAULT_SINCE_COMMITS = 500

# Below this many commits touching the source file, a weight is a coincidence
# -- 1.0 over 2 commits is noise. At or above it, it is a pattern worth a
# reviewer's attention.
DEFAULT_MIN_SAMPLES = 5

# A commit touching more files than this is a bulk reformat, a vendored
# dependency bump, or similar mechanical change. Counting it would couple
# every file it touches to every other file, drowning out real signal.
BULK_COMMIT_FILE_THRESHOLD = 50

GIT_TIMEOUT_SECONDS = 60


class GitError(RuntimeError):
    """A git invocation failed or returned something unusable."""


@dataclass(frozen=True)
class Node:
    """A `file` node. This module never produces symbol-level nodes -- those
    are CodeGraph/SCIP's job."""

    id: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "kind": "file"}


@dataclass(frozen=True)
class Edge:
    """A `co_changes_with` edge. `weight` and `samples` are always required
    for this edge kind, so they are plain fields rather than optional ones."""

    src: str
    dst: str
    weight: float
    samples: int

    def to_dict(self) -> dict[str, object]:
        return {
            "src": self.src,
            "dst": self.dst,
            "kind": "co_changes_with",
            "weight": self.weight,
            "samples": self.samples,
        }


@dataclass(frozen=True)
class Graph:
    version: int
    repo: str
    commit: str
    built_at: str
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "repo": self.repo,
            "commit": self.commit,
            "built_at": self.built_at,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }


@dataclass(frozen=True)
class _Commit:
    """One parsed block of `git log --name-only` output."""

    sha: str
    files: tuple[str, ...]


def _normalize_path(path: str) -> str:
    """Forward slashes always -- node ids must match across platforms."""
    return path.replace("\\", "/")


def _parse_log(raw: str) -> tuple[tuple[_Commit, ...], int]:
    """Parse `git log --name-only --pretty=format:%H` output into commits.

    Output is blank-line-separated blocks: a block's first line is the
    commit sha, the remaining lines are changed file paths. Git's `%H` is
    always a single token with no embedded whitespace, so a "sha" line that
    contains whitespace means the block structure broke (e.g. two records
    ran together) -- that block is dropped and counted as malformed rather
    than mined under a garbage id. Never raises: one corrupt commit must
    never take down the whole mine.

    Returns (commits, malformed_count).
    """
    commits: list[_Commit] = []
    malformed = 0
    for block in re.split(r"\n\s*\n", raw.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        sha, *files = lines
        if len(sha.split()) != 1:
            malformed += 1
            continue
        commits.append(_Commit(sha=sha, files=tuple(_normalize_path(f) for f in files)))
    return tuple(commits), malformed


def build_cochange_edges(
    git: GitRunner,
    ref: str,
    since_commits: int,
    min_samples: int,
) -> tuple[Edge, ...]:
    """Mine `git log` for files that change together.

    Directional: weight(src->dst) = commits touching both / commits touching
    src. The two directions have different denominators, so src->dst and
    dst->src can (and usually do) carry different weights; both are emitted
    independently, each filtered by its own `samples` against `min_samples`.
    """
    raw = git(["log", ref, "--name-only", "--pretty=format:%H", "-n", str(since_commits)])
    commits, _malformed = _parse_log(raw)

    touches: dict[str, int] = defaultdict(int)
    co_touches: dict[tuple[str, str], int] = defaultdict(int)
    for commit in commits:
        files = sorted(set(commit.files))
        if len(files) > BULK_COMMIT_FILE_THRESHOLD:
            continue  # bulk commit: treat as if it never happened
        for path in files:
            touches[path] += 1
        for src, dst in permutations(files, 2):
            co_touches[(src, dst)] += 1

    edges = [
        Edge(src=src, dst=dst, weight=together / touches[src], samples=touches[src])
        for (src, dst), together in co_touches.items()
        if touches[src] >= min_samples
    ]
    return tuple(sorted(edges, key=lambda edge: (edge.src, edge.dst)))


def _resolve_commit(git: GitRunner, ref: str) -> str:
    """Full sha for `ref`. Load-bearing: the reviewer refuses a graph whose
    `commit` is not an ancestor of the PR base (docs/KNOWLEDGE-GRAPH.md)."""
    return git(["rev-parse", ref]).strip()


def _parse_remote_slug(url: str) -> str | None:
    """`owner/name` from an https/ssh git remote URL, any host. None when the
    URL doesn't parse into at least two path segments to take."""
    url = url.strip()
    if "://" in url:
        path = url.split("://", 1)[1].partition("/")[2]
    elif "@" in url and ":" in url:
        path = url.split(":", 1)[1]
    else:
        return None

    if path.endswith(".git"):
        path = path[: -len(".git")]
    parts = [part for part in path.strip("/").split("/") if part]
    return "/".join(parts[-2:]) if len(parts) >= 2 else None


def _resolve_repo_slug(repo: Path, git: GitRunner) -> str:
    """`owner/name` parsed from the origin remote; the checkout's own folder
    name when there is no remote or the URL doesn't parse, so the graph
    always carries a usable repo id."""
    try:
        raw_url = git(["config", "--get", "remote.origin.url"])
    except GitError:
        return repo.name
    return _parse_remote_slug(raw_url) or repo.name


def build_graph(
    repo: Path,
    git: GitRunner,
    *,
    ref: str = DEFAULT_REF,
    since_commits: int = DEFAULT_SINCE_COMMITS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    now: datetime | None = None,
) -> Graph:
    """Assemble the full graph: mine co-change edges, emit a `file` node for
    every path any kept edge references, and stamp repo/commit/built_at."""
    edges = build_cochange_edges(git, ref, since_commits, min_samples)
    node_ids = sorted({edge.src for edge in edges} | {edge.dst for edge in edges})
    nodes = tuple(Node(id=node_id) for node_id in node_ids)

    resolved_now = now or datetime.now(timezone.utc)
    return Graph(
        version=GRAPH_VERSION,
        repo=_resolve_repo_slug(repo, git),
        commit=_resolve_commit(git, ref),
        built_at=resolved_now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        nodes=nodes,
        edges=edges,
    )


def write_graph_atomic(path: Path, graph: Graph) -> None:
    """Write tmp-then-replace, mirroring prime_pr_review/state.py, so a
    reader never observes a half-written graph."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(graph.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def default_runner(repo: Path) -> GitRunner:
    """A `GitRunner` that shells out to the real `git` CLI with `cwd=repo`."""

    def run(args: Sequence[str]) -> str:
        if shutil.which("git") is None:
            raise GitError("The `git` CLI is not installed or not on PATH.")
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=repo,
                capture_output=True,
                text=True,
                # Locale-independent: file paths in git log are UTF-8.
                encoding="utf-8",
                errors="replace",
                timeout=GIT_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise GitError(f"`git {' '.join(args)}` timed out after {GIT_TIMEOUT_SECONDS}s") from exc

        if result.returncode != 0:
            raise GitError(
                f"`git {' '.join(args)}` failed with exit code {result.returncode}: "
                f"{result.stderr.strip()}"
            )
        return result.stdout

    return run


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine co-change edges from git history (Tier 0 of the knowledge graph)."
    )
    parser.add_argument("--repo", required=True, help="Path to the git checkout to mine.")
    parser.add_argument("--out", required=True, help="Path to write the graph JSON to.")
    parser.add_argument(
        "--ref",
        default=DEFAULT_REF,
        help="Git ref to mine (default: HEAD). Use e.g. origin/main to mine a fetched "
        "default branch without touching the working tree.",
    )
    parser.add_argument("--since-commits", type=int, default=DEFAULT_SINCE_COMMITS)
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    git_runner: GitRunner | None = None,
    now: datetime | None = None,
) -> int:
    args = _parse_args(argv)
    repo = Path(args.repo).resolve()
    git = git_runner or default_runner(repo)

    try:
        graph = build_graph(
            repo,
            git,
            ref=args.ref,
            since_commits=args.since_commits,
            min_samples=args.min_samples,
            now=now,
        )
    except GitError as exc:
        print(f"git error: {exc}", file=sys.stderr)
        return 1

    write_graph_atomic(Path(args.out), graph)
    print(f"{len(graph.nodes)} files, {len(graph.edges)} edges, commit {graph.commit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
