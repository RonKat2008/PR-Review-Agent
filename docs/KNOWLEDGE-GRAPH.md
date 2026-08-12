# Knowledge Graph — format and how to produce one

The reviewer can consume a graph of the repository under review. This document is the
contract: produce a file in this shape by any means you like, point `config.toml` at it,
and the agent will use it.

```toml
[review]
graph_path = "path/to/graph.json"
```

---

## Why a graph at all

`git grep` matches text. Search for `save` and you hit `save`, `autosave`, `saveDraft`,
and a comment mentioning saving. A graph answers structural questions directly: what
calls this, what imports this, what tests this.

And it answers one question nothing else can — see **co-change** below.

---

## Format

A single JSON file. Version 1.

```jsonc
{
  "version": 1,
  "repo": "owner/name",
  "commit": "a1b2c3d4...",          // the commit this graph reflects — REQUIRED
  "built_at": "2026-08-11T04:00:00Z",
  "nodes": [
    { "id": "shop/orders.py", "kind": "file" },
    { "id": "shop/orders.py::total_price",
      "kind": "function", "file": "shop/orders.py", "line": 5 }
  ],
  "edges": [
    { "src": "shop/invoice.py::build", "dst": "shop/orders.py::total_price",
      "kind": "calls" },
    { "src": "shop/parser.py", "dst": "grammar.toml",
      "kind": "co_changes_with", "weight": 0.875, "samples": 16 }
  ]
}
```

### Node ids

| Kind | `id` convention |
|---|---|
| `file` | repo-relative path, forward slashes: `shop/orders.py` |
| `function`, `class`, `method` | `path::symbol` — `shop/orders.py::total_price` |
| `test` | same as file or function, with `kind: "test"` |

Forward slashes always, even on Windows. Ids must be stable across builds — they are how
the reviewer joins the graph to a diff.

### Edge kinds

| Kind | Meaning | Source |
|---|---|---|
| `imports` | src imports dst | AST |
| `calls` | src calls dst | AST |
| `inherits` | src subclasses dst | AST |
| `defines` | file defines symbol | AST |
| `tests` | src is a test for dst | AST + naming |
| `co_changes_with` | changed together historically | **git history** |

`co_changes_with` carries two extra fields:
- `weight` — 0.0–1.0, fraction of commits touching src that also touched dst
- `samples` — how many commits that fraction is based on. **Required.** A weight of 1.0
  over 2 commits is noise; over 40 it is a rule.

Unknown edge kinds are ignored rather than rejected, so you can add your own.

### The `commit` field is load-bearing

The agent refuses a graph whose `commit` is not an ancestor of the PR's base. A stale
graph is worse than no graph: it reports call sites that no longer exist, with total
confidence. Rebuild it in CI, or accept that it degrades to no-graph.

---

## The co-change edge is the one that earns this

Static analysis finds coupling the code *declares*. Git history finds coupling the code
*hides*:

- the config file that must change with the parser
- the schema and its migration
- the fixture that goes stale
- the doc nobody updates

That produces a finding class no analyzer can reach:

> ⚠️ `parser.py` and `grammar.toml` changed together in **14 of the last 16** commits.
> This PR changes `parser.py` alone.

That is frequently a real bug, and reading the diff would never reveal it. It is also
the cheapest part of the graph to build — pure `git log`, no language parsing, works on
any repo in any language.

---

## Architecture — four tiers, adopted in order

The reviewer consumes only the JSON schema above. Every producer gets an adapter. That
single constraint is what makes the tiers below independently adoptable: you can start
at Tier 1 and add Tier 2 later without touching any review logic.

| Tier | Layer | Tool | Adopt when |
|---|---|---|---|
| **0** | Temporal coupling | ours (`build_cochange.py`) | **now** — nothing else does it, zero dependencies |
| **1** | Syntactic structure | CodeGraph (tree-sitter + SQLite) | **now** — 20+ languages, no setup |
| **2** | Semantic symbols | SCIP indexers | when tree-sitter resolution starts producing false call edges |
| **3** | Data flow / taint | CodeQL | only if licensed — see below |

### Why Tier 2 exists

Tree-sitter is **syntactic**: it parses reliably but cannot always resolve *which*
`save` a call refers to across module boundaries. SCIP does **semantic** resolution.

That distinction is unusually important here. This reviewer's core claim is "this PR
breaks these four callers." If the call edges are guesses, the output is confidently
wrong — the exact failure mode the entire plan is built to avoid. Precision is the
product, so Tier 2 is a real upgrade rather than polish.

SCIP indexer maturity varies by language; `scip-python`, `scip-typescript`, and
`scip-java` are the solid ones.

### Tier 3 is license-gated — check before designing around it

CodeQL would upgrade the SQL-injection finding from *pattern match* to *proven taint
path*. But:

> The CodeQL CLI is free on **public** repositories. Private repositories require a
> GitHub Code Security license; analyzing closed-source code requires a separate
> commercial license.

For a private company repo that is a hard gate. Also, CodeQL database builds take
minutes to hours, so even when licensed it belongs on security-relevant diffs only,
never on every sweep.

### What we are deliberately NOT doing

**A graph database.** SQLite handles repo-scale graphs comfortably and CodeGraph already
demonstrates that shape. Neo4j or similar buys query expressiveness we have no measured
bottleneck for, at the cost of an operational dependency. Revisit when there is a
profile showing a problem, not before.

---

## How to produce one — don't hand-roll it

The graph has two halves, and **no single tool does both**. Use a mature tool for the
static half; the temporal half is ours because nothing covers it.

### Half 1 — static structure: use CodeGraph

[CodeGraph](https://github.com/colbymchenry/codegraph) — MIT, 100% local, 20+ languages,
built explicitly for Claude Code and similar agents.

```bash
npm i -g @colbymchenry/codegraph      # or the standalone installer, no Node needed
codegraph init /path/to/checkout      # builds .codegraph/codegraph.db
codegraph sync                        # incremental update
```

It ships exactly the queries this reviewer needs, with `--json` on each:

| Command | Replaces |
|---|---|
| `codegraph callers <symbol>` | the `git grep` in P1 — precise, no over-matching |
| `codegraph callees <symbol>` | forward dependency traversal |
| `codegraph impact <symbol>` | **most of P9 blast radius** |
| `codegraph explore <query>` | context retrieval for the prompt |

Storage is SQLite at `.codegraph/codegraph.db`. Language coverage includes TS/JS,
Python, Go, Rust, Java, C#, PHP, Ruby, C/C++, Swift, Kotlin, Scala, Dart, Solidity, and
more — far past what a hand-rolled Python `ast` walker would ever reach.

**It does not track git history, commit metadata, or co-change coupling.** Hence half 2.

### Half 2 — temporal coupling: mine it from git

This is the half that produces findings nothing static can, and CodeGraph explicitly
does not cover it. Two options:

**Bundled miner** (no dependencies, recommended):
```bash
python scripts/build_cochange.py --repo /path/to/checkout --out cochange.json
```

**[code-maat](https://github.com/adamtornhill/code-maat)** — Adam Tornhill's tool from
*Your Code as a Crime Scene*, the original implementation of logical-coupling analysis.
More analyses than we need (hotspots, code age, sum-of-coupling) and battle-tested, but
it is a Clojure/JVM tool, so it needs Java installed. Outputs CSV:

```bash
git log --pretty=format:'[%h] %an %ad %s' --date=short --name-only > evo.log
maat -l evo.log -c git -a coupling > coupling.csv
```

Use code-maat if you already have a JVM and want the wider analysis suite; use the
bundled miner if you want zero install.

### Half 3 — things no tool can infer

The schema is small enough to hand-write, which is the point. Use it to encode coupling
you *know about* and no analyzer can see: "these two services share a wire format",
"this constant must match that config".

### Merging

Concatenate `edges` from each source into one file. Ids are the join key, so keep path
conventions identical (repo-relative, forward slashes).

### In CI

Cheapest steady state: build on merge to the default branch, upload as an artifact,
download it in the review workflow. The initial index is the expensive part; `codegraph
sync` and incremental co-change updates are not.

---

## What the agent does with it

| Capability | Without graph | With graph |
|---|---|---|
| Call sites (P1) | `git grep` on the name — over-matches | real `calls` edges |
| Blast radius (P9) | grep hits, noisy | precise callers, k-hop |
| **Co-change** | not possible | ⚠️ findings, as above |
| Reviewer suggestion | — | who has context on this file |

Missing graph, malformed graph, and stale graph all degrade to current behavior and are
reported in the sweep as a note. The graph is an enhancement; nothing depends on it.
