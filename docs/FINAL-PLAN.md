# Final Plan — Knowledge Graph to Finished Agent

**Date:** 2026-08-12
**Targets:** two large private repos — a TypeScript frontend and a Python backend.
One agent instance reviews either; nothing may assume a single language.
**Supersedes:** the phasing in IMPROVEMENT-PLAN.md §7. Design rationale there still applies.

---

## 0. Where the system stands today

Verified state at commit `a3f6339`, 359 tests passing:

| Working, proven live | Built, not proven | Missing |
|---|---|---|
| Sweep pipeline, both lanes | Inline comments (quota-blocked before a live run) | Ensemble (P2) — `ensemble.py` is a half-written orphan |
| Safety gates incl. two-layer idempotency | P1 context + P9 blast (gated on `repo_root`) | Feedback loop (P6) |
| Intent check (P8), rendered in output | Structured template (rendered live once) | GitHub Actions (Phase 4) |
| Live posting on the demo repo | P3 linters — **built, never wired** | Knowledge graph (all of it) |
| | | Scored eval harness (`score_demo.py`) |

Known scale problems for the target repos, found earlier and not yet fixed:
`gh pr list --limit 50` silently truncates; enrichment costs ~4 model calls/PR
(free-tier 429 at three PRs); polling re-lists everything every cycle.

---

## 1. Architecture decisions, fixed

These were argued earlier; recording them as settled so nothing relitigates them.

1. **Graph consumers see only the JSON schema** in `docs/KNOWLEDGE-GRAPH.md`.
   Producers are adapters. This is what makes every tier swappable.
2. **SCIP is first-pass, not a later tier**, because both target languages are
   dynamic. `scip-typescript` and `scip-python` are the two most mature indexers —
   the languages chose the tools for us. CodeGraph remains the fallback where SCIP
   fails to index.
3. **Co-change mining is ours** (`build_cochange.py`). Nothing else does it.
4. **No graph database.** SQLite/JSON until a measured bottleneck says otherwise.
5. **CodeQL is out** unless a GitHub Code Security license shows up. License-gated
   on private repos; do not design around it.
6. **Event-driven over polling** at this repo size. Actions is the delivery target;
   the local sweep stays as the dev/test harness.
7. **Graphs build in CI on merge to default, cached as artifacts.** Never in the
   review path. A stale graph is refused (commit ancestry check), degrading to
   no-graph with a visible note.

---

## 2. Phases

Ordered so each phase de-risks the next. Sonnet-lane subagents per module, same
file-ownership discipline as before; integration and review stay with the orchestrator.

### Phase A — Foundations made honest (small, do first)

The cheap items that block everything downstream or are currently lying about scale.

| # | Task | Detail |
|---|---|---|
| A1 | Pagination | `list_open_prs`/`list_merged_prs` paginate past 50. A truncated *listing* means PRs silently never reviewed — the worst failure shape this system can have |
| A2 | Wire P3 | `analysis.py` into `_build_payload`: run linters, inject `render()` + the "do not re-report" rule ids into the prompt, thread `corroboration` through |
| A3 | Resolve the orphan | Delete `ensemble.py` (P2 rebuilds clean in Phase D) |
| A4 | `score_demo.py` | Assert the demo answer key mechanically — 4/4 plus PR #6 silence — so quality regressions become test failures, not vibes |
| A5 | Multi-repo config | `[[repos]]` array replacing the single `[repo]` block: per-repo `owner/name`, `repo_root`, `graph_path`, language hints. One agent, two repos, per your requirement |

**Gate to leave A:** suite green, `score_demo.py` passes, both target repos listable
end-to-end (dry-run) without truncation.

### Phase B — Knowledge graph, production path

| # | Task | Detail |
|---|---|---|
| B1 | `prime_pr_review/graph.py` — the consumer | Load/validate schema; **refuse stale** (graph commit must be ancestor of PR base — shell `git merge-base --is-ancestor` via injected runner); k-hop neighborhood for changed symbols; co-change lookup; `render()` for the prompt. Missing/stale/malformed all degrade to a note |
| B2 | `scripts/build_cochange.py` | Tier 0 miner from the earlier dispatch brief: bulk-commit exclusion (~50 files), min-samples floor (5), directional weights, deterministic output, injected git runner. Salvage assertions from the deleted `test_build_graph.py` (recoverable at `ebf87d7`) |
| B3 | SCIP adapters | `adapters/scip.py`: parse a `scip snapshot`/index for `scip-typescript` + `scip-python` output → schema JSON. One adapter, two indexers |
| B4 | CodeGraph adapter | `adapters/codegraph.py`: shell `codegraph callers/impact --json` → schema. Fallback for whatever SCIP can't index |
| B5 | Merge tool | `scripts/merge_graphs.py`: concatenate edge sets, verify id conventions agree, stamp the youngest common commit |
| B6 | Rewire P1 + P9 | Call sites and blast radius consult the graph **when present**; `git grep` stays as explicit fallback with the degradation note. New finding class: co-change warnings ("these 2 files change together 87% of the time; this PR touches one") |

**Gate to leave B:** on a real checkout of the Python repo, blast radius resolves an
aliased import that `git grep` provably misses; co-change warning fires on a known
coupled pair; stale-graph refusal exercised by test and by hand.

### Phase C — Event-driven delivery (Actions)

| # | Task | Detail |
|---|---|---|
| C1 | Per-PR entry point | `run_sweep.py --pr N` — review exactly one PR, skip listing entirely. This is what an event trigger calls |
| C2 | Workflow | Per the killed agent's brief, unchanged: `pull_request_target` with `persist-credentials: false`, never execute PR code, least-privilege permissions, concurrency-cancel per PR, `workflow_dispatch` for manual runs. Diff is read as text only |
| C3 | Graph in CI | Second workflow on push-to-default: build SCIP index + co-change, merge, upload artifact. Review workflow downloads it; missing artifact = degraded review, visible note |
| C4 | Selective enrichment | By diff shape, to control the 4-calls/PR cost: blast radius only when signatures/exports change; intent only above a size threshold or when title/diff mismatch is plausible; docs-only PRs get the cheap pass. Budget knobs in config |

**Gate to leave C:** a PR opened on the demo repo gets its review from Actions with no
local machine involved; a doc-only PR consumes ≤1 model call.

### Phase D — Precision (rebuild of the killed work)

| # | Task | Detail |
|---|---|---|
| D1 | Ensemble (P2) | Rebuild per the original brief: N=3, match on (file, line-bucket, severity), keep at ≥2/3, confidence = observed agreement. Fixes the measured 95%-on-everything problem. `size=1` is the off switch — and the cost lever for busy days |
| D2 | Feedback loop (P6) | Rebuild per brief: 👎 reactions + dismissal phrases → rejection store → prompt guidance + suppression (suppressed findings returned, never dropped). The only mechanism that improves the bot over time |
| D3 | Calibration report | From `reviews/` front matter + rejections: acceptance rate by area/severity. Feeds `min_confidence` tuning with data instead of guesses. (P11's useful core; per-author stats stay off) |

### Phase E — Validation and go-live

| # | Task | Detail |
|---|---|---|
| E1 | Real-corpus eval | 20+ *merged* PRs from the actual target repos, replayed in dry-run. Read every review. This is the number that decides go-live — not the planted demo |
| E2 | Threshold tuning | Set `min_confidence`, ensemble size, budgets from E1's observed precision |
| E3 | Staged rollout | Backend repo first (Python = strongest graph support): dry-run 1 week → digest-only → inline comments on, `allow_request_changes` stays off → frontend repo repeats the ladder |
| E4 | Docs close-out | README, GITHUB-ACTIONS.md, KNOWLEDGE-GRAPH.md reconciled to what shipped |

**Go-live bar (E1):** zero fabricated findings (file/line that doesn't exist) — a
single one blocks; ≥60% of findings a maintainer would act on; PR-#6-style silence
on cosmetic PRs. Miss the bar → tune → re-run E1. Never go live on a missed bar.

---

## 3. What you do vs. what I do

**You (once, ~15 min):**
- `npm i -g @sourcegraph/scip-typescript` in the frontend repo, `pip install scip-python` for the backend; run each indexer once and confirm an index file appears
- Local checkouts of both repos on disk (paths → `[[repos]]` config)
- Repo names + `GITHUB_TOKEN` when you're ready to point at them; Actions secrets (`GEMINI_API_KEY`) when C lands
- The E3 go/no-go calls

**Me:** everything else, same delegation model — Sonnet agents build modules against
written contracts, I integrate, review, and keep the suite green.

## 4. Order of execution and why

**A → B → C → D → E.** A is cheap and two of its items (A1, A5) block big-repo use
entirely. B before C because the Actions workflow should ship already graph-aware
rather than be rewired later. D after C because ensemble triples model calls —
pointless to tune before selective enrichment (C4) exists to pay for it. E last and
strictly gated.

Single biggest risk, unchanged from day one: **false positives eroding trust.** Every
phase carries a piece of the mitigation — corroboration (A2), precise edges (B), scoped
delivery (C), agreement + memory (D), and a measured bar before anyone else sees a
comment (E).
