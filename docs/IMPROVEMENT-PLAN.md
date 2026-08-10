# Improvement Plan — Automated PR Review

**Status:** proposed, awaiting confirmation
**Date:** 2026-08-09
**Baseline:** `9936f29` — 131 tests, 91% coverage, 4/4 on the demo answer key

---

## 1. Requirements restatement

1. Research prime-agent deeply and identify capabilities worth adopting
2. Propose improvements that materially raise review quality, with implementation detail
3. Design a structured PR review template that includes actionable code-improvement suggestions
4. Connect the system to GitHub as a bot

---

## 2. Where we actually are

| Working | Not working / untested |
|---|---|
| Selection → dedupe → diff filter → review → gates → sink | prime-agent not in the loop at all |
| 4/4 on planted defects, correct silence on clean PR | `reviewers.py` at 0% coverage |
| Dry-run, budget, idempotency marker, self-exclusion | Live posting never executed |
| Error containment (4 model failures, sweep survived) | Merged lane never run |
| Head-SHA re-review semantics | Webhook disabled |

### Findings from the prime-agent research

| Finding | Source | Consequence |
|---|---|---|
| `rlm()` returns an admission handle, never the answer | `rlm.md` §2 | **`SKILL.md` is wrong today.** Results come via `agent_message.send(..., receiver_role="parent")` or files |
| Children can call `rlm()` themselves; depth configurable | `rlm.md` §2 | Multi-perspective review is native, not something to build |
| Python-backed skills expose typed callables into the kernel | `rlm.md` §3 | `prime_pr_review` can be imported directly as a skill package |
| Child registry survives compaction and restart | `rlm.md` §2 | Long sweeps are resumable |
| "It is a durable control environment, **not a security sandbox**" | `rlm.md` Trust Model | **Reviewing fork PRs means executing against untrusted content.** Hard constraint |

---

## 3. The single biggest quality limitation

**The reviewer sees only the diff.** Nothing else.

A diff in isolation cannot answer the questions that matter most:

- Did this change a function signature and leave callers broken?
- Does a helper already exist that this reimplements?
- Was there a test pinning the behavior this just changed?
- Is this `except:` swallowing an error the caller depends on?

The demo scored 4/4 because the planted defects were *locally visible*. Real regressions usually are not. Every proposal below is ranked by how much it closes this gap.

---

## 4. Proposed improvements

### P1 — Repository context injection ★ highest value

**Problem:** the model reasons about a diff with no surrounding code.

**Implementation** — new `prime_pr_review/context.py`:

| Signal | How | Budget |
|---|---|---|
| Full changed file | `gh api .../contents/{path}?ref={sha}` | ~40% |
| Call sites of changed symbols | Parse `def`/`class` from diff hunks, `git grep -n` each | ~25% |
| Sibling tests | Glob `test_*{stem}*`, `{stem}_test.*`, `tests/**/*{stem}*` | ~20% |
| Repo conventions | `CONTRIBUTING.md`, `CLAUDE.md`, `AGENTS.md`, lint config | ~15% |

Gemini Flash gives a 1M-token window, so this fits comfortably. Assemble into a `ReviewContext` dataclass; the reviewer signature becomes `reviewer(pr, context, lane)`.

**Effect:** enables the entire class of contract-break and missed-caller findings that are currently invisible.

### P2 — Ensemble agreement replaces self-reported confidence ★ fixes a known defect

**Problem:** the demo returned **95% confidence on all four PRs, including the empty verdict.** `min_confidence` is currently filtering nothing.

**Implementation:** run each PR through N=3 subagents at temperature ~0.4, then:

```
agreement = (subagents reporting this finding) / N
```

- Match findings on `(file, normalized_line_range, severity_bucket)`
- Keep findings with `agreement >= 2/3`
- `verdict.confidence` becomes the agreement ratio — an *observed* quantity, not a self-assessment

**Effect:** turns the confidence gate into a real precision mechanism. Cost is 3× per PR, which Flash pricing absorbs; make `ensemble_size` configurable and allow `1` to disable.

### P3 — Static analysis pre-pass ★ cheapest quality win

Run deterministic tooling before the model, and pass results in as grounding:

| Tool | Catches |
|---|---|
| `ruff` | Unused vars, shadowing, mutable defaults, bare except |
| `bandit` | Hardcoded secrets, `subprocess(shell=True)`, weak crypto |
| `semgrep` | Injection, path traversal, taint flows |
| `mypy` | Type contract breaks |

Two effects, both valuable: findings become **evidence** the model can cite instead of hallucinate, and the prompt gains a rule — *do not report what the linter already reports* — which removes the highest-volume category of noise.

### P4 — Inline comments with committable suggestions ★ answers the "improvement suggestions" ask

Stop posting one summary blob. Use GitHub's review API:

```
POST /repos/{owner}/{repo}/pulls/{n}/reviews
{
  "event": "COMMENT",
  "body": "<summary>",
  "comments": [
    {"path": "shop/orders.py", "line": 6, "side": "RIGHT", "body": "..."}
  ]
}
```

We already capture `file` and `line` on every `Finding` — the data is there, only delivery changes.

Then have the model emit a `suggestion` field, rendered as:

````
```suggestion
    return sum(item["price"] * item["qty"] for item in items)
```
````

GitHub renders that as a **one-click "Commit suggestion" button**. This is the difference between a bot that complains and a bot that fixes.

Requires a `line_end` for multi-line replacement and validation that the line exists in the diff — GitHub rejects comments on unchanged lines.

### P5 — Severity-driven review event

| Worst severity | Event | Effect |
|---|---|---|
| CRITICAL | `REQUEST_CHANGES` | Blocks merge if branch protection requires review |
| HIGH | `COMMENT` | Visible, non-blocking |
| MEDIUM / LOW | `COMMENT`, collapsed in `<details>` | Present, not noisy |
| none | `APPROVE` or silence | Configurable |

Gate `REQUEST_CHANGES` behind explicit config — an automated block is a strong action.

### P6 — False-positive feedback loop

Read reactions on prior bot comments. A 👎 or a reply containing a configured phrase marks the finding rejected; persist to `state/rejections.json` keyed by `(file, normalized_claim)`. Inject recent rejections into subsequent prompts as *"previously rejected by maintainers — do not re-report unless materially different."*

Cheap to build, and it is the only mechanism here that improves the bot over time.

### P8 — Intent alignment: does the code do what the PR claims? ★ highest value

**Problem:** nothing currently checks the diff against the PR's *stated purpose*. A PR
titled "fix typo in README" could change an auth check and the reviewer would evaluate
the auth change on its own merits without ever asking why it is in this PR.

**What it catches:**

| Category | Example |
|---|---|
| Debug residue | `print()`, `console.log`, commented-out blocks |
| Scope creep | A 400-line refactor bundled into a one-line bugfix |
| Accidental commits | `.env`, `credentials.json`, IDE config, a stray `node_modules` |
| Contradiction | Title says "add validation", diff *removes* a check |
| **Hidden change** | A permission or dependency edit buried in a cosmetic PR |

The last one is a supply-chain attack pattern and is the reason this check earns a
CRITICAL severity path.

**Implementation — two passes, and the ordering is load-bearing:**

```
Pass 1  inputs: title, body, linked issue, commit messages, branch name
        NO DIFF
        output: {"intent": "...", "expected_files": [...], "out_of_scope": [...]}

Pass 2  inputs: the intent statement from pass 1, plus the diff
        output: per-hunk {serves_intent: bool, reason: str}
```

**Why the diff is withheld in pass 1:** if the model sees the title and the diff
together and is asked "do these match?", it rationalizes a match — it reads the diff and
constructs a story where the title covers it. Forcing it to commit to an interpretation
of intent *first*, then judge the diff against that fixed statement, is dramatically
harder to rationalize around. This is the difference between the check working and
being theater.

**New verdict field:**

```json
"scope": {
  "intent": "Fix the off-by-one in total_price",
  "aligned": false,
  "unrelated": [
    {"file": "shop/auth.py", "lines": "22-31", "severity": "HIGH",
     "claim": "Changes the session timeout. Unrelated to the stated fix.",
     "evidence": "PR title and body mention only total_price."}
  ]
}
```

Severity guidance for the prompt: `LOW` debug residue · `MEDIUM` scope creep ·
`HIGH` unrelated behavior change · `CRITICAL` unrelated change to auth, permissions,
crypto, dependencies, or CI config.

### P9 — Blast radius: what else does this break? ★ highest value

**Problem:** P1 gives the model call sites as context. That is passive — it may or may
not use them. P9 makes it an explicit, structured obligation with its own output.

**Implementation — deterministic first, model last:**

```
Step 1  (Python, no model)   extract changed symbols from diff hunks
                             -> functions, classes, constants, exported names
Step 2  (git grep, no model) find every reference outside the diff
                             -> {symbol: [file:line, ...]}
Step 3  (model)              for each (symbol, caller) pair, judge:
                             does this change break this caller?
```

Steps 1 and 2 must be deterministic. The model should never be *guessing* who the
callers are — it should be handed the list and asked only to judge each one. Guessed
call sites are the most confident-sounding hallucination this system could produce.

**Breakage categories to enumerate in the prompt:**

| Kind | Detect by |
|---|---|
| Signature change | Params added, removed, reordered, or made required |
| Return shape change | Different type, extra/missing key, `None` newly possible |
| Exception change | Now raises where it did not, or swallows where it did not |
| Semantic change | Same signature, different behavior — the hardest and most dangerous |
| Removal / rename | Public symbol gone; callers unpatched |
| Constant / config | Value read elsewhere and now different |
| Schema / migration | Column change other queries still assume |

**New verdict field:**

```json
"blast_radius": [
  {"symbol": "total_price", "kind": "signature_change",
   "change": "added required parameter tax_rate",
   "breaks": [
     {"file": "shop/invoice.py", "line": 44, "severity": "HIGH",
      "claim": "Calls total_price(items) with one argument; will raise TypeError."}
   ],
   "unbroken_callers": 3}
]
```

Reporting `unbroken_callers` matters: "checked 4 call sites, 1 breaks" is a far more
trustworthy statement than a bare finding, and it tells the reviewer the check actually
ran rather than silently finding nothing.

**Language coverage:** Python via `ast` is exact. Everything else starts as `git grep`
on the symbol name, which over-matches. Accept the noise initially — a false "check this
caller" costs a glance; a missed break costs an incident.

### P10 — Repository knowledge graph ★ upgrades P1 and P9 from string matching to structure

**Problem:** P1 finds call sites with `git grep`, which matches text. Search for `save`
and you hit `save`, `autosave`, `saveDraft`, and a comment. P9 inherits the same
imprecision. Neither can answer "what else depends on this" — only "what else mentions
this string."

A persistent graph of the repository answers structural questions directly.

**Schema.** Two node kinds and two very different edge kinds:

```
NODES    file · module · function · class · test

EDGES
  static     imports · calls · inherits · tests · defines
             (from AST — precise, explicit coupling)

  temporal   co_changes_with(weight, sample_size)
             (from git history — implicit coupling nothing static can see)
```

**The temporal edges are the ones that earn this feature.** Static analysis finds
coupling the code declares. Git history finds coupling the code *hides*: the config file
that must change with the parser, the schema and its migration, the fixture that goes
stale, the doc nobody updates. Mining `git log --name-only` for files that change
together produces a finding class no analyzer can reach:

> ⚠️ `parser.py` and `grammar.toml` changed together in **14 of the last 16** commits.
> This PR changes `parser.py` alone.

That is frequently a real bug, and no amount of reading the diff would reveal it.

**Implementation:**

| Step | How |
|---|---|
| Build | `ast` for Python (exact); tree-sitter for other languages |
| Co-change mining | `git log --name-only --pretty=format:%H` → pairwise co-occurrence, weighted by recency |
| Store | SQLite, `nodes(id, kind, path, name)` + `edges(src, dst, kind, weight)` — no server, and it diffs well |
| Update | Incremental. Reparse only files touched since the last stored commit SHA |
| Query | For changed symbols, pull the k-hop neighborhood (k=2 default) and render into the prompt |

**What it upgrades:**
- **P1** — call sites become graph edges, not grep hits. Over-matching disappears.
- **P9** — blast radius traverses real `calls` edges. Precision improves; the "accept
  noise deliberately" tradeoff can be relaxed.
- **New** — co-change findings, described above.

**Prior art already installed:** GSD ships a `gsd-graphify` skill that builds a project
knowledge graph into `.planning/graphs/`. Worth reading before building from scratch —
this may be largely a matter of pointing an existing tool at the reviewed repo.

**Cost note:** the graph is built once and updated incrementally, so it is cheap per
sweep. The initial build on a large repo is the expensive part; do it in CI, cache it.

### P11 — Contribution and review-outcome tracking

**The useful core:** the system already produces per-PR data it currently throws away —
what was found, what was confirmed, what was rejected, what shipped and then broke. Kept
over time, that data answers questions worth answering.

**The hazard, stated once.** Ranking people as "best" and "worst" on this data has three
problems, and I would not ship the leaderboard version by default:

1. **The signal is not ground truth.** These are LLM findings. We measured this exact
   system returning 95% confidence on all four demo PRs *including one where it found
   nothing*. Ranking humans on an unvalidated model's output is building a performance
   system on a foundation we know is not yet calibrated.
2. **Defect counts track difficulty and volume, not skill.** Whoever maintains the
   gnarliest legacy module will always look worst. The person who ships nothing looks
   best. This is the classic failure of defect-density metrics.
3. **Measurement changes behavior.** A visible ranking teaches people to avoid hard
   work, split PRs to dilute counts, and stop volunteering for risky refactors. Also
   worth checking your jurisdiction — automated evaluation of workers carries real
   compliance obligations in several.

**The reframe that keeps the value:** track **outcomes on changes and code areas**,
not rankings of humans. Same data, different unit of analysis, and it is actually more
useful for the agent.

**What to record**, per PR, in `state/contributions.db`:

| Metric | Why it is trustworthy |
|---|---|
| **Reverted or hotfixed within N days** | The only real ground-truth defect signal here. A human decided it was broken. |
| Findings raised / confirmed / rejected | Measures **the agent's** calibration, not the author's |
| Review iterations before merge | Proxy for clarity of the change |
| PR size, files touched, blast radius | The **difficulty normalizer** — without it every other metric is noise |
| Test coverage delta | Did the change carry its own verification |
| Area of code touched | Where defects actually cluster |

**What to do with it — three uses that need no ranking:**

1. **Calibrate the agent.** If findings in `shop/legacy/` are rejected 80% of the time,
   the agent is miscalibrated for that area — raise its confidence bar there. This makes
   the bot better and blames nobody.
2. **Route review depth.** Spend more model budget on areas with a high historical
   revert rate, and on changes with large blast radius. Skip deep review on
   documentation-only PRs.
3. **Suggest reviewers.** "Who has the most context on this file" is a graph+history
   query (pairs naturally with P10) and is genuinely helpful.

**Config, defaulting to safe:**

```toml
[contributions]
enabled = true
track_reverts = true
# Per-person aggregates. OFF by default. See the hazard notes in P11 before enabling.
per_author_stats = false
# Never post contributor statistics into a PR comment.
visibility = "local"    # local | digest | never
```

If you do want per-author data, my recommendation is: keep it **local and private**,
normalize every metric by PR size and area difficulty, report ranges rather than ranks,
and never surface it in a PR comment where it becomes a public judgment of a colleague.

### P7 — Per-file fan-out for large PRs

Current behavior truncates at `max_diff_bytes` on file boundaries — a 50-file PR gets partially reviewed with no signal about what was dropped. Instead, above a threshold, spawn one subagent per file and merge. Uses the same `rlm()` recursion as P2.

---

## 5. PR review template

Rendered by `render_markdown`, driven by the fields above.

```markdown
<!-- prime-agent-review:{head_sha} -->
## 🔍 Automated Review

`{n} files` · `{additions}+ {deletions}-` · agreement `{ratio}` · `{model}`

> **⛔ 1 blocking issue** — SQL injection in `shop/customers.py`

---

### ⛔ Blocking

**CRITICAL · `shop/customers.py:15` · SQL injection**

User input is interpolated into a query string. The rest of this module
already uses parameterized queries, so this is inconsistent with its own
file.

```suggestion
    query = "SELECT id, name, email FROM customers WHERE name LIKE ?"
    return conn.execute(query, (f"%{name}%",)).fetchall()
```

*Confirmed by 3/3 reviewers · corroborated by `bandit:B608`*

---

### 💡 Non-blocking

<details><summary>2 suggestions</summary>

**MEDIUM · `shop/orders.py:13`** — bare `except` returns `None` implicitly...

</details>

---

### 🎯 Scope

**Stated intent:** fix the off-by-one in `total_price`

⚠️ **1 change does not serve that intent**

- **HIGH · `shop/auth.py:22-31`** — changes the session timeout from 30m to 24h.
  Nothing in the title or body mentions auth. Either split this into its own PR
  or say why it belongs here.

---

### 🕸️ Coupling

⚠️ `parser.py` and `grammar.toml` changed together in **14 of the last 16** commits.
This PR changes `parser.py` alone. Intentional?

---

### 💥 Blast radius

Checked **4 call sites** of the 2 changed symbols. **1 breaks.**

| Caller | Status |
|---|---|
| `shop/invoice.py:44` | ⛔ calls `total_price(items)` — now raises `TypeError` |
| `shop/cart.py:12` | ✅ already passes both arguments |
| `tests/test_orders.py:8,31` | ✅ updated in this PR |

---

### ✅ Fixes in this PR

- Guards `order_summary` against a missing order (`TypeError` on `None` subscript)

---

### 🧪 Tests

`shop/customers.py::search_customers` is new and has no test.
Suggested case: a name containing `'` should not alter the query.

---
<sub>[Docs](.) · react 👎 to suppress a finding · `@prime-bot recheck` to re-run</sub>
```

**Design rules:**
1. **Verdict first** — a reviewer decides in three seconds whether to care
2. **Blocking and non-blocking visually separated**; non-blocking collapsed
3. **Every finding carries evidence** — agreement ratio and/or linter corroboration
4. **Suggestions are committable**, not prose
5. **Fixes are acknowledged** — a bot that only criticizes gets muted
6. **Feedback affordance in the footer** — closes the P6 loop

---

## 6. GitHub bot integration

### Options

| | Latency | Hosting | Permissions needed | Identity |
|---|---|---|---|---|
| **A. Scheduled poller** (today) | minutes–hours | none | PAT only | your account |
| **B. GitHub Actions** | seconds | none (GitHub runs it) | write access to add workflow | `github-actions[bot]` |
| **C. GitHub App + webhook** | seconds | you host a receiver | app install | true bot identity |

### Recommendation: A → B → C

**Start with A.** It already works, and critically — it needs **no repository permissions beyond a token.** For a repo you contribute to but do not administer, this is the only option you can unilaterally deploy.

**Move to B when you have write access.** `.github/workflows/pr-review.yml`:

```yaml
on:
  pull_request_target:      # needed for fork PRs to have a writable token
    types: [opened, synchronize, reopened]

permissions:
  pull-requests: write
  contents: read

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.event.pull_request.head.sha }}
          persist-credentials: false      # never expose the token to PR code
      - uses: actions/setup-python@v5
        with: { python-version: '3.13' }
      - run: pip install -e .
      - env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
        run: python scripts/run_sweep.py --lane open --pr ${{ github.event.number }}
```

> ⚠️ **`pull_request_target` runs with repository secrets against fork-authored code.** This is the single most exploited GitHub Actions misconfiguration. Never check out and *execute* PR code under it. We only read the diff as text — that is safe — but the checkout must use `persist-credentials: false` and nothing from the PR may be run. This constraint compounds with prime-agent's own "not a security sandbox" warning.

**C only if** you want one bot serving many repos, or a real bot identity rather than `github-actions[bot]`.

### Where prime-agent fits

Notably: **B removes prime-agent's scheduling role entirely** — GitHub's event system is a better trigger than cron. What remains valuable is `rlm()` recursion for P2 and P7. That is worth being honest about: prime-agent's contribution here is subagent orchestration, not scheduling.

---

## 7. Phasing

| Phase | Contents | Effect | Complexity |
|---|---|---|---|
| **0** | Fix `SKILL.md` `rlm()` bug; test `reviewers.py`; run merged lane; one live post to the demo repo | Correctness baseline | **S** |
| **1** | P1 repo context · P3 static pre-pass · **P8 intent** · **P9 blast radius** | The big quality jump | **M/L** |
| **2** | P4 inline comments + suggestions · P5 severity events · new template | The visible product | **M** |
| **3** | P2 ensemble · P7 fan-out | Precision | **M** |
| **4** | GitHub Actions (option B) | Real bot | **S** |
| **5** | P6 feedback loop | Improves over time | **S** |
| **6** | **P10 knowledge graph** | Structure replaces string matching; co-change findings | **L** |
| **7** | **P11 outcome tracking** | Calibration and review routing | **M** |

Phase 0 is not optional — three of its items are known-broken or never-executed paths.

**Recommended stopping point: end of Phase 2.** That is a genuinely useful bot. Phases 3–5 are refinement.

---

## 8. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `pull_request_target` secret exfiltration via fork PR | Medium | **Critical** | Never execute PR code; `persist-credentials: false`; diff read as text only |
| False positives erode trust | **High** | High | Ensemble gate; linter corroboration; P6 rejections; keep `dry_run` until measured |
| Context injection inflates cost | Medium | Medium | Token budget per section; Flash pricing; watermark already prevents re-review |
| Suggestion blocks that don't apply cleanly | Medium | Low | Validate line is in the diff; omit `suggestion` when unsure |
| `REQUEST_CHANGES` blocking a teammate wrongly | Low | **High** | Off by default; CRITICAL only; requires explicit opt-in |
| prime-agent Windows instability | **High** | Low | Already bypassed — pipeline is plain Python |
| Demo answer key overfits | **High** | Medium | Expand corpus to 20+ PRs incl. real merged history before trusting numbers |
| Knowledge graph goes stale and misleads | Medium | Medium | Store the commit SHA the graph was built at; refuse to use a graph older than the base commit rather than serving stale edges |
| Graph build cost on a large repo | Medium | Low | Build once in CI and cache; sweeps only update incrementally |
| **Contributor metrics used as performance review** | **Medium** | **High** | `per_author_stats` off by default; `visibility = "local"`; never posted to a PR; normalize by size and area difficulty; see the P11 hazard notes |
| Ranking built on uncalibrated LLM findings | **High** if enabled | **High** | Gate any per-author reporting behind a measured finding-acceptance rate; revert/hotfix is the only ground-truth signal |

---

## 9. Validation

```bash
python -m pytest                                    # gate: 80% coverage
python scripts/run_sweep.py --lane open --fresh     # demo repo, expect 4/4
python scripts/run_sweep.py --lane merged --fresh   # never yet run
python -m prime_pr_review check
```

Add a scored harness — `scripts/score_demo.py` — asserting the answer key mechanically rather than by eye, so quality regressions surface as test failures.

---

## 10. Acceptance

- [ ] Phase 0: `SKILL.md` corrected, `reviewers.py` covered, merged lane run, one live comment posted
- [ ] Phase 1: reviewer receives file context, call sites, tests, conventions; linter output grounds findings
- [ ] Phase 2: inline line-anchored comments with committable suggestions; severity drives review event
- [ ] Demo corpus ≥ 20 PRs; `score_demo.py` passes
- [ ] No finding posts without either ensemble agreement or linter corroboration
