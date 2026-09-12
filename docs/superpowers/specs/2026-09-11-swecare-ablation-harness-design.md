# SWE-CARE Ablation Harness — Design

**Date:** 2026-09-11
**Status:** approved in chat, awaiting implementation plan
**Depends on:** citation validator (P14, deterministic half), uncommitted as of this date

## 1. Goal

Measure, on real pull requests with human-labeled ground truth, whether this
reviewer's adversarial design (three-lab ensemble, judge-merge, skeptic
refutation, citation validation) beats a single-pass review — and by how much.
Output is one ablation table with precision, recall, fabrication rate, cost, and
wall time per arm, reproducible from recorded artifacts without re-running any
model.

Non-goals: tuning thresholds, rating the enrichment passes that need a local
checkout (context, blast, unwired exports, linters), or measuring silence on
cosmetic PRs. SWE-CARE has no such set; those are stated limitations in the
report.

## 2. Corpus

`inclusionAI/SWE-CARE`, `test` split (671 rows, Python only, Apache 2.0),
fetched from the Hugging Face datasets-server API and cached locally as JSON.

Fields used per row:

| Field | Use |
|---|---|
| `instance_id`, `repo`, `pull_number` | identity, artifact directory name |
| `title`, `body` | PR metadata for the reviewer and intent pass |
| `base_commit` | reported only |
| `commit_to_review.head_commit` | head SHA: idempotency key, citation head-file fetch |
| `commit_to_review.head_commit_message` | commit message for the intent pass |
| `commit_to_review.patch_to_review` | **the diff the reviewer sees** (the mid-review commit humans commented on) |
| `reference_review_comments[]` | ground truth: `path`, `line`, `start_line`, `original_line`, `original_start_line`, `text` |

Sampling: deterministic by `--seed` (default 0), size `--count` (default 200).
Filters, applied before sampling: `language == "Python"`, `len(patch_to_review)
<= 60_000` bytes, at least one reference comment with a non-empty `path`.
Rows failing filters are counted and reported, never silently dropped.

## 3. Architecture

```
scripts/eval_swecare.py                 CLI: run | score | report
prime_pr_review/providers.py            Prime Inference client, CostMeter
prime_pr_review/evaluation/
    corpus.py                           load, filter, sample SWE-CARE rows
    runner.py                           corpus-backed fake GhRunner
    recording.py                        recording Reviewer/ModelFn + replay ModelFn
    arms.py                             offline arm reassembly
    scoring.py                          matcher, per-PR and aggregate metrics
    report.py                           JSON summary + markdown table
eval/
    corpus/swecare-test.json            cached rows (gitignored)
    runs/<run_id>/<instance_id>/        recordings (gitignored)
    runs/<run_id>/meter.json            spend so far (gitignored)
docs/eval/<run_id>.md, .json            committed results
```

One live run per PR through the real `sweep_lane`. Every model call is
recorded. All ablation arms are recomputed from recordings.

### 3.1 providers.py

- `prime_model_fn(model: str, meter: CostMeter, client: httpx.Client | None) -> ModelFn`
  POST `https://api.pinference.ai/api/v1/chat/completions`, bearer key,
  `messages=[{"role":"user","content":prompt}]`, `temperature=0`. Returns
  `choices[0].message.content`. Retries 429/5xx/timeouts with exponential
  backoff (5 attempts, 2s base, jitter). Any other failure raises
  `ProviderError`.
- `prime_reviewer(model, meter, prompts_dir) -> Reviewer`: reads
  `<lane>_pr.md`, `build_prompt(template, pr, payload)` (same helper
  `reviewers.gemini_reviewer` uses), delegates to the model fn.
- `resolve_prime_key()`: `PRIME_API_KEY` env var, else `~/.prime/config.json`
  `api_key`, else raise with an instruction.
- `fetch_pricing(client) -> dict[model, (in_usd_per_mtok, out_usd_per_mtok)]`
  from `GET /models`. Missing model → `ProviderError` at startup, not mid-run.
- `CostMeter(cap_usd, pricing)`: `record(model, prompt_tokens, completion_tokens)`
  accumulates `spent_usd` and per-model counters; `check()` raises
  `BudgetExceeded` when `spent_usd > cap_usd`. `to_json()/from_json()` for
  persistence between resumed runs. Default cap `--cap-usd 40`.

The Prime lineup mirrors `skills/pr-review/SKILL.md`: seats
`deepseek/deepseek-v4-pro`, `qwen/qwen3.8-max`, `z-ai/glm-5.2`; intent and
blast `deepseek/deepseek-v4-flash`; skeptic and judge `deepseek/deepseek-v4-pro`.

### 3.2 corpus.py

- `fetch_rows(split="test", cache: Path) -> tuple[Row, ...]`: pages the
  datasets-server `rows` endpoint (100 per page) into the cache file once;
  later calls read the cache.
- `Row` frozen dataclass with the fields in §2 plus
  `reference_comments: tuple[ReferenceComment, ...]`.
- `ReferenceComment(path, line: int | None, start_line: int | None, text)`.
  `line` resolves as `line` then `original_line`; `start_line` likewise.
- `select(rows, count, seed, max_patch_bytes) -> tuple[Row, ...]` applies
  §2 filters then `random.Random(seed).sample`.

### 3.3 runner.py

`corpus_runner(row: Row, fallback: GhRunner) -> GhRunner` answers, by args:

| `gh` call | Response |
|---|---|
| `pr list …` | one-element JSON array in `PR_FIELDS` shape: number, title, author `"swe-care"`, `headRefOid` = review head SHA, `baseRefName` = `"main"`, url, additions/deletions/changedFiles computed from the patch, `mergedAt` null |
| `pr view … --json` | same object plus `body`, `commits: [{messageHeadline: head_commit_message}]` |
| `pr diff …` | `patch_to_review` verbatim |
| `pr checks …` | `"[]"` (CI unknown) |
| `api …/issues/…/comments` | `"[]"` |
| `api …/contents/<path>?ref=<sha>` | delegated to `fallback` (real `gh`) so citation validation gets exact head line counts; a failure degrades to unverified, as in production. Every response (or failure) is appended to the instance's `head_files.json` as `{path: raw gh content response | null}` so the offline `citations=on` arm replays it without network |
| anything else | `GitHubError("eval runner: unexpected gh call: …")` |

The runner is pure over `row` apart from that one append, and logs nothing.

### 3.4 recording.py

- `Recorder(dir: Path)` writes `calls/<seq:03d>-<role>.json` with
  `{role, model, prompt_sha256, prompt, response, prompt_tokens,
  completion_tokens, seconds}`. Roles: `seat`, `aux` (the intent pass; the
  blast-radius walk is skipped in the diff-only harness), `judge`, `skeptic`.
- `recording_reviewer(seat_models: Sequence[str], make_model_fn, recorder) -> Reviewer`:
  the ensemble calls the reviewer `size` times per PR; calls are assigned to
  seats round-robin in order (call k → `seat_models[k % len]`), so seat identity
  is recoverable from the sequence number.
- `recording_model_fn(role, model, inner, recorder) -> ModelFn`.
- `replay_model_fn(dir, role) -> ModelFn`: looks up by `prompt_sha256`; a miss
  raises `ReplayMiss` (a prompt that differs from the live run means the arm
  is not a pure replay — the scorer reports it, never guesses).
- `replay_reviewer(dir, seat_index) -> Reviewer` returns that seat's recorded
  response regardless of prompt (payload is identical by construction).
- A `done` marker file is written last; `run` skips instances that have it.

### 3.5 arms.py

All arms consume one instance directory and return a `Verdict`.

| Arm | How |
|---|---|
| `seat-<i>` (i = 1..3) | `parse_verdict(seat i response)` |
| `single` | mean of the three seat arms' metrics (reported as the single-pass baseline) |
| `ensemble` | `ensemble_review_detailed(size=3, min_agreement=1, judge_fn=None)` with `replay_reviewer` |
| `ensemble+judge` | same with `judge_fn = replay_model_fn("judge")` |
| `full` | `ensemble+judge` then `refute_findings(model_fn=replay_model_fn("skeptic"))`; refuted findings are treated as not reported |

Each arm is scored twice: `citations=off` (verdict as produced) and
`citations=on` (`validate_citations` applied with head line counts recorded
during the live run in `head_files.json`). Production order is validation
before refutation; for the offline ladder validation is applied last so the
`off` variant measures the raw fabrication rate. The report states this.

The live run itself is the `full` arm; a scorer check asserts the live
verdict equals the replayed `full` verdict and reports any drift.

### 3.6 scoring.py

- `match(finding, ref, window=5) -> bool`: `finding.file == ref.path` and
  `finding.line` within `[lo - window, hi + window]` where `lo = ref.start_line
  or ref.line`, `hi = ref.line`. A finding with `line is None` matches at file
  level only.
- Per instance, per arm, per citations mode:
  `findings`, `matched_findings`, `refs`, `matched_refs`, `fabricated`
  (= findings the validator drops), `file_level_matched_findings`.
- Aggregate (micro-averaged over instances): precision = matched findings /
  findings; recall = matched refs / refs; fabrication rate = fabricated /
  findings on `citations=off`; file-level precision as a secondary column;
  findings per PR; cost USD and seconds per PR from recordings.
- Also per severity bucket for the `full` arm.

### 3.7 report.py

`summary.json` (all per-instance and aggregate numbers, run config, seed,
model lineup, pricing snapshot, drift count) and `<run_id>.md` with:
the ablation ladder table, the citations on/off delta table, the severity
table, scope limitations (§1), and the exact command to reproduce.

### 3.8 CLI

```
prime-review eval run    --count 200 --seed 0 --cap-usd 40 [--run-id ...] [--resume]
prime-review eval score  --run-id <id>
prime-review eval report --run-id <id>
```

`run` = fetch corpus → select → for each row: build runner, config
(`dry_run=True`, `pr_comment=False`, `read_only=True`, `ensemble_size=3`,
`min_agreement=1`, `judge_merge=True`, `check_refute=True`,
`validate_citations=True`, `check_intent=True`, `repo_root=""`,
`graph_path=""`), `Enrichment(model_fn=intent/blast fn, skeptic_fn, judge_fn)`
→ `sweep_lane(config, LANE_OPEN, reviewer, State.empty(), runner=…,
reviews_dir=eval/runs/<id>/reviews)` → write `outcome.json` (verdict as
JSON, notes, seconds) → `meter.check()`. One PR failing is recorded as an
error and the run continues. `BudgetExceeded` stops the run cleanly with the
meter persisted; `--resume` continues.

`cli.py` gains the `eval` subcommand mapped to `scripts/eval_swecare.py`.

## 4. Error handling

- Provider: retries as above; after retries, the seat fails, which the
  ensemble already treats as one failed run.
- Corpus fetch failure: fail fast before any spend.
- Missing head file for a citation: unverified, not dropped (production
  behavior).
- Replay miss: reported per instance; the arm's metric for that instance is
  excluded and the exclusion count is in the summary.
- Budget: hard stop; nothing is retried past the cap.

## 5. Testing

No network in tests. Fixture rows come from the datasets-server sample saved
during design (`tests/fixtures/swecare_rows.json`, 5 rows, trimmed patches).

- corpus: filter rules, deterministic sampling, cache read/write, comment
  line resolution.
- runner: each `gh` call shape; unexpected call raises; content fetch delegates.
- recording: sequence numbering, seat round-robin, replay hit and miss, done marker.
- arms: with recorded fixtures, `ensemble` reproduces `ensemble_review_detailed`
  on the seats; `full` equals the live verdict.
- scoring: window edges, `line is None`, file-level, zero-findings PR,
  zero-refs guard (recall undefined → excluded and counted).
- providers: retry on 429 then success, `BudgetExceeded`, pricing lookup
  failure at startup, key resolution order. httpx mocked via
  `httpx.MockTransport`.
- CLI: `run` on two fixture rows with a fake provider writes the expected
  directory layout; `score` and `report` on that layout produce a table.

Coverage stays above the 80% gate; target 90% on the new modules.

## 6. Cost and runtime

Median review patch is ~6 KB. Per PR: three seats plus judge plus
per-finding skeptic on ~5k-token payloads, intent on flash. Estimated
$0.05–0.10 per PR, $10–20 for 200 PRs, under the $40 cap. Sequential
execution; roughly 200 PRs × ~60 s ≈ 3–4 hours. Concurrency is a later
option, not in this spec.

## 7. Out of scope, recorded for later

- Local checkout per PR for context/blast/exports/linters (`--with-checkout`).
- Silence set (cosmetic PRs with zero reference comments).
- Human-comment relevance filtering (nits vs. defects) — a labeling pass.
- Per-repo breakdown and threshold tuning from the results (plan E2).
