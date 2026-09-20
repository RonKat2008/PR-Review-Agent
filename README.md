# PR Review Agent

An automated pull-request reviewer that finds **bugs a diff introduces** — not style, not
naming, not "consider extracting this."

Three LLMs from three different labs review the same evidence blind. A judge model merges
findings that describe one defect. A skeptic model tries to refute what survives. Every
cited file and line is verified to exist before anything is reported.

**1,010 tests · 97% coverage · benchmarked on 200 real pull requests**

---

## Results

Measured on 200 real pull requests from the public
[SWE-CARE](https://huggingface.co/datasets/inclusionAI/SWE-CARE) test split, scored against
the human review comments on each PR. Every model call was recorded once, and each pipeline
variant is replayed offline from those recordings, so the comparison below is reproducible
without re-running a model.

| pipeline | PRs | precision | recall | findings/PR | fabricated citations | $/PR |
|---|---|---|---|---|---|---|
| strongest single model | 192 | 0.24 | 0.24 | 1.80 | 0.00 | 0.08 |
| single model (mean of 3) | 185 | 0.25 | 0.20 | 1.39 | 0.00 | 0.05 |
| **3-lab ensemble** | 192 | 0.24 | **0.34** | 3.45 | 0.00 | 0.14 |
| + judge-merge | 192 | 0.22 | 0.30 | 2.59 | 0.00 | 0.15 |
| + skeptic (full pipeline) | 192 | 0.23 | 0.27 | 2.17 | 0.00 | 0.21 |

**The ensemble is the result.** Three model families voting blind find ~40% more of what
human reviewers flagged than the strongest single model, at unchanged precision.

**The adversarial passes are a negative result.** Judge-merge and the skeptic cut findings
per PR by a third — real noise reduction — but do **not** improve precision against human
comments. Published as measured rather than quietly dropped.

**Citations hold up.** Effectively zero findings pointing at a file or line that does not
exist, across every arm.

### How much to trust these numbers

A finding "matches" a human comment when it lands on the same file within ±5 lines. That
tolerance drives the absolute scores, so they are reported with it, never without:

| match tolerance | ensemble precision | ensemble recall |
|---|---|---|
| exact line | 0.09 | 0.15 |
| ±5 lines (headline) | 0.24 | 0.34 |
| ±20 lines | 0.38 | 0.49 |
| same file, any line | 0.65 | 0.66 |

The ensemble's lead holds at **every** tolerance. As a coincidence control, shifting every
finding by 37 lines collapses matching to 0.02–0.04, so real matches run about ten times
above chance.

Two honest caveats. Precision is a lower bound: a real bug no human commented on counts
against it. And a location match is not proof of an issue match — reading a sample of
matched pairs, roughly half describe the same underlying defect.

Full ladder, per-severity breakdown, cost and wall time per arm, and every limitation:
**[`docs/eval/swecare-200-s0.md`](docs/eval/swecare-200-s0.md)**.

---

## How it works

```
PR ──► gather evidence (deterministic, no model)
       │  changed files, or ±80-line windows around each hunk
       │  call sites of changed symbols   (git grep)
       │  sibling tests, repo conventions
       │  co-change graph                 (mined from git history)
       │  CI status + failing log tail    (gh)
       │  linters, filtered to touched lines (ruff/bandit/mypy)
       ▼
    ┌─ seat 1 ─┐
    ├─ seat 2 ─┤  three labs, blind to each other
    └─ seat 3 ─┘
       ▼
   judge-merge      one defect reported at two lines becomes one finding
       ▼
   agreement        confidence = how many seats found it, not self-report
       ▼
   citation check   file/line must exist at the PR head, or the finding is dropped
       ▼
   skeptic          tries to disprove each finding; refuted ones are challenged, never deleted
       ▼
   render + gates   local markdown, or line-anchored GitHub comments with suggestions
```

Two review lanes: **`open`** answers "what breaks if this merges?" and **`merged`** answers
"what did this fix, and what regressed?"

### Design decisions worth knowing

**Confidence is measured, not asked for.** Early runs had the model self-report confidence;
it returned 95% on every PR, including ones where it found nothing. Confidence is now the
observed agreement rate across independent seats.

**Evidence is deterministic before it is interpreted.** Call sites come from `git grep`,
coupling comes from git history, CI state comes from the API. The model reasons over facts
it did not invent.

**A stale knowledge graph is refused, not trusted.** If git says the graph predates the PR's
base commit, it is dropped with a visible note, because a stale graph fails convincingly.

**Refuted findings are marked, never deleted.** The skeptic moves a finding into a
"challenged" section and out of inline comments. The maintainer still sees it.

---

## Quick start

```bash
uv venv --python 3.13
uv pip install -e ".[dev]"
gh auth login                  # the pipeline reads PRs through the gh CLI
cp .env.example .env           # GITHUB_TOKEN, optional webhook URL
```

Point it at a repo in `config.toml`, then preflight everything except the model:

```bash
prime-review check             # names exactly what is missing, exits non-zero until fixed
```

Review a single PR:

```bash
prime-review pr owner/repo 1234
```

Sweep a lane:

```bash
prime-review sweep --lane open --repo owner/repo
```

Reviews land in `reviews/PR-<number>-<sha>.md` as a six-section report: intent, changes by
file, issues, proposed changes, what to test, verdict.

### Where the models come from

| path | reviewer | credential |
|---|---|---|
| inside prime-agent | `rlm()` subagents, one per seat | none — the session's models |
| headless (`prime-review sweep`) | Gemini | `GEMINI_API_KEY` |
| evaluation harness | Prime Inference (OpenAI-compatible) | `PRIME_API_KEY` or `~/.prime/config.json` |

Scheduled sweeps run inside prime-agent — see
[`skills/pr-review/SKILL.md`](skills/pr-review/SKILL.md) for the invocation and cron
registration.

---

## Evaluation harness

The benchmark is part of the repo, not a one-off script:

```bash
prime-review eval run    --count 200 --seed 0 --cap-usd 40
prime-review eval score  --run-id <id>
prime-review eval report --run-id <id>
```

`run` reviews each sampled PR live and records every model call — prompt, response, tokens,
latency. `score` rebuilds six pipeline variants from those recordings and scores each one.
`report` renders the ladder to `docs/eval/`.

That split is the point: **one paid run, unlimited re-scoring.** Changing the matching rule,
adding an arm, or re-checking a metric costs nothing. The harness also tracks spend against a
hard cap, resumes interrupted runs from per-instance markers, and can re-issue only the calls
that failed.

---

## Safety

Ships refusing to post. Every gate is enforced in one place, `sinks.evaluate_comment_gates`:

| gate | effect |
|---|---|
| `read_only` (per repo) | hard write ban, checked **before** `dry_run` — flipping dry-run is not enough |
| `dry_run` | reviews are written locally, never posted |
| `min_confidence` | findings below the agreement threshold stay local |
| `max_comments_per_sweep` | hard cap, so a bad prompt cannot spray a repo |
| idempotency marker | never comments twice on the same head commit |
| `bot_login` | skips PRs the agent itself authored |

A single PR failing never aborts a sweep; the error is recorded on the report and that PR is
left unmarked so it retries next run.

---

## Configuration

`config.toml`, with secrets only ever in the environment:

| knob | what it does |
|---|---|
| `ensemble_size` / `min_agreement` | seats per PR, and how many must agree to keep a finding |
| `check_refute` / `judge_merge` | the skeptic and judge passes |
| `validate_citations` | drop findings citing a file or line that does not exist |
| `check_intent` | two-pass check that the diff does what the PR claims |
| `graph_path` | co-change graph, refused if stale |
| `repo_root` | local checkout; without it, call sites and blast radius are skipped |
| `ignore_paths` / `max_diff_bytes` | lockfile filtering and token guards |

---

## Layout

```
prime_pr_review/             sweep, ensemble, judge, refute, citations, context, graph,
                             blast radius, CI, exports, feedback, sinks, template
prime_pr_review/evaluation/  corpus, record/replay, ablation arms, scoring, reporting
scripts/                     run_sweep · eval_swecare · replay_corpus · build_cochange
skills/pr-review/            SKILL.md + review, intent, blast, judge, skeptic prompts
docs/eval/                   published benchmark results
docs/superpowers/            design specs and implementation plans
tests/                       1,010 tests — every external call is injected, none touch the network
```

```bash
python -m pytest              # 1,010 tests, ~3s, no network, no credentials
```

---

## Status and limits

Working end to end, with the honest boundaries:

- **Nothing has posted to a public repo.** Every configured target is `read_only`.
- **The `merged` lane has not been run live.** Only the `open` lane has real mileage.
- **The benchmark is diff-only.** Context gathering, blast radius and linters need a local
  checkout, so they are excluded from the measured numbers and the pipeline scores *worse*
  there than in normal use.
- **The evaluation swapped one seat.** Qwen3.8-Max reasons without bound (~7 min, $0.10/PR),
  so the benchmark ran GPT-5.4-mini in its place.
