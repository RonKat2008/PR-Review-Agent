---
name: pr-review
description: Review open and recently merged pull requests for bugs introduced and bugs fixed. Drives the prime_pr_review pipeline from the kernel with rlm() subagents as the reviewer ensemble — the session's prime-inference models do the judging. Use when asked to review a PR, sweep a repo's PRs, or replay a PR corpus.
---

# PR Review Sweep

Automated bug review of example-org/service-a and example-org/service-b, in two lanes:

- **`open`** — pre-merge gate. "What breaks if this merges?"
- **`merged`** — retrospective. "What did this fix, and what regressed?"

The deterministic pipeline (diff filtering, call-site discovery, blast radius,
CI/graph/lint evidence, safety gates, rendering, watermarks) lives in the
`prime_pr_review` package. THIS session supplies the judgment: each review is an
`rlm(...)` child agent, so the models below do the work — no separate API key.

Both example-org repos are configured `read_only = true` in `config.toml`:
reviews land in `reviews/` locally and are NEVER posted to GitHub, regardless
of dry_run or who drives. Do not change that setting.

## Model lineup (ensemble seats + auxiliary passes)

| Role | Model |
|---|---|
| Ensemble seat 1 | `prime-inference/deepseek/deepseek-v4-pro` |
| Ensemble seat 2 | `prime-inference/qwen/qwen3.8-max` |
| Ensemble seat 3 | `prime-inference/z-ai/glm-5.2` |
| Intent + blast passes | `prime-inference/deepseek/deepseek-v4-flash` |

Three different labs vote on every PR; a finding must survive 2-of-3 agreement
(`ensemble_size`/`min_agreement` in config.toml). Typical cost: ~$0.25/PR.

## One-time kernel setup

```python
%pip install -q -e /home/user/Projects/pr-review-agent ruff bandit mypy
```

Also required once per machine: `gh auth login` (the pipeline reads PRs via the
gh CLI). Verify everything with:

```python
%%bash
cd /home/user/Projects/pr-review-agent && ./.venv/bin/prime-review check
```

## Before each headless run

The prime-agent daemon degrades after a few `-p` sessions (worker connections
start dying with "Daemon worker client closed" — upstream bug). Reset it first:

```bash
prime-agent shutdown --force
```

A fresh daemon starts automatically on the next invocation. Run long reviews
under `caffeinate -i` so the Mac cannot sleep mid-review.

## The recipe

Run these cells in order. Cell 1 — shared plumbing (define once per session):

```python
import asyncio, json, os, sys, time, uuid
from dataclasses import replace
from pathlib import Path

AGENT = Path("/home/user/Projects/pr-review-agent")
PROMPTS = AGENT / "skills" / "pr-review" / "prompts"
VERDICTS = AGENT / "state" / "verdicts"

from prime_pr_review import github
from prime_pr_review.analysis import run_analysis
from prime_pr_review.config import load_config, require_repo, resolve_active
from prime_pr_review.feedback import load_rejections
from prime_pr_review.graph import strict_runner
from prime_pr_review.reviewers import build_prompt
from prime_pr_review.state import LANE_OPEN, LANE_MERGED, load_state, save_state
from prime_pr_review.sweep import Enrichment, sweep_lane

# rlm(model=...) needs the FULL selector, provider prefix included -- a bare
# "deepseek/deepseek-v4-pro" is rejected as unknown.
ENSEMBLE_SEATS = [
    "prime-inference/deepseek/deepseek-v4-pro",
    "prime-inference/qwen/qwen3.8-max",
    "prime-inference/z-ai/glm-5.2",
]
AUX_MODEL = "prime-inference/deepseek/deepseek-v4-flash"
REPO_ROOT = ""  # set by run_sweep; told to every child so it never hunts for the checkout

# The kernel loop. sweep_lane is synchronous and runs on a worker thread
# (asyncio.to_thread below); each spawn is marshalled back onto this loop.
LOOP = asyncio.get_running_loop()

def _spawn(prompt, name, model):
    """Spawn one child whose only deliverable is writing its verdict file.

    The payload is handed over as a FILE the child loads itself, never embedded
    in the task message: large task messages get copied into the session
    transcript, and the daemon's supervisor<->worker snapshot transfer dies on
    large frames ("Daemon worker client closed" mid-run — observed repeatedly
    at ~400KB task messages, never with small ones). rlm() returns an admission
    handle immediately, NEVER the child's answer — the verdict comes back
    through the output file, for the same reason in reverse.
    """
    out = VERDICTS / f"{name}.json"
    payload = VERDICTS / f"{name}.payload.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    payload.write_text(
        f"{prompt}\n\n## Local checkout\n\n"
        f"The reviewed repository is checked out at `{REPO_ROOT}` (base branch). "
        "Read files there directly if you need surrounding code. Do NOT search "
        "the filesystem for it (no os.walk/glob/find over home directories).\n",
        encoding="utf-8",
    )
    task = (
        f"Your complete review instructions and evidence are in the file:\n\n"
        f"    {payload}\n\n"
        "Load ALL of it into your context first: read the file in slices of at "
        "most 1500 lines (print each slice) until end of file — do not skim or "
        "summarize while reading. Then follow its instructions exactly.\n\n"
        "## Output — file handoff (mandatory)\n\n"
        f"Write ONLY the JSON object the instructions describe to this exact "
        f"file:\n\n    {out}\n\n"
        "Complete JSON in a single write, no code fence in the file, no prose "
        "before or after it. When the file is written, reply with no prose."
    )
    try:
        asyncio.run_coroutine_threadsafe(rlm(task, name=name, model=model), LOOP).result(60)
    except Exception as exc:
        print(f"[spawn failed] {name} ({model}): {exc!r}")
        raise
    return out


def _await_file(out, timeout=600, poll=2):
    """Poll until the verdict file is non-empty AND stable across two checks
    (a size still changing means the child is mid-write). Raises on timeout so
    the sweep records an error for this run rather than hanging."""
    deadline, last = time.monotonic() + timeout, -1
    while time.monotonic() < deadline:
        size = out.stat().st_size if out.is_file() else 0
        if size > 0 and size == last:
            return out.read_text(encoding="utf-8")
        last = size if size > 0 else -1
        time.sleep(poll)
    raise TimeoutError(f"no verdict written to {out} within {timeout}s")

def _spawn_and_wait(prompt, name, model):
    return _await_file(_spawn(prompt, name, model))

# Reviewer: the ensemble calls this ensemble_size times per PR. The FIRST call
# for a PR spawns EVERY seat at once (three different labs, in parallel) and
# collects all their files; later calls just hand back the next seat's verdict.
# Wall time is the slowest seat, not the sum of three -- and a seat that fails
# counts as exactly one failed ensemble run, never a re-spawn of the others.
_pending = {}
def reviewer(pr, payload, lane):
    key = (lane, pr.number, pr.head_sha)
    if key not in _pending:
        template = (PROMPTS / f"{lane}_pr.md").read_text(encoding="utf-8")
        prompt = build_prompt(template, pr, payload)
        outs = [
            _spawn(prompt, f"review-{lane}-{pr.number}-{uuid.uuid4().hex[:6]}", model)
            for model in ENSEMBLE_SEATS
        ]
        results = []
        for out in outs:
            try:
                results.append(_await_file(out))
            except Exception as exc:  # one slow/failed seat must not sink the others
                print(f"[seat failed] {out.name}: {exc!r}")
                results.append(exc)
        _pending[key] = results
    result = _pending[key].pop(0)
    if not _pending[key]:
        del _pending[key]
    if isinstance(result, Exception):
        raise result
    return result

# ModelFn for the intent and blast passes (they build their own prompts).
def model_fn(prompt):
    return _spawn_and_wait(prompt, f"pass-{uuid.uuid4().hex[:6]}", AUX_MODEL)

def build_enrichment(config):
    root = Path(config.review.repo_root)
    return Enrichment(
        model_fn=model_fn,
        repo_root=root,
        prompts_dir=PROMPTS,
        # Strict on purpose: for merge-base --is-ancestor, exit 1 means
        # "stale graph, refuse" — the lenient grep runner would swallow it.
        git_runner=strict_runner(root),
        analysis_fn=run_analysis,
        rejections=load_rejections(AGENT / "state" / "rejections.json"),
    )

async def run_sweep(repo_selector, lane=LANE_OPEN, runner=github.default_runner):
    global REPO_ROOT
    config = resolve_active(load_config(AGENT / "config.toml"), repo_selector)
    repo = require_repo(config)
    REPO_ROOT = config.review.repo_root
    # graph_path in config.toml is relative to the AGENT repo, but the sweep
    # runs with cwd = the reviewed checkout (see below); resolve it first.
    if config.review.graph_path and not Path(config.review.graph_path).is_absolute():
        config = replace(config, review=replace(
            config.review, graph_path=str(AGENT / config.review.graph_path)))
    state_path = AGENT / "state" / f"watermark-{repo.slug.replace('/', '-')}.json"
    state = load_state(state_path)

    # gh inside the kernel may not reach the keychain login a terminal has;
    # give it the token from the agent's .env explicitly (GH_TOKEN wins).
    for line in (AGENT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("GITHUB_TOKEN=") and line.partition("=")[2].strip():
            os.environ["GH_TOKEN"] = os.environ["GITHUB_TOKEN"] = line.partition("=")[2].strip()

    # Call-site grep and the linters resolve paths against the process cwd, so
    # cwd must be the reviewed checkout for the sweep's duration. The kernel
    # venv's bin dir goes on PATH so ruff/bandit/mypy are found.
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    prev = os.getcwd()
    os.chdir(config.review.repo_root)
    try:
        report, state = await asyncio.to_thread(
            sweep_lane, config, lane, reviewer, state,
            runner=runner, enrichment=build_enrichment(config),
            reviews_dir=AGENT / "reviews",
        )
    finally:
        os.chdir(prev)

    save_state(state, state_path)
    for line in report.summaries():
        print(line)
    print(f"{report.considered} considered | {report.reviewed} reviewed | "
          f"{report.skipped} skipped | {report.errors} errors")
    return report
```

Cell 2 — sweep a lane:

```python
report = await run_sweep("service-b", LANE_OPEN)
```

Or review exactly one PR (single-PR runs go through the open-lane machinery on
purpose — the merged lane's lookback filter could silently drop the very PR
just resolved):

```python
async def run_single_pr(repo_selector, number):
    config = resolve_active(load_config(AGENT / "config.toml"), repo_selector)
    repo = require_repo(config)
    pr = github.get_pr(repo.slug, number)
    pr_json = json.dumps([{
        "number": pr.number, "title": pr.title, "author": {"login": pr.author},
        "headRefOid": pr.head_sha, "baseRefName": pr.base_ref, "url": pr.url,
        "additions": pr.additions, "deletions": pr.deletions,
        "changedFiles": pr.changed_files, "mergedAt": pr.merged_at,
    }])
    runner = github.single_pr_runner(github.default_runner, pr_json)
    return await run_sweep(repo_selector, LANE_OPEN, runner=runner)

report = await run_single_pr("service-b", 2567)
```

Read the results in `reviews/PR-<number>-<sha8>.md`. The JSON front matter's
`notes` field records every degraded or skipped enrichment — a review that ran
with less context than intended says so there.

## Scheduling

```bash
prime-agent schedule add worker "0 */4 * * *" -- "Load the pr-review skill and sweep the open lane for service-b"
prime-agent schedule add worker "0 9 * * 1-5" -- "Load the pr-review skill and sweep the merged lane for service-b"
prime-agent schedule list
```

The head-SHA watermark makes repeat sweeps idempotent: a PR is re-reviewed only
when new commits change its head. Keep the cron strings in sync with
`[schedule]` in config.toml.

## Headless fallback (no TUI)

`prime-review sweep --repo service-b` runs the same pipeline with the
Gemini-backed reviewer (needs `GEMINI_API_KEY`). The replay harness and demo
scorer are `prime-review replay` / `prime-review score`.

## Safety gates

Every gate is enforced in `prime_pr_review/sinks.py:evaluate_comment_gates`:

| Gate | Effect |
|---|---|
| `read_only` (both example-org repos) | Hard write-ban, checked BEFORE dry_run — flipping dry_run is never enough to post |
| `dry_run` | Nothing posts to GitHub |
| `min_confidence` | With the ensemble on, confidence is the observed agreement ratio |
| `max_comments_per_sweep` | Hard cap; a bad prompt cannot spray a repo |
| Idempotency marker | Never comments twice on the same head SHA |
| `bot_login` | Skips PRs the agent itself authored |

One PR failing never aborts a sweep — failures are recorded on the report.
