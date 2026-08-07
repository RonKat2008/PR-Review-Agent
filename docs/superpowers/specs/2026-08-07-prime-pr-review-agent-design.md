# PRIME PR Review Agent — Design

**Date:** 2026-08-07
**Status:** Approved, implemented

## Problem

Automatically review pull requests on a repository the author contributes to, and
report two things: bugs a PR introduces, and bugs a PR fixes. Cover both open PRs
(before merge) and recently merged PRs (after the fact).

## Approach

A [prime-agent](https://github.com/PrimeIntellect-ai/prime-agent) skill. prime-agent
supplies three primitives this job needs, so none are rebuilt here:

- **Cron scheduling** — `prime-agent schedule add worker "<cron>" -- "<prompt>"`
- **Persistent detached workers** — schedules survive terminal detach and worker restart
- **RLM subagents** — one review context per PR, fanned out concurrently

Windows runs natively: prime-agent probes `C:\Program Files\Git\bin\bash.exe`, which is
present. No WSL.

## Architecture

```
prime-agent worker (persistent)
  └─ cron tick ──> skills/pr-review ──> prime_pr_review.sweep.sweep_lane()
        ├─ select candidates   (gh pr list, per lane)
        ├─ skip                (already reviewed at head SHA | bot-authored)
        ├─ fetch + filter diff (drop lockfiles/build output, cap size)
        ├─ review              (injected reviewer -> RLM subagent -> verdict JSON)
        ├─ sink                (local file always; PR comment gated; webhook digest)
        └─ advance watermark
```

### Modules

| Module | Responsibility |
|---|---|
| `config.py` | Load/validate `config.toml`; demand repo and secrets at point of use |
| `github.py` | `gh` CLI wrapper; all subprocess use behind one injectable runner |
| `diffs.py` | Split unified diffs per file; filter ignored paths; cap size on file boundaries |
| `review.py` | Verdict schema, tolerant JSON parsing, confidence gate, comment rendering |
| `state.py` | Per-lane watermarks, head-SHA dedupe, idempotency marker |
| `sinks.py` | Local audit trail, gated PR comments, webhook digest |
| `sweep.py` | Orchestration; converts per-PR failure into recorded outcome |
| `__main__.py` | `check` preflight — validates everything except the model |

### Key decisions

**Structured verdicts, not prose.** Subagents return a fixed JSON shape
(`introduces[]`, `fixes[]`, `confidence`). Prose cannot be thresholded, deduplicated,
or counted against a rate cap — the structure is what makes every safety gate possible.

**The reviewer is injected.** `sweep_lane` takes a `reviewer` callable. Inside
prime-agent it is backed by `rlm(...)`; in tests it is a stub. That single seam is why
the entire pipeline is testable without a model, a token, or a network.

**Two lanes, one machine.** Open and merged share all machinery and differ only in
candidate selection and prompt. The merged prompt additionally weights interaction
effects between co-merged PRs, since post-hoc regressions rarely come from one diff
alone.

**Dedupe on head SHA, not PR number.** A PR is re-reviewed exactly when new commits
land, and never otherwise.

## Safety

Output is public, so gates are defense-in-depth. All are enforced in one pure function,
`sinks.evaluate_comment_gates`, kept separate from the side effect so each is directly
testable.

| Gate | Effect |
|---|---|
| `dry_run` (ships `true`) | Nothing reaches GitHub |
| `min_confidence` | Low-confidence verdicts stay local |
| Silent-verdict check | "Found nothing" never posts |
| `max_comments_per_sweep` | A bad prompt cannot spray the repo |
| Idempotency marker | Hidden `<!-- prime-agent-review:<sha> -->`; never comments twice |
| `bot_login` | Skips self-authored PRs |

If existing comments cannot be read, the sweep **refuses to post** rather than risk a
duplicate. The local file sink always runs — every verdict is on disk whether or not it
was allowed out.

Prompts are calibrated for precision over recall: one confident wrong finding costs more
trust than ten missed subtle bugs.

## Secrets

`GITHUB_TOKEN` and `PRIME_REVIEW_WEBHOOK_URL` come from the environment only. `.env` is
gitignored; `require_secrets` fails at startup naming every missing variable at once.

## Error handling

One failing PR never aborts a sweep. Diff-fetch failures, unparseable verdicts, and
subagent crashes become recorded outcomes on the report and appear in the digest. A PR
that errored is **not** marked reviewed, so it retries next sweep.

## Testing

131 tests, 97% coverage. Everything external — gh, filesystem, webhook, model — is
injected. Explicit cases cover every safety gate, head-SHA re-review, budget exhaustion,
and partial-failure isolation.

## Deferred

- Real repo, tokens, and webhook URL (supplied later; `check` validates them)
- Registering the cron schedules against a live worker
- Calibrating `min_confidence` against observed output before leaving dry-run
