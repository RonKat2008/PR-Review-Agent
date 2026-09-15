# service-a Prime Agent — PR Review

A [prime-agent](https://github.com/PrimeIntellect-ai/prime-agent) skill that reviews
pull requests on a schedule and reports **bugs introduced** and **bugs fixed**.

Two lanes:

- **`open`** — pre-merge gate: "what breaks if this merges?"
- **`merged`** — retrospective: "what did this fix, and what regressed?"

## Status

**Working.** First live sweep scored 4/4 against a controlled repo with planted
defects — including correctly staying silent on a behavior-preserving change.

See **[RESULTS.md](RESULTS.md)** for the full reviews and caveats.

Nothing has posted to GitHub — `dry_run = true`, and the real target repos are
configured `read_only`.

## Evaluation

Measured on **200 real pull requests** sampled from the public
[SWE-CARE](https://huggingface.co/datasets/inclusionAI/SWE-CARE) test split, scored
against the human review comments on each PR (a finding matches a comment on the same
file within ±5 lines). Every model call was recorded once and each ablation arm is
rebuilt offline from those recordings, so the table below is reproducible without
re-running a model.

| arm | PRs | precision | recall | findings/PR | fabrication | $/PR |
|---|---|---|---|---|---|---|
| single seat (mean of 3) | 185 | 0.25 | 0.20 | 1.39 | 0.00 | 0.05 |
| 3-lab ensemble | 192 | 0.24 | **0.34** | 3.45 | 0.00 | 0.14 |
| + judge-merge | 192 | 0.22 | 0.30 | 2.59 | 0.00 | 0.15 |
| + skeptic (full pipeline) | 192 | 0.23 | 0.27 | 2.17 | 0.00 | 0.21 |

What it shows:

- **The ensemble is the win.** Three model families voting blind raise recall from
  0.20 (average seat) and 0.24 (strongest seat) to 0.34, at unchanged precision.
- **The adversarial passes trade recall for noise.** Judge-merge and the skeptic cut
  findings per PR from 3.45 to 2.17, but do not raise precision against human comments.
- **Citations are sound.** Near-zero fabricated file/line citations across every arm.
- **Precision near 0.25 is in line with published benchmarks** for automated review
  against human comments, and is a lower bound: a real defect no human commented on
  counts as a false positive.

Full ladder, per-severity breakdown, cost and wall time per arm, and every limitation:
[`docs/eval/swecare-200-s0.md`](docs/eval/swecare-200-s0.md). Reproduce with
`prime-review eval run --count 200 --seed 0`, then `eval score` and `eval report`.

## Setup

```bash
uv venv --python 3.13
uv pip install -e ".[dev]"
```

Then, once you have credentials:

1. Set `[repo] owner` and `name` in `config.toml`
2. Set `[review] bot_login` to the account the agent posts as
3. `cp .env.example .env` and fill in `GITHUB_TOKEN` + `PRIME_REVIEW_WEBHOOK_URL`
4. Install [`gh`](https://cli.github.com/) and run `gh auth login`

Verify everything except the model is wired up:

```bash
python -m prime_pr_review check
```

It names precisely what is missing and exits non-zero until all of it is resolved.

## Running

Sweeps run inside prime-agent, where an RLM subagent supplies the reviewer. See
[`skills/pr-review/SKILL.md`](skills/pr-review/SKILL.md) for the invocation and the
schedule registration commands.

## Safety

Ships with `dry_run = true`: reviews are written to `reviews/` and pushed to the
webhook, but **not** posted on PRs. Read a few sweeps, then flip one line in
`config.toml`.

Other gates: confidence threshold, per-sweep comment cap, idempotency marker (never
comments twice on the same head SHA), and self-authored-PR exclusion. All enforced in
`sinks.evaluate_comment_gates`.

## Layout

```
prime_pr_review/             config, gh wrapper, diffs, verdicts, state, sinks, sweep,
                             citations, ensemble, judge, refute, Prime Inference provider
prime_pr_review/evaluation/  SWE-CARE corpus, recording/replay, ablation arms, scoring
scripts/eval_swecare.py      prime-review eval run | score | report
skills/pr-review/            SKILL.md + review, intent, blast, judge and skeptic prompts
tests/                       1,000+ tests, everything external injected
docs/eval/                   published evaluation results
docs/superpowers/            design specs and implementation plans
```

## Tests

```bash
python -m pytest
```
