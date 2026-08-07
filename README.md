# service-a Prime Agent — PR Review

A [prime-agent](https://github.com/PrimeIntellect-ai/prime-agent) skill that reviews
pull requests on a schedule and reports **bugs introduced** and **bugs fixed**.

Two lanes:

- **`open`** — pre-merge gate: "what breaks if this merges?"
- **`merged`** — retrospective: "what did this fix, and what regressed?"

## Status

Built and tested. **Not yet connected** — needs a repo, a token, and a webhook URL.

```
131 tests passing, 97% coverage
```

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
prime_pr_review/       config, gh wrapper, diffs, verdicts, state, sinks, sweep
skills/pr-review/      SKILL.md + the two review prompts
tests/                 131 tests, everything external injected
docs/superpowers/specs/ design doc
```

## Tests

```bash
python -m pytest
```
