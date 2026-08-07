---
name: pr-review
description: Review open and recently merged pull requests for bugs introduced and bugs fixed. Fetches PR diffs via the gh CLI, fans out one RLM subagent per PR, and delivers structured verdicts to local files, a webhook, and (when not in dry-run) PR comments.
---

# PR Review Sweep

Automated bug review over a GitHub repository, in two lanes:

- **`open`** — pre-merge gate. "What breaks if this merges?"
- **`merged`** — retrospective. "What did this fix, and what regressed?"

## Before the first run

1. Fill in `[repo] owner` and `name` in `config.toml`
2. Copy `.env.example` to `.env` and set `GITHUB_TOKEN` and `PRIME_REVIEW_WEBHOOK_URL`
3. Set `[review] bot_login` to the account this agent posts as
4. Confirm `gh` is installed and authenticated: `gh auth status`

`config.toml` ships with `dry_run = true`. Reviews are written to `reviews/` and pushed
to the webhook, but **not** posted on PRs. Flip it to `false` once you trust the output.

## Running a sweep

The sweep needs a `reviewer` — a callable that takes a PR, its filtered diff, and the
lane, and returns the verdict JSON. Inside prime-agent, back it with an RLM subagent so
each PR is reviewed in its own context:

```python
from pathlib import Path
from prime_pr_review.config import load_config, require_repo, require_secrets
from prime_pr_review.state import LANE_OPEN, load_state, save_state
from prime_pr_review.sweep import sweep_lane
from prime_pr_review.sinks import send_webhook

config = load_config("config.toml")
require_repo(config)
secrets = require_secrets(config)

PROMPTS = Path("skills/pr-review/prompts")

async def reviewer(pr, diff, lane):
    template = (PROMPTS / f"{lane}_pr.md").read_text(encoding="utf-8")
    return await rlm(
        f"{template}\n\n"
        f"## PR #{pr.number}: {pr.title}\n"
        f"author: {pr.author} | base: {pr.base_ref} | {pr.changed_files} files\n\n"
        f"## Diff\n\n```diff\n{diff}\n```"
    )

state = load_state()
report, state = sweep_lane(config, LANE_OPEN, reviewer, state)
save_state(state)

send_webhook(config, secrets, report.summaries())
print(f"{report.considered} considered, {report.reviewed} reviewed, "
      f"{report.posted} posted, {report.errors} errors")
```

Fan the PRs out concurrently rather than serially when the lane returns more than a
few — each subagent is independent.

## Scheduling

```bash
prime-agent schedule add worker "0 */4 * * *" -- "Run the pr-review skill, open lane"
prime-agent schedule add worker "0 9 * * 1-5" -- "Run the pr-review skill, merged lane"
prime-agent schedule list --all
```

Cron expressions live in `config.toml` under `[schedule]`; keep them in sync with what
you register.

## Safety gates

Every gate is enforced in `prime_pr_review/sinks.py:evaluate_comment_gates`:

| Gate | Effect |
|---|---|
| `dry_run` | Nothing posts to GitHub |
| `min_confidence` | Low-confidence verdicts stay local |
| `max_comments_per_sweep` | Hard cap; a bad prompt cannot spray the repo |
| Idempotency marker | Never comments twice on the same head SHA |
| `bot_login` | Skips PRs the agent itself authored |

A PR is re-reviewed when new commits change its head SHA, and only then.

## Failure behavior

One PR failing never aborts a sweep. Failures are recorded on the report and appear in
the digest. If existing comments cannot be read, the sweep refuses to post rather than
risk a duplicate.
