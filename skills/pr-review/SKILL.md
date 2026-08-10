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
VERDICTS = Path("state/verdicts")

async def reviewer(pr, diff, lane):
    """Spawn a child agent and wait for the verdict file it writes.

    IMPORTANT: `rlm(...)` returns an admission handle immediately. It does NOT
    return the child's answer. Results come back only through `agent_message`
    replies or files — here, a file, because the payload is structured JSON.
    """
    VERDICTS.mkdir(parents=True, exist_ok=True)
    out_path = (VERDICTS / f"{lane}-{pr.number}-{pr.head_sha[:8]}.json").resolve()
    out_path.unlink(missing_ok=True)

    template = (PROMPTS / f"{lane}_pr.md").read_text(encoding="utf-8")
    await rlm(
        f"{template}\n\n"
        f"## PR #{pr.number}: {pr.title}\n"
        f"author: {pr.author} | base: {pr.base_ref} | {pr.changed_files} files\n\n"
        f"## Diff\n\n```diff\n{diff}\n```\n\n"
        f"## Output\n\n"
        f"Write ONLY the JSON verdict to `{out_path}`. Reply with no prose.",
        name=f"review-{lane}-{pr.number}",
    )

    return await _await_verdict(out_path)


async def _await_verdict(path, timeout=300, interval=2):
    """Poll for the child's verdict file. Raises on timeout so the sweep records
    an error for this PR and moves on rather than hanging the whole run."""
    waited = 0
    while waited < timeout:
        if path.is_file() and path.stat().st_size > 0:
            return path.read_text(encoding="utf-8")
        await asyncio.sleep(interval)
        waited += interval
    raise TimeoutError(f"No verdict written to {path} within {timeout}s")

state = load_state()
report, state = sweep_lane(config, LANE_OPEN, reviewer, state)
save_state(state)

send_webhook(config, secrets, report.summaries())
print(f"{report.considered} considered, {report.reviewed} reviewed, "
      f"{report.posted} posted, {report.errors} errors")
```

### Why the file handoff

`rlm(...)` admits the child and returns a handle carrying `rlm_child_id`, `name`,
`session_dir`, and `model` — **never the child's answer**. Awaiting the call gives you
the handle, not a review. Children report back explicitly:

```python
await agent_message.send(text, receiver_role="parent")   # prose
```

For a structured verdict a file is the better channel: it survives compaction, it is
inspectable after the fact, and it does not need parsing out of a message stream.

Spawning is non-blocking, so for a lane returning many PRs, spawn every child first and
only then wait on the files — that is where the per-PR context isolation pays off.

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
