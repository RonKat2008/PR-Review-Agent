# GitHub Actions delivery (Phase C)

Two workflows: `.github/workflows/pr-review.yml` (reviews one PR per run) and
`.github/workflows/build-graph.yml` (mines a co-change knowledge graph). Both
are described here as what they actually do today, plus what changes once
they are installed elsewhere. See `docs/FINAL-PLAN.md` §2 Phase C for how this
fits the larger plan, and `docs/KNOWLEDGE-GRAPH.md` for the graph schema.

---

## 1. Deployment model — read this first

**These workflow files live in this agent repo (`pr-review-agent`) as the
deployable unit. They are not, by themselves, a live deployment against
example-org/service-a or example-org/service-b.**

A GitHub Actions token (`secrets.GITHUB_TOKEN`) is always scoped to exactly
the one repository whose workflow is running. It cannot read or comment on
PRs in a *different* repository, even one owned by the same account, and
this agent repo (`RonKat2008/pr-review-agent`) is not in the same GitHub
organization as the two real targets (`example-org/service-a`,
`example-org/service-b`). Practically, that means:

- As committed here, `pr-review.yml`'s automatic `pull_request_target`
  trigger reviews PRs opened against **whichever repo currently hosts this
  file** — today, that is this agent repo. It cannot see a example-org PR.
- `build-graph.yml` likewise mines the git history of **whichever repo it is
  checked out in** (`--repo .`). Run from here, it produces a graph of this
  agent repo's own history, not service-a's or service-b's.

**Installing into a target repo is the owner's manual step, and this repo
never performs it automatically.** To make either workflow live against
example-org/service-a or example-org/service-b, the owner copies the relevant
file(s) directly into *that* repo's own `.github/workflows/`:

| Workflow | Files to copy | Why both |
|---|---|---|
| `build-graph.yml` | `build-graph.yml` **and** `scripts/build_cochange.py` | The script imports only the standard library (see its module docstring) specifically so it is copyable on its own; the workflow has no `pip install` step because of this. |
| `pr-review.yml` | `pr-review.yml`, plus a way to obtain `prime_pr_review/`, `scripts/run_sweep.py`, `skills/pr-review/prompts/`, and `config.toml` on the runner | Unlike the co-change miner, the reviewer is a real installed package (`pip install -e .`) with prompt files alongside it — there is no standalone-script shortcut for it today. **This is an open decision, not yet solved by anything in this repo** — see §7. |

Once installed directly into a target repo, both files work unmodified: they
already resolve `--repo`/history to `github.repository` / the local
checkout, never a hardcoded slug, and `pr-review.yml`'s graph-download step
looks for `build-graph.yml`'s artifact in the *same* repository it runs in —
so a target repo that has both files installed is self-contained.

Until that copy happens, the only two things you can exercise against this
agent repo's own Actions tab are (a) the workflow mechanics themselves —
checkout, install, degrade-gracefully-without-a-graph, artifact upload, job
summary — and (b) `workflow_dispatch` runs whose `repo` input is this agent
repo itself. See §5.

---

## 2. Secrets — exactly what to create, and where

Both are **repository secrets** on whichever repo is hosting the workflow
file at the time (`RonKat2008/pr-review-agent` today; the target repo,
once installed there): **Settings → Secrets and variables → Actions → New
repository secret**.

| Secret | Used by | Notes |
|---|---|---|
| `GEMINI_API_KEY` | `pr-review.yml` only | Same key `resolve_api_key()` reads locally via the `GEMINI_API_KEY` env var. Create it explicitly — there is no fallback in CI (the `~/.prime/agent/auth.json` fallback only exists on a machine that has it). |
| `GITHUB_TOKEN` | Both workflows, implicitly | **Do not create this one.** It is the built-in, auto-generated, per-run token GitHub always provides as `secrets.GITHUB_TOKEN` — repository secrets cannot even be named `GITHUB_TOKEN` (the `GITHUB_` prefix is reserved). `pr-review.yml` reads it into `GH_TOKEN` for the `gh` CLI; `build-graph.yml` never calls `gh` at all (it only runs local `git` commands), so it does not need it in `env:` explicitly — `actions/checkout` uses it internally regardless. |

`build-graph.yml` needs **no secrets beyond the automatic `GITHUB_TOKEN`** —
it never calls `gh` or any external API.

The `permissions:` block declared in each workflow file (`pull-requests:
write, contents: read` for `pr-review.yml`; `contents: read` only for
`build-graph.yml`) is sufficient on its own; nothing needs to be changed
under **Settings → Actions → General → Workflow permissions** for these two
workflows specifically (an explicit `permissions:` block always overrides
that repo-level default).

---

## 3. The `pull_request_target` risk, in plain language

`pull_request_target` runs with **this repo's secrets** (`GEMINI_API_KEY`,
the write-scoped `GITHUB_TOKEN`) even when the PR that triggered it was
opened by a total stranger, from a fork, containing whatever code they
wrote. That combination — real secrets, plus a trigger anyone can pull just
by opening a PR — is the single most exploited GitHub Actions
misconfiguration: if the workflow ever checks out and *runs* the PR's code
(installs its dependencies, builds it, executes its tests), that code runs
with access to the secrets and can exfiltrate them.

**Reading the PR's diff as text is safe. Executing anything from the PR is
not.** `pr-review.yml` only ever does the former:

- The checkout step has no `ref:` override, so it checks out this repo's own
  base branch — never the PR's head commit or branch (see the checkout
  step's own comment in the YAML for exactly how).
- `pip install -e .` installs this repo's own `pyproject.toml`, from that
  same base-branch checkout. The PR's code is never present on disk.
- The PR's actual changes reach the reviewer exactly once, as a text blob,
  via `gh pr diff` inside `run_sweep.py` — never cloned, built, linted
  against a local copy, or executed in any form.

This is also why the P3 static-analysis pre-pass (ruff/bandit/mypy) and the
P1 context-gathering / P9 blast-radius passes are inert in this workflow:
all three require `review.repo_root` to point at a local checkout of the
*reviewed* repo (they shell out to `git grep` or read files on disk), and
this workflow never creates one. That is a side effect of the "text only"
design, not a bug to fix — see §7.

---

## 4. `read_only` mode: artifacts and job summaries, not comments

`config.toml` sets `read_only = true` on both `[[repos]]` entries for
`example-org/service-a` and `example-org/service-b` (the owner's standing
instruction: never upload anything to either repo). `sinks.py`'s
`evaluate_comment_gates` checks this **before** `dry_run`, on purpose — it
is a standing write-ban on the repo itself, not a mode a config flag can
override.

**This is intentional, and it is the expected mode until the owner flips
`read_only` to `false`.** A sweep against either repo still runs to
completion and still writes a review to `reviews/*.md` — the comment sink is
simply refused. Under Actions, that means the review is only ever visible
as:

1. The **`pr-review-<pr>-<run>` artifact** (the `reviews/` directory,
   uploaded every run, retained 90 days) — download it from the run's
   Summary page, or via `gh run download`.
2. The **job summary** — the newest review file's contents, appended
   directly to the run's page under GitHub's own "Summary" tab.

No comment ever lands on the PR in this mode, regardless of confidence,
severity, or how many findings there are.

---

## 5. Testing with `workflow_dispatch` before enabling the PR trigger

Both workflows are triggered manually the same way: **Actions tab → select
the workflow → "Run workflow"**.

For `pr-review.yml` specifically, `repo` and `pr` are both required inputs,
so a manual run is fully explicit about what it reviews — nothing is
inferred from an ambient PR event. Recommended order:

1. Confirm `config.toml`'s `review.dry_run = true` (it is, by default) —
   this means nothing can post even if every other gate would allow it, so a
   dispatch run is safe to try before you trust the output.
2. **Actions → PR Review → Run workflow.** Pick the branch to run from
   (`main`), then fill in `repo` (an `owner/name` matching a `config.toml`
   `[[repos]]` entry) and `pr` (an existing PR number in that repo).
3. Watch the run. Every step should complete even with no knowledge graph
   present (the graph-download step notices and continues; §7 explains why
   the graph is inert either way until `repo_root` is set).
4. Open the run's Summary page: the rendered review should be there, and the
   `reviews/` artifact should contain the same file.
5. Only after a dispatch run looks right should you consider flipping
   `review.dry_run` to `false`, and/or performing the install-into-target-repo
   step from §1 so the automatic `pull_request_target` trigger has something
   real to fire on.

Note the one real constraint from §1: a `workflow_dispatch` run's `gh` calls
still only work for whichever repo the token is scoped to (the repo hosting
the workflow at dispatch time). A dispatch run against `example-org/service-a`
triggered *from this agent repo* will fail to read that PR — dispatch has to
happen from wherever the file is actually installed for the token to have
access.

`build-graph.yml`'s `workflow_dispatch` takes no inputs — a manual run mines
whatever is checked out (this repo, until installed elsewhere) and uploads
the `cochange-graph` artifact exactly as the scheduled/push-triggered runs
do.

---

## 6. Rollback

**Fastest: disable the workflow, no code change needed.** Actions tab →
select the workflow (`PR Review` or `Build Knowledge Graph`) → "…" menu (top
right) → **Disable workflow**. This stops every trigger — automatic and
manual — immediately, and is reversible the same way ("Enable workflow").
Nothing in-flight is affected retroactively; a run already executing
finishes normally.

If you want the trigger gone at the source instead (e.g. before a longer
pause), comment out or remove the `pull_request_target:` block under `on:`
in a follow-up commit — `workflow_dispatch` can be left in place either way,
since it never fires on its own.

There is no state to roll back beyond the workflow itself: nothing here
writes to the repo (`contents: read` only), and `state/*.json` is
gitignored, so no watermark persists across runs to clean up (see §7).

---

## 7. Known limitations (candidates for follow-up work)

These are properties of the current design, not bugs in these two files —
recorded here so they are not silently rediscovered later.

1. **The reviewer's Python source has no cross-repo install path yet.**
   `build_cochange.py` is deliberately dependency-free so it can be copied
   standalone into a target repo (§1). `prime_pr_review` has no equivalent —
   installing `pr-review.yml` into `example-org/service-a` or
   `example-org/service-b` needs a way to get the package onto that runner
   (vendoring the source, a git submodule, publishing it as an installable
   package, or a cross-org PAT stored under a non-reserved secret name).
   None of those is chosen yet; this file only documents that the decision
   is still open.

2. **The knowledge-graph plumbing is wired but currently inert against real
   targets.** `graph.py`'s freshness check runs
   `git merge-base --is-ancestor <graph commit> origin/<base ref>` against
   `review.repo_root` — which requires a local checkout of the *reviewed*
   repo. `config.toml`'s `[[repos]]` entries leave `repo_root = ""` (`TODO`,
   not yet set by the owner), so `_graph_section` in `sweep.py` returns
   "graph skipped: no git runner configured" before `graph_path` is even
   consulted — independent of whether `pr-review.yml`'s download step found
   an artifact. `run_sweep.py` has no `--graph-path` or `--repo-root` CLI
   flag to override this per-invocation from CI, so today's checked-in
   `graphs/service-a-cochange.json` and `graphs/service-b-cochange.json` (built
   locally, once) remain the only usable graphs anywhere, CI included, until
   `repo_root` is set.

3. **CI state is cold on every run.** `state/*.json` is gitignored on
   purpose (per-machine watermarks), so a fresh checkout never has one:
   `run_sweep.py --pr N`'s watermark pre-check (`is_reviewed`) can never
   short-circuit a re-run of the same head SHA in Actions, unlike a
   long-lived local machine. This is not a correctness gap — the marker
   embedded in posted comments (`has_marker`, checked live against the PR's
   existing comments) is what actually prevents a duplicate comment, and
   that check does not depend on local state — but a re-triggered run at an
   unchanged head SHA still spends a full model call before reaching it.
   `concurrency: cancel-in-progress` (already in `pr-review.yml`) covers the
   common case (rapid pushes); a manually re-run job at the same SHA is the
   remaining, minor cost case.

---

## 8. `dry_run` × `read_only` × branch protection

`sinks.py`'s `evaluate_comment_gates` checks these in a fixed order
(`read_only` before `dry_run`, deliberately — see §4). This table shows the
net effect of every combination that matters:

| `read_only` | `dry_run` | `allow_request_changes` | Comments posted? | Can block a merge? |
|---|---|---|---|---|
| `true` | *(any)* | *(any)* | **Never.** Refused before `dry_run` is even checked. | No |
| `false` | `true` | *(any)* | **Never.** Review still runs; still written to `reviews/*.md` and this workflow's artifact/summary. | No |
| `false` | `false` | `false` (default) | Yes — as `COMMENT`-event PR reviews only, never `REQUEST_CHANGES`. | No — a plain comment review is never treated as blocking, regardless of branch protection rules. |
| `false` | `false` | `true` | Yes, and a `CRITICAL` finding or a broken caller may submit `REQUEST_CHANGES`. | Only if the target branch's protection rule requires approving reviews / disallows unresolved "changes requested" — and only if this bot's GitHub identity is not exempted from that rule. Off by default (`allow_request_changes = false`) precisely so this is an explicit, deliberate choice, not an inherited default. |

Both `[[repos]]` entries for example-org today are `read_only = true`, so row 1
applies regardless of anything else in `config.toml` — including the
`dry_run` and `allow_request_changes` values shown in the other rows.
