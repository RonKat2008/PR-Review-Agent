"""Run a sweep against the configured repo using a Gemini-backed reviewer.

    python scripts/run_sweep.py --lane open
    python scripts/run_sweep.py --repo example-org/service-b --pr 42

This is the headless path. It needs no prime-agent runtime, no TUI, and no daemon;
prime-agent's role is scheduling this, not performing it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prime_pr_review import github  # noqa: E402
from prime_pr_review.analysis import run_analysis  # noqa: E402
from prime_pr_review.config import (  # noqa: E402
    ConfigError,
    load_config,
    require_repo,
    resolve_active,
)
from prime_pr_review.feedback import FeedbackError, load_rejections  # noqa: E402
from prime_pr_review.graph import strict_runner  # noqa: E402
from prime_pr_review.reviewers import gemini_model_fn, gemini_reviewer  # noqa: E402
from prime_pr_review.state import (  # noqa: E402
    LANE_MERGED,
    LANE_OPEN,
    load_state,
    save_state,
)
from prime_pr_review.sweep import Enrichment, sweep_lane  # noqa: E402

AUTH_FILE = Path.home() / ".prime" / "agent" / "auth.json"

# Every agent-owned path is resolved against the agent's own directory, never the
# process working directory. Call-site discovery requires running from inside the
# reviewed repo's checkout, so cwd belongs to the repo under review — prompts,
# state, and reviews must not follow it there.
AGENT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = AGENT_ROOT / "skills" / "pr-review" / "prompts"
LEGACY_STATE_FILE = AGENT_ROOT / "state" / "watermark.json"
REVIEWS_DIR = AGENT_ROOT / "reviews"
REJECTIONS_FILE = AGENT_ROOT / "state" / "rejections.json"


def state_file_for(repo_slug: str) -> tuple[Path, Path]:
    """(load_from, save_to) watermark paths for one repo.

    PR numbers are only unique within one repository — a shared watermark would
    put service-a #100 and service-b #100 in the same namespace. Saves always go to
    the per-repo file; loading falls back to the legacy shared file exactly once
    (when no per-repo file exists yet) so demo history migrates rather than
    being re-reviewed.
    """
    per_repo = AGENT_ROOT / "state" / f"watermark-{repo_slug.replace('/', '-')}.json"
    load_from = per_repo
    if not per_repo.exists() and LEGACY_STATE_FILE.exists():
        load_from = LEGACY_STATE_FILE
    return load_from, per_repo


def resolve_api_key() -> str:
    """Prefer the environment; fall back to the prime-agent auth file."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key

    try:
        entry = json.loads(AUTH_FILE.read_text(encoding="utf-8")).get("google", {})
    except (OSError, json.JSONDecodeError):
        entry = {}

    key = str(entry.get("key", "")).strip()
    # The auth file may store an env var *name* rather than a literal key.
    if key and not key.startswith("AQ.") and not key.startswith("AIza"):
        key = os.environ.get(key, "").strip()

    if not key:
        raise SystemExit(
            "No Gemini key found. Set GEMINI_API_KEY, or put a literal key in "
            f"{AUTH_FILE} under \"google\"."
        )
    return key


def _single_pr_list_json(pr: github.PullRequest) -> str:
    """Serialize an already-fetched PullRequest back into the JSON array shape
    `gh pr list --json <PR_FIELDS>` would have produced for it.

    Feeds `github.single_pr_runner`: candidate selection inside `sweep_lane`
    always lists, so answering that one `pr list` call with this array is what
    lets `--pr N` reuse `sweep_lane`'s per-PR machinery -- dedup, diff fetch,
    rendering, posting, idempotency -- rather than a second, parallel
    implementation of any of it.
    """
    return json.dumps(
        [
            {
                "number": pr.number,
                "title": pr.title,
                "author": {"login": pr.author},
                "headRefOid": pr.head_sha,
                "baseRefName": pr.base_ref,
                "url": pr.url,
                "additions": pr.additions,
                "deletions": pr.deletions,
                "changedFiles": pr.changed_files,
                "mergedAt": pr.merged_at,
            }
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a PR review sweep")
    parser.add_argument("--lane", choices=(LANE_OPEN, LANE_MERGED), default=LANE_OPEN)
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument(
        "--repo",
        default="",
        help=(
            "Select one config.toml [[repos]] entry by 'owner/name' or bare "
            "name (case-insensitive). Required when more than one is "
            "configured; unused when config.toml has none (the flat [repo] "
            "block applies instead)."
        ),
    )
    parser.add_argument(
        "--repo-root",
        default="",
        help=(
            "Override review.repo_root for this run. Config values are local "
            "machine paths; under Actions the workspace IS the target checkout, "
            "so CI passes this explicitly."
        ),
    )
    parser.add_argument(
        "--graph-path",
        default="",
        help="Override review.graph_path for this run (e.g. a CI-downloaded artifact).",
    )
    lane_source = parser.add_mutually_exclusive_group()
    lane_source.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore the watermark and re-review everything.",
    )
    lane_source.add_argument(
        "--pr",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Review exactly PR N and skip listing entirely -- the entry "
            "point an event trigger calls."
        ),
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        config = resolve_active(config, args.repo)
        repo = require_repo(config)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1

    if args.repo_root or args.graph_path:
        from dataclasses import replace as _replace

        review = config.review
        if args.repo_root:
            review = _replace(review, repo_root=args.repo_root)
        if args.graph_path:
            review = _replace(review, graph_path=args.graph_path)
        config = _replace(config, review=review)

    print(f"Resolved repo: {repo.slug} (read_only={repo.read_only})")

    # A single PR is always swept through the "open" lane machinery, regardless
    # of --lane or the PR's real state: list_merged_prs additionally filters
    # results by merged_at, which would silently drop the very PR
    # single_pr_runner just resolved if it isn't reliably populated.
    # list_open_prs applies no such filter, so it is always safe here -- the
    # lane label only changes which watermark key the state file records the
    # review under.
    lane = LANE_OPEN if args.pr is not None else args.lane

    runner = github.default_runner
    if args.pr is not None:
        try:
            target_pr = github.get_pr(repo.slug, args.pr, runner)
        except github.GitHubError as exc:
            print(f"Could not fetch PR #{args.pr}: {exc}", file=sys.stderr)
            return 1
        runner = github.single_pr_runner(runner, _single_pr_list_json(target_pr))

    state_load_path, state_save_path = state_file_for(repo.slug)
    state = load_state(state_load_path)
    if args.fresh:
        from prime_pr_review.state import State

        state = State.empty()

    try:
        rejections = load_rejections(REJECTIONS_FILE)
    except FeedbackError as exc:
        print(f"Warning: rejection store unreadable, continuing without it: {exc}")
        rejections = ()

    mode = "DRY RUN (nothing posts)" if config.review.dry_run else "LIVE (will comment)"
    print(f"Sweeping {repo.slug} | lane={lane} | model={args.model} | {mode}\n")

    api_key = resolve_api_key()
    reviewer = gemini_reviewer(api_key, model=args.model, prompts_dir=PROMPTS_DIR)
    root_path = Path(config.review.repo_root or ".")
    enrichment = Enrichment(
        model_fn=gemini_model_fn(api_key, model=args.model),
        repo_root=root_path,
        prompts_dir=PROMPTS_DIR,
        # Strict on purpose: for `merge-base --is-ancestor`, exit 1 means "stale
        # graph, refuse" — the lenient grep runner would swallow it.
        git_runner=strict_runner(root_path) if config.review.repo_root else None,
        analysis_fn=run_analysis,
        rejections=rejections,
    )

    report, state = sweep_lane(
        config,
        lane,
        reviewer,
        state,
        runner=runner,
        enrichment=enrichment,
        reviews_dir=REVIEWS_DIR,
    )
    save_state(state, state_save_path)

    for line in report.summaries():
        print(f"  {line}")

    print(
        f"\n{report.considered} considered | {report.reviewed} reviewed | "
        f"{report.posted} posted | {report.skipped} skipped | {report.errors} errors"
    )
    print("Reviews written to reviews/")
    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())
