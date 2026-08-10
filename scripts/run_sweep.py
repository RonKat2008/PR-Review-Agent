"""Run a sweep against the configured repo using a Gemini-backed reviewer.

    python scripts/run_sweep.py --lane open

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

from prime_pr_review.config import ConfigError, load_config, require_repo  # noqa: E402
from prime_pr_review.reviewers import gemini_model_fn, gemini_reviewer  # noqa: E402
from prime_pr_review.state import (  # noqa: E402
    LANE_MERGED,
    LANE_OPEN,
    load_state,
    save_state,
)
from prime_pr_review.sweep import Enrichment, sweep_lane  # noqa: E402

AUTH_FILE = Path.home() / ".prime" / "agent" / "auth.json"


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a PR review sweep")
    parser.add_argument("--lane", choices=(LANE_OPEN, LANE_MERGED), default=LANE_OPEN)
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore the watermark and re-review everything.",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        repo = require_repo(config)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1

    state = load_state()
    if args.fresh:
        from prime_pr_review.state import State

        state = State.empty()

    mode = "DRY RUN (nothing posts)" if config.review.dry_run else "LIVE (will comment)"
    print(f"Sweeping {repo.slug} | lane={args.lane} | model={args.model} | {mode}\n")

    api_key = resolve_api_key()
    reviewer = gemini_reviewer(api_key, model=args.model)
    enrichment = Enrichment(
        model_fn=gemini_model_fn(api_key, model=args.model),
        repo_root=Path(config.review.repo_root or "."),
    )

    report, state = sweep_lane(config, args.lane, reviewer, state, enrichment=enrichment)
    save_state(state)

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
