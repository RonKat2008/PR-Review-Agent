"""Preflight CLI: `python -m prime_pr_review check`.

Real sweeps run inside prime-agent, where an RLM subagent supplies the reviewer.
This entry point exists to validate that everything *around* the model is wired up
correctly — config, secrets, gh auth, repo reachability — before the first scheduled run.
"""

from __future__ import annotations

import argparse
import shutil
import sys

from . import github
from .config import Config, ConfigError, load_config, require_repo, require_secrets

OK = "  ok    "
FAIL = "  FAIL  "
WARN = "  warn  "


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="prime_pr_review")
    parser.add_argument("command", choices=("check",), help="only `check` is supported")
    parser.add_argument("--config", default="config.toml", help="path to config.toml")
    args = parser.parse_args(argv)

    if args.command == "check":
        return run_check(args.config)
    return 1


def run_check(config_path: str) -> int:
    """Validate the whole non-model surface. Returns a process exit code."""
    print("Prime PR Review — preflight\n")

    config = _check_config(config_path)
    if config is None:
        return 1

    failures = 0
    failures += 0 if _check_repo(config) else 1
    failures += 0 if _check_secrets(config) else 1
    failures += 0 if _check_gh(config) else 1

    print()
    if failures:
        print(f"{failures} check(s) failed. Resolve the above before scheduling a sweep.")
        return 1

    mode = "DRY RUN — nothing will post to GitHub" if config.review.dry_run else "LIVE — will post PR comments"
    print(f"All checks passed. Mode: {mode}")
    return 0


def _check_config(config_path: str) -> Config | None:
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        print(f"{FAIL} config: {exc}")
        return None
    print(f"{OK} config: loaded {config_path}")
    return config


def _check_repo(config: Config) -> bool:
    try:
        repo = require_repo(config)
    except ConfigError as exc:
        print(f"{FAIL} repo: {exc}")
        return False
    print(f"{OK} repo: {repo.slug}")
    return True


def _check_secrets(config: Config) -> bool:
    try:
        require_secrets(config)
    except ConfigError as exc:
        print(f"{FAIL} secrets: {exc}")
        return False
    print(f"{OK} secrets: present")
    return True


def _check_gh(config: Config) -> bool:
    if shutil.which("gh") is None:
        print(f"{FAIL} gh: not installed. See https://cli.github.com/")
        return False

    try:
        login = github.authenticated_login()
    except github.GitHubError as exc:
        print(f"{FAIL} gh: not authenticated ({exc})")
        return False
    print(f"{OK} gh: authenticated as {login}")

    if not config.review.bot_login:
        print(f"{WARN} bot_login is unset — the agent will not skip its own PRs")
    elif config.review.bot_login != login:
        print(
            f"{WARN} bot_login is {config.review.bot_login!r} "
            f"but gh is authenticated as {login!r}"
        )

    if not config.repo.is_set:
        return True

    try:
        github.list_open_prs(config.repo.slug, limit=1)
    except github.GitHubError as exc:
        print(f"{FAIL} repo access: cannot list PRs on {config.repo.slug} ({exc})")
        return False
    print(f"{OK} repo access: can list pull requests")
    return True


if __name__ == "__main__":
    sys.exit(main())
