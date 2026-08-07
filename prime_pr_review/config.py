"""Configuration loading and validation.

Config loads successfully even when the repo and secrets are unset, so the rest of
the system is testable before credentials exist. The values are demanded at the
point of use via `require_repo` / `require_secrets`, which fail with an explicit
message naming exactly what is missing.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("config.toml")

ENV_GITHUB_TOKEN = "GITHUB_TOKEN"
ENV_WEBHOOK_URL = "PRIME_REVIEW_WEBHOOK_URL"

WEBHOOK_KINDS = ("slack", "discord", "generic")

MIN_CONFIDENCE_FLOOR = 0.0
MIN_CONFIDENCE_CEILING = 1.0


class ConfigError(ValueError):
    """Configuration is missing, malformed, or internally inconsistent."""


@dataclass(frozen=True)
class RepoConfig:
    owner: str
    name: str

    @property
    def is_set(self) -> bool:
        return bool(self.owner and self.name)

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.name}"


@dataclass(frozen=True)
class ScheduleConfig:
    open_prs: str
    merged_prs: str


@dataclass(frozen=True)
class ReviewConfig:
    dry_run: bool
    min_confidence: float
    max_comments_per_sweep: int
    merged_lookback_days: int
    bot_login: str
    ignore_paths: tuple[str, ...]
    max_diff_bytes: int


@dataclass(frozen=True)
class SinkConfig:
    pr_comment: bool
    webhook: bool
    local_file: bool
    webhook_kind: str


@dataclass(frozen=True)
class Config:
    repo: RepoConfig
    schedule: ScheduleConfig
    review: ReviewConfig
    sinks: SinkConfig


@dataclass(frozen=True)
class Secrets:
    github_token: str
    webhook_url: str | None


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Config:
    """Read and validate config.toml. Does not touch the environment."""
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")

    try:
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Config file is not valid TOML: {config_path}: {exc}") from exc

    config = _build_config(raw)
    _validate(config)
    return config


def _build_config(raw: dict) -> Config:
    repo = raw.get("repo", {})
    schedule = raw.get("schedule", {})
    review = raw.get("review", {})
    sinks = raw.get("sinks", {})

    return Config(
        repo=RepoConfig(
            owner=str(repo.get("owner", "")).strip(),
            name=str(repo.get("name", "")).strip(),
        ),
        schedule=ScheduleConfig(
            open_prs=str(schedule.get("open_prs", "0 */4 * * *")),
            merged_prs=str(schedule.get("merged_prs", "0 9 * * 1-5")),
        ),
        review=ReviewConfig(
            dry_run=bool(review.get("dry_run", True)),
            min_confidence=float(review.get("min_confidence", 0.7)),
            max_comments_per_sweep=int(review.get("max_comments_per_sweep", 5)),
            merged_lookback_days=int(review.get("merged_lookback_days", 7)),
            bot_login=str(review.get("bot_login", "")).strip(),
            ignore_paths=tuple(review.get("ignore_paths", ())),
            max_diff_bytes=int(review.get("max_diff_bytes", 200_000)),
        ),
        sinks=SinkConfig(
            pr_comment=bool(sinks.get("pr_comment", True)),
            webhook=bool(sinks.get("webhook", True)),
            local_file=bool(sinks.get("local_file", True)),
            webhook_kind=str(sinks.get("webhook_kind", "slack")),
        ),
    )


def _validate(config: Config) -> None:
    review = config.review
    problems: list[str] = []

    if not MIN_CONFIDENCE_FLOOR <= review.min_confidence <= MIN_CONFIDENCE_CEILING:
        problems.append(
            f"review.min_confidence must be between {MIN_CONFIDENCE_FLOOR} and "
            f"{MIN_CONFIDENCE_CEILING}, got {review.min_confidence}"
        )
    if review.max_comments_per_sweep < 0:
        problems.append(
            f"review.max_comments_per_sweep must be >= 0, got {review.max_comments_per_sweep}"
        )
    if review.merged_lookback_days <= 0:
        problems.append(
            f"review.merged_lookback_days must be > 0, got {review.merged_lookback_days}"
        )
    if review.max_diff_bytes <= 0:
        problems.append(f"review.max_diff_bytes must be > 0, got {review.max_diff_bytes}")
    if config.sinks.webhook_kind not in WEBHOOK_KINDS:
        problems.append(
            f"sinks.webhook.kind must be one of {WEBHOOK_KINDS}, "
            f"got {config.sinks.webhook_kind!r}"
        )

    if problems:
        raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(problems))


def require_repo(config: Config) -> RepoConfig:
    """Return the repo config, or fail with instructions if it is still a placeholder."""
    if not config.repo.is_set:
        raise ConfigError(
            "No target repository configured. Set [repo] owner and name in config.toml."
        )
    return config.repo


def require_secrets(config: Config, env: dict[str, str] | None = None) -> Secrets:
    """Read secrets from the environment, demanding only what this config actually uses."""
    source = os.environ if env is None else env
    missing: list[str] = []

    token = source.get(ENV_GITHUB_TOKEN, "").strip()
    if not token:
        missing.append(f"{ENV_GITHUB_TOKEN} (required to read pull requests)")

    webhook_url = source.get(ENV_WEBHOOK_URL, "").strip() or None
    if config.sinks.webhook and not webhook_url:
        missing.append(f"{ENV_WEBHOOK_URL} (required because sinks.webhook = true)")

    if missing:
        raise ConfigError(
            "Missing required environment variables:\n  - "
            + "\n  - ".join(missing)
            + "\n\nCopy .env.example to .env and fill it in."
        )

    return Secrets(github_token=token, webhook_url=webhook_url)
