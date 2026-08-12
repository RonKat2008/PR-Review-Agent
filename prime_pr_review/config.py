"""Configuration loading and validation.

Config loads successfully even when the repo and secrets are unset, so the rest of
the system is testable before credentials exist. The values are demanded at the
point of use via `require_repo` / `require_secrets`, which fail with an explicit
message naming exactly what is missing.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, replace
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
    # Hard write-ban for this repo: the comment sink is refused regardless of
    # dry_run or any other setting. For repos the agent may read but must never
    # post to (owner's standing instruction, e.g. example-org/service-a and
    # example-org/service-b).
    read_only: bool = False

    @property
    def is_set(self) -> bool:
        return bool(self.owner and self.name)

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.name}"


@dataclass(frozen=True)
class RepoEntry:
    """One `[[repos]]` entry: a target repo plus its per-repo overrides.

    `resolve_active` turns exactly one entry into the active Config by
    replacing `[repo]` and overriding `review.repo_root` / `review.graph_path`
    with these values -- but only when they are non-empty (see
    `resolve_active`), so an entry that leaves them blank does not blank out a
    value the flat `[review]` block already had.
    """

    repo: RepoConfig
    repo_root: str = ""
    graph_path: str = ""


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
    # Local checkout of the REVIEWED repository. Call-site discovery uses `git grep`
    # and sibling-test discovery globs the filesystem, so both need the target repo
    # on disk. Empty means unavailable — pointing this at the wrong checkout would
    # return call sites from an unrelated codebase, which is worse than none.
    repo_root: str = ""
    # Feed the reviewer full changed files, call sites, sibling tests, and repo
    # conventions alongside the diff. Requires repo_root for the local parts.
    gather_context: bool = True
    # Byte budget for that context, split across its sections.
    max_context_bytes: int = 400_000
    # Run the two-pass intent check: does the diff do what the PR claims?
    check_intent: bool = True
    # Run blast-radius analysis: what else does this change break? Needs repo_root,
    # since call-site discovery shells out to `git grep`.
    check_blast: bool = True
    # Knowledge-graph file for the reviewed repo (docs/KNOWLEDGE-GRAPH.md schema).
    # Empty means no graph: the review degrades to grep-based context with a note.
    # A stale graph (commit not an ancestor of the PR base) is refused, not used.
    graph_path: str = ""
    # Allow a CRITICAL finding or a broken caller to submit REQUEST_CHANGES, which
    # blocks the merge under branch protection. Off by default: automatically
    # blocking a colleague's merge is a strong action and should be chosen, not
    # inherited from a default.
    allow_request_changes: bool = False
    # Selective enrichment (C4): every skipped pass is a model call saved.
    # Intent checking runs only when the diff touches at least this many files
    # (0 = always). Single-file diffs rarely hide scope creep.
    intent_min_files: int = 0
    # Diffs where EVERY file matches one of these globs skip the intent and
    # blast passes entirely — a docs-only PR cannot break callers, and its
    # "intent" is its text. The base review still runs.
    docs_globs: tuple[str, ...] = ("**/*.md", "**/*.rst", "**/*.txt", "docs/**")
    # CI awareness (P12): inject build status + failing-log tail as evidence.
    check_ci: bool = True
    # Unwired-export detection (P13): diff-added public symbols with zero
    # production callers. Needs repo_root (grep runs in the checkout).
    check_exports: bool = True
    # Ensemble (P2): number of independent review runs per PR. 1 = off (single
    # call, self-reported confidence — known to be uncalibrated). 3 = the
    # measured fix, at 3x review cost. min_confidence only becomes a real gate
    # when this is > 1, because only then is confidence an observed quantity.
    ensemble_size: int = 1
    # Findings must appear in at least this many runs to survive (absolute count).
    min_agreement: int = 2


@dataclass(frozen=True)
class SinkConfig:
    pr_comment: bool
    webhook: bool
    local_file: bool
    webhook_kind: str
    # Deliver findings as line-anchored review comments with committable
    # suggestions rather than one summary comment at the bottom of the PR.
    inline_comments: bool = True


@dataclass(frozen=True)
class Config:
    repo: RepoConfig
    schedule: ScheduleConfig
    review: ReviewConfig
    sinks: SinkConfig
    # Multi-repo targets (A5). Empty when config.toml has no [[repos]] entries,
    # in which case [repo] above is the whole story. `resolve_active` folds
    # exactly one entry onto a copy of this Config, so sweep.py never has to
    # know multi-repo config exists.
    repos: tuple[RepoEntry, ...] = ()


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
    repos = raw.get("repos", [])

    return Config(
        repo=RepoConfig(
            owner=str(repo.get("owner", "")).strip(),
            name=str(repo.get("name", "")).strip(),
            read_only=bool(repo.get("read_only", False)),
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
            repo_root=str(review.get("repo_root", "")).strip(),
            gather_context=bool(review.get("gather_context", True)),
            max_context_bytes=int(review.get("max_context_bytes", 400_000)),
            check_intent=bool(review.get("check_intent", True)),
            check_blast=bool(review.get("check_blast", True)),
            allow_request_changes=bool(review.get("allow_request_changes", False)),
            graph_path=str(review.get("graph_path", "")).strip(),
            intent_min_files=int(review.get("intent_min_files", 0)),
            check_ci=bool(review.get("check_ci", True)),
            check_exports=bool(review.get("check_exports", True)),
            ensemble_size=int(review.get("ensemble_size", 1)),
            min_agreement=int(review.get("min_agreement", 2)),
            docs_globs=tuple(
                review.get("docs_globs", ("**/*.md", "**/*.rst", "**/*.txt", "docs/**"))
            ),
        ),
        sinks=SinkConfig(
            pr_comment=bool(sinks.get("pr_comment", True)),
            webhook=bool(sinks.get("webhook", True)),
            local_file=bool(sinks.get("local_file", True)),
            webhook_kind=str(sinks.get("webhook_kind", "slack")),
            inline_comments=bool(sinks.get("inline_comments", True)),
        ),
        repos=tuple(_build_repo_entry(entry) for entry in repos),
    )


def _build_repo_entry(raw: dict) -> RepoEntry:
    return RepoEntry(
        repo=RepoConfig(
            owner=str(raw.get("owner", "")).strip(),
            name=str(raw.get("name", "")).strip(),
            read_only=bool(raw.get("read_only", False)),
        ),
        repo_root=str(raw.get("repo_root", "")).strip(),
        graph_path=str(raw.get("graph_path", "")).strip(),
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
    if review.ensemble_size < 1:
        problems.append(f"review.ensemble_size must be >= 1, got {review.ensemble_size}")
    # min_agreement is only meaningful when the ensemble is on; with size 1 the
    # single run is authoritative and the knob is ignored.
    elif review.ensemble_size > 1 and not (1 <= review.min_agreement <= review.ensemble_size):
        problems.append(
            "review.min_agreement must be between 1 and ensemble_size, "
            f"got {review.min_agreement} with ensemble_size {review.ensemble_size}"
        )
    if config.sinks.webhook_kind not in WEBHOOK_KINDS:
        problems.append(
            f"sinks.webhook.kind must be one of {WEBHOOK_KINDS}, "
            f"got {config.sinks.webhook_kind!r}"
        )
    problems.extend(_duplicate_repo_problems(config.repos))

    if problems:
        raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(problems))


def _duplicate_repo_problems(repos: tuple[RepoEntry, ...]) -> list[str]:
    """[[repos]] entries sharing an owner/name pair (case-insensitively) would
    make `resolve_active` ambiguous by full slug -- reject at load time rather
    than let selection silently pick whichever came first."""
    seen: set[str] = set()
    duplicates: set[str] = set()
    for entry in repos:
        key = entry.repo.slug.lower()
        if key in seen:
            duplicates.add(entry.repo.slug)
        seen.add(key)

    if not duplicates:
        return []
    return [
        "repos entries must have unique owner/name pairs; duplicated: "
        + ", ".join(sorted(duplicates))
    ]


def require_repo(config: Config) -> RepoConfig:
    """Return the repo config, or fail with instructions if it is still a placeholder."""
    if not config.repo.is_set:
        raise ConfigError(
            "No target repository configured. Set [repo] owner and name in config.toml."
        )
    return config.repo


def resolve_active(config: Config, selector: str = "") -> Config:
    """Materialize a single-repo Config from `config.repos` for `selector`.

    The rest of the system -- sweep_lane included -- only ever consumes a
    single active `[repo]`. This picks one `[[repos]]` entry and folds it onto
    a NEW Config via `dataclasses.replace`, so nothing downstream has to know
    multi-repo config exists.

    `selector` matches an entry's full `owner/name` slug or its bare `name`,
    case-insensitively. Resolution rules:
      - No `[[repos]]` entries at all: `config` is returned unchanged -- the
        flat `[repo]` block is the whole story. A `selector` in this case is
        an error, since there is nothing to select from.
      - No selector, exactly one entry: that entry is used.
      - No selector, more than one entry: ConfigError telling the caller to
        pass `--repo`.
      - A selector matching no entry: ConfigError listing the available
        `owner/name` slugs.

    `repo_root` / `graph_path` from the chosen entry override the equivalent
    `review.*` fields only when the entry actually sets them (non-empty).
    """
    if not config.repos:
        if selector:
            raise ConfigError(
                f"No [[repos]] entries configured; cannot select {selector!r}. "
                "Add [[repos]] entries to config.toml, or omit --repo to use "
                "the [repo] fallback."
            )
        return config

    entry = _select_repo_entry(config.repos, selector)
    return replace(
        config,
        repo=entry.repo,
        review=replace(
            config.review,
            repo_root=entry.repo_root or config.review.repo_root,
            graph_path=entry.graph_path or config.review.graph_path,
        ),
    )


def _select_repo_entry(repos: tuple[RepoEntry, ...], selector: str) -> RepoEntry:
    """Pick one entry from `repos`, applying the "no selector" rules."""
    available = ", ".join(entry.repo.slug for entry in repos)

    if not selector:
        if len(repos) == 1:
            return repos[0]
        raise ConfigError(
            f"Multiple [[repos]] entries configured ({available}); pass "
            "--repo to choose one."
        )

    needle = selector.strip().lower()
    for entry in repos:
        if needle == entry.repo.slug.lower() or needle == entry.repo.name.lower():
            return entry

    raise ConfigError(f"Unknown repo {selector!r}. Available: {available}")


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
