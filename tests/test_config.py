"""Config loading, validation, and the fail-loud behavior around missing credentials."""

from __future__ import annotations

from dataclasses import replace

import pytest

from prime_pr_review.config import (
    Config,
    ConfigError,
    RepoConfig,
    RepoEntry,
    load_config,
    require_repo,
    require_secrets,
    resolve_active,
)

from .conftest import make_config

MINIMAL_TOML = """
[repo]
owner = "acme"
name = "widget"

[review]
dry_run = true
min_confidence = 0.8
"""

REPOS_TOML = """
[repo]
owner = "acme"
name = "widget"

[[repos]]
owner = "example-org"
name = "service-a"
read_only = true
repo_root = "/checkouts/service-a"
graph_path = "graphs/service-a-cochange.json"

[[repos]]
owner = "example-org"
name = "service-b"
read_only = true
graph_path = "graphs/service-b-cochange.json"
"""

service-a_ENTRY = RepoEntry(
    repo=RepoConfig(owner="example-org", name="service-a", read_only=True),
    repo_root="/checkouts/service-a",
    graph_path="graphs/service-a-cochange.json",
)
SERVICE_B_ENTRY = RepoEntry(
    repo=RepoConfig(owner="example-org", name="service-b", read_only=True),
    graph_path="graphs/service-b-cochange.json",
)


def _with_repos(*entries: RepoEntry) -> Config:
    """A Config carrying the given [[repos]] entries, for resolve_active tests
    that exercise the function directly without round-tripping TOML."""
    return replace(make_config(), repos=entries)


def _write(tmp_path, text: str):
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_values_from_toml(tmp_path):
    # Arrange
    path = _write(tmp_path, MINIMAL_TOML)

    # Act
    config = load_config(path)

    # Assert
    assert config.repo.slug == "acme/widget"
    assert config.review.dry_run is True
    assert config.review.min_confidence == 0.8


def test_applies_defaults_for_omitted_values(tmp_path):
    config = load_config(_write(tmp_path, MINIMAL_TOML))

    assert config.review.max_comments_per_sweep == 5
    assert config.sinks.webhook_kind == "slack"
    assert config.review.max_diff_bytes == 200_000


def test_shipped_config_is_valid():
    """The config.toml committed to the repo must actually parse."""
    config = load_config("config.toml")

    assert config.review.dry_run is True, "shipped config must start in dry-run"


def test_repo_read_only_parses_from_toml(tmp_path):
    toml = '[repo]\nowner = "example-org"\nname = "service-a"\nread_only = true\n'

    config = load_config(_write(tmp_path, toml))

    assert config.repo.read_only is True


def test_repo_read_only_defaults_to_false(tmp_path):
    config = load_config(_write(tmp_path, MINIMAL_TOML))

    assert config.repo.read_only is False


def test_raises_when_config_file_missing(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


def test_raises_on_malformed_toml(tmp_path):
    with pytest.raises(ConfigError, match="not valid TOML"):
        load_config(_write(tmp_path, "this is [not toml"))


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("min_confidence", 1.5, "min_confidence"),
        ("min_confidence", -0.1, "min_confidence"),
        ("max_comments_per_sweep", -1, "max_comments_per_sweep"),
        ("merged_lookback_days", 0, "merged_lookback_days"),
        ("max_diff_bytes", 0, "max_diff_bytes"),
    ],
)
def test_rejects_out_of_range_values(tmp_path, field, value, message):
    toml = f'[repo]\nowner="a"\nname="b"\n\n[review]\n{field} = {value}\n'

    with pytest.raises(ConfigError, match=message):
        load_config(_write(tmp_path, toml))


def test_rejects_unknown_webhook_kind(tmp_path):
    toml = '[repo]\nowner="a"\nname="b"\n\n[sinks]\nwebhook_kind = "carrier-pigeon"\n'

    with pytest.raises(ConfigError, match="webhook.kind|webhook_kind|one of"):
        load_config(_write(tmp_path, toml))


def test_config_loads_without_repo_so_system_is_testable_before_setup(tmp_path):
    """The repo is a placeholder until the user fills it in; loading must still work."""
    config = load_config(_write(tmp_path, "[review]\ndry_run = true\n"))

    assert config.repo.is_set is False


def test_require_repo_fails_with_actionable_message():
    config = make_config(owner="", name="")

    with pytest.raises(ConfigError, match="config.toml"):
        require_repo(config)


def test_require_repo_returns_repo_when_set():
    assert require_repo(make_config()).slug == "acme/widget"


def test_require_secrets_reports_every_missing_variable_at_once():
    config = make_config(webhook=True)

    with pytest.raises(ConfigError) as exc:
        require_secrets(config, env={})

    message = str(exc.value)
    assert "GITHUB_TOKEN" in message
    assert "PRIME_REVIEW_WEBHOOK_URL" in message


def test_require_secrets_ignores_webhook_url_when_webhook_disabled():
    config = make_config(webhook=False)

    secrets = require_secrets(config, env={"GITHUB_TOKEN": "ghp_x"})

    assert secrets.github_token == "ghp_x"
    assert secrets.webhook_url is None


def test_require_secrets_treats_blank_token_as_missing():
    with pytest.raises(ConfigError, match="GITHUB_TOKEN"):
        require_secrets(make_config(webhook=False), env={"GITHUB_TOKEN": "   "})


# --- [[repos]] parsing (A5) --------------------------------------------------


def test_repos_array_parses_every_entry(tmp_path):
    config = load_config(_write(tmp_path, REPOS_TOML))

    assert [entry.repo.slug for entry in config.repos] == [
        "example-org/service-a",
        "example-org/service-b",
    ]


def test_repos_entry_read_only_true_is_honored(tmp_path):
    config = load_config(_write(tmp_path, REPOS_TOML))

    assert all(entry.repo.read_only for entry in config.repos)


def test_repos_entry_read_only_defaults_to_false(tmp_path):
    toml = '[[repos]]\nowner = "a"\nname = "b"\n'

    config = load_config(_write(tmp_path, toml))

    assert config.repos[0].repo.read_only is False


def test_repos_entry_repo_root_and_graph_path_default_to_empty(tmp_path):
    toml = '[[repos]]\nowner = "a"\nname = "b"\n'

    config = load_config(_write(tmp_path, toml))

    assert config.repos[0].repo_root == ""
    assert config.repos[0].graph_path == ""


def test_repos_entry_carries_its_own_repo_root_and_graph_path(tmp_path):
    config = load_config(_write(tmp_path, REPOS_TOML))

    service-a = config.repos[0]
    assert service-a.repo_root == "/checkouts/service-a"
    assert service-a.graph_path == "graphs/service-a-cochange.json"


def test_flat_repo_block_still_parses_when_repos_array_is_present(tmp_path):
    """[[repos]] is additive -- [repo] keeps parsing exactly as before."""
    config = load_config(_write(tmp_path, REPOS_TOML))

    assert config.repo.slug == "acme/widget"


def test_duplicate_repo_entries_are_rejected(tmp_path):
    toml = (
        '[[repos]]\nowner = "example-org"\nname = "service-a"\n\n'
        '[[repos]]\nowner = "example-org"\nname = "service-a"\n'
    )

    with pytest.raises(ConfigError, match="unique"):
        load_config(_write(tmp_path, toml))


def test_shipped_config_has_both_target_repos_as_read_only():
    """Owner's standing instruction: never write to either target repo."""
    config = load_config("config.toml")

    read_only_by_slug = {entry.repo.slug: entry.repo.read_only for entry in config.repos}

    assert read_only_by_slug == {
        "example-org/service-a": True,
        "example-org/service-b": True,
    }


# --- resolve_active -----------------------------------------------------------


def test_resolve_active_matches_full_slug_case_insensitively():
    config = _with_repos(service-a_ENTRY, SERVICE_B_ENTRY)

    resolved = resolve_active(config, "example-org/service-a")

    assert resolved.repo.slug == "example-org/service-a"


def test_resolve_active_matches_bare_name_case_insensitively():
    config = _with_repos(service-a_ENTRY, SERVICE_B_ENTRY)

    resolved = resolve_active(config, "service-b")

    assert resolved.repo.slug == "example-org/service-b"


def test_resolve_active_overrides_repo_root_and_graph_path_from_the_entry():
    config = _with_repos(service-a_ENTRY, SERVICE_B_ENTRY)

    resolved = resolve_active(config, "service-a")

    assert resolved.review.repo_root == "/checkouts/service-a"
    assert resolved.review.graph_path == "graphs/service-a-cochange.json"
    assert resolved is not config


def test_resolve_active_keeps_existing_repo_root_when_entry_leaves_it_empty():
    base = make_config()
    config = replace(
        base,
        review=replace(base.review, repo_root="/existing"),
        repos=(SERVICE_B_ENTRY,),  # sets graph_path but not repo_root
    )

    resolved = resolve_active(config, "service-b")

    assert resolved.review.repo_root == "/existing"
    assert resolved.review.graph_path == "graphs/service-b-cochange.json"


def test_resolve_active_raises_for_unknown_selector_and_lists_available_names():
    config = _with_repos(service-a_ENTRY, SERVICE_B_ENTRY)

    with pytest.raises(ConfigError, match="example-org/service-a"):
        resolve_active(config, "nonexistent")


def test_resolve_active_without_selector_and_multiple_entries_raises():
    config = _with_repos(service-a_ENTRY, SERVICE_B_ENTRY)

    with pytest.raises(ConfigError, match="--repo"):
        resolve_active(config)


def test_resolve_active_without_selector_and_single_entry_auto_selects():
    config = _with_repos(service-a_ENTRY)

    resolved = resolve_active(config)

    assert resolved.repo.slug == "example-org/service-a"


def test_resolve_active_without_selector_and_no_repos_returns_flat_fallback_unchanged():
    config = make_config(owner="acme", name="widget")

    resolved = resolve_active(config)

    assert resolved.repo.slug == "acme/widget"
    assert resolved is config


def test_resolve_active_with_selector_but_no_repos_entries_raises():
    config = make_config()

    with pytest.raises(ConfigError, match="entries configured"):
        resolve_active(config, "anything")


def test_resolve_active_end_to_end_from_parsed_toml(tmp_path):
    config = load_config(_write(tmp_path, REPOS_TOML))

    resolved = resolve_active(config, "example-org/service-b")

    assert resolved.repo.read_only is True
    assert resolved.review.graph_path == "graphs/service-b-cochange.json"
