"""Config loading, validation, and the fail-loud behavior around missing credentials."""

from __future__ import annotations

import pytest

from prime_pr_review.config import (
    ConfigError,
    load_config,
    require_repo,
    require_secrets,
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
