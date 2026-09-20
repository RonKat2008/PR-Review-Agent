"""The prime-review console entry: dispatch, sugar, and cwd-independent defaults."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from prime_pr_review import cli


@pytest.fixture
def captured(monkeypatch):
    """Replace script loading with a recorder so no script actually runs."""
    calls: list[list[str]] = []

    def fake_load(script: str):
        def fake_main():
            calls.append([script, *sys.argv[1:]])
            return 0

        return SimpleNamespace(main=fake_main)

    monkeypatch.setattr(cli, "_load", fake_load)
    return calls


def test_help_exits_zero_and_prints_usage(capsys):
    assert cli.main(["--help"]) == 0
    assert "prime-review <command>" in capsys.readouterr().out


def test_unknown_command_exits_two(capsys):
    assert cli.main(["frobnicate"]) == 2
    assert "unknown command" in capsys.readouterr().err


def test_pr_sugar_expands_to_a_single_pr_sweep(captured):
    cli.main(["pr", "ExampleOrg/ServiceB", "2567"])

    script, *argv = captured[0]
    assert script == "run_sweep"
    assert argv[:4] == ["--repo", "ExampleOrg/ServiceB", "--pr", "2567"]


def test_pr_sugar_without_a_number_exits_two(capsys):
    assert cli.main(["pr", "ExampleOrg/ServiceB"]) == 2


def test_sweep_pins_config_to_the_agent_root_when_absent(captured):
    cli.main(["sweep", "--repo", "ServiceB"])

    argv = captured[0][1:]
    assert "--config" in argv
    config_value = argv[argv.index("--config") + 1]
    assert config_value.endswith("config.toml")
    assert str(cli.AGENT_ROOT) in config_value


def test_explicit_config_is_never_overridden(captured):
    cli.main(["replay", "--config", "elsewhere.toml"])

    argv = captured[0][1:]
    assert argv.count("--config") == 1
    assert "elsewhere.toml" in argv


def test_score_pins_reviews_dir(captured):
    cli.main(["score"])

    argv = captured[0][1:]
    assert "--reviews-dir" in argv
    assert str(cli.AGENT_ROOT) in argv[argv.index("--reviews-dir") + 1]
