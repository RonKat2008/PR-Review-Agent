"""Tests for the E1 replay harness (scripts/replay_corpus.py).

Every test injects a fake `gh` runner and a stub reviewer -- nothing here
touches a network, a model, or the real repo's `reviews/`/`state/`
directories. The module is imported directly (`from scripts import
replay_corpus`), the same way `tests/test_score_demo.py` imports
`scripts/score_demo.py`: `scripts/` has no package `__init__.py`, so pytest's
rootdir-based sys.path insertion makes it importable as a plain namespace
package.

The single most important property this harness has is that it never posts
to GitHub, no matter what config.toml says. That guarantee is tested at three
layers: the pure transform (`_lock_down`), the core pipeline (`run_replay`,
which locks down unconditionally), and the full CLI (`main`).
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from prime_pr_review.github import GitHubError
from prime_pr_review.state import State
from prime_pr_review.sweep import Enrichment, PullRequestOutcome, SweepReport

from scripts import replay_corpus

from .conftest import (
    FakeGh,
    SAMPLE_DIFF,
    VERDICT_EMPTY,
    VERDICT_WITH_BUG,
    is_pr_diff,
    is_pr_list,
    make_config,
    make_pr,
    pr_list_json,
)

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)

VERDICT_RICH = (
    '{"introduces":['
    '{"file":"src/app.py","line":10,"severity":"HIGH","claim":"bug one","evidence":"e1"},'
    '{"file":"src/app.py","line":20,"severity":"LOW","claim":"bug two","evidence":"e2"}'
    '],"fixes":[],"confidence":0.9,'
    '"scope":{"intent":"do X","aligned":false,"unrelated":['
    '{"file":"src/other.py","lines":"1-2","severity":"LOW","claim":"scope issue","evidence":"e3"}'
    ']},'
    '"blast_radius":[{"symbol":"foo","kind":"function","change":"sig",'
    '"breaks":[{"file":"src/caller.py","line":9,"severity":"HIGH","claim":"broken"}],'
    '"unbroken_callers":0}]}'
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _gh_with(*prs, diff: str = SAMPLE_DIFF) -> FakeGh:
    """A fake `gh` answering the initial merged-PR listing and every diff
    fetch. Deliberately registers NO handler for a comment/review post: any
    such call reaching this runner raises, which is the structural proof
    that the harness never posts.
    """
    return FakeGh().on(is_pr_list, pr_list_json(*prs)).on(is_pr_diff, diff)


def _write_config(tmp_path: Path) -> Path:
    """A minimal single-repo config.toml. `dry_run`/`pr_comment` are set to
    the "config says live" values on purpose, so tests exercise the lock-down
    rather than merely agreeing with an already-safe default. The three
    enrichment passes are off so a test never has to also stub a model_fn.
    """
    text = "\n".join(
        [
            "[repo]",
            'owner = "acme"',
            'name = "widget"',
            "read_only = false",
            "",
            "[review]",
            "dry_run = false",
            "check_intent = false",
            "check_blast = false",
            "gather_context = false",
            "",
            "[sinks]",
            "pr_comment = true",
            "webhook = false",
            "local_file = true",
            "",
        ]
    )
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _row_for(report: str, pr_number: int) -> str:
    """The single scorecard line for a PR. Fails loudly if the row is missing."""
    marker = f"| #{pr_number} "
    matches = [line for line in report.splitlines() if line.startswith(marker)]
    assert matches, f"no scorecard row for PR#{pr_number} in report:\n{report}"
    return matches[0]


def _section_for(report: str, pr_number: int) -> str:
    """The full per-PR section (heading through the next '## PR #' heading)."""
    marker = f"## PR #{pr_number} "
    lines = report.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(marker))
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("## PR #")),
        len(lines),
    )
    return "\n".join(lines[start:end])


# --------------------------------------------------------------------------
# _lock_down -- the pure safety transform
# --------------------------------------------------------------------------


def test_lock_down_forces_dry_run_true_even_when_config_says_live():
    config = make_config(dry_run=False)

    locked = replay_corpus._lock_down(config)

    assert locked.review.dry_run is True


def test_lock_down_disables_pr_comment_sink_even_when_config_enables_it():
    config = make_config(pr_comment=True)

    locked = replay_corpus._lock_down(config)

    assert locked.sinks.pr_comment is False


def test_lock_down_does_not_touch_unrelated_fields():
    config = make_config(dry_run=False, pr_comment=True, graph_path="graphs/x.json")

    locked = replay_corpus._lock_down(config)

    assert locked.repo == config.repo
    assert locked.review.graph_path == "graphs/x.json"
    assert locked.sinks.webhook == config.sinks.webhook


# --------------------------------------------------------------------------
# _single_pr_list_json -- duplicated serializer contract
# --------------------------------------------------------------------------


def test_single_pr_list_json_matches_the_gh_pr_list_json_shape():
    pr = make_pr(number=42, title="Solo PR", author="carol", merged_at="2026-01-01T00:00:00Z")

    payload = json.loads(replay_corpus._single_pr_list_json(pr))

    assert payload == [
        {
            "number": 42,
            "title": "Solo PR",
            "author": {"login": "carol"},
            "headRefOid": pr.head_sha,
            "baseRefName": pr.base_ref,
            "url": pr.url,
            "additions": pr.additions,
            "deletions": pr.deletions,
            "changedFiles": pr.changed_files,
            "mergedAt": "2026-01-01T00:00:00Z",
        }
    ]


# --------------------------------------------------------------------------
# resolve_api_key
# --------------------------------------------------------------------------


def test_resolve_api_key_prefers_the_environment_variable(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")

    assert replay_corpus.resolve_api_key() == "env-key"


def test_resolve_api_key_reads_a_literal_key_from_the_auth_file(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({"google": {"key": "AIzaFromFile"}}), encoding="utf-8")
    monkeypatch.setattr(replay_corpus, "AUTH_FILE", auth_file)

    assert replay_corpus.resolve_api_key() == "AIzaFromFile"


def test_resolve_api_key_treats_a_non_literal_auth_value_as_an_env_var_name(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("MY_OTHER_KEY_VAR", "indirect-key")
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({"google": {"key": "MY_OTHER_KEY_VAR"}}), encoding="utf-8")
    monkeypatch.setattr(replay_corpus, "AUTH_FILE", auth_file)

    assert replay_corpus.resolve_api_key() == "indirect-key"


def test_resolve_api_key_raises_systemexit_when_nothing_is_configured(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(replay_corpus, "AUTH_FILE", tmp_path / "does-not-exist.json")

    with pytest.raises(SystemExit):
        replay_corpus.resolve_api_key()


# --------------------------------------------------------------------------
# build_enrichment -- mirrors run_sweep.py's construction
# --------------------------------------------------------------------------


def test_build_enrichment_reads_rejections_from_the_configured_path(monkeypatch, tmp_path):
    rejections_file = tmp_path / "rejections.json"
    rejections_file.write_text(
        json.dumps(
            [
                {
                    "file": "src/app.py",
                    "claim_fingerprint": "abc123",
                    "reason": "false positive",
                    "pr_number": 7,
                    "rejected_at": "2026-01-01T00:00:00+00:00",
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(replay_corpus, "REJECTIONS_FILE", rejections_file)

    enrichment = replay_corpus.build_enrichment(make_config(), "fake-key", "test-model")

    assert len(enrichment.rejections) == 1
    assert enrichment.rejections[0].pr_number == 7
    assert enrichment.model_fn is not None


def test_build_enrichment_has_no_git_runner_without_a_repo_root(monkeypatch, tmp_path):
    monkeypatch.setattr(replay_corpus, "REJECTIONS_FILE", tmp_path / "rejections.json")

    enrichment = replay_corpus.build_enrichment(make_config(), "fake-key", "test-model")

    assert enrichment.git_runner is None


def test_build_enrichment_sets_a_strict_git_runner_when_repo_root_is_configured(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(replay_corpus, "REJECTIONS_FILE", tmp_path / "rejections.json")
    config = make_config()
    config = replace(config, review=replace(config.review, repo_root=str(tmp_path)))

    enrichment = replay_corpus.build_enrichment(config, "fake-key", "test-model")

    assert enrichment.git_runner is not None
    assert enrichment.repo_root == tmp_path


def test_build_enrichment_continues_with_no_rejections_when_the_file_is_malformed(
    monkeypatch, tmp_path, capsys
):
    rejections_file = tmp_path / "rejections.json"
    rejections_file.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr(replay_corpus, "REJECTIONS_FILE", rejections_file)

    enrichment = replay_corpus.build_enrichment(make_config(), "fake-key", "test-model")

    assert enrichment.rejections == ()
    assert "Warning" in capsys.readouterr().out


# --------------------------------------------------------------------------
# _replay_one -- defensive branches around sweep_lane
# --------------------------------------------------------------------------


def test_replay_one_records_a_crash_without_propagating(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(replay_corpus, "sweep_lane", boom)

    result = replay_corpus._replay_one(
        make_config(), make_pr(number=1), lambda pr, diff, lane: VERDICT_WITH_BUG,
        None, FakeGh(), Path("unused"),
    )

    assert result.outcome is None
    assert "replay crashed" in result.error
    assert "kaboom" in result.error


def test_replay_one_records_an_error_when_sweep_lane_returns_no_outcome(monkeypatch):
    def empty_sweep(*args, **kwargs):
        return SweepReport(lane="open"), State.empty()

    monkeypatch.setattr(replay_corpus, "sweep_lane", empty_sweep)

    result = replay_corpus._replay_one(
        make_config(), make_pr(number=1), lambda pr, diff, lane: VERDICT_WITH_BUG,
        None, FakeGh(), Path("unused"),
    )

    assert result.outcome is None
    assert "no outcome" in result.error


# --------------------------------------------------------------------------
# run_replay -- the safety guarantee, end to end
# --------------------------------------------------------------------------


def test_run_replay_never_posts_even_when_config_says_live(tmp_path):
    gh = _gh_with(make_pr(number=1, merged_at="2026-07-01T00:00:00Z"))
    config = make_config(dry_run=False, pr_comment=True)

    run = replay_corpus.run_replay(
        config, "acme/widget", 1, "test-model",
        lambda pr, diff, lane: VERDICT_WITH_BUG, None,
        runner=gh, reviews_dir=tmp_path, now=NOW,
    )

    assert run.config.review.dry_run is True
    assert run.config.sinks.pr_comment is False
    assert run.results[0].outcome.posted is False
    assert gh.calls_matching("pr comment") == []
    assert gh.calls_matching("/reviews") == []


def test_run_replay_uses_a_fresh_state_per_pr_so_nothing_is_skipped_as_already_reviewed(tmp_path):
    """Every PR in a corpus must actually be reviewed -- state never
    accumulates skip history across the replay loop."""
    gh = _gh_with(
        make_pr(number=1, head_sha="sha1", merged_at="2026-07-01T00:00:00Z"),
        make_pr(number=2, head_sha="sha2", merged_at="2026-07-02T00:00:00Z"),
    )

    run = replay_corpus.run_replay(
        make_config(), "acme/widget", 2, "test-model",
        lambda pr, diff, lane: VERDICT_WITH_BUG, None,
        runner=gh, reviews_dir=tmp_path, now=NOW,
    )

    # Both were actually reviewed (verdict present); neither was skipped
    # pre-review -- the only reason held back is the forced dry-run gate.
    assert all(r.outcome.verdict is not None for r in run.results)
    assert all(r.outcome.posted is False for r in run.results)


def test_run_replay_trims_to_count_keeping_newest_first(tmp_path):
    prs = [
        make_pr(number=n, head_sha=f"sha{n}", merged_at="2026-07-01T00:00:00Z")
        for n in (105, 104, 103, 102, 101)
    ]
    gh = _gh_with(*prs)

    run = replay_corpus.run_replay(
        make_config(), "acme/widget", 2, "test-model",
        lambda pr, diff, lane: VERDICT_EMPTY, None,
        runner=gh, reviews_dir=tmp_path, now=NOW,
    )

    assert [r.pr.number for r in run.results] == [105, 104]


def test_run_replay_reports_a_shortfall_when_fewer_merged_prs_exist_than_requested(tmp_path):
    gh = _gh_with(
        make_pr(number=1, merged_at="2026-07-01T00:00:00Z"),
        make_pr(number=2, head_sha="sha2", merged_at="2026-07-02T00:00:00Z"),
    )

    run = replay_corpus.run_replay(
        make_config(), "acme/widget", 5, "test-model",
        lambda pr, diff, lane: VERDICT_EMPTY, None,
        runner=gh, reviews_dir=tmp_path, now=NOW,
    )

    assert len(run.results) == 2
    assert run.count_requested == 5


def test_run_replay_records_a_reviewer_error_without_aborting_the_rest(tmp_path):
    def flaky(pr, diff, lane):
        if pr.number == 1:
            return "not json at all"
        return VERDICT_EMPTY

    gh = _gh_with(
        make_pr(number=1, merged_at="2026-07-01T00:00:00Z"),
        make_pr(number=2, head_sha="sha2", merged_at="2026-07-02T00:00:00Z"),
    )

    run = replay_corpus.run_replay(
        make_config(), "acme/widget", 2, "test-model", flaky, None,
        runner=gh, reviews_dir=tmp_path, now=NOW,
    )

    assert len(run.errors) == 1
    assert run.errors[0].pr.number == 1
    assert len(run.results) == 2, "the second PR must still be reviewed"


# --------------------------------------------------------------------------
# render_report -- header, scorecard, per-PR bodies, reading guide
# --------------------------------------------------------------------------


def test_report_header_includes_repo_model_counts_timestamp_and_error_total(tmp_path):
    gh = _gh_with(make_pr(number=1, merged_at="2026-06-01T00:00:00Z"))

    run = replay_corpus.run_replay(
        make_config(), "acme/widget", 3, "gemini-flash-latest",
        lambda pr, diff, lane: VERDICT_EMPTY, None,
        runner=gh, reviews_dir=tmp_path, now=NOW,
    )
    report = replay_corpus.render_report(run)

    assert "acme/widget" in report
    assert "gemini-flash-latest" in report
    assert "count requested**: 3" in report
    assert "count reviewed**: 1" in report
    assert NOW.isoformat() in report
    assert "total errors**: 0" in report
    assert "DRY RUN" in report


def test_report_notes_the_shortfall_in_the_header(tmp_path):
    gh = _gh_with(make_pr(number=1, merged_at="2026-06-01T00:00:00Z"))

    run = replay_corpus.run_replay(
        make_config(), "acme/widget", 5, "test-model",
        lambda pr, diff, lane: VERDICT_EMPTY, None,
        runner=gh, reviews_dir=tmp_path, now=NOW,
    )
    report = replay_corpus.render_report(run)

    assert "shortfall" in report.lower()
    assert "requested 5" in report
    assert "only 1" in report


def test_report_omits_shortfall_note_when_count_is_fully_satisfied(tmp_path):
    gh = _gh_with(make_pr(number=1, merged_at="2026-06-01T00:00:00Z"))

    run = replay_corpus.run_replay(
        make_config(), "acme/widget", 1, "test-model",
        lambda pr, diff, lane: VERDICT_EMPTY, None,
        runner=gh, reviews_dir=tmp_path, now=NOW,
    )
    report = replay_corpus.render_report(run)

    assert "shortfall" not in report.lower()


def test_scorecard_row_reports_findings_blocking_scope_and_broken_callers(tmp_path):
    gh = _gh_with(make_pr(number=9, merged_at="2026-06-01T00:00:00Z"))

    run = replay_corpus.run_replay(
        make_config(), "acme/widget", 1, "test-model",
        lambda pr, diff, lane: VERDICT_RICH, None,
        runner=gh, reviews_dir=tmp_path, now=NOW,
    )
    report = replay_corpus.render_report(run)
    row = _row_for(report, 9)

    # files=1 (make_pr default) | findings=2 | blocking=1 | scope-unrelated=1 | callers broken=1
    assert "| 1 | 2 | 1 | 1 | 1 |" in row


def test_scorecard_row_uses_dashes_when_a_pr_was_skipped_not_errored(tmp_path):
    lock_only_diff = "diff --git a/uv.lock b/uv.lock\n+version = 2\n"
    gh = _gh_with(make_pr(number=3, merged_at="2026-06-01T00:00:00Z"), diff=lock_only_diff)
    config = make_config(ignore_paths=("**/*.lock",))

    run = replay_corpus.run_replay(
        config, "acme/widget", 1, "test-model",
        lambda pr, diff, lane: VERDICT_WITH_BUG, None,
        runner=gh, reviews_dir=tmp_path, now=NOW,
    )
    report = replay_corpus.render_report(run)
    row = _row_for(report, 3)

    assert "| - | - | - | - |" in row
    assert "skipped: diff empty after filtering" in row


def test_scorecard_row_shows_the_error_when_a_pr_errored(tmp_path):
    gh = _gh_with(make_pr(number=4, merged_at="2026-06-01T00:00:00Z"))

    run = replay_corpus.run_replay(
        make_config(), "acme/widget", 1, "test-model",
        lambda pr, diff, lane: "not json at all", None,
        runner=gh, reviews_dir=tmp_path, now=NOW,
    )
    report = replay_corpus.render_report(run)
    row = _row_for(report, 4)

    assert "ERROR" in row
    assert "unusable verdict" in row


def test_scorecard_row_notes_are_truncated_beyond_the_char_limit():
    long_note = "x" * 300
    outcome = PullRequestOutcome(pr=make_pr(number=1), lane="open", notes=(long_note,))
    result = replay_corpus.ReplayResult(pr=make_pr(number=1), outcome=outcome)

    notes = replay_corpus._row_notes(result)

    assert len(notes) == replay_corpus.NOTES_TRUNCATE_CHARS
    assert notes.endswith("…")


def test_pr_body_embeds_the_local_file_the_sink_wrote(tmp_path):
    gh = _gh_with(make_pr(number=5, title="Fix widget speed", merged_at="2026-06-01T00:00:00Z"))

    run = replay_corpus.run_replay(
        make_config(local_file=True), "acme/widget", 1, "test-model",
        lambda pr, diff, lane: VERDICT_WITH_BUG, None,
        runner=gh, reviews_dir=tmp_path, now=NOW,
    )
    report = replay_corpus.render_report(run)
    section = _section_for(report, 5)

    assert "Fix widget speed" in section
    assert "Off-by-one in the loop bound" in section
    assert list(tmp_path.glob("PR-5-*.md")), "the local file the report read from should exist"


def test_pr_body_falls_back_to_rerendering_when_local_file_sink_is_disabled(tmp_path):
    gh = _gh_with(make_pr(number=6, merged_at="2026-06-01T00:00:00Z"))

    run = replay_corpus.run_replay(
        make_config(local_file=False), "acme/widget", 1, "test-model",
        lambda pr, diff, lane: VERDICT_WITH_BUG, None,
        runner=gh, reviews_dir=tmp_path, now=NOW,
    )
    report = replay_corpus.render_report(run)
    section = _section_for(report, 6)

    assert "Off-by-one in the loop bound" in section
    assert list(tmp_path.glob("PR-6-*.md")) == [], "local sink was disabled; nothing should be on disk"


def test_pr_body_shows_the_skip_reason_when_there_is_no_verdict(tmp_path):
    lock_only_diff = "diff --git a/uv.lock b/uv.lock\n+version = 2\n"
    gh = _gh_with(make_pr(number=7, merged_at="2026-06-01T00:00:00Z"), diff=lock_only_diff)
    config = make_config(ignore_paths=("**/*.lock",))

    run = replay_corpus.run_replay(
        config, "acme/widget", 1, "test-model",
        lambda pr, diff, lane: VERDICT_WITH_BUG, None,
        runner=gh, reviews_dir=tmp_path, now=NOW,
    )
    report = replay_corpus.render_report(run)
    section = _section_for(report, 7)

    assert "Skipped" in section
    assert "diff empty after filtering" in section


def test_pr_body_shows_the_replay_error_when_there_is_no_outcome_at_all(monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(replay_corpus, "sweep_lane", boom)
    gh = _gh_with(make_pr(number=8, merged_at="2026-06-01T00:00:00Z"))

    run = replay_corpus.run_replay(
        make_config(), "acme/widget", 1, "test-model",
        lambda pr, diff, lane: VERDICT_WITH_BUG, None,
        runner=gh, reviews_dir=tmp_path, now=NOW,
    )
    report = replay_corpus.render_report(run)
    section = _section_for(report, 8)

    assert "Replay error" in section
    assert "kaboom" in section


def test_report_includes_the_reading_guide_checklist(tmp_path):
    gh = _gh_with(make_pr(number=1, merged_at="2026-06-01T00:00:00Z"))

    run = replay_corpus.run_replay(
        make_config(), "acme/widget", 1, "test-model",
        lambda pr, diff, lane: VERDICT_EMPTY, None,
        runner=gh, reviews_dir=tmp_path, now=NOW,
    )
    report = replay_corpus.render_report(run)

    assert "Reading guide" in report
    assert "Zero fabricated findings" in report
    assert "60%" in report
    assert "silent on cosmetic" in report


# --------------------------------------------------------------------------
# _graph_status
# --------------------------------------------------------------------------


def test_graph_status_reports_not_configured_when_graph_path_is_empty(tmp_path):
    gh = _gh_with(make_pr(number=1, merged_at="2026-06-01T00:00:00Z"))

    run = replay_corpus.run_replay(
        make_config(graph_path=""), "acme/widget", 1, "test-model",
        lambda pr, diff, lane: VERDICT_EMPTY, None,
        runner=gh, reviews_dir=tmp_path, now=NOW,
    )

    assert replay_corpus._graph_status(run) == "not configured"


def test_graph_status_reports_degradation_from_notes(tmp_path):
    gh = _gh_with(make_pr(number=1, merged_at="2026-06-01T00:00:00Z"))

    run = replay_corpus.run_replay(
        make_config(graph_path="graphs/x.json"), "acme/widget", 1, "test-model",
        lambda pr, diff, lane: VERDICT_EMPTY, Enrichment(),  # no git_runner configured
        runner=gh, reviews_dir=tmp_path, now=NOW,
    )

    status = replay_corpus._graph_status(run)
    assert "degraded" in status
    assert "graph skipped" in status


def test_graph_status_reports_no_degradation_when_nothing_was_noted(tmp_path):
    """`graph_path` is set, but with `enrichment=None` the graph section never
    even runs -- so there is nothing to note either way. This is the flip
    side of the "derived from notes" design: silence reads as clean."""
    gh = _gh_with(make_pr(number=1, merged_at="2026-06-01T00:00:00Z"))

    run = replay_corpus.run_replay(
        make_config(graph_path="graphs/x.json"), "acme/widget", 1, "test-model",
        lambda pr, diff, lane: VERDICT_EMPTY, None,
        runner=gh, reviews_dir=tmp_path, now=NOW,
    )

    status = replay_corpus._graph_status(run)
    assert status == "configured (`graphs/x.json`), no degradation noted"


# --------------------------------------------------------------------------
# write_report / _report_filename
# --------------------------------------------------------------------------


def test_report_filename_matches_the_expected_pattern():
    now = datetime(2026, 3, 4, tzinfo=timezone.utc)

    assert (
        replay_corpus._report_filename("example-org/service-b", now)
        == "replay-example-org-service-b-20260304.md"
    )


def test_write_report_creates_the_output_directory_and_file(tmp_path):
    gh = _gh_with(make_pr(number=1, merged_at="2026-06-01T00:00:00Z"))
    run = replay_corpus.run_replay(
        make_config(), "acme/widget", 1, "test-model",
        lambda pr, diff, lane: VERDICT_EMPTY, None,
        runner=gh, reviews_dir=tmp_path / "reviews", now=NOW,
    )
    out_dir = tmp_path / "nested" / "reports"

    path = replay_corpus.write_report(run, out_dir)

    assert path.parent == out_dir
    assert path.name == "replay-acme-widget-20260812.md"
    assert "Replay report" in path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# main -- the full CLI
# --------------------------------------------------------------------------


def test_main_end_to_end_writes_a_report_and_returns_zero_on_success(tmp_path, capsys):
    gh = _gh_with(
        make_pr(number=11, merged_at="2026-06-01T00:00:00Z"),
        make_pr(number=12, head_sha="sha12", merged_at="2026-06-02T00:00:00Z"),
    )
    config_path = _write_config(tmp_path)
    out_dir = tmp_path / "reports"

    exit_code = replay_corpus.main(
        ["--config", str(config_path), "--count", "2", "--out", str(out_dir)],
        runner=gh,
        reviewer=lambda pr, diff, lane: VERDICT_WITH_BUG,
        enrichment=None,
        reviews_dir=tmp_path / "reviews",
    )

    out = capsys.readouterr().out
    files = list(out_dir.glob("replay-acme-widget-*.md"))
    assert exit_code == 0
    assert len(files) == 1
    assert "Report written to" in out
    assert "2/2 PR(s) reviewed | 0 error(s)" in out


def test_main_returns_exit_1_and_names_the_pr_when_a_review_errors(tmp_path, capsys):
    gh = _gh_with(make_pr(number=201, merged_at="2026-06-01T00:00:00Z"))
    config_path = _write_config(tmp_path)

    exit_code = replay_corpus.main(
        ["--config", str(config_path), "--count", "1", "--out", str(tmp_path / "reports")],
        runner=gh,
        reviewer=lambda pr, diff, lane: "not json at all, no braces",
        enrichment=None,
        reviews_dir=tmp_path / "reviews",
    )

    err = capsys.readouterr().err
    assert exit_code == 1
    assert "PR #201" in err


def test_main_returns_exit_1_when_config_file_is_missing(tmp_path, capsys):
    exit_code = replay_corpus.main(
        ["--config", str(tmp_path / "does-not-exist.toml")],
        reviewer=lambda pr, diff, lane: VERDICT_EMPTY,
        enrichment=None,
    )

    err = capsys.readouterr().err
    assert exit_code == 1
    assert "Config error" in err


def test_main_returns_exit_1_when_repo_selector_is_ambiguous(tmp_path, capsys):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[[repos]]",
                'owner = "acme"',
                'name = "one"',
                "",
                "[[repos]]",
                'owner = "acme"',
                'name = "two"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = replay_corpus.main(
        ["--config", str(config_path)],
        reviewer=lambda pr, diff, lane: VERDICT_EMPTY,
        enrichment=None,
    )

    err = capsys.readouterr().err
    assert exit_code == 1
    assert "Config error" in err


def test_main_returns_exit_1_when_listing_merged_prs_fails(tmp_path, capsys):
    gh = FakeGh()  # no `pr list` handler registered
    config_path = _write_config(tmp_path)

    exit_code = replay_corpus.main(
        ["--config", str(config_path), "--count", "3"],
        runner=gh,
        reviewer=lambda pr, diff, lane: VERDICT_EMPTY,
        enrichment=None,
    )

    err = capsys.readouterr().err
    assert exit_code == 1
    assert "Could not list merged PRs" in err


def test_main_notes_the_shortfall_on_stdout(tmp_path, capsys):
    gh = _gh_with(make_pr(number=1, merged_at="2026-06-01T00:00:00Z"))
    config_path = _write_config(tmp_path)

    exit_code = replay_corpus.main(
        ["--config", str(config_path), "--count", "5", "--out", str(tmp_path / "reports")],
        runner=gh,
        reviewer=lambda pr, diff, lane: VERDICT_EMPTY,
        enrichment=None,
        reviews_dir=tmp_path / "reviews",
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "only 1 merged PR(s)" in out
    assert "requested 5" in out


def test_main_never_writes_anything_under_the_agent_state_dir(tmp_path, monkeypatch):
    """The core safety property behind 'never mark a PR reviewed for the real
    pipeline': nothing under a stand-in for the agent's state directory is
    ever created, even on a full, successful CLI run that exercises the real
    (non-injected) enrichment construction -- which is the one code path that
    ever reads anything under state/."""
    fake_state_dir = tmp_path / "agent-state-stand-in"
    monkeypatch.setattr(replay_corpus, "REJECTIONS_FILE", fake_state_dir / "rejections.json")
    config_path = _write_config(tmp_path)
    gh = _gh_with(make_pr(number=1, merged_at="2026-06-01T00:00:00Z"))

    exit_code = replay_corpus.main(
        ["--config", str(config_path), "--count", "1", "--out", str(tmp_path / "reports")],
        runner=gh,
        reviewer=lambda pr, diff, lane: VERDICT_WITH_BUG,
        api_key_resolver=lambda: "fake-key",
        reviews_dir=tmp_path / "reviews",
    )

    assert exit_code == 0
    assert not fake_state_dir.exists(), "replay must never write anything under the state dir"


def test_main_builds_the_real_reviewer_when_none_is_injected(tmp_path):
    """No merged PRs at all -- the real reviewer gets constructed (exercising
    that branch of `main`) but is never invoked, so this stays network-free."""
    gh = FakeGh().on(is_pr_list, "[]")
    config_path = _write_config(tmp_path)

    exit_code = replay_corpus.main(
        ["--config", str(config_path), "--count", "1", "--out", str(tmp_path / "reports")],
        runner=gh,
        enrichment=Enrichment(),
        api_key_resolver=lambda: "fake-key",
        reviews_dir=tmp_path / "reviews",
    )

    assert exit_code == 0


def test_main_never_posts_even_through_the_full_cli_with_a_live_config(tmp_path):
    """`_write_config` sets dry_run=false and pr_comment=true -- 'config says
    live' -- yet no comment/review call may reach the fake runner."""
    gh = _gh_with(make_pr(number=1, merged_at="2026-06-01T00:00:00Z"))
    config_path = _write_config(tmp_path)

    exit_code = replay_corpus.main(
        ["--config", str(config_path), "--count", "1", "--out", str(tmp_path / "reports")],
        runner=gh,
        reviewer=lambda pr, diff, lane: VERDICT_WITH_BUG,
        enrichment=None,
        reviews_dir=tmp_path / "reviews",
    )

    assert exit_code == 0
    assert gh.calls_matching("pr comment") == []
    assert gh.calls_matching("/reviews") == []


def test_open_state_lists_open_prs_and_suffixes_the_report_name():
    """--state open previews live candidates without touching merged history."""
    from datetime import datetime, timezone

    from scripts.replay_corpus import _report_filename, _select_prs
    from tests.conftest import FakeGh, is_pr_list, make_pr, pr_list_json

    gh = FakeGh().on(is_pr_list, pr_list_json(make_pr(number=8), make_pr(number=7)))
    now = datetime(2026, 8, 12, tzinfo=timezone.utc)

    picked = _select_prs("acme/widget", gh, 1, now, pr_state="open")

    assert [p.number for p in picked] == [8], "newest first, trimmed to count"
    listing_args = " ".join(gh.calls[0][0])
    assert "--state open" in listing_args
    assert _report_filename("acme/widget", now, "open") == "replay-acme-widget-open-20260812.md"
    assert _report_filename("acme/widget", now) == "replay-acme-widget-20260812.md"
