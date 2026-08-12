"""End-to-end sweep behavior against a fake gh and a stub reviewer."""

from __future__ import annotations

from datetime import datetime, timezone

from prime_pr_review.github import GitHubError
from prime_pr_review.state import (
    LANE_MERGED,
    LANE_OPEN,
    State,
    is_reviewed,
    mark_reviewed,
)
from dataclasses import replace

from prime_pr_review.sweep import Enrichment, sweep_lane

from .conftest import (
    FakeGh,
    SAMPLE_DIFF,
    VERDICT_EMPTY,
    VERDICT_LOW_CONFIDENCE,
    VERDICT_WITH_BUG,
    is_list_comments,
    is_post_review,
    is_pr_comment,
    is_pr_diff,
    is_pr_list,
    make_config,
    make_pr,
    pr_list_json,
)

NOW = datetime(2026, 8, 7, 12, 0, tzinfo=timezone.utc)


def reviewer_returning(raw: str):
    def reviewer(pr, diff, lane):
        return raw
    return reviewer


def gh_with(*prs, diff: str = SAMPLE_DIFF, comments: str = "") -> FakeGh:
    return (
        FakeGh()
        .on(is_pr_list, pr_list_json(*prs))
        .on(is_pr_diff, diff)
        .on(is_list_comments, comments)
        .on(is_post_review, "")
        .on(is_pr_comment, "")
    )


def test_reviews_an_open_pr_end_to_end(tmp_path):
    # Arrange
    gh = gh_with(make_pr(number=1))

    # Act
    report, state = sweep_lane(
        make_config(), LANE_OPEN, reviewer_returning(VERDICT_WITH_BUG),
        State.empty(), gh, tmp_path, NOW,
    )

    # Assert
    assert report.considered == 1
    assert report.reviewed == 1
    assert report.posted == 1
    assert is_reviewed(state, LANE_OPEN, 1, "abcdef1234567890")


def test_writes_a_local_review_file(tmp_path):
    gh = gh_with(make_pr(number=7))

    sweep_lane(
        make_config(), LANE_OPEN, reviewer_returning(VERDICT_WITH_BUG),
        State.empty(), gh, tmp_path, NOW,
    )

    assert list(tmp_path.glob("PR-7-*.md"))


def test_skips_a_pr_already_reviewed_at_the_same_sha(tmp_path):
    gh = gh_with(make_pr(number=1, head_sha="same-sha"))
    state = mark_reviewed(State.empty(), LANE_OPEN, 1, "same-sha")

    report, _ = sweep_lane(
        make_config(), LANE_OPEN, reviewer_returning(VERDICT_WITH_BUG),
        state, gh, tmp_path, NOW,
    )

    assert report.reviewed == 0
    assert report.skipped == 1
    assert gh.calls_matching("pr diff") == [], "must not fetch a diff for a skipped PR"


def test_re_reviews_after_new_commits_change_the_head_sha(tmp_path):
    gh = gh_with(make_pr(number=1, head_sha="new-sha"))
    state = mark_reviewed(State.empty(), LANE_OPEN, 1, "old-sha")

    report, _ = sweep_lane(
        make_config(), LANE_OPEN, reviewer_returning(VERDICT_WITH_BUG),
        state, gh, tmp_path, NOW,
    )

    assert report.reviewed == 1


def test_skips_bot_authored_prs_before_spending_tokens(tmp_path):
    gh = gh_with(make_pr(number=1, author="prime-bot"))
    calls: list = []

    def tracking_reviewer(pr, diff, lane):
        calls.append(pr.number)
        return VERDICT_WITH_BUG

    report, _ = sweep_lane(
        make_config(bot_login="prime-bot"), LANE_OPEN, tracking_reviewer,
        State.empty(), gh, tmp_path, NOW,
    )

    assert report.skipped == 1
    assert calls == [], "reviewer must never run on a skipped PR"


def _graph_file(tmp_path, commit="graphsha"):
    import json

    path = tmp_path / "graph.json"
    path.write_text(json.dumps({
        "version": 1,
        "repo": "acme/widget",
        "commit": commit,
        "built_at": "2026-08-12T00:00:00+00:00",
        "nodes": [{"id": "src/app.py", "kind": "file"},
                  {"id": "grammar.toml", "kind": "file"}],
        "edges": [{"src": "src/app.py", "dst": "grammar.toml",
                   "kind": "co_changes_with", "weight": 0.9, "samples": 20}],
    }), encoding="utf-8")
    return path


def test_fresh_graph_evidence_is_injected_into_the_review_payload(tmp_path):
    """A usable graph adds its rendered section; SAMPLE_DIFF touches src/app.py,
    whose strong co-change partner grammar.toml is absent from the diff."""
    gh = gh_with(make_pr(number=1))
    seen: list[str] = []

    def reviewer(pr, payload, lane):
        seen.append(payload)
        return VERDICT_WITH_BUG

    enrichment = Enrichment(git_runner=lambda args: "")  # ancestry check passes

    sweep_lane(
        make_config(graph_path=str(_graph_file(tmp_path))), LANE_OPEN, reviewer,
        State.empty(), gh, tmp_path, NOW, enrichment=enrichment,
    )

    assert "## Knowledge graph" in seen[0]
    assert "grammar.toml" in seen[0], "the co-change warning should name the partner"


def test_graph_activity_is_counted_in_notes_and_front_matter(tmp_path):
    """'Loaded and said nothing' vs 'loaded and warned' must be visible per PR."""
    import json as _json

    gh = gh_with(make_pr(number=1))
    enrichment = Enrichment(git_runner=lambda args: "")

    report, _ = sweep_lane(
        make_config(graph_path=str(_graph_file(tmp_path))), LANE_OPEN,
        reviewer_returning(VERDICT_WITH_BUG), State.empty(), gh, tmp_path, NOW,
        enrichment=enrichment,
    )

    outcome = report.outcomes[0]
    graph_notes = [n for n in outcome.notes if n.startswith("graph:")]
    assert graph_notes, f"expected a graph activity note, got {outcome.notes}"
    assert "1 co-change warning(s)" in graph_notes[0]

    front = _json.loads(
        outcome.local_path.read_text(encoding="utf-8").split("<!--")[1].split("-->")[0]
    )
    assert any(n.startswith("graph:") for n in front["notes"])


def test_stale_graph_is_refused_with_a_visible_note(tmp_path):
    """When ancestry fails the graph is NOT used and the outcome says why."""
    from prime_pr_review.context import GitError

    gh = gh_with(make_pr(number=1))
    seen: list[str] = []

    def reviewer(pr, payload, lane):
        seen.append(payload)
        return VERDICT_WITH_BUG

    def refusing_runner(args):
        raise GitError("exit 1: not an ancestor")

    enrichment = Enrichment(git_runner=refusing_runner)

    report, _ = sweep_lane(
        make_config(graph_path=str(_graph_file(tmp_path))), LANE_OPEN, reviewer,
        State.empty(), gh, tmp_path, NOW, enrichment=enrichment,
    )

    assert "## Knowledge graph" not in seen[0]
    assert any("graph" in n for n in report.outcomes[0].notes)


def test_static_analysis_findings_are_injected_when_repo_root_is_set(tmp_path):
    from prime_pr_review.analysis import AnalysisResult, LintFinding
    from prime_pr_review.review import Severity

    gh = gh_with(make_pr(number=1))
    seen: list[str] = []

    def reviewer(pr, payload, lane):
        seen.append(payload)
        return VERDICT_WITH_BUG

    def fake_analysis(paths, diff_lines):
        return AnalysisResult(findings=(
            LintFinding(tool="bandit", rule_id="B608", file="src/app.py",
                        line=2, severity=Severity.CRITICAL, message="sql injection"),
        ))

    config = make_config()
    config = replace(config, review=replace(config.review, repo_root=str(tmp_path)))
    enrichment = Enrichment(analysis_fn=fake_analysis)

    sweep_lane(config, LANE_OPEN, reviewer, State.empty(), gh, tmp_path, NOW,
               enrichment=enrichment)

    assert "B608" in seen[0], "linter evidence should reach the model"


def test_ensemble_size_three_runs_the_reviewer_three_times_and_votes(tmp_path):
    gh = gh_with(make_pr(number=1))
    calls: list[str] = []

    def reviewer(pr, payload, lane):
        calls.append(lane)
        return VERDICT_WITH_BUG

    config = make_config()
    config = replace(config, review=replace(config.review, ensemble_size=3))

    report, _ = sweep_lane(
        config, LANE_OPEN, reviewer, State.empty(), gh, tmp_path, NOW,
    )

    assert len(calls) == 3
    verdict = report.outcomes[0].verdict
    assert verdict is not None
    assert verdict.confidence == 1.0, "3/3 agreement is an observed 100%"
    assert "3/3 reviewers" in verdict.introduces[0].corroboration


def test_ensemble_off_by_default_keeps_a_single_call(tmp_path):
    gh = gh_with(make_pr(number=1))
    calls: list[str] = []

    def reviewer(pr, payload, lane):
        calls.append(lane)
        return VERDICT_WITH_BUG

    sweep_lane(
        make_config(), LANE_OPEN, reviewer, State.empty(), gh, tmp_path, NOW,
    )

    assert len(calls) == 1


def test_docs_only_diff_skips_intent_and_blast_with_notes(tmp_path):
    """C4: prose cannot break callers; both model-costing passes are skipped."""
    docs_diff = "diff --git a/docs/guide.md b/docs/guide.md\n+hello\n"
    gh = gh_with(make_pr(number=1), diff=docs_diff)

    report, _ = sweep_lane(
        make_config(ignore_paths=()), LANE_OPEN, reviewer_returning(VERDICT_WITH_BUG),
        State.empty(), gh, tmp_path, NOW,
        enrichment=Enrichment(model_fn=lambda prompt: "{}"),
    )

    notes = report.outcomes[0].notes
    assert any("intent check skipped: docs-only" in n for n in notes)


def test_intent_min_files_threshold_skips_small_diffs(tmp_path):
    gh = gh_with(make_pr(number=1))  # SAMPLE_DIFF touches 1 kept file
    config = make_config()
    config = replace(config, review=replace(config.review, intent_min_files=3))

    report, _ = sweep_lane(
        config, LANE_OPEN, reviewer_returning(VERDICT_WITH_BUG),
        State.empty(), gh, tmp_path, NOW,
        enrichment=Enrichment(model_fn=lambda prompt: "{}"),
    )

    assert any("below intent_min_files" in n for n in report.outcomes[0].notes)


def test_rejected_findings_are_suppressed_with_an_audit_note(tmp_path):
    """A finding maintainers rejected (P6) is filtered out, and the outcome says so."""
    from prime_pr_review.feedback import Rejection, claim_fingerprint

    gh = gh_with(make_pr(number=1))
    rejection = Rejection(
        file="src/app.py",
        claim_fingerprint=claim_fingerprint("Off-by-one in the loop bound"),
        reason="false positive",
        pr_number=99,
        rejected_at="2026-08-11T00:00:00+00:00",
    )

    report, _ = sweep_lane(
        make_config(), LANE_OPEN, reviewer_returning(VERDICT_WITH_BUG),
        State.empty(), gh, tmp_path, NOW,
        enrichment=Enrichment(rejections=(rejection,)),
    )

    outcome = report.outcomes[0]
    assert outcome.verdict is not None
    assert outcome.verdict.introduces == (), "the rejected finding must not survive"
    assert any("suppressed by maintainer feedback" in n for n in outcome.notes)


def test_rejection_guidance_reaches_the_review_prompt(tmp_path):
    from prime_pr_review.feedback import Rejection, claim_fingerprint

    gh = gh_with(make_pr(number=1))
    seen: list[str] = []

    def reviewer(pr, payload, lane):
        seen.append(payload)
        return VERDICT_WITH_BUG

    rejection = Rejection(
        file="src/other.py",
        claim_fingerprint=claim_fingerprint("some other complaint"),
        reason="wontfix",
        pr_number=7,
        rejected_at="2026-08-11T00:00:00+00:00",
    )

    sweep_lane(
        make_config(), LANE_OPEN, reviewer, State.empty(), gh, tmp_path, NOW,
        enrichment=Enrichment(rejections=(rejection,)),
    )

    assert "rejected" in seen[0].lower(), "guidance block should be in the payload"


def test_findings_are_delivered_as_line_anchored_review_comments(tmp_path):
    """Delivery goes through the reviews API, not a single trailing comment."""
    gh = gh_with(make_pr(number=1))

    report, _ = sweep_lane(
        make_config(), LANE_OPEN, reviewer_returning(VERDICT_WITH_BUG),
        State.empty(), gh, tmp_path, NOW,
    )

    assert report.posted == 1
    assert gh.calls_matching("/reviews"), "expected a reviews API call"
    assert gh.calls_matching("pr comment") == [], "must not also post a summary comment"


def test_inline_comments_can_be_turned_off(tmp_path):
    gh = gh_with(make_pr(number=1))

    sweep_lane(
        make_config(inline_comments=False), LANE_OPEN,
        reviewer_returning(VERDICT_WITH_BUG), State.empty(), gh, tmp_path, NOW,
    )

    assert gh.calls_matching("pr comment"), "expected the summary-comment fallback"
    assert gh.calls_matching("/reviews") == []


def test_dry_run_writes_locally_but_never_posts(tmp_path):
    gh = gh_with(make_pr(number=1))

    report, _ = sweep_lane(
        make_config(dry_run=True), LANE_OPEN, reviewer_returning(VERDICT_WITH_BUG),
        State.empty(), gh, tmp_path, NOW,
    )

    assert report.reviewed == 1
    assert report.posted == 0
    assert gh.calls_matching("pr comment") == []
    assert list(tmp_path.glob("PR-1-*.md"))


def test_comment_budget_caps_posts_across_many_prs(tmp_path):
    gh = gh_with(*[make_pr(number=n, head_sha=f"sha{n}") for n in range(1, 6)])

    report, _ = sweep_lane(
        make_config(max_comments=2), LANE_OPEN, reviewer_returning(VERDICT_WITH_BUG),
        State.empty(), gh, tmp_path, NOW,
    )

    assert report.reviewed == 5
    assert report.posted == 2


def test_low_confidence_verdict_is_reviewed_but_not_posted(tmp_path):
    gh = gh_with(make_pr(number=1))

    report, _ = sweep_lane(
        make_config(min_confidence=0.7), LANE_OPEN,
        reviewer_returning(VERDICT_LOW_CONFIDENCE), State.empty(), gh, tmp_path, NOW,
    )

    assert report.reviewed == 1
    assert report.posted == 0
    assert list(tmp_path.glob("PR-1-*.md")), "held reviews still land in the audit trail"


def test_silent_verdict_is_recorded_but_not_posted(tmp_path):
    gh = gh_with(make_pr(number=1))

    report, _ = sweep_lane(
        make_config(), LANE_OPEN, reviewer_returning(VERDICT_EMPTY),
        State.empty(), gh, tmp_path, NOW,
    )

    assert report.reviewed == 1
    assert report.posted == 0


def test_skips_a_pr_whose_diff_is_all_ignored_paths(tmp_path):
    lock_only = "diff --git a/uv.lock b/uv.lock\n+version = 2\n"
    gh = gh_with(make_pr(number=1), diff=lock_only)

    report, _ = sweep_lane(
        make_config(ignore_paths=("**/*.lock",)), LANE_OPEN,
        reviewer_returning(VERDICT_WITH_BUG), State.empty(), gh, tmp_path, NOW,
    )

    assert report.reviewed == 0
    assert report.skipped == 1


def test_a_failing_reviewer_does_not_abort_the_sweep(tmp_path):
    gh = gh_with(make_pr(number=1, head_sha="s1"), make_pr(number=2, head_sha="s2"))

    def flaky(pr, diff, lane):
        if pr.number == 1:
            raise RuntimeError("subagent exploded")
        return VERDICT_WITH_BUG

    report, _ = sweep_lane(
        make_config(), LANE_OPEN, flaky, State.empty(), gh, tmp_path, NOW
    )

    assert report.errors == 1
    assert report.reviewed == 1, "the second PR must still be reviewed"


def test_a_failed_pr_is_not_marked_reviewed_so_it_retries_next_sweep(tmp_path):
    gh = gh_with(make_pr(number=1, head_sha="s1"))

    def always_fails(pr, diff, lane):
        raise RuntimeError("nope")

    _, state = sweep_lane(
        make_config(), LANE_OPEN, always_fails, State.empty(), gh, tmp_path, NOW
    )

    assert is_reviewed(state, LANE_OPEN, 1, "s1") is False


def test_unparseable_verdict_is_recorded_as_an_error(tmp_path):
    gh = gh_with(make_pr(number=1))

    report, _ = sweep_lane(
        make_config(), LANE_OPEN, reviewer_returning("I think it looks fine"),
        State.empty(), gh, tmp_path, NOW,
    )

    assert report.errors == 1
    assert "unusable verdict" in report.outcomes[0].summary_line()


def test_diff_fetch_failure_is_recorded_without_aborting(tmp_path):
    gh = FakeGh().on(is_pr_list, pr_list_json(make_pr(number=1)))  # no diff handler

    report, _ = sweep_lane(
        make_config(), LANE_OPEN, reviewer_returning(VERDICT_WITH_BUG),
        State.empty(), gh, tmp_path, NOW,
    )

    assert report.errors == 1
    assert "diff fetch failed" in report.outcomes[0].error


def test_merged_lane_queries_with_lookback_and_sets_cursor(tmp_path):
    gh = gh_with(make_pr(number=1, merged_at="2026-08-06T10:00:00Z"))

    report, state = sweep_lane(
        make_config(merged_lookback_days=7), LANE_MERGED,
        reviewer_returning(VERDICT_WITH_BUG), State.empty(), gh, tmp_path, NOW,
    )

    assert report.reviewed == 1
    assert state.merged_cursor == NOW.isoformat()
    # Filtering is client-side on mergedAt, not via --search: the search index is
    # eventually consistent and would hide a just-merged PR from this sweep.
    args = " ".join(gh.calls[0][0])
    assert "--state merged" in args
    assert "--search" not in args


def test_open_lane_does_not_set_a_merged_cursor(tmp_path):
    gh = gh_with(make_pr(number=1))

    _, state = sweep_lane(
        make_config(), LANE_OPEN, reviewer_returning(VERDICT_WITH_BUG),
        State.empty(), gh, tmp_path, NOW,
    )

    assert state.merged_cursor is None


def test_empty_repo_produces_an_empty_report(tmp_path):
    gh = FakeGh().on(is_pr_list, "[]")

    report, state = sweep_lane(
        make_config(), LANE_OPEN, reviewer_returning(VERDICT_WITH_BUG),
        State.empty(), gh, tmp_path, NOW,
    )

    assert report.considered == 0
    assert report.summaries() == ()


def test_summary_lines_describe_each_outcome(tmp_path):
    gh = gh_with(make_pr(number=11))

    report, _ = sweep_lane(
        make_config(), LANE_OPEN, reviewer_returning(VERDICT_WITH_BUG),
        State.empty(), gh, tmp_path, NOW,
    )

    line = report.summaries()[0]
    assert "PR #11" in line
    assert "HIGH" in line
    assert "posted" in line
