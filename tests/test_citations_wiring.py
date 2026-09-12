"""Sweep wiring for the citation validator: it runs on `verdict.introduces`
before any other post-processing pass, is gated by `review.validate_citations`,
and fetches an exact head-SHA line count only for a file with a finding whose
line falls outside the diff's commentable lines — so the common path makes
zero API calls."""

from __future__ import annotations

import base64
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime

from prime_pr_review.review import Finding, Severity, Verdict
from prime_pr_review.state import LANE_OPEN, State
from prime_pr_review.sweep import Enrichment, _validate_citations, sweep_lane

from .conftest import (
    SAMPLE_DIFF,
    FakeGh,
    is_list_comments,
    is_post_review,
    is_pr_comment,
    is_pr_diff,
    is_pr_list,
    make_config,
    make_pr,
    pr_list_json,
)

NOW = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)

DIFF = (
    "diff --git a/src/app.py b/src/app.py\n"
    "index 1111111..2222222 100644\n"
    "--- a/src/app.py\n"
    "+++ b/src/app.py\n"
    "@@ -8,5 +8,5 @@\n"
    " context\n"
    " context\n"
    "-old\n"
    "+new\n"
    " context\n"
)

REPO_SLUG = "acme/widget"
HEAD_SHA = "deadbeef00"


def _finding(
    *,
    file: str = "src/app.py",
    line: int | None = 10,
    claim: str = "Off-by-one in the loop bound",
) -> Finding:
    return Finding(
        file=file,
        line=line,
        severity=Severity.HIGH,
        claim=claim,
        evidence="the loop runs one extra time",
    )


def _verdict(*findings: Finding) -> Verdict:
    return Verdict(introduces=tuple(findings), fixes=(), confidence=0.9)


def _gh_content(text: str) -> str:
    return base64.b64encode(text.encode()).decode() + "\n"


def _is_contents_call(path: str) -> Callable[[Sequence[str]], bool]:
    def predicate(args: Sequence[str]) -> bool:
        return args[0] == "api" and f"contents/{path}" in " ".join(args)

    return predicate


def _forbidden_runner(args: Sequence[str], stdin: str | None = None) -> str:
    raise AssertionError(f"unexpected gh call: {' '.join(args)}")


class TestFlagOn:
    def test_fabricated_file_finding_is_dropped_with_a_note_and_no_api_call(self) -> None:
        config = make_config()
        verdict = _verdict(_finding(file="not/in/diff.py", claim="fabricated"))

        result, notes = _validate_citations(
            config, DIFF, verdict, REPO_SLUG, HEAD_SHA, _forbidden_runner
        )

        assert result.introduces == ()
        assert any("not/in/diff.py" in n for n in notes)
        assert any("fabricated" in n for n in notes)


class TestFlagOff:
    def test_flag_off_returns_verdict_unchanged_with_no_notes(self) -> None:
        config = make_config()
        config = replace(config, review=replace(config.review, validate_citations=False))
        verdict = _verdict(_finding(file="not/in/diff.py"))

        result, notes = _validate_citations(
            config, DIFF, verdict, REPO_SLUG, HEAD_SHA, _forbidden_runner
        )

        assert result is verdict
        assert notes == ()


class TestZeroApiCallsOnTheCommonPath:
    def test_a_line_verifiable_from_the_diff_makes_no_api_call(self) -> None:
        config = make_config()
        verdict = _verdict(_finding(line=10))  # "+new", on the diff itself

        result, notes = _validate_citations(
            config, DIFF, verdict, REPO_SLUG, HEAD_SHA, _forbidden_runner
        )

        assert len(result.introduces) == 1
        assert notes == ("citations: 1 checked, 0 dropped, 0 unverified",)


class TestHeadCountFetchOnlyWhenNeeded:
    def test_line_outside_hunks_fetches_head_count_and_drops_beyond_it(self) -> None:
        gh = FakeGh().on(_is_contents_call("src/app.py"), _gh_content("a\nb\nc\nd\ne\n"))
        config = make_config()
        # 5 lines at head; line 20 cannot exist.
        verdict = _verdict(_finding(line=20))

        result, notes = _validate_citations(config, DIFF, verdict, REPO_SLUG, HEAD_SHA, gh)

        assert result.introduces == ()
        assert any("beyond file length" in n for n in notes)
        assert any("5 lines at head" in n for n in notes)
        assert gh.calls_matching("contents/src/app.py")

    def test_line_outside_hunks_within_head_count_is_kept_unverified(self) -> None:
        gh = FakeGh().on(_is_contents_call("src/app.py"), _gh_content("a\nb\n" * 10))
        config = make_config()
        verdict = _verdict(_finding(line=14))

        result, notes = _validate_citations(config, DIFF, verdict, REPO_SLUG, HEAD_SHA, gh)

        assert len(result.introduces) == 1
        assert any(n.startswith("citations:") and "1 unverified" in n for n in notes)

    def test_a_fetch_failure_leaves_the_finding_unverified_not_dropped(self) -> None:
        config = make_config()
        verdict = _verdict(_finding(line=14))

        def raising_runner(args: Sequence[str], stdin: str | None = None) -> str:
            from prime_pr_review.github import GitHubError

            raise GitHubError("boom")

        result, notes = _validate_citations(
            config, DIFF, verdict, REPO_SLUG, HEAD_SHA, raising_runner
        )

        assert len(result.introduces) == 1
        assert any("1 unverified" in n for n in notes)


class TestAbsolutePathNormalizationUsesConfigRepoRoot:
    def test_absolute_finding_path_is_normalized_via_repo_root(self, tmp_path) -> None:
        config = make_config()
        config = replace(config, review=replace(config.review, repo_root=str(tmp_path)))
        finding = _finding(file=str(tmp_path / "src" / "app.py"), line=10)
        verdict = _verdict(finding)

        result, _notes = _validate_citations(
            config, DIFF, verdict, REPO_SLUG, HEAD_SHA, _forbidden_runner
        )

        assert len(result.introduces) == 1
        assert result.introduces[0].file == "src/app.py"


class TestEndToEnd:
    def test_fabricated_and_ignored_path_findings_never_reach_the_skeptic(self, tmp_path) -> None:
        """A real finding survives; a fabricated-file finding and one on a
        lockfile path (ignored, so it is absent from `filtered.text`) are
        both dropped before the skeptic pass ever sees their claim text."""
        gh = (
            FakeGh()
            .on(is_pr_list, pr_list_json(make_pr(number=1)))
            .on(is_pr_diff, SAMPLE_DIFF)
            .on(is_list_comments, "")
            .on(is_post_review, "")
            .on(is_pr_comment, "")
        )

        verdict_json = (
            '{"introduces":['
            '{"file":"src/app.py","line":1,"severity":"HIGH",'
            '"claim":"Off-by-one in the loop bound","evidence":"real evidence"},'
            '{"file":"src/does_not_exist.py","line":5,"severity":"HIGH",'
            '"claim":"FABRICATED_CLAIM_MARKER","evidence":"invented"},'
            '{"file":"uv.lock","line":1,"severity":"HIGH",'
            '"claim":"LOCKFILE_CLAIM_MARKER","evidence":"invented"}'
            "],"
            '"fixes":[],"confidence":0.9}'
        )

        def reviewer(pr, payload, lane):
            return verdict_json

        skeptic_prompts: list[str] = []

        def recording_skeptic(prompt: str) -> str:
            skeptic_prompts.append(prompt)
            return "{}"

        config = make_config()
        assert config.review.check_refute is True

        report, _ = sweep_lane(
            config,
            LANE_OPEN,
            reviewer,
            State.empty(),
            gh,
            tmp_path,
            NOW,
            enrichment=Enrichment(skeptic_fn=recording_skeptic),
        )

        outcome = report.outcomes[0]
        assert outcome.verdict is not None
        assert len(outcome.verdict.introduces) == 1
        assert outcome.verdict.introduces[0].file == "src/app.py"

        assert any(
            "src/does_not_exist.py:5" in n and "file not in diff" in n for n in outcome.notes
        )
        assert any("uv.lock:1" in n and "file not in diff" in n for n in outcome.notes)
        assert any(n.startswith("citations: 3 checked, 2 dropped") for n in outcome.notes)

        assert skeptic_prompts, "the skeptic should still run over the surviving finding"
        for prompt in skeptic_prompts:
            assert "FABRICATED_CLAIM_MARKER" not in prompt
            assert "LOCKFILE_CLAIM_MARKER" not in prompt
