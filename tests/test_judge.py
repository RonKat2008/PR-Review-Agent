"""Judge-merge (P15): semantic clustering of ensemble finding groups, with the
deterministic guardrails that keep the judge merge-only, same-file-only, and
unable to sink the review when it misbehaves."""

from __future__ import annotations

import json

import pytest

from prime_pr_review.judge import (
    Candidate,
    JudgeError,
    build_judge_prompt,
    propose_clusters,
)

PROMPTS_DIR = "skills/pr-review/prompts"


def _candidate(
    index: int,
    *,
    file: str = "src/app.py",
    line: int | None = 10,
    severity: str = "HIGH",
    claim: str = "Off-by-one in the loop bound",
    evidence: str = "the loop runs one extra time",
) -> Candidate:
    return Candidate(
        index=index, file=file, line=line, severity=severity, claim=claim, evidence=evidence
    )


def _model(response: str):
    """A ModelFn stub that also records the prompt it was given."""
    calls: list[str] = []

    def fn(prompt: str) -> str:
        calls.append(prompt)
        return response

    fn.calls = calls  # type: ignore[attr-defined]
    return fn


class TestProposeClusters:
    def test_merges_same_file_groups(self) -> None:
        candidates = (
            _candidate(0, line=10),
            _candidate(1, line=40, claim="Loop bound is off by one"),
            _candidate(2, file="src/other.py"),
        )
        fn = _model(json.dumps({"clusters": [[0, 1]]}))

        clusters = propose_clusters(candidates, fn, PROMPTS_DIR)

        assert clusters == ((0, 1),)

    def test_prompt_contains_every_candidate(self) -> None:
        candidates = (_candidate(0), _candidate(1, file="src/other.py", claim="Null deref"))
        fn = _model(json.dumps({"clusters": []}))

        propose_clusters(candidates, fn, PROMPTS_DIR)

        prompt = fn.calls[0]
        assert "src/app.py" in prompt
        assert "src/other.py" in prompt
        assert "Null deref" in prompt

    def test_cross_file_cluster_is_rejected(self) -> None:
        candidates = (_candidate(0, file="src/a.py"), _candidate(1, file="src/b.py"))
        fn = _model(json.dumps({"clusters": [[0, 1]]}))

        with pytest.raises(JudgeError, match="different files"):
            propose_clusters(candidates, fn, PROMPTS_DIR)

    def test_unknown_index_is_rejected(self) -> None:
        candidates = (_candidate(0), _candidate(1))
        fn = _model(json.dumps({"clusters": [[0, 7]]}))

        with pytest.raises(JudgeError, match="unknown"):
            propose_clusters(candidates, fn, PROMPTS_DIR)

    def test_overlapping_clusters_are_rejected(self) -> None:
        candidates = (_candidate(0), _candidate(1), _candidate(2))
        fn = _model(json.dumps({"clusters": [[0, 1], [1, 2]]}))

        with pytest.raises(JudgeError, match="more than one cluster"):
            propose_clusters(candidates, fn, PROMPTS_DIR)

    def test_singleton_clusters_are_dropped_not_rejected(self) -> None:
        """A judge that lists singletons is being chatty, not wrong."""
        candidates = (_candidate(0), _candidate(1))
        fn = _model(json.dumps({"clusters": [[0], [1]]}))

        assert propose_clusters(candidates, fn, PROMPTS_DIR) == ()

    def test_non_json_response_raises(self) -> None:
        candidates = (_candidate(0), _candidate(1))
        fn = _model("I could not decide.")

        with pytest.raises(JudgeError):
            propose_clusters(candidates, fn, PROMPTS_DIR)

    def test_fenced_json_is_tolerated(self) -> None:
        candidates = (_candidate(0), _candidate(1))
        fn = _model('```json\n{"clusters": [[0, 1]]}\n```')

        assert propose_clusters(candidates, fn, PROMPTS_DIR) == ((0, 1),)


class TestBuildJudgePrompt:
    def test_numbers_candidates_by_index(self) -> None:
        prompt = build_judge_prompt(
            "TEMPLATE", (_candidate(0), _candidate(1, claim="Other bug"))
        )
        assert prompt.startswith("TEMPLATE")
        assert "[0]" in prompt
        assert "[1]" in prompt
