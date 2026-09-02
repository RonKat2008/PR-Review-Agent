"""Adversarial refutation (P14): the skeptic pass that challenges findings with
concrete grounds — and fails open, never silently killing a finding it could
not properly examine."""

from __future__ import annotations

import json

from prime_pr_review.refute import build_refute_prompt, refute_findings
from prime_pr_review.review import Finding, Severity

PROMPTS_DIR = "skills/pr-review/prompts"
DIFF = "diff --git a/src/app.py b/src/app.py\n@@ -1,3 +1,3 @@\n-old\n+new\n"


def _finding(
    *,
    file: str = "src/app.py",
    line: int | None = 10,
    severity: Severity = Severity.HIGH,
    claim: str = "Off-by-one in the loop bound",
    evidence: str = "the loop runs one extra time",
    suggestion: str = "",
) -> Finding:
    return Finding(
        file=file,
        line=line,
        severity=severity,
        claim=claim,
        evidence=evidence,
        suggestion=suggestion,
    )


def _scripted(*responses: str | Exception):
    """A ModelFn stub returning (or raising) the next scripted response."""
    queue = list(responses)
    calls: list[str] = []

    def fn(prompt: str) -> str:
        assert queue, "skeptic called more times than scripted"
        calls.append(prompt)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    fn.calls = calls  # type: ignore[attr-defined]
    return fn


def _refuted(reasoning: str = "the flagged line is inside a dead branch") -> str:
    return json.dumps({"refuted": True, "reasoning": reasoning})


def _upheld() -> str:
    return json.dumps({"refuted": False, "reasoning": "the claim holds"})


class TestRefuteFindings:
    def test_refuted_finding_is_annotated_not_dropped(self) -> None:
        fn = _scripted(_refuted())

        kept, notes = refute_findings((_finding(),), DIFF, fn, PROMPTS_DIR)

        assert len(kept) == 1
        assert kept[0].refuted is True
        assert "dead branch" in kept[0].refutation
        assert any("1/1" in note for note in notes)

    def test_upheld_finding_is_unchanged(self) -> None:
        original = _finding()
        fn = _scripted(_upheld())

        kept, notes = refute_findings((original,), DIFF, fn, PROMPTS_DIR)

        assert kept == (original,)
        assert any("0/1" in note for note in notes)

    def test_skeptic_crash_fails_open_with_note(self) -> None:
        original = _finding()
        fn = _scripted(RuntimeError("subagent died"))

        kept, notes = refute_findings((original,), DIFF, fn, PROMPTS_DIR)

        assert kept == (original,)
        assert kept[0].refuted is False
        assert any("skeptic failed" in note for note in notes)

    def test_refutation_without_reasoning_fails_open(self) -> None:
        """A skeptic that cannot say WHY does not get to kill a finding."""
        original = _finding()
        fn = _scripted(json.dumps({"refuted": True, "reasoning": ""}))

        kept, notes = refute_findings((original,), DIFF, fn, PROMPTS_DIR)

        assert kept[0].refuted is False
        assert any("skeptic failed" in note for note in notes)

    def test_one_crash_does_not_sink_the_other_findings(self) -> None:
        findings = (_finding(line=10), _finding(line=50, claim="Null deref"))
        fn = _scripted(RuntimeError("boom"), _refuted())

        kept, _ = refute_findings(findings, DIFF, fn, PROMPTS_DIR)

        assert kept[0].refuted is False
        assert kept[1].refuted is True

    def test_no_findings_means_no_calls_and_no_notes(self) -> None:
        fn = _scripted()

        kept, notes = refute_findings((), DIFF, fn, PROMPTS_DIR)

        assert kept == ()
        assert notes == ()

    def test_fenced_json_is_tolerated(self) -> None:
        fn = _scripted('```json\n{"refuted": true, "reasoning": "guarded by the if"}\n```')

        kept, _ = refute_findings((_finding(),), DIFF, fn, PROMPTS_DIR)

        assert kept[0].refuted is True


class TestBuildRefutePrompt:
    def test_contains_finding_and_diff(self) -> None:
        prompt = build_refute_prompt("TEMPLATE", _finding(suggestion="fixed = 1"), DIFF)

        assert prompt.startswith("TEMPLATE")
        assert "src/app.py" in prompt
        assert "Off-by-one" in prompt
        assert "fixed = 1" in prompt
        assert DIFF.strip() in prompt
