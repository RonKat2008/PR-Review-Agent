"""Intent alignment (P8): the two-pass check and its parsing contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from prime_pr_review.intent import (
    IntentError,
    IntentStatement,
    build_pass1_prompt,
    build_pass2_prompt,
    parse_intent,
    parse_scope_response,
    run_intent_check,
)
from prime_pr_review.review import Scope, Severity, VerdictError

from .conftest import make_pr

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "pr-review" / "prompts"

# A marker that appears only inside the diff, never in PR metadata. Its presence
# or absence in a captured prompt is what proves which pass actually saw it.
DIFF_ONLY_MARKER = "DIFF_ONLY_MARKER_7f3c"

SAMPLE_DIFF = f"""diff --git a/src/app.py b/src/app.py
index 1111111..2222222 100644
--- a/src/app.py
+++ b/src/app.py
@@ -1,3 +1,3 @@
-total = sum(values)
+total = sum(values[:-1])  # {DIFF_ONLY_MARKER}
"""

PASS1_RESPONSE = (
    '{"intent": "Fixes a typo in the README installation instructions.",'
    '"expected_files": ["README.md"],'
    '"out_of_scope": ["Does not touch any code behavior"]}'
)

PASS2_RESPONSE_ALIGNED = '{"aligned": true, "unrelated": []}'

PASS2_RESPONSE_UNRELATED = (
    '{"aligned": false, "unrelated": [{"file": "src/auth.py", "lines": "10-14",'
    '"severity": "CRITICAL", "claim": "Adds an unrelated auth bypass",'
    '"evidence": "New branch skips the permission check; unrelated to a README fix"}]}'
)

PASS2_RESPONSE_CONTRADICTORY = (
    '{"aligned": true, "unrelated": [{"file": "src/app.py", "lines": "3",'
    '"severity": "LOW", "claim": "Leftover debug print", "evidence": "print() added, no ticket"}]}'
)


@dataclass(frozen=True)
class StubPR:
    """A PR with the full metadata pass 1 wants — title, body, commits, branch."""

    number: int = 42
    title: str = "Fix typo in README"
    author: str = "octocat"
    branch_name: str = "fix/readme-typo"
    body: str = "Fixes a small typo in the installation section."
    commit_messages: tuple[str, ...] = ("Fix typo",)


class RecordingModel:
    """A model_fn stub that returns canned responses in order and records every prompt."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._responses.pop(0)


# --- the load-bearing test -------------------------------------------------


def test_pass_one_prompt_contains_no_diff_content():
    """Pass 1 must never see the diff. If this test breaks, the whole check is theater."""
    model = RecordingModel([PASS1_RESPONSE, PASS2_RESPONSE_ALIGNED])

    run_intent_check(StubPR(), SAMPLE_DIFF, model, PROMPTS_DIR)

    pass1_prompt = model.prompts[0]
    assert DIFF_ONLY_MARKER not in pass1_prompt
    assert "```diff" not in pass1_prompt


def test_pass_two_prompt_contains_the_diff_and_the_pass_one_intent():
    model = RecordingModel([PASS1_RESPONSE, PASS2_RESPONSE_ALIGNED])

    run_intent_check(StubPR(), SAMPLE_DIFF, model, PROMPTS_DIR)

    pass2_prompt = model.prompts[1]
    assert DIFF_ONLY_MARKER in pass2_prompt
    assert "Fixes a typo in the README installation instructions." in pass2_prompt


def test_run_intent_check_calls_model_fn_exactly_twice():
    model = RecordingModel([PASS1_RESPONSE, PASS2_RESPONSE_ALIGNED])

    run_intent_check(StubPR(), SAMPLE_DIFF, model, PROMPTS_DIR)

    assert len(model.prompts) == 2


# --- run_intent_check end to end -------------------------------------------


def test_run_intent_check_returns_aligned_scope_when_pass_two_finds_nothing():
    model = RecordingModel([PASS1_RESPONSE, PASS2_RESPONSE_ALIGNED])

    scope = run_intent_check(StubPR(), SAMPLE_DIFF, model, PROMPTS_DIR)

    assert isinstance(scope, Scope)
    assert scope.aligned is True
    assert scope.unrelated == ()
    assert scope.intent == "Fixes a typo in the README installation instructions."


def test_run_intent_check_returns_unaligned_scope_with_the_flagged_issue():
    model = RecordingModel([PASS1_RESPONSE, PASS2_RESPONSE_UNRELATED])

    scope = run_intent_check(StubPR(), SAMPLE_DIFF, model, PROMPTS_DIR)

    assert scope.aligned is False
    assert len(scope.unrelated) == 1
    issue = scope.unrelated[0]
    assert issue.file == "src/auth.py"
    assert issue.severity is Severity.CRITICAL
    assert "auth bypass" in issue.claim


def test_run_intent_check_forces_unaligned_when_issues_contradict_the_aligned_flag():
    """Mirrors review._parse_scope: trust the findings over a model's self-contradiction."""
    model = RecordingModel([PASS1_RESPONSE, PASS2_RESPONSE_CONTRADICTORY])

    scope = run_intent_check(StubPR(), SAMPLE_DIFF, model, PROMPTS_DIR)

    assert scope.aligned is False
    assert len(scope.unrelated) == 1


def test_run_intent_check_stops_before_pass_two_when_pass_one_is_malformed():
    calls: list[str] = []

    def model_fn(prompt: str) -> str:
        calls.append(prompt)
        return "not json at all"

    with pytest.raises(IntentError):
        run_intent_check(StubPR(), SAMPLE_DIFF, model_fn, PROMPTS_DIR)

    assert len(calls) == 1


def test_run_intent_check_falls_back_gracefully_for_a_pr_missing_optional_fields():
    """A PullRequest without body/commit_messages/branch_name must not crash pass 1."""
    model = RecordingModel([PASS1_RESPONSE, PASS2_RESPONSE_ALIGNED])
    pr = make_pr(title="Bump dependency version")

    run_intent_check(pr, SAMPLE_DIFF, model, PROMPTS_DIR)

    pass1_prompt = model.prompts[0]
    assert "(unknown)" in pass1_prompt
    assert "(no description provided)" in pass1_prompt
    assert "(no commit messages provided)" in pass1_prompt


def test_run_intent_check_raises_intent_error_when_prompt_file_is_missing(tmp_path):
    model = RecordingModel([PASS1_RESPONSE, PASS2_RESPONSE_ALIGNED])

    with pytest.raises(IntentError, match="Could not read prompt"):
        run_intent_check(StubPR(), SAMPLE_DIFF, model, tmp_path)

    assert model.prompts == []


# --- build_pass1_prompt / build_pass2_prompt --------------------------------


def test_build_pass1_prompt_includes_title_body_commits_and_branch():
    prompt = build_pass1_prompt("TEMPLATE", StubPR())

    assert "Fix typo in README" in prompt
    assert "fix/readme-typo" in prompt
    assert "Fixes a small typo in the installation section." in prompt
    assert "- Fix typo" in prompt
    assert "octocat" in prompt


def test_build_pass2_prompt_includes_expected_files_and_out_of_scope():
    statement = IntentStatement(
        intent="Adds pagination to the search endpoint.",
        expected_files=("api/search.py",),
        out_of_scope=("Does not change the response schema",),
    )

    prompt = build_pass2_prompt("TEMPLATE", statement, SAMPLE_DIFF)

    assert "api/search.py" in prompt
    assert "Does not change the response schema" in prompt
    assert DIFF_ONLY_MARKER in prompt


def test_build_pass2_prompt_uses_placeholders_when_expected_files_are_empty():
    statement = IntentStatement(intent="Vague one-liner PR.")

    prompt = build_pass2_prompt("TEMPLATE", statement, SAMPLE_DIFF)

    assert "(none specified)" in prompt


# --- parse_intent ------------------------------------------------------------


def test_parse_intent_parses_expected_files_and_out_of_scope():
    statement = parse_intent(PASS1_RESPONSE)

    assert statement.intent == "Fixes a typo in the README installation instructions."
    assert statement.expected_files == ("README.md",)
    assert statement.out_of_scope == ("Does not touch any code behavior",)


def test_parse_intent_strips_a_json_code_fence():
    fenced = f"```json\n{PASS1_RESPONSE}\n```"

    statement = parse_intent(fenced)

    assert statement.intent == "Fixes a typo in the README installation instructions."


def test_parse_intent_raises_intent_error_on_invalid_json():
    with pytest.raises(IntentError, match="not usable"):
        parse_intent("this is not json")


def test_parse_intent_raises_intent_error_on_empty_response():
    with pytest.raises(IntentError, match="not usable"):
        parse_intent("   ")


def test_parse_intent_raises_intent_error_when_intent_field_is_missing():
    with pytest.raises(IntentError, match="non-empty 'intent'"):
        parse_intent('{"expected_files": [], "out_of_scope": []}')


def test_parse_intent_raises_intent_error_when_intent_field_is_blank():
    with pytest.raises(IntentError, match="non-empty 'intent'"):
        parse_intent('{"intent": "   "}')


def test_parse_intent_treats_non_list_expected_files_as_empty():
    statement = parse_intent('{"intent": "Does something.", "expected_files": "README.md"}')

    assert statement.expected_files == ()


def test_parse_intent_drops_blank_entries_from_out_of_scope():
    statement = parse_intent('{"intent": "Does something.", "out_of_scope": ["", "  ", "real one"]}')

    assert statement.out_of_scope == ("real one",)


def test_parse_intent_defaults_are_empty_tuples_when_fields_are_absent():
    statement = parse_intent('{"intent": "Minimal response."}')

    assert statement.expected_files == ()
    assert statement.out_of_scope == ()


# --- parse_scope_response -----------------------------------------------------


def test_parse_scope_response_uses_the_pass_one_intent_not_a_pass_two_echo():
    """Pass 2's contract has no 'intent' key; even if a model adds one, pass 1 wins."""
    raw = '{"aligned": true, "unrelated": [], "intent": "a pass-2 hallucination"}'

    scope = parse_scope_response(raw, "the real pass-1 intent")

    assert scope.intent == "the real pass-1 intent"


def test_parse_scope_response_strips_a_json_code_fence():
    fenced = f"```json\n{PASS2_RESPONSE_ALIGNED}\n```"

    scope = parse_scope_response(fenced, "some intent")

    assert scope.aligned is True


def test_parse_scope_response_raises_verdict_error_on_invalid_json():
    with pytest.raises(VerdictError, match="not valid JSON"):
        parse_scope_response("not json", "some intent")


@pytest.mark.parametrize(
    "severity_text, expected",
    [
        ("LOW", Severity.LOW),
        ("MEDIUM", Severity.MEDIUM),
        ("HIGH", Severity.HIGH),
        ("CRITICAL", Severity.CRITICAL),
    ],
)
def test_parse_scope_response_maps_each_severity_level(severity_text, expected):
    raw = (
        '{"aligned": false, "unrelated": [{"file": "a.py", "lines": "1",'
        f'"severity": "{severity_text}", "claim": "x", "evidence": "y"}}]}}'
    )

    scope = parse_scope_response(raw, "some intent")

    assert scope.unrelated[0].severity is expected


def test_parse_scope_response_raises_verdict_error_on_unknown_severity():
    raw = (
        '{"aligned": false, "unrelated": [{"file": "a.py", "lines": "1",'
        '"severity": "WHOOPS", "claim": "x", "evidence": "y"}]}'
    )

    with pytest.raises(VerdictError, match="Unknown severity"):
        parse_scope_response(raw, "some intent")


def test_parse_scope_response_raises_verdict_error_on_issue_without_a_claim():
    raw = (
        '{"aligned": false, "unrelated": [{"file": "a.py", "lines": "1",'
        '"severity": "LOW", "claim": "  ", "evidence": "y"}]}'
    )

    with pytest.raises(VerdictError, match="non-empty 'claim'"):
        parse_scope_response(raw, "some intent")


# --- IntentStatement -----------------------------------------------------------


def test_intent_statement_defaults_to_empty_tuples():
    statement = IntentStatement(intent="Just an intent, nothing else.")

    assert statement.expected_files == ()
    assert statement.out_of_scope == ()
