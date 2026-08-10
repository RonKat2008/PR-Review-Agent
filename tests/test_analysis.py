"""Static analysis pre-pass: parsing real tool output, diff-line filtering, and
severity mapping. Every tool invocation is faked, so nothing here shells out.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import pytest

from prime_pr_review import analysis
from prime_pr_review.analysis import (
    AnalysisError,
    AnalysisResult,
    LintFinding,
    ToolRun,
    WhichFn,
    default_analysis_runner,
    run_analysis,
)
from prime_pr_review.review import Severity

# --- realistic recorded tool output --------------------------------------------

RUFF_JSON = json.dumps(
    [
        {
            "cell": None,
            "code": "F401",
            "end_location": {"column": 20, "row": 1},
            "filename": "src/app.py",
            "fix": None,
            "location": {"column": 8, "row": 1},
            "message": "`os` imported but unused",
            "noqa_row": 1,
            "url": "https://docs.astral.sh/ruff/rules/unused-import",
        },
        {
            "cell": None,
            "code": "S608",
            "end_location": {"column": 60, "row": 12},
            "filename": "src/app.py",
            "fix": None,
            "location": {"column": 13, "row": 12},
            "message": "Possible SQL injection vector through string-based query construction",
            "noqa_row": 12,
            "url": "https://docs.astral.sh/ruff/rules/hardcoded-sql-expression",
        },
        {
            "cell": None,
            "code": "E501",
            "end_location": {"column": 90, "row": 20},
            "filename": "src/app.py",
            "fix": None,
            "location": {"column": 89, "row": 20},
            "message": "Line too long (90 > 88)",
            "noqa_row": 20,
            "url": None,
        },
    ]
)

BANDIT_JSON = json.dumps(
    {
        "errors": [],
        "generated_at": "2024-01-01T00:00:00Z",
        "results": [
            {
                "code": '8         query = "SELECT * FROM users WHERE id = " + user_id\n',
                "col_offset": 12,
                "end_col_offset": 60,
                "filename": "src/db.py",
                "issue_confidence": "MEDIUM",
                "issue_cwe": {"id": 89, "link": "https://cwe.mitre.org/data/definitions/89.html"},
                "issue_severity": "MEDIUM",
                "issue_text": "Possible SQL injection vector through string-based query construction.",
                "line_number": 8,
                "line_range": [8],
                "more_info": "https://bandit.readthedocs.io/en/1.7.x/plugins/b608.html",
                "test_id": "B608",
                "test_name": "hardcoded_sql_expressions",
            }
        ],
    }
)

BANDIT_JSON_WITH_TOOL_ERROR = json.dumps(
    {
        "errors": [{"filename": "src/broken.py", "reason": "SyntaxError: invalid syntax"}],
        "results": [],
    }
)

MYPY_TEXT = (
    'src/app.py:10: error: Incompatible types in assignment (expression has type "str", '
    'variable has type "int")  [assignment]\n'
    'src/app.py:15: note: Revealed type is "builtins.int"\n'
    'src/other.py:3: error: Name "foo" is not defined  [name-defined]\n'
)

ALL_DIFF_LINES = frozenset(
    {
        ("src/app.py", 1),
        ("src/app.py", 12),
        ("src/app.py", 20),
        ("src/app.py", 10),
        ("src/db.py", 8),
        ("src/other.py", 3),
    }
)

ALL_TOOLS = ("ruff", "bandit", "mypy")


# --- fakes -----------------------------------------------------------------------


@dataclass
class FakeRunner:
    """An analysis runner keyed on the tool binary. Records every call."""

    responses: dict[str, ToolRun] = field(default_factory=dict)
    raise_for: frozenset[str] = frozenset()
    calls: list[list[str]] = field(default_factory=list)

    def __call__(self, args: Sequence[str]) -> ToolRun:
        self.calls.append(list(args))
        binary = args[0]
        if binary in self.raise_for:
            raise AnalysisError(f"{binary} exploded")
        return self.responses.get(binary, ToolRun(stdout="", returncode=0))


def fake_which(available: Iterable[str]) -> WhichFn:
    """A `shutil.which` stand-in that only "finds" the given binaries."""
    available_set = frozenset(available)

    def which(binary: str) -> str | None:
        return f"/usr/bin/{binary}" if binary in available_set else None

    return which


def clean_runner(*, extra: dict[str, ToolRun] | None = None) -> FakeRunner:
    """A runner where every tool ran and found nothing, unless overridden."""
    responses = {
        "ruff": ToolRun(stdout="[]", returncode=0),
        "bandit": ToolRun(stdout=json.dumps({"errors": [], "results": []}), returncode=0),
        "mypy": ToolRun(stdout="", returncode=0),
    }
    responses.update(extra or {})
    return FakeRunner(responses=responses)


# --- parsing real output formats -------------------------------------------------


def test_ruff_findings_are_parsed_from_real_json_output():
    runner = clean_runner(extra={"ruff": ToolRun(stdout=RUFF_JSON, returncode=1)})

    result = run_analysis(["src/app.py"], ALL_DIFF_LINES, runner, fake_which(ALL_TOOLS))

    ruff_findings = {f.rule_id: f for f in result.findings if f.tool == "ruff"}
    assert ruff_findings["F401"].file == "src/app.py"
    assert ruff_findings["F401"].line == 1
    assert ruff_findings["F401"].message == "`os` imported but unused"
    assert ruff_findings["F401"].severity is Severity.MEDIUM


def test_bandit_findings_are_parsed_from_real_json_output():
    runner = clean_runner(extra={"bandit": ToolRun(stdout=BANDIT_JSON, returncode=1)})

    result = run_analysis(["src/db.py"], ALL_DIFF_LINES, runner, fake_which(ALL_TOOLS))

    [finding] = [f for f in result.findings if f.tool == "bandit"]
    assert finding.rule_id == "B608"
    assert finding.file == "src/db.py"
    assert finding.line == 8
    assert finding.severity is Severity.HIGH
    assert "SQL injection" in finding.message


def test_mypy_findings_are_parsed_from_real_text_output():
    runner = clean_runner(extra={"mypy": ToolRun(stdout=MYPY_TEXT, returncode=1)})

    result = run_analysis(["src/app.py", "src/other.py"], ALL_DIFF_LINES, runner, fake_which(ALL_TOOLS))

    mypy_findings = {(f.file, f.line): f for f in result.findings if f.tool == "mypy"}
    assert mypy_findings[("src/app.py", 10)].rule_id == "assignment"
    assert mypy_findings[("src/other.py", 3)].rule_id == "name-defined"
    assert mypy_findings[("src/other.py", 3)].severity is Severity.MEDIUM


def test_mypy_note_lines_are_not_reported_as_findings():
    runner = clean_runner(extra={"mypy": ToolRun(stdout=MYPY_TEXT, returncode=1)})

    result = run_analysis(["src/app.py"], ALL_DIFF_LINES, runner, fake_which(ALL_TOOLS))

    assert ("src/app.py", 15) not in {(f.file, f.line) for f in result.findings if f.tool == "mypy"}


def test_mypy_lines_that_do_not_match_the_expected_format_are_skipped_not_errors():
    garbage = "Found 1 error in 1 file (checked 3 source files)\nSuccess: no issues found\n"
    runner = clean_runner(extra={"mypy": ToolRun(stdout=garbage, returncode=0)})

    result = run_analysis(["src/app.py"], ALL_DIFF_LINES, runner, fake_which(ALL_TOOLS))

    assert not [f for f in result.findings if f.tool == "mypy"]
    assert result.errors == ()


def test_ruff_item_with_missing_location_defaults_line_to_zero_without_crashing():
    payload = json.dumps([{"code": "E501", "filename": "src/app.py", "message": "too long"}])
    runner = clean_runner(extra={"ruff": ToolRun(stdout=payload, returncode=1)})

    result = run_analysis(
        ["src/app.py"], frozenset({("src/app.py", 0)}), runner, fake_which(ALL_TOOLS)
    )

    assert result.errors == ()
    [finding] = result.findings
    assert finding.line == 0


# --- missing tools ------------------------------------------------------------


def test_missing_tool_binary_is_skipped_cleanly_and_recorded_as_unavailable():
    runner = clean_runner()

    result = run_analysis(["src/app.py"], ALL_DIFF_LINES, runner, fake_which({"ruff", "bandit"}))

    assert result.unavailable == ("mypy",)
    assert result.errors == ()
    assert {c[0] for c in runner.calls} == {"ruff", "bandit"}


def test_no_tools_available_yields_only_unavailable_entries_and_no_findings():
    runner = clean_runner()

    result = run_analysis(["src/app.py"], ALL_DIFF_LINES, runner, fake_which(()))

    assert result.findings == ()
    assert set(result.unavailable) == set(ALL_TOOLS)
    assert runner.calls == []


# --- diff-line filtering --------------------------------------------------------


def test_findings_outside_the_diff_are_filtered_out():
    runner = clean_runner(extra={"ruff": ToolRun(stdout=RUFF_JSON, returncode=1)})
    # Only line 1 of src/app.py was actually touched by the diff.
    narrow_diff_lines = frozenset({("src/app.py", 1)})

    result = run_analysis(["src/app.py"], narrow_diff_lines, runner, fake_which(ALL_TOOLS))

    assert [f.rule_id for f in result.findings if f.tool == "ruff"] == ["F401"]


def test_findings_with_no_diff_lines_at_all_are_all_filtered_out():
    runner = clean_runner(extra={"ruff": ToolRun(stdout=RUFF_JSON, returncode=1)})

    result = run_analysis(["src/app.py"], frozenset(), runner, fake_which(ALL_TOOLS))

    assert result.findings == ()


# --- non-python paths -----------------------------------------------------------


def test_no_python_paths_short_circuits_without_invoking_any_tool():
    runner = clean_runner()

    result = run_analysis(["README.md", "uv.lock"], ALL_DIFF_LINES, runner, fake_which(ALL_TOOLS))

    assert result == AnalysisResult()
    assert runner.calls == []


def test_non_python_paths_are_excluded_from_the_tool_command_line():
    runner = clean_runner()

    run_analysis(["src/app.py", "README.md"], ALL_DIFF_LINES, runner, fake_which(ALL_TOOLS))

    ruff_call = next(c for c in runner.calls if c[0] == "ruff")
    assert "README.md" not in ruff_call
    assert "src/app.py" in ruff_call


def test_command_arguments_are_built_correctly_for_each_tool():
    runner = clean_runner()

    run_analysis(["src/app.py"], ALL_DIFF_LINES, runner, fake_which(ALL_TOOLS))

    calls = {c[0]: c for c in runner.calls}
    assert calls["ruff"] == ["ruff", "check", "--output-format", "json", "src/app.py"]
    assert calls["bandit"] == ["bandit", "-f", "json", "-r", "src/app.py"]
    assert calls["mypy"] == ["mypy", "--no-error-summary", "src/app.py"]


# --- crashes and malformed output -------------------------------------------------


def test_malformed_ruff_json_is_recorded_as_an_error_not_raised():
    runner = clean_runner(extra={"ruff": ToolRun(stdout="not json {{{", returncode=0)})

    result = run_analysis(["src/app.py"], ALL_DIFF_LINES, runner, fake_which(ALL_TOOLS))

    assert result.findings == ()
    assert any("ruff" in err for err in result.errors)


def test_ruff_json_that_is_not_a_list_is_recorded_as_an_error():
    runner = clean_runner(extra={"ruff": ToolRun(stdout='{"not": "a list"}', returncode=0)})

    result = run_analysis(["src/app.py"], ALL_DIFF_LINES, runner, fake_which(ALL_TOOLS))

    assert any("ruff" in err for err in result.errors)


def test_bandit_tool_level_errors_are_recorded_without_becoming_findings():
    runner = clean_runner(extra={"bandit": ToolRun(stdout=BANDIT_JSON_WITH_TOOL_ERROR, returncode=1)})

    result = run_analysis(["src/app.py"], ALL_DIFF_LINES, runner, fake_which(ALL_TOOLS))

    assert not [f for f in result.findings if f.tool == "bandit"]
    assert any("SyntaxError" in err and "src/broken.py" in err for err in result.errors)


def test_a_tool_crash_returncode_is_recorded_as_an_error_and_yields_no_findings():
    runner = clean_runner(
        extra={"mypy": ToolRun(stdout="", returncode=2, stderr="mypy: fatal error: bad config")}
    )

    result = run_analysis(["src/app.py"], ALL_DIFF_LINES, runner, fake_which(ALL_TOOLS))

    assert not [f for f in result.findings if f.tool == "mypy"]
    assert any("exited with code 2" in err and "bad config" in err for err in result.errors)


def test_a_tool_crash_with_no_stderr_falls_back_to_a_generic_message():
    runner = clean_runner(extra={"mypy": ToolRun(stdout="", returncode=2, stderr="")})

    result = run_analysis(["src/app.py"], ALL_DIFF_LINES, runner, fake_which(ALL_TOOLS))

    assert any("mypy" in err and "no output" in err for err in result.errors)


def test_a_runner_that_raises_for_one_tool_does_not_block_the_others():
    runner = clean_runner(extra={"ruff": ToolRun(stdout=RUFF_JSON, returncode=1)})
    runner.raise_for = frozenset({"bandit"})

    result = run_analysis(["src/app.py"], ALL_DIFF_LINES, runner, fake_which(ALL_TOOLS))

    assert any(f.tool == "ruff" for f in result.findings)
    assert any("bandit" in err for err in result.errors)


def test_bandit_json_that_is_not_an_object_is_recorded_as_an_error():
    runner = clean_runner(extra={"bandit": ToolRun(stdout="[]", returncode=0)})

    result = run_analysis(["src/app.py"], ALL_DIFF_LINES, runner, fake_which(ALL_TOOLS))

    assert any("bandit" in err for err in result.errors)


# --- severity mapping -------------------------------------------------------------


@pytest.mark.parametrize(
    "code,expected",
    [
        ("S608", Severity.HIGH),
        ("E999", Severity.HIGH),
        ("F821", Severity.HIGH),  # F82x: undefined name
        ("F401", Severity.MEDIUM),
        ("E501", Severity.LOW),
        ("D100", Severity.LOW),
    ],
)
def test_ruff_severity_mapping_by_rule_code(code, expected):
    assert analysis._ruff_severity(code) is expected


@pytest.mark.parametrize(
    "level,expected",
    [
        ("HIGH", Severity.CRITICAL),
        ("MEDIUM", Severity.HIGH),
        ("LOW", Severity.MEDIUM),
        ("", Severity.LOW),
    ],
)
def test_bandit_severity_mapping_by_issue_severity(level, expected):
    assert analysis._bandit_severity(level) is expected


@pytest.mark.parametrize("word,expected", [("error", Severity.MEDIUM), ("warning", Severity.LOW)])
def test_mypy_severity_mapping_by_diagnostic_word(word, expected):
    assert analysis._mypy_severity(word) is expected


# --- rule_ids / corroboration -----------------------------------------------------


def test_rule_ids_returns_tool_colon_rule_id_for_every_finding():
    result = AnalysisResult(
        findings=(
            LintFinding("ruff", "F401", "a.py", 1, Severity.MEDIUM, "unused"),
            LintFinding("bandit", "B608", "b.py", 2, Severity.HIGH, "sql"),
        )
    )

    assert result.rule_ids() == {"ruff:F401", "bandit:B608"}


def test_corroboration_falls_back_to_bare_tool_name_when_rule_id_is_empty():
    finding = LintFinding("mypy", "", "a.py", 1, Severity.MEDIUM, "type error")

    assert finding.corroboration == "mypy"


def test_lintfinding_is_frozen():
    finding = LintFinding("ruff", "F401", "a.py", 1, Severity.MEDIUM, "unused")

    with pytest.raises(AttributeError):
        finding.line = 2  # type: ignore[misc]


def test_analysis_result_is_frozen():
    result = AnalysisResult()

    with pytest.raises(AttributeError):
        result.findings = ()  # type: ignore[misc]


# --- is_empty ----------------------------------------------------------------


def test_is_empty_is_true_when_nothing_ran_and_nothing_was_found():
    assert AnalysisResult().is_empty is True


def test_is_empty_is_false_once_any_section_has_content():
    assert AnalysisResult(unavailable=("mypy",)).is_empty is False


def test_a_fully_clean_run_across_all_tools_yields_no_findings_no_errors():
    runner = clean_runner()

    result = run_analysis(["src/app.py"], ALL_DIFF_LINES, runner, fake_which(ALL_TOOLS))

    assert result == AnalysisResult()


# --- render --------------------------------------------------------------------


def test_render_includes_the_top_level_header():
    markdown = AnalysisResult().render()

    assert markdown.startswith("## Static analysis findings")


def test_render_reports_none_found_when_there_are_no_findings():
    markdown = AnalysisResult().render()

    assert "### Findings" in markdown
    assert "_None found._" in markdown


def test_render_omits_unavailable_and_errors_sections_when_empty():
    markdown = AnalysisResult().render()

    assert "Tools unavailable" not in markdown
    assert "Tool errors" not in markdown


def test_render_lists_findings_with_severity_location_rule_and_message():
    result = AnalysisResult(
        findings=(LintFinding("bandit", "B608", "src/db.py", 8, Severity.HIGH, "SQL injection"),)
    )

    markdown = result.render()

    assert "**HIGH**" in markdown
    assert "`src/db.py:8`" in markdown
    assert "`bandit:B608`" in markdown
    assert "SQL injection" in markdown


def test_render_sorts_findings_by_severity_then_file_then_line():
    result = AnalysisResult(
        findings=(
            LintFinding("ruff", "E501", "b.py", 1, Severity.LOW, "low issue"),
            LintFinding("bandit", "B608", "a.py", 5, Severity.CRITICAL, "critical issue"),
            LintFinding("mypy", "assignment", "a.py", 1, Severity.MEDIUM, "medium issue"),
        )
    )

    markdown = result.render()

    assert markdown.index("critical issue") < markdown.index("medium issue") < markdown.index(
        "low issue"
    )


def test_render_lists_unavailable_tools_when_present():
    markdown = AnalysisResult(unavailable=("mypy",)).render()

    assert "### Tools unavailable" in markdown
    assert "`mypy` not installed; skipped" in markdown


def test_render_lists_tool_errors_when_present():
    markdown = AnalysisResult(errors=("ruff: malformed output: boom",)).render()

    assert "### Tool errors" in markdown
    assert "ruff: malformed output: boom" in markdown


def test_render_appends_a_do_not_repeat_footer_only_when_findings_exist():
    with_findings = AnalysisResult(
        findings=(LintFinding("ruff", "F401", "a.py", 1, Severity.MEDIUM, "unused"),)
    ).render()
    without_findings = AnalysisResult().render()

    assert "Do not repeat them" in with_findings
    assert "Do not repeat them" not in without_findings


# --- default runner (subprocess wrapping, mocked so nothing shells out) -----------


def test_default_analysis_runner_wraps_a_successful_subprocess_call(monkeypatch):
    class FakeCompleted:
        stdout = "[]"
        stderr = ""
        returncode = 0

    captured: dict = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeCompleted()

    monkeypatch.setattr(analysis.subprocess, "run", fake_run)

    result = default_analysis_runner(["ruff", "check"])

    assert result == ToolRun(stdout="[]", returncode=0, stderr="")
    assert captured["args"] == ["ruff", "check"]
    assert captured["kwargs"]["capture_output"] is True


def test_default_analysis_runner_wraps_a_timeout_as_an_analysis_error(monkeypatch):
    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="ruff", timeout=1)

    monkeypatch.setattr(analysis.subprocess, "run", raise_timeout)

    with pytest.raises(AnalysisError):
        default_analysis_runner(["ruff", "check"])


def test_default_analysis_runner_wraps_an_os_error_as_an_analysis_error(monkeypatch):
    def raise_os_error(*args, **kwargs):
        raise OSError("binary vanished")

    monkeypatch.setattr(analysis.subprocess, "run", raise_os_error)

    with pytest.raises(AnalysisError):
        default_analysis_runner(["ruff", "check"])
