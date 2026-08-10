"""Static analysis pre-pass.

The model hallucinates findings a linter would catch deterministically, and burns
attention re-reporting lint-level issues real tools already caught for free. This
module runs ruff, bandit, and mypy over the changed files, normalizes their output
into one shape, and filters it down to lines the diff actually touched — a lint
error on an untouched line is noise, not a finding.

Every tool is optional. A missing binary is not an error, it is simply skipped and
recorded as unavailable. All subprocess use goes through one injected runner, so
this module is testable against recorded tool output without a network, a token,
or the tools themselves installed.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence, Set
from dataclasses import dataclass

from .review import Severity

WhichFn = Callable[[str], str | None]

ANALYSIS_TIMEOUT_SECONDS = 60

MYPY_LINE_PATTERN = re.compile(
    r"^(?P<file>.+?):(?P<line>\d+):(?:\d+:)?\s*"
    r"(?P<severity>error|warning|note):\s*"
    r"(?P<message>.*?)"
    r"(?:\s*\[(?P<code>[\w-]+)\])?$"
)

_BANDIT_SEVERITY_MAP = {
    "HIGH": Severity.CRITICAL,
    "MEDIUM": Severity.HIGH,
    "LOW": Severity.MEDIUM,
}


class AnalysisError(RuntimeError):
    """An analyzer invocation failed unexpectedly (not just "found issues")."""


@dataclass(frozen=True)
class LintFinding:
    """One finding, normalized onto the same shape and Severity scale regardless
    of which tool produced it."""

    tool: str
    rule_id: str
    file: str
    line: int
    severity: Severity
    message: str

    @property
    def corroboration(self) -> str:
        """The `tool:rule_id` tag `review.Finding.corroboration` expects, e.g.
        "bandit:B608". Falls back to the bare tool name if a rule has none."""
        return f"{self.tool}:{self.rule_id}" if self.rule_id else self.tool


@dataclass(frozen=True)
class ToolRun:
    """Raw output of one analyzer invocation."""

    stdout: str
    returncode: int
    stderr: str = ""


AnalysisRunner = Callable[[Sequence[str]], ToolRun]


@dataclass(frozen=True)
class AnalysisResult:
    findings: tuple[LintFinding, ...] = ()
    # Tool binaries that were not on PATH, skipped cleanly.
    unavailable: tuple[str, ...] = ()
    # Crashes and malformed output, one entry per failure. Never raised.
    errors: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not (self.findings or self.unavailable or self.errors)

    def rule_ids(self) -> frozenset[str]:
        """Rules already reported deterministically. The prompt uses this to tell
        the model not to repeat them."""
        return frozenset(f.corroboration for f in self.findings)

    def render(self) -> str:
        """Render as the markdown block appended to the review prompt."""
        parts = [
            "## Static analysis findings",
            "",
            _render_findings(self.findings),
            _render_unavailable(self.unavailable),
            _render_errors(self.errors),
            _render_footer(self.findings),
        ]
        return "\n".join(p for p in parts if p is not None).rstrip() + "\n"


@dataclass(frozen=True)
class _ToolSpec:
    binary: str
    ok_returncodes: frozenset[int]
    build_args: Callable[[Sequence[str]], tuple[str, ...]]
    parse: Callable[[str], tuple[tuple[LintFinding, ...], tuple[str, ...]]]


def default_analysis_runner(args: Sequence[str]) -> ToolRun:
    """Run one analyzer and capture its output.

    A nonzero exit is normal here — ruff, bandit, and mypy all exit nonzero when
    they *find* issues, not only when they crash. Distinguishing the two is
    `run_analysis`'s job, not this function's.
    """
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=ANALYSIS_TIMEOUT_SECONDS,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise AnalysisError(f"`{' '.join(args)}` failed: {exc}") from exc
    return ToolRun(stdout=result.stdout, returncode=result.returncode, stderr=result.stderr)


def run_analysis(
    paths: Sequence[str],
    diff_lines: Set[tuple[str, int]],
    runner: AnalysisRunner = default_analysis_runner,
    which: WhichFn = shutil.which,
) -> AnalysisResult:
    """Run every available analyzer over `paths`, keeping only findings that land
    on a (file, line) present in `diff_lines`."""
    python_paths = tuple(p for p in paths if p.endswith(".py"))
    if not python_paths:
        return AnalysisResult()

    findings: list[LintFinding] = []
    unavailable: list[str] = []
    errors: list[str] = []

    for spec in TOOL_SPECS:
        if which(spec.binary) is None:
            unavailable.append(spec.binary)
            continue
        tool_findings, tool_errors = _run_one_tool(spec, python_paths, runner)
        findings.extend(f for f in tool_findings if (f.file, f.line) in diff_lines)
        errors.extend(tool_errors)

    return AnalysisResult(
        findings=tuple(findings), unavailable=tuple(unavailable), errors=tuple(errors)
    )


def _run_one_tool(
    spec: _ToolSpec, paths: Sequence[str], runner: AnalysisRunner
) -> tuple[tuple[LintFinding, ...], tuple[str, ...]]:
    args = spec.build_args(paths)
    try:
        result = runner(args)
    except AnalysisError as exc:
        return (), (f"{spec.binary}: {exc}",)

    if result.returncode not in spec.ok_returncodes:
        detail = result.stderr.strip() or result.stdout.strip() or "no output"
        return (), (f"{spec.binary}: exited with code {result.returncode}: {detail}",)

    try:
        return spec.parse(result.stdout)
    except (
        AnalysisError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
    ) as exc:
        return (), (f"{spec.binary}: malformed output: {exc}",)


# --- ruff ----------------------------------------------------------------------


def _parse_ruff(raw: str) -> tuple[tuple[LintFinding, ...], tuple[str, ...]]:
    payload = json.loads(raw) if raw.strip() else []
    if not isinstance(payload, list):
        raise AnalysisError(f"expected a JSON array, got {type(payload).__name__}")
    return tuple(_ruff_finding(item) for item in payload), ()


def _ruff_finding(item: dict) -> LintFinding:
    code = str(item.get("code") or "")
    location = item.get("location") or {}
    return LintFinding(
        tool="ruff",
        rule_id=code,
        file=str(item.get("filename", "")).replace("\\", "/"),
        line=int(location.get("row", 0)),
        severity=_ruff_severity(code),
        message=str(item.get("message", "")).strip(),
    )


def _ruff_severity(code: str) -> Severity:
    """Ruff's JSON output carries no severity field, only a rule code, so the code
    has to stand in for one. `S`-prefixed rules are flake8-bandit's security checks
    surfaced through ruff and are treated like bandit's own HIGH; `E9`/`F82` mean the
    file doesn't parse or references an undefined name, both fatal at runtime; plain
    `F` codes (pyflakes) are usually real bugs; everything else is style."""
    if code.startswith("S") or code.startswith("E9") or code.startswith("F82"):
        return Severity.HIGH
    if code.startswith("F"):
        return Severity.MEDIUM
    return Severity.LOW


# --- bandit ----------------------------------------------------------------------


def _parse_bandit(raw: str) -> tuple[tuple[LintFinding, ...], tuple[str, ...]]:
    payload = json.loads(raw) if raw.strip() else {}
    if not isinstance(payload, dict):
        raise AnalysisError(f"expected a JSON object, got {type(payload).__name__}")
    findings = tuple(_bandit_finding(item) for item in payload.get("results", []))
    errors = tuple(_bandit_error(item) for item in payload.get("errors", []))
    return findings, errors


def _bandit_finding(item: dict) -> LintFinding:
    return LintFinding(
        tool="bandit",
        rule_id=str(item.get("test_id", "")),
        file=str(item.get("filename", "")).replace("\\", "/"),
        line=int(item.get("line_number", 0)),
        severity=_bandit_severity(str(item.get("issue_severity", "")).upper()),
        message=str(item.get("issue_text", "")).strip(),
    )


def _bandit_severity(level: str) -> Severity:
    """Bandit only fires on real security patterns, so its own HIGH/MEDIUM/LOW are
    shifted up a notch relative to a plain style linter's."""
    return _BANDIT_SEVERITY_MAP.get(level, Severity.LOW)


def _bandit_error(item: dict) -> str:
    filename = str(item.get("filename", "?"))
    reason = str(item.get("reason", "unknown error"))
    return f"bandit: {reason} ({filename})"


# --- mypy ------------------------------------------------------------------------


def _parse_mypy(raw: str) -> tuple[tuple[LintFinding, ...], tuple[str, ...]]:
    """Text output, one diagnostic per line. Lines that don't match — blank lines,
    "Success: no issues found", anything unexpected — are skipped, not errors."""
    findings = tuple(f for f in (_mypy_finding(line) for line in raw.splitlines()) if f is not None)
    return findings, ()


def _mypy_finding(line: str) -> LintFinding | None:
    match = MYPY_LINE_PATTERN.match(line.strip())
    if match is None or match.group("severity") == "note":
        return None
    return LintFinding(
        tool="mypy",
        rule_id=match.group("code") or "",
        file=match.group("file").strip().replace("\\", "/"),
        line=int(match.group("line")),
        severity=_mypy_severity(match.group("severity")),
        message=match.group("message").strip(),
    )


def _mypy_severity(word: str) -> Severity:
    return Severity.MEDIUM if word == "error" else Severity.LOW


TOOL_SPECS: tuple[_ToolSpec, ...] = (
    _ToolSpec(
        binary="ruff",
        ok_returncodes=frozenset({0, 1}),
        build_args=lambda paths: ("ruff", "check", "--output-format", "json", *paths),
        parse=_parse_ruff,
    ),
    _ToolSpec(
        binary="bandit",
        ok_returncodes=frozenset({0, 1}),
        build_args=lambda paths: ("bandit", "-f", "json", "-r", *paths),
        parse=_parse_bandit,
    ),
    _ToolSpec(
        binary="mypy",
        ok_returncodes=frozenset({0, 1}),
        build_args=lambda paths: ("mypy", "--no-error-summary", *paths),
        parse=_parse_mypy,
    ),
)


# --- rendering ---------------------------------------------------------------


def _render_findings(findings: tuple[LintFinding, ...]) -> str:
    lines = ["### Findings", ""]
    if not findings:
        lines.append("_None found._")
    else:
        for finding in _sorted_findings(findings):
            lines.append(
                f"- **{finding.severity.value}** `{finding.file}:{finding.line}` "
                f"`{finding.corroboration}` — {finding.message}"
            )
    lines.append("")
    return "\n".join(lines)


def _sorted_findings(findings: tuple[LintFinding, ...]) -> tuple[LintFinding, ...]:
    return tuple(sorted(findings, key=lambda f: (f.severity.rank, f.file, f.line)))


def _render_unavailable(unavailable: tuple[str, ...]) -> str | None:
    if not unavailable:
        return None
    lines = ["### Tools unavailable", ""]
    lines.extend(f"- `{name}` not installed; skipped" for name in unavailable)
    lines.append("")
    return "\n".join(lines)


def _render_errors(errors: tuple[str, ...]) -> str | None:
    if not errors:
        return None
    lines = ["### Tool errors", ""]
    lines.extend(f"- {err}" for err in errors)
    lines.append("")
    return "\n".join(lines)


def _render_footer(findings: tuple[LintFinding, ...]) -> str | None:
    if not findings:
        return None
    return (
        "_The findings above came from deterministic static analysis, not the model. "
        "Do not repeat them as new findings._\n"
    )
