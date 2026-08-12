"""CI status awareness (P12).

A reviewer that only reads the diff has no idea whether the PR's own test suite
is currently passing. A competing reviewer that opens the Actions run, names the
failing step, and quotes the exact error reframes its whole verdict around that
fact before saying anything else; a reviewer that stays silent about a red build
looks naive by comparison. This module is the deterministic half of that
capability: fetch CI status and a best-effort failure-log excerpt through `gh`,
entirely through an injected runner, and render both into markdown for the
review prompt and the review header.

CI data is enrichment, never load-bearing: any `gh` failure, timeout, or
payload that doesn't parse degrades to an empty/unknown result rather than
raising. CI awareness must never be the thing that breaks a sweep.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .github import GhRunner, default_runner

# The `gh pr checks --json` field set this module relies on. All five are real
# fields (confirmed against `gh pr checks --help`, gh 2.97): `bucket` is gh's
# own pass/fail/pending/skipping/cancel categorization of `state`, the raw
# upstream check/commit-status value.
CHECK_FIELDS = "name,state,bucket,link,description"

# The `gh run list --json` field set used by the failure-excerpt fallback path.
RUN_LIST_FIELDS = "databaseId,headSha"

# A GitHub Actions check's `link` looks like `.../actions/runs/<id>/job/<id>`.
_RUN_ID_PATTERN = re.compile(r"/actions/runs/(\d+)")

# `gh pr checks --json`'s own `bucket` categorization, mapped onto our status
# vocabulary. Per `gh pr checks --help`: "bucket ... categorizes the `state`
# field into `pass`, `fail`, `pending`, `skipping`, or `cancel`."
_BUCKET_TO_STATUS: dict[str, str] = {
    "pass": "pass",
    "fail": "fail",
    "pending": "pending",
    "skipping": "skipped",
    "cancel": "cancelled",
}

# Fallback used only when `bucket` is absent or unrecognized: gh's raw check
# and workflow-run state/conclusion vocabulary (see `gh run list --help`'s
# `--status` values), mapped onto our five statuses. Kept conservative --
# anything not clearly a pass or a hard failure defaults to "pending" rather
# than guessing.
_STATE_TO_STATUS: dict[str, str] = {
    "success": "pass",
    "neutral": "pass",
    "failure": "fail",
    "error": "fail",
    "timed_out": "fail",
    "startup_failure": "fail",
    "cancelled": "cancelled",
    "canceled": "cancelled",
    "skipped": "skipped",
    "pending": "pending",
    "queued": "pending",
    "requested": "pending",
    "waiting": "pending",
    "in_progress": "pending",
    "action_required": "pending",
    "stale": "pending",
}


@dataclass(frozen=True)
class CheckResult:
    """One named check from `gh pr checks`, normalized onto a fixed status set."""

    name: str
    status: str  # "pass" | "fail" | "pending" | "skipped" | "cancelled"
    required: bool
    summary: str = ""


@dataclass(frozen=True)
class CIStatus:
    """CI status for one PR: every check `gh pr checks` reported."""

    checks: tuple[CheckResult, ...] = ()

    @property
    def state(self) -> str:
        """"passing" | "failing" | "pending" | "unknown".

        A single failing check outweighs everything else; a pending check
        outweighs an otherwise-clean run; no checks at all means nothing is
        known about CI, not that it passed.
        """
        if not self.checks:
            return "unknown"
        if any(c.status == "fail" for c in self.checks):
            return "failing"
        if any(c.status == "pending" for c in self.checks):
            return "pending"
        return "passing"

    @property
    def failing(self) -> tuple[CheckResult, ...]:
        return tuple(c for c in self.checks if c.status == "fail")


def fetch_ci_status(
    repo_slug: str,
    pr_number: int,
    runner: GhRunner = default_runner,
) -> CIStatus:
    """CI status for one PR, straight from `gh pr checks`.

    CI data is enrichment, not a gate: a `gh` failure, a timeout, or a payload
    that doesn't parse must never break a sweep, so any of those degrades to
    `CIStatus()` (state "unknown") rather than raising.
    """
    try:
        raw = runner(
            ["pr", "checks", str(pr_number), "--repo", repo_slug, "--json", CHECK_FIELDS],
            None,
        )
        return _parse_checks(raw)
    except Exception:  # noqa: BLE001 - CI data is enrichment; must never break a sweep
        return CIStatus()


def _parse_checks(raw: str) -> CIStatus:
    if not raw.strip():
        return CIStatus()
    payload = json.loads(raw)
    if not isinstance(payload, list):
        return CIStatus()
    parsed = (_parse_check(item) for item in payload if isinstance(item, dict))
    return CIStatus(checks=tuple(c for c in parsed if c is not None))


def _parse_check(item: dict) -> CheckResult | None:
    """One row of `gh pr checks --json` output, or None to skip a row with no
    usable name. Every other field degrades to a tolerant default rather than
    raising -- one odd row must not cost the rest of the table."""
    name = str(item.get("name", "")).strip()
    if not name:
        return None
    return CheckResult(
        name=name,
        status=_normalize_status(item),
        required=_parse_required(item),
        summary=str(item.get("description") or "").strip(),
    )


def _normalize_status(item: dict) -> str:
    """A check's gh `bucket`, falling back to the raw `state` field, falling
    back to "pending" when neither is recognized. Never raises: an
    unrecognized value degrades to the most conservative status rather than a
    guess at pass or fail.
    """
    bucket = str(item.get("bucket", "")).strip().lower()
    if bucket in _BUCKET_TO_STATUS:
        return _BUCKET_TO_STATUS[bucket]
    state = str(item.get("state", "")).strip().lower()
    return _STATE_TO_STATUS.get(state, "pending")


def _parse_required(item: dict) -> bool:
    """`gh pr checks --json` (gh 2.97) has no per-check "required" column --
    `--required` is a server-side output *filter*, not a field it emits.
    Read an explicit "required"/"isRequired" key if a payload happens to
    carry one (forward-compatible, and lets tests exercise the field);
    default False otherwise.
    """
    if "required" in item:
        return bool(item["required"])
    if "isRequired" in item:
        return bool(item["isRequired"])
    return False


def fetch_failure_excerpt(
    repo_slug: str,
    pr_number: int,
    runner: GhRunner = default_runner,
    max_lines: int = 40,
) -> str:
    """Best-effort tail of the failing run's log.

    Never load-bearing: no failing check, no usable run id, `gh` itself
    failing, empty logs -- any of those degrades to "". The failing-checks
    table `render` already produces is the load-bearing part of CI
    awareness; this is color on top of it.
    """
    try:
        run_id = _find_failing_run_id(repo_slug, pr_number, runner)
        if run_id is None:
            return ""
        raw = runner(["run", "view", run_id, "--repo", repo_slug, "--log-failed"], None)
    except Exception:  # noqa: BLE001 - best-effort color; must never raise into the sweep
        return ""
    return _tail_lines(raw, max_lines)


def _find_failing_run_id(repo_slug: str, pr_number: int, runner: GhRunner) -> str | None:
    """A failing run's id, preferring a failing check's own `link` (exact --
    no ambiguity about which run belongs to this PR) and falling back to the
    most recent failed run in the repo when no link resolves to one.

    Only attempted at all once this PR is confirmed to have a failing check
    -- otherwise the repo-wide fallback below could attribute some other PR's
    failure to this one.
    """
    raw = runner(
        ["pr", "checks", str(pr_number), "--repo", repo_slug, "--json", CHECK_FIELDS],
        None,
    )
    checks = json.loads(raw) if raw.strip() else []
    failing_items = [
        item
        for item in (checks if isinstance(checks, list) else ())
        if isinstance(item, dict) and _normalize_status(item) == "fail"
    ]
    if not failing_items:
        return None

    for item in failing_items:
        run_id = _run_id_from_link(str(item.get("link", "")))
        if run_id:
            return run_id

    # No failing check's link resolved to a run id (e.g. a non-Actions status
    # check). Best-effort fallback: the most recent failed run in the repo.
    # Not perfectly PR-scoped, but this whole function is explicitly
    # best-effort color, never the load-bearing part.
    raw = runner(
        [
            "run", "list",
            "--repo", repo_slug,
            "--status", "failure",
            "--limit", "1",
            "--json", RUN_LIST_FIELDS,
        ],
        None,
    )
    runs = json.loads(raw) if raw.strip() else []
    if isinstance(runs, list) and runs and isinstance(runs[0], dict):
        database_id = runs[0].get("databaseId")
        if database_id is not None:
            return str(database_id)
    return None


def _run_id_from_link(link: str) -> str | None:
    match = _RUN_ID_PATTERN.search(link)
    return match.group(1) if match else None


def _tail_lines(text: str, max_lines: int) -> str:
    """The last `max_lines` lines of `text` -- error tails matter more than
    headers. `max_lines <= 0` returns "" explicitly; `lines[-0:]` would
    otherwise silently return the whole thing.
    """
    if max_lines <= 0:
        return ""
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


def render(status: CIStatus, excerpt: str) -> str:
    """Markdown block for the review prompt: a state line, a table of failing
    checks, and a fenced failure-log excerpt when there is one to show.

    "" whenever `status.state` is "unknown" -- there is nothing trustworthy to
    say about CI, so saying nothing is the honest choice.
    """
    if status.state == "unknown":
        return ""

    lines = [f"## CI status: {status.state}", ""]

    if status.failing:
        lines.append("| Check | Summary |")
        lines.append("| --- | --- |")
        for check in status.failing:
            lines.append(f"| `{check.name}` | {check.summary or '(no summary provided)'} |")
        lines.append("")

    if excerpt.strip():
        lines.append("```")
        lines.append(excerpt.rstrip("\n"))
        lines.append("```")
        lines.append("")

    lines.append(
        "_Do not re-report what CI already reports; frame findings knowing "
        f"CI is {status.state}._"
    )
    return "\n".join(lines) + "\n"


def activity_note(status: CIStatus) -> str:
    """One-line sweep note: "ci: failing (2 checks)" / "ci: passing" /
    "ci: pending" / "" when unknown -- the same `notes` convention
    `sweep.py`/`sinks.write_local` already use for enrichment activity.
    """
    if status.state == "unknown":
        return ""
    if status.state == "failing":
        return f"ci: failing ({len(status.failing)} checks)"
    return f"ci: {status.state}"
