"""Adversarial refutation (P14): a skeptic model tries to disprove each finding.

Recall mode (`min_agreement = 1`) keeps every finding any ensemble seat
reports, because requiring finders to agree measurably dropped real bugs — the
seats find DISJOINT defects, so their intersection is nearly empty. That trade
buys recall by giving up the ensemble's only precision mechanism. This pass is
the replacement: precision comes from a dedicated adversary, not from finders
happening to overlap.

Each finding gets one skeptic call — the finding plus the diff it was made
about — with a single job: refute it with concrete grounds (the flagged path
is guarded, the behavior is pre-existing, the line was misread, the code is
test-only). Three rules keep the skeptic honest:

  - **Refute only with evidence.** Uncertainty upholds the finding. A skeptic
    that defaulted to "refuted" on doubt would re-drop exactly the single-seat
    findings recall mode exists to keep.
  - **Annotate, never drop.** A refuted finding stays in the verdict, marked
    `refuted` with the skeptic's reasoning — the local review shows the
    challenge and the reader adjudicates. Downstream, refuted findings are
    excluded from inline PR comments, blocking status, and suggestion blocks.
  - **Fail open.** A skeptic that crashes, times out, or refutes without
    reasoning costs a note, not a finding. A broken adversary must never
    silently suppress a real bug.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

from .review import Finding, VerdictError, _extract_json

ModelFn = Callable[[str], str]

PROMPT_FILE = "refute.md"


class RefuteError(RuntimeError):
    """The skeptic returned something that is not a usable refutation verdict."""


def refute_findings(
    findings: Sequence[Finding],
    diff: str,
    model_fn: ModelFn,
    prompts_dir: Path | str,
) -> tuple[tuple[Finding, ...], tuple[str, ...]]:
    """Run the skeptic against every finding. Returns the findings (refuted ones
    annotated in place, order preserved) and the activity notes.

    The count note always appears when any finding was examined — "ran and
    challenged nothing" and "never ran" must not look identical in the review's
    front matter.
    """
    if not findings:
        return (), ()

    template = _read_prompt(Path(prompts_dir))
    kept: list[Finding] = []
    notes: list[str] = []
    challenged = 0

    for finding in findings:
        try:
            response = model_fn(build_refute_prompt(template, finding, diff))
            verdict = _parse_refutation(response)
        except Exception as exc:  # noqa: BLE001 - fail open: a broken skeptic costs a note
            notes.append(f"skeptic failed for {finding.file}: {exc}")
            kept.append(finding)
            continue

        if verdict is None:
            kept.append(finding)
        else:
            challenged += 1
            kept.append(replace(finding, refuted=True, refutation=verdict))

    notes.append(f"skeptic: {challenged}/{len(findings)} finding(s) challenged")
    return tuple(kept), tuple(notes)


def build_refute_prompt(template: str, finding: Finding, diff: str) -> str:
    location = f"{finding.file}:{finding.line}" if finding.line else finding.file
    lines = [
        template,
        "",
        "## The finding under challenge",
        "",
        f"- location: `{location}`",
        f"- severity: {finding.severity.value}",
        f"- claim: {finding.claim}",
    ]
    if finding.evidence:
        lines.append(f"- evidence: {finding.evidence}")
    if finding.has_suggestion:
        lines.append(f"- proposed fix:\n\n```\n{finding.suggestion}\n```")
    lines += ["", "## The diff the finding is about", "", f"```diff\n{diff}\n```"]
    return "\n".join(lines)


def _read_prompt(directory: Path) -> str:
    path = directory / PROMPT_FILE
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RefuteError(f"Could not read prompt {path}: {exc}") from exc


def _parse_refutation(raw: str) -> str | None:
    """The skeptic's verdict: the refutation reasoning, or None when upheld.

    A refutation without reasoning raises — the caller's fail-open handling
    turns that into "finding kept, note recorded". A skeptic that cannot say
    WHY does not get to kill a finding.
    """
    try:
        payload = _extract_json(raw)
    except VerdictError as exc:
        raise RefuteError(f"Skeptic response is not usable: {exc}") from exc

    if not bool(payload.get("refuted", False)):
        return None

    reasoning = str(payload.get("reasoning", "")).strip()
    if not reasoning:
        raise RefuteError("Skeptic refuted without reasoning — refutation ignored")
    return reasoning
