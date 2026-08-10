"""Intent alignment (P8): does the diff do what the PR claims it does?

Nothing else in this package checks the diff against the PR's stated purpose. A
PR titled "fix typo" could change an auth check and nothing here would ask why.
This module answers that question in two independent passes.

DESIGN CONSTRAINT — do not collapse this into one model call:

    Pass 1 sees only what the PR *says* about itself — title, description, commit
    messages, branch name — and commits, in writing, to an interpretation of intent.
    It is never shown the diff.

    Pass 2 sees that written intent plus the diff, and must argue against it to
    find anything unrelated.

    If a single call saw the title and the diff together and were asked "does this
    diff match the title?", it would read the diff first and construct a story in
    which the title covers whatever it sees — models are extremely good at
    post-hoc rationalization, and a title is loose enough to cover almost
    anything in hindsight. Forcing pass 1 to commit to an intent statement *before*
    any code is visible means pass 2 has to argue against words the model already
    wrote, rather than freely reconciling title and diff at the same time. That
    ordering is the entire difference between this check working and being
    theater. If you are tempted to merge the two passes into one prompt for
    efficiency, don't — you would be deleting the one thing that makes it work.

The pass-2 output contract (`aligned` / `unrelated`) is owned by `review.py`'s
`Scope` / `ScopeIssue` dataclasses and `_parse_scope` parser — this module reuses
them rather than redefining the shape.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .review import Scope, VerdictError, _extract_json, _parse_scope

ModelFn = Callable[[str], str]

PASS1_PROMPT_FILE = "intent_pass1.md"
PASS2_PROMPT_FILE = "intent_pass2.md"


class IntentError(RuntimeError):
    """Pass 1 returned something that is not a usable intent statement."""


@dataclass(frozen=True)
class IntentStatement:
    """Pass 1's committed interpretation of the PR, written before any diff is seen."""

    intent: str
    expected_files: tuple[str, ...] = ()
    out_of_scope: tuple[str, ...] = ()


def run_intent_check(
    pr: object,
    diff: str,
    model_fn: ModelFn,
    prompts_dir: Path | str,
) -> Scope:
    """Run both passes, in order, and return the resulting Scope.

    Pass 1 is given `pr` only — never `diff`. See the module docstring for why
    that ordering is load-bearing rather than incidental.
    """
    directory = Path(prompts_dir)
    statement = _run_pass1(pr, model_fn, directory)
    return _run_pass2(statement, diff, model_fn, directory)


def _run_pass1(pr: object, model_fn: ModelFn, directory: Path) -> IntentStatement:
    template = _read_prompt(directory, PASS1_PROMPT_FILE)
    prompt = build_pass1_prompt(template, pr)
    return parse_intent(model_fn(prompt))


def _run_pass2(
    statement: IntentStatement, diff: str, model_fn: ModelFn, directory: Path
) -> Scope:
    template = _read_prompt(directory, PASS2_PROMPT_FILE)
    prompt = build_pass2_prompt(template, statement, diff)
    return parse_scope_response(model_fn(prompt), statement.intent)


def _read_prompt(directory: Path, filename: str) -> str:
    path = directory / filename
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise IntentError(f"Could not read prompt {path}: {exc}") from exc


def build_pass1_prompt(template: str, pr: object) -> str:
    """Assemble the pass-1 prompt from PR metadata only.

    Deliberately excludes `diff` from its signature: there is nothing to pass
    even if a caller wanted to.
    """
    commits = _pr_commit_messages(pr)
    commit_block = "\n".join(f"- {m}" for m in commits) or "(no commit messages provided)"
    return (
        f"{template}\n\n"
        f"## Pull request #{getattr(pr, 'number', '?')}: {_pr_text(pr, 'title')}\n\n"
        f"- author: {_pr_text(pr, 'author')}\n"
        f"- branch: {_pr_text(pr, 'branch_name') or '(unknown)'}\n\n"
        f"## Description\n\n{_pr_text(pr, 'body') or '(no description provided)'}\n\n"
        f"## Commit messages\n\n{commit_block}\n"
    )


def build_pass2_prompt(template: str, statement: IntentStatement, diff: str) -> str:
    """Assemble the pass-2 prompt: the committed intent (pass 1) plus the diff."""
    expected = ", ".join(statement.expected_files) or "(none specified)"
    out_of_scope = ", ".join(statement.out_of_scope) or "(none specified)"
    return (
        f"{template}\n\n"
        f"## Stated intent (written before this diff was shown)\n\n{statement.intent}\n\n"
        f"- expected files: {expected}\n"
        f"- explicitly out of scope: {out_of_scope}\n\n"
        f"## Diff\n\n```diff\n{diff}\n```\n"
    )


def parse_intent(raw: str) -> IntentStatement:
    """Parse pass 1's response. Malformed input fails loudly — never a guessed intent."""
    try:
        payload = _extract_json(raw)
    except VerdictError as exc:
        raise IntentError(f"Pass 1 response is not usable: {exc}") from exc

    intent = str(payload.get("intent", "")).strip()
    if not intent:
        raise IntentError("Pass 1 response is missing a non-empty 'intent'")

    return IntentStatement(
        intent=intent,
        expected_files=_string_tuple(payload.get("expected_files")),
        out_of_scope=_string_tuple(payload.get("out_of_scope")),
    )


def parse_scope_response(raw: str, intent: str) -> Scope:
    """Parse pass 2's response via `review._parse_scope`, the one place that contract lives.

    Pass 2's own JSON never carries `intent` — the contract is `aligned` /
    `unrelated` only. The pass-1 intent is threaded in here so the resulting
    `Scope.intent` reflects what was decided before the diff was seen, not
    anything pass 2 might echo back.
    """
    payload = _extract_json(raw)
    scope = _parse_scope({**payload, "intent": intent})
    if scope is None:
        raise VerdictError("Pass 2 response must be a JSON object")
    return scope


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _pr_text(pr: object, name: str) -> str:
    return str(getattr(pr, name, "") or "").strip()


def _pr_commit_messages(pr: object) -> tuple[str, ...]:
    messages = getattr(pr, "commit_messages", ()) or ()
    return tuple(str(m).strip() for m in messages if str(m).strip())
