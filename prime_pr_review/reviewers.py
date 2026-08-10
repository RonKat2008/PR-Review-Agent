"""Reviewer implementations.

A reviewer is the single injected seam where a model is called:
`(PullRequest, diff, lane) -> raw verdict JSON`. Everything else in this package
is deterministic. Swapping providers means swapping one function.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx

from .github import PullRequest

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
# Alias that tracks the current Flash model. Pinned versions get retired for new
# users, so the alias is the durable choice for an agent that runs unattended.
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"
DEFAULT_PROMPTS_DIR = Path("skills/pr-review/prompts")
REQUEST_TIMEOUT_SECONDS = 180


class ReviewerError(RuntimeError):
    """The model could not be reached or returned an unusable response."""


def build_prompt(template: str, pr: PullRequest, diff: str) -> str:
    """Assemble the full reviewer prompt for one pull request."""
    return (
        f"{template}\n\n"
        f"## Pull request #{pr.number}: {pr.title}\n\n"
        f"- author: {pr.author}\n"
        f"- base branch: {pr.base_ref}\n"
        f"- files changed: {pr.changed_files}\n"
        f"- lines: +{pr.additions} / -{pr.deletions}\n\n"
        f"## Diff\n\n```diff\n{diff}\n```\n"
    )


def gemini_reviewer(
    api_key: str,
    model: str = DEFAULT_GEMINI_MODEL,
    prompts_dir: Path | str = DEFAULT_PROMPTS_DIR,
    client: httpx.Client | None = None,
):
    """Build a reviewer backed by the Gemini API.

    `responseMimeType: application/json` constrains the model to emit a bare JSON
    object, which is exactly the verdict shape the pipeline expects.
    """
    directory = Path(prompts_dir)

    def reviewer(pr: PullRequest, diff: str, lane: str) -> str:
        template_path = directory / f"{lane}_pr.md"
        try:
            template = template_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ReviewerError(f"Could not read prompt {template_path}: {exc}") from exc

        return post_gemini(api_key, model, build_prompt(template, pr, diff), client)

    return reviewer


def gemini_model_fn(
    api_key: str,
    model: str = DEFAULT_GEMINI_MODEL,
    client: httpx.Client | None = None,
) -> Callable[[str], str]:
    """A raw prompt-to-text callable.

    Used by analysis passes that are not the review itself — the intent check and
    blast-radius judgement build their own prompts and do not want the reviewer's
    lane template wrapped around them.
    """

    def call(prompt: str) -> str:
        return post_gemini(api_key, model, prompt, client)

    return call


def post_gemini(
    api_key: str,
    model: str,
    prompt: str,
    client: httpx.Client | None = None,
) -> str:
    """One Gemini call. `responseMimeType: application/json` constrains the model to
    emit a bare JSON object, which is the shape every consumer here expects."""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
        },
    }

    owns_client = client is None
    http = client or httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS)
    try:
        response = http.post(
            GEMINI_ENDPOINT.format(model=model),
            params={"key": api_key},
            json=payload,
        )
        if response.status_code >= 400:
            # Name the model explicitly rather than trusting the upstream body to
            # mention it. A retired model id already cost one full sweep; an opaque
            # gateway error would otherwise leave no clue which model was wrong.
            raise ReviewerError(
                f"Gemini returned {response.status_code} for model {model!r}: "
                f"{response.text[:300]}"
            )
        return _extract_text(response.json())
    except httpx.HTTPError as exc:
        raise ReviewerError(f"Gemini request failed: {exc}") from exc
    finally:
        if owns_client:
            http.close()


def _extract_text(body: dict) -> str:
    """Pull the generated text out of a Gemini response."""
    candidates = body.get("candidates") or []
    if not candidates:
        blocked = (body.get("promptFeedback") or {}).get("blockReason")
        if blocked:
            raise ReviewerError(f"Gemini blocked the prompt: {blocked}")
        raise ReviewerError("Gemini returned no candidates")

    parts = ((candidates[0].get("content") or {}).get("parts")) or []
    text = "".join(part.get("text", "") for part in parts)
    if not text.strip():
        finish = candidates[0].get("finishReason", "unknown")
        raise ReviewerError(f"Gemini returned empty text (finishReason={finish})")
    return text
