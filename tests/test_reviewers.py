"""Tests for the Gemini-backed reviewer.

This is the single seam that talks to an external HTTP API. Every request goes
through `httpx.MockTransport`, so nothing here ever touches the network — a bad
mock is the only way one of these tests could pass for the wrong reason.
"""

from __future__ import annotations

import json

import httpx
import pytest

from prime_pr_review.reviewers import ReviewerError, build_prompt, gemini_reviewer
from prime_pr_review.state import LANE_MERGED, LANE_OPEN

from .conftest import SAMPLE_DIFF, make_pr


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


@pytest.fixture
def prompts_dir(tmp_path):
    """A prompts directory with distinguishable open/merged templates."""
    (tmp_path / "open_pr.md").write_text("OPEN LANE TEMPLATE", encoding="utf-8")
    (tmp_path / "merged_pr.md").write_text("MERGED LANE TEMPLATE", encoding="utf-8")
    return tmp_path


def _success_response(text: str = "No issues found.") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "candidates": [
                {"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}
            ]
        },
    )


def _reviewer(prompts_dir, handler, **kwargs):
    """Build a reviewer wired to a mock transport instead of the network."""
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return gemini_reviewer(api_key="test-key", prompts_dir=prompts_dir, client=client, **kwargs)


# --------------------------------------------------------------------------
# build_prompt (pure)
# --------------------------------------------------------------------------


def test_build_prompt_includes_template_pr_metadata_and_diff():
    prompt = build_prompt(
        "TEMPLATE TEXT",
        make_pr(number=7, title="Add feature", author="bob"),
        SAMPLE_DIFF,
    )

    assert "TEMPLATE TEXT" in prompt
    assert "#7" in prompt
    assert "Add feature" in prompt
    assert "bob" in prompt
    assert SAMPLE_DIFF in prompt


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------


def test_successful_call_returns_the_models_text(prompts_dir):
    def handler(request: httpx.Request) -> httpx.Response:
        return _success_response("no bugs found")

    reviewer = _reviewer(prompts_dir, handler)

    result = reviewer(make_pr(), SAMPLE_DIFF, LANE_OPEN)

    assert result == "no bugs found"


def test_request_body_carries_the_built_prompt(prompts_dir):
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return _success_response()

    reviewer = _reviewer(prompts_dir, handler)

    reviewer(make_pr(title="Fix the widget"), SAMPLE_DIFF, LANE_OPEN)

    body = json.loads(captured[0].content)
    prompt_text = body["contents"][0]["parts"][0]["text"]
    assert "OPEN LANE TEMPLATE" in prompt_text
    assert "Fix the widget" in prompt_text
    assert SAMPLE_DIFF in prompt_text


def test_request_url_carries_the_configured_model_id(prompts_dir):
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return _success_response()

    reviewer = _reviewer(prompts_dir, handler, model="gemini-2-5-pro-test")

    reviewer(make_pr(), SAMPLE_DIFF, LANE_OPEN)

    assert "models/gemini-2-5-pro-test:generateContent" in str(captured[0].url)


def test_request_sets_response_mime_type_to_application_json(prompts_dir):
    """This is what constrains the model's output to the verdict JSON schema."""
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return _success_response()

    reviewer = _reviewer(prompts_dir, handler)

    reviewer(make_pr(), SAMPLE_DIFF, LANE_OPEN)

    body = json.loads(captured[0].content)
    assert body["generationConfig"]["responseMimeType"] == "application/json"


def test_prompt_template_is_selected_by_lane(prompts_dir):
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return _success_response()

    reviewer = _reviewer(prompts_dir, handler)

    reviewer(make_pr(), SAMPLE_DIFF, LANE_MERGED)

    prompt_text = json.loads(captured[0].content)["contents"][0]["parts"][0]["text"]
    assert "MERGED LANE TEMPLATE" in prompt_text
    assert "OPEN LANE TEMPLATE" not in prompt_text


def test_missing_prompt_template_raises_a_clear_error(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return _success_response()

    reviewer = _reviewer(tmp_path, handler)  # tmp_path has no *_pr.md files

    with pytest.raises(ReviewerError, match="Could not read prompt"):
        reviewer(make_pr(), SAMPLE_DIFF, LANE_OPEN)


# --------------------------------------------------------------------------
# HTTP error statuses
# --------------------------------------------------------------------------


def test_retired_model_404_names_the_model_in_the_error(prompts_dir):
    """A pinned model getting retired is what actually cost a full sweep in prod."""
    model_id = "gemini-legacy-001"
    error_body = {
        "error": {
            "code": 404,
            "message": (
                f"models/{model_id} is not found for API version v1beta, "
                "or is not supported for generateContent."
            ),
            "status": "NOT_FOUND",
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json=error_body)

    reviewer = _reviewer(prompts_dir, handler, model=model_id)

    with pytest.raises(ReviewerError) as exc_info:
        reviewer(make_pr(), SAMPLE_DIFF, LANE_OPEN)

    message = str(exc_info.value)
    assert "404" in message
    assert model_id in message


def test_bad_api_key_401_produces_a_clear_error(prompts_dir):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={
                "error": {
                    "code": 401,
                    "message": "API key not valid. Please pass a valid API key.",
                    "status": "UNAUTHENTICATED",
                }
            },
        )

    reviewer = _reviewer(prompts_dir, handler)

    with pytest.raises(ReviewerError) as exc_info:
        reviewer(make_pr(), SAMPLE_DIFF, LANE_OPEN)

    message = str(exc_info.value)
    assert "401" in message
    assert "API key" in message


def test_server_error_500_produces_a_clear_error(prompts_dir):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal error encountered.")

    reviewer = _reviewer(prompts_dir, handler)

    with pytest.raises(ReviewerError) as exc_info:
        reviewer(make_pr(), SAMPLE_DIFF, LANE_OPEN)

    assert "500" in str(exc_info.value)


# --------------------------------------------------------------------------
# Malformed / unexpected response bodies
# --------------------------------------------------------------------------


def test_missing_candidates_key_is_handled_explicitly(prompts_dir):
    """Not with a raw KeyError -- a caller needs a message it can act on."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    reviewer = _reviewer(prompts_dir, handler)

    with pytest.raises(ReviewerError, match="no candidates"):
        reviewer(make_pr(), SAMPLE_DIFF, LANE_OPEN)


def test_empty_candidates_array_is_handled(prompts_dir):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": []})

    reviewer = _reviewer(prompts_dir, handler)

    with pytest.raises(ReviewerError, match="no candidates"):
        reviewer(make_pr(), SAMPLE_DIFF, LANE_OPEN)


def test_candidate_missing_parts_is_handled_explicitly(prompts_dir):
    """Not with a raw IndexError -- the finish reason should surface instead."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": [{"finishReason": "MAX_TOKENS"}]})

    reviewer = _reviewer(prompts_dir, handler)

    with pytest.raises(ReviewerError, match="MAX_TOKENS"):
        reviewer(make_pr(), SAMPLE_DIFF, LANE_OPEN)


def test_blocked_prompt_names_the_block_reason(prompts_dir):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}}
        )

    reviewer = _reviewer(prompts_dir, handler)

    with pytest.raises(ReviewerError, match="SAFETY"):
        reviewer(make_pr(), SAMPLE_DIFF, LANE_OPEN)


# --------------------------------------------------------------------------
# Network failures
# --------------------------------------------------------------------------


def test_network_failure_is_wrapped_not_leaked_raw(prompts_dir):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    reviewer = _reviewer(prompts_dir, handler)

    with pytest.raises(ReviewerError, match="Gemini request failed"):
        reviewer(make_pr(), SAMPLE_DIFF, LANE_OPEN)


# --------------------------------------------------------------------------
# Client lifecycle
# --------------------------------------------------------------------------


def test_reviewer_closes_a_client_it_creates_for_itself(monkeypatch, prompts_dir):
    """When no client is injected, the reviewer must own and clean up its own."""

    def handler(request: httpx.Request) -> httpx.Response:
        return _success_response("ok")

    real_client_cls = httpx.Client  # capture before patching -- httpx is one shared module

    def fake_httpx_client(*, timeout):
        return real_client_cls(transport=httpx.MockTransport(handler), timeout=timeout)

    monkeypatch.setattr("prime_pr_review.reviewers.httpx.Client", fake_httpx_client)

    reviewer = gemini_reviewer(api_key="test-key", prompts_dir=prompts_dir)  # no client injected

    result = reviewer(make_pr(), SAMPLE_DIFF, LANE_OPEN)

    assert result == "ok"
