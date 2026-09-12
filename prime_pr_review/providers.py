"""Prime Inference provider: an OpenAI-compatible chat client with retries, a
pricing table read from the API, and a hard budget meter.

Every call reports its token usage into a `MeterBox` so the caller can stop a
long run the moment spend passes the cap. The meter itself is immutable; the
box is the one mutable cell, replaced (never mutated) after each call."""
from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import httpx

from .github import PullRequest
from .reviewers import build_prompt

BASE_URL = "https://api.pinference.ai/api/v1"
DEFAULT_CONFIG_PATH = Path.home() / ".prime" / "config.json"
REQUEST_TIMEOUT_SECONDS = 300
MAX_ATTEMPTS = 5
BACKOFF_BASE_SECONDS = 2.0
MAX_COMPLETION_TOKENS = 16_000
RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})

ModelFn = Callable[[str], str]
Reviewer = Callable[[PullRequest, str, str], str]
Pricing = Mapping[str, tuple[float, float]]  # model -> (usd per M input tok, usd per M output tok)


class ProviderError(RuntimeError):
    """The provider returned something unusable, or configuration is missing."""


class BudgetExceeded(RuntimeError):
    """Spend passed the cap; the caller must stop issuing calls."""


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class CostMeter:
    cap_usd: float
    pricing: Pricing
    spent_usd: float = 0.0
    tokens: Mapping[str, tuple[int, int]] = field(default_factory=dict)

    def record(self, model: str, usage: Usage) -> CostMeter:
        if model not in self.pricing:
            raise ProviderError(f"no pricing for model {model!r}")
        in_rate, out_rate = self.pricing[model]
        cost = usage.prompt_tokens * in_rate / 1e6 + usage.completion_tokens * out_rate / 1e6
        prev_in, prev_out = self.tokens.get(model, (0, 0))
        tokens = {**self.tokens, model: (prev_in + usage.prompt_tokens, prev_out + usage.completion_tokens)}
        return replace(self, spent_usd=self.spent_usd + cost, tokens=tokens)

    def check(self) -> None:
        if self.spent_usd > self.cap_usd:
            raise BudgetExceeded(f"spent ${self.spent_usd:.2f} > cap ${self.cap_usd:.2f}")

    def to_json(self) -> str:
        return json.dumps({"cap_usd": self.cap_usd, "spent_usd": self.spent_usd,
                           "tokens": {m: list(t) for m, t in self.tokens.items()}}, indent=1)

    @staticmethod
    def from_json(text: str, pricing: Pricing) -> CostMeter:
        raw = json.loads(text)
        return CostMeter(cap_usd=float(raw["cap_usd"]), pricing=pricing, spent_usd=float(raw["spent_usd"]),
                         tokens={m: (int(a), int(b)) for m, (a, b) in raw.get("tokens", {}).items()})


@dataclass
class MeterBox:
    """The single mutable cell: holds the current (immutable) meter."""
    meter: CostMeter


def resolve_prime_key(env: Mapping[str, str] | None = None, config_path: Path = DEFAULT_CONFIG_PATH) -> str:
    env = os.environ if env is None else env
    key = str(env.get("PRIME_API_KEY", "")).strip()
    if key:
        return key
    try:
        key = str(json.loads(config_path.read_text(encoding="utf-8")).get("api_key", "")).strip()
    except (OSError, json.JSONDecodeError):
        key = ""
    if not key:
        raise ProviderError(f"No Prime key: set PRIME_API_KEY or put api_key in {config_path}")
    return key


def make_client(key: str) -> httpx.Client:
    return httpx.Client(base_url=BASE_URL, headers={"Authorization": f"Bearer {key}"},
                        timeout=REQUEST_TIMEOUT_SECONDS)


def fetch_pricing(client: httpx.Client, models: Sequence[str]) -> dict[str, tuple[float, float]]:
    response = client.get("/models")
    if response.status_code != 200:
        raise ProviderError(f"GET /models -> {response.status_code}")
    listed = {m.get("id"): m.get("pricing") or {} for m in response.json().get("data", [])}
    pricing: dict[str, tuple[float, float]] = {}
    for model in models:
        entry = listed.get(model)
        if not entry or entry.get("input_usd_per_mtok") is None:
            raise ProviderError(f"model {model!r} not available or unpriced on Prime Inference")
        pricing[model] = (float(entry["input_usd_per_mtok"]), float(entry["output_usd_per_mtok"]))
    return pricing


def chat(client: httpx.Client, model: str, prompt: str,
         sleep: Callable[[float], None] = time.sleep,
         extra: Mapping[str, object] | None = None) -> tuple[str, Usage]:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
            "max_tokens": MAX_COMPLETION_TOKENS}
    if extra:
        body = {**body, **extra}
    last = "no attempts"
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = client.post("/chat/completions", json=body)
        except httpx.HTTPError as exc:
            last = repr(exc)
        else:
            if response.status_code == 200:
                return _extract(response.json())
            last = f"HTTP {response.status_code}: {response.text[:200]}"
            if response.status_code not in RETRYABLE_STATUS:
                break
        if attempt < MAX_ATTEMPTS - 1:
            sleep(BACKOFF_BASE_SECONDS * (2 ** attempt) + random.uniform(0, 1))
    raise ProviderError(f"{model}: {last}")


def _extract(payload: dict) -> tuple[str, Usage]:
    try:
        choice = payload["choices"][0]
        message = choice["message"]
        content = message["content"]
        if content is None or not str(content).strip():
            finish_reason = choice.get("finish_reason")
            reasoning_chars = len(message.get("reasoning") or "")
            raise ProviderError(
                f"empty content (finish_reason={finish_reason}, reasoning_chars={reasoning_chars})"
            )
        text = str(content)
        usage = payload.get("usage") or {}
        return text, Usage(int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)))
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(f"unexpected response shape: {payload!r}"[:300]) from exc


def prime_model_fn(client: httpx.Client, model: str, box: MeterBox,
                   extra: Mapping[str, object] | None = None) -> ModelFn:
    def model_fn(prompt: str) -> str:
        text, usage = chat(client, model, prompt, extra=extra)
        box.meter = box.meter.record(model, usage)
        return text
    return model_fn


def prime_reviewer(client: httpx.Client, model: str, box: MeterBox, prompts_dir: Path | str,
                   extra: Mapping[str, object] | None = None) -> Reviewer:
    model_fn = prime_model_fn(client, model, box, extra=extra)
    def reviewer(pr: PullRequest, payload: str, lane: str) -> str:
        template = (Path(prompts_dir) / f"{lane}_pr.md").read_text(encoding="utf-8")
        return model_fn(build_prompt(template, pr, payload))
    return reviewer
