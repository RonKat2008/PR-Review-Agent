from __future__ import annotations

import json

import httpx
import pytest

from prime_pr_review.providers import (
    BASE_URL,
    BudgetExceeded,
    CostMeter,
    MeterBox,
    ProviderError,
    Usage,
    chat,
    fetch_pricing,
    prime_model_fn,
    prime_reviewer,
    resolve_prime_key,
)

from .conftest import make_pr

PRICING = {"m/a": (2.0, 4.0)}


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), base_url=BASE_URL)


def _ok(text="hello", pt=100, ct=50):
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}],
                                     "usage": {"prompt_tokens": pt, "completion_tokens": ct}})


def test_cost_meter_accumulates_and_caps():
    m = CostMeter(cap_usd=0.001, pricing=PRICING).record("m/a", Usage(1_000_000, 0))
    assert m.spent_usd == pytest.approx(2.0)
    with pytest.raises(BudgetExceeded):
        m.check()


def test_cost_meter_round_trips_json():
    m = CostMeter(cap_usd=5.0, pricing=PRICING).record("m/a", Usage(10, 20))
    back = CostMeter.from_json(m.to_json(), PRICING)
    assert back.spent_usd == pytest.approx(m.spent_usd) and dict(back.tokens) == dict(m.tokens)


def test_cost_meter_unknown_model_raises():
    with pytest.raises(ProviderError):
        CostMeter(cap_usd=1, pricing=PRICING).record("nope", Usage(1, 1))


def test_fetch_pricing_reads_models_endpoint():
    def handler(req):
        assert req.url.path.endswith("/models")
        return httpx.Response(200, json={"data": [{"id": "m/a", "pricing": {"input_usd_per_mtok": 2.0, "output_usd_per_mtok": 4.0}}]})
    assert fetch_pricing(_client(handler), ["m/a"]) == PRICING


def test_fetch_pricing_missing_model_fails_fast():
    with pytest.raises(ProviderError, match="m/zzz"):
        fetch_pricing(_client(lambda r: httpx.Response(200, json={"data": []})), ["m/zzz"])


def test_chat_retries_429_then_succeeds():
    seen = []
    def handler(req):
        seen.append(json.loads(req.content))
        return httpx.Response(429) if len(seen) == 1 else _ok("done")
    text, usage = chat(_client(handler), "m/a", "prompt", sleep=lambda s: None)
    assert text == "done" and usage == Usage(100, 50)
    assert seen[0]["model"] == "m/a" and seen[0]["messages"][0]["content"] == "prompt"
    assert seen[0]["temperature"] == 0


def test_chat_gives_up_after_max_attempts():
    with pytest.raises(ProviderError):
        chat(_client(lambda r: httpx.Response(503)), "m/a", "p", sleep=lambda s: None)


def test_model_fn_records_usage_into_box():
    box = MeterBox(CostMeter(cap_usd=10, pricing=PRICING))
    fn = prime_model_fn(_client(lambda r: _ok("x", 1_000_000, 0)), "m/a", box)
    assert fn("p") == "x" and box.meter.spent_usd == pytest.approx(2.0)


def test_reviewer_builds_lane_prompt(tmp_path):
    (tmp_path / "open_pr.md").write_text("TEMPLATE", encoding="utf-8")
    seen = []
    def handler(req):
        seen.append(json.loads(req.content)["messages"][0]["content"]); return _ok("v")
    box = MeterBox(CostMeter(cap_usd=10, pricing=PRICING))
    reviewer = prime_reviewer(_client(handler), "m/a", box, tmp_path)
    assert reviewer(make_pr(), "DIFF", "open") == "v"
    assert seen[0].startswith("TEMPLATE") and "DIFF" in seen[0]


def test_resolve_key_prefers_env_then_config(tmp_path):
    cfg = tmp_path / "config.json"; cfg.write_text(json.dumps({"api_key": "from-file"}))
    assert resolve_prime_key({"PRIME_API_KEY": "from-env"}, cfg) == "from-env"
    assert resolve_prime_key({}, cfg) == "from-file"
    with pytest.raises(ProviderError):
        resolve_prime_key({}, tmp_path / "missing.json")
