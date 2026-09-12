# SWE-CARE Ablation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the real review pipeline once per SWE-CARE PR on the Prime Inference lineup, record every model call, and score four ablation arms (single seat, ensemble, ensemble+judge, full with skeptic) with citation validation on and off, against human review comments.

**Architecture:** A corpus-backed fake `gh` runner feeds `sweep_lane` unchanged. A recording reviewer and model functions persist every prompt/response per PR. Offline, the recorded seats/judge/skeptic outputs are replayed through the existing `ensemble_review_detailed`, `refute_findings`, and `validate_citations` functions to rebuild each arm at zero model cost, then scored against reference comment locations.

**Tech Stack:** Python 3.13, httpx (already a dependency), pytest with `httpx.MockTransport`, the Hugging Face datasets-server REST API, Prime Inference OpenAI-compatible chat endpoint.

**Spec:** `docs/superpowers/specs/2026-09-11-swecare-ablation-harness-design.md`

## Global Constraints

- Run tests with `.venv/bin/python -m pytest -q`; coverage gate is 80% (`pyproject.toml`), target 90% on new modules.
- Lint every touched file with `.venv/bin/ruff check <files>`; it must pass.
- Style: frozen dataclasses, pure functions, `dataclasses.replace`, no mutation of inputs, functions under 50 lines, files under 400 lines, module docstring stating intent.
- No network in tests. Every external call goes through an injectable callable.
- Never post to GitHub: eval config always sets `dry_run=True`, `sinks.pr_comment=False`, `repo.read_only=True`.
- Secrets: the Prime key comes from `PRIME_API_KEY` or `~/.prime/config.json`; never log or write it.
- Default budget cap `--cap-usd 40`; the meter is checked after every PR.
- Model lineup (verbatim from spec): seats `deepseek/deepseek-v4-pro`, `qwen/qwen3.8-max`, `z-ai/glm-5.2`; intent/blast `deepseek/deepseek-v4-flash`; skeptic and judge `deepseek/deepseek-v4-pro`.
- A pre-tool hook ("Fact-Forcing Gate") may block the first Bash/Write/Edit call and ask for facts; state them in text and retry the same call.
- Commit after each task with a conventional message; do not push.

### Existing APIs the plan uses (verified signatures)

```python
from prime_pr_review.review import Finding, Verdict, Severity, parse_verdict, VerdictError
# Finding(file, line, severity, claim, evidence, suggestion="", line_end=None, refuted=False, refutation="")
# Verdict(introduces: tuple[Finding,...], fixes: tuple[FixClaim,...], confidence: float, ...)
from prime_pr_review.github import PullRequest, GhRunner, GitHubError
# PullRequest(number, title, author, head_sha, base_ref, url, additions, deletions, changed_files, merged_at)
# GhRunner = Callable[[Sequence[str], str | None], str]
from prime_pr_review.reviewers import build_prompt          # (template, pr, diff) -> str
from prime_pr_review.ensemble import ensemble_review_detailed
# (pr, payload, lane, reviewer, size=3, min_agreement=2, judge_fn=None, prompts_dir=...) -> (Verdict, notes)
from prime_pr_review.refute import refute_findings           # (findings, diff, model_fn, prompts_dir) -> (findings, notes)
from prime_pr_review.citations import validate_citations, head_line_counts, paths_needing_head_counts
# validate_citations(findings, diff, head_counts: Mapping[str,int], repo_root: str|None) -> (findings, notes)
# head_line_counts(repo_slug, head_sha, paths, runner) -> dict[str,int]
from prime_pr_review.sweep import sweep_lane, Enrichment, SweepReport
# sweep_lane(config, lane, reviewer, state, runner=..., reviews_dir=..., enrichment=...) -> (SweepReport, State)
# Enrichment(model_fn=None, prompts_dir=..., skeptic_fn=None, judge_fn=None, ...)
from prime_pr_review.state import State, LANE_OPEN            # State.empty()
from prime_pr_review.config import load_config, Config       # dataclasses.replace on config.review / config.sinks / config.repo
# tests/conftest.py: make_config(owner=..., name=..., ...), make_pr(...)
```

Note on intent: `intent.run_intent_check` reads `pr.body` and `pr.commit_messages` via `getattr` with empty defaults, and `PullRequest` has neither, so in the headless path intent sees the title only. The eval inherits that; the report states it. The runner therefore does not need to answer `pr view`.

---

### Task 1: Corpus loader and fixture

**Files:**
- Create: `prime_pr_review/evaluation/__init__.py` (one-line docstring)
- Create: `prime_pr_review/evaluation/corpus.py`
- Create: `tests/fixtures/swecare_rows.json`
- Test: `tests/test_eval_corpus.py`

**Interfaces:**
- Produces: `Row`, `ReferenceComment`, `parse_rows(payload: dict) -> tuple[Row, ...]`, `select(rows, count, seed, max_patch_bytes=60_000) -> tuple[Row, ...]`, `fetch_rows(cache: Path, fetch_page: Callable[[int, int], dict], split="test") -> tuple[Row, ...]`.

- [ ] **Step 1: Build the fixture from the design-time sample**

```bash
.venv/bin/python - <<'EOF'
import json, pathlib
src = pathlib.Path("/private/tmp/claude-501/-Users-user-Projects-pr-review-agent/804c9a92-bd6c-4bb4-8fd0-facd2e35b596/scratchpad/swe_care_first_rows.json")
rows = json.loads(src.read_text())["test"]["rows"][:5]
for r in rows:
    row = r["row"]
    row["commit_to_review"]["patch_to_review"] = row["commit_to_review"]["patch_to_review"][:4000]
    row["merged_patch"] = ""
    row["body"] = (row.get("body") or "")[:500]
out = pathlib.Path("tests/fixtures/swecare_rows.json"); out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps({"rows": rows}, indent=1))
print(len(rows), "rows,", out.stat().st_size, "bytes")
EOF
```
Expected: `5 rows, <under 60000> bytes`. If the scratchpad file is gone, fetch one page: `GET https://datasets-server.huggingface.co/rows?dataset=inclusionAI/SWE-CARE&config=default&split=test&offset=0&length=5` and save its JSON as `{"rows": [...]}` after the same trimming.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_eval_corpus.py
from __future__ import annotations
import json
from pathlib import Path
import pytest
from prime_pr_review.evaluation.corpus import Row, ReferenceComment, parse_rows, select, fetch_rows

FIXTURE = Path(__file__).parent / "fixtures" / "swecare_rows.json"

@pytest.fixture
def rows() -> tuple[Row, ...]:
    return parse_rows(json.loads(FIXTURE.read_text(encoding="utf-8")))

def test_parse_rows_maps_every_field(rows):
    row = rows[0]
    assert row.instance_id and "/" in row.repo and row.pull_number > 0
    assert row.head_sha and row.patch.startswith("diff --git")
    assert isinstance(row.reference_comments, tuple)

def test_reference_comment_line_falls_back_to_original_line():
    payload = {"rows": [{"row": _row(comments=[{"path": "a.py", "line": None, "original_line": 7,
                                                 "start_line": None, "original_start_line": 5, "text": "x",
                                                 "diff_hunk": ""}])}]}
    (row,) = parse_rows(payload)
    assert row.reference_comments == (ReferenceComment(path="a.py", line=7, start_line=5, text="x"),)

def test_select_is_deterministic_and_filters(rows):
    a = select(rows, count=3, seed=0)
    b = select(rows, count=3, seed=0)
    assert a == b and len(a) == 3
    assert select(rows, count=3, seed=0, max_patch_bytes=10) == ()

def test_select_requires_a_comment_with_a_path():
    payload = {"rows": [{"row": _row(comments=[{"path": "", "line": 1, "original_line": 1,
                                                 "start_line": None, "original_start_line": None,
                                                 "text": "", "diff_hunk": ""}])}]}
    assert select(parse_rows(payload), count=1, seed=0) == ()

def test_fetch_rows_pages_then_caches(tmp_path):
    calls: list[tuple[int, int]] = []
    page = json.loads(FIXTURE.read_text(encoding="utf-8"))["rows"]
    def fetch_page(offset: int, length: int) -> dict:
        calls.append((offset, length))
        return {"rows": page if offset == 0 else [], "num_rows_total": len(page)}
    cache = tmp_path / "c.json"
    first = fetch_rows(cache, fetch_page)
    second = fetch_rows(cache, fetch_page)
    assert len(first) == len(page) and first == second
    assert calls == [(0, 100)]

def _row(*, comments):
    return {"instance_id": "o__r-1@abc", "repo": "o/r", "language": "Python", "pull_number": 1,
            "title": "t", "body": "b", "base_commit": "base",
            "commit_to_review": {"head_commit": "abc", "head_commit_message": "m",
                                 "patch_to_review": "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n"},
            "reference_review_comments": comments}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_eval_corpus.py -q --no-cov`
Expected: FAIL with `ModuleNotFoundError: prime_pr_review.evaluation`

- [ ] **Step 4: Implement corpus.py**

```python
"""SWE-CARE corpus: load rows from the Hugging Face datasets-server, cache them,
filter to what the harness can score, and sample deterministically.

`parse_rows` keeps everything; only `select` filters, so the CLI can report how
many rows each filter removed."""
from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

PAGE_LENGTH = 100
DEFAULT_MAX_PATCH_BYTES = 60_000
FetchPage = Callable[[int, int], dict]  # (offset, length) -> datasets-server JSON


@dataclass(frozen=True)
class ReferenceComment:
    path: str
    line: int | None
    start_line: int | None
    text: str


@dataclass(frozen=True)
class Row:
    instance_id: str
    repo: str
    language: str
    pull_number: int
    title: str
    body: str
    base_commit: str
    head_sha: str
    head_commit_message: str
    patch: str
    reference_comments: tuple[ReferenceComment, ...]

    @property
    def scorable(self) -> bool:
        return any(c.path for c in self.reference_comments)


def parse_rows(payload: dict) -> tuple[Row, ...]:
    return tuple(_parse_row(item["row"]) for item in payload.get("rows", ()))


def select(
    rows: tuple[Row, ...], count: int, seed: int, max_patch_bytes: int = DEFAULT_MAX_PATCH_BYTES
) -> tuple[Row, ...]:
    eligible = [
        r for r in rows
        if r.language == "Python" and len(r.patch.encode("utf-8")) <= max_patch_bytes and r.scorable
    ]
    if len(eligible) <= count:
        return tuple(eligible)
    return tuple(random.Random(seed).sample(eligible, count))


def fetch_rows(cache: Path, fetch_page: FetchPage, split: str = "test") -> tuple[Row, ...]:
    if cache.is_file():
        return parse_rows(json.loads(cache.read_text(encoding="utf-8")))
    items: list[dict] = []
    offset = 0
    while True:
        page = fetch_page(offset, PAGE_LENGTH)
        batch = page.get("rows", [])
        items.extend(batch)
        offset += len(batch)
        if not batch or offset >= int(page.get("num_rows_total", offset)):
            break
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"split": split, "rows": items}), encoding="utf-8")
    return parse_rows({"rows": items})


def _parse_row(raw: dict) -> Row:
    review = raw.get("commit_to_review") or {}
    return Row(
        instance_id=str(raw["instance_id"]),
        repo=str(raw["repo"]),
        language=str(raw.get("language", "")),
        pull_number=int(raw["pull_number"]),
        title=str(raw.get("title") or ""),
        body=str(raw.get("body") or ""),
        base_commit=str(raw.get("base_commit") or ""),
        head_sha=str(review.get("head_commit") or ""),
        head_commit_message=str(review.get("head_commit_message") or ""),
        patch=str(review.get("patch_to_review") or ""),
        reference_comments=tuple(_parse_comment(c) for c in raw.get("reference_review_comments") or ()),
    )


def _parse_comment(raw: dict) -> ReferenceComment:
    line = raw.get("line") if raw.get("line") is not None else raw.get("original_line")
    start = raw.get("start_line") if raw.get("start_line") is not None else raw.get("original_start_line")
    return ReferenceComment(
        path=str(raw.get("path") or ""),
        line=int(line) if line is not None else None,
        start_line=int(start) if start is not None else None,
        text=str(raw.get("text") or ""),
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_eval_corpus.py -q --no-cov && .venv/bin/ruff check prime_pr_review/evaluation tests/test_eval_corpus.py`
Expected: all PASS, ruff clean.

- [ ] **Step 6: Commit**

```bash
git add prime_pr_review/evaluation tests/fixtures/swecare_rows.json tests/test_eval_corpus.py
git commit -m "feat(eval): SWE-CARE corpus loader with deterministic sampling"
```

---

### Task 2: Prime Inference provider and cost meter

**Files:**
- Create: `prime_pr_review/providers.py`
- Test: `tests/test_providers.py`

**Interfaces:**
- Produces: `ProviderError`, `BudgetExceeded`, `Usage(prompt_tokens, completion_tokens)`, `Pricing = Mapping[str, tuple[float, float]]`, `CostMeter(cap_usd, pricing)` with `.record(model, usage) -> CostMeter`, `.spent_usd`, `.tokens`, `.check()`, `.to_json()`, `CostMeter.from_json(text, pricing)`; `MeterBox(meter)` (the one mutable holder); `fetch_pricing(client, models) -> dict`; `chat(client, model, prompt, sleep=time.sleep) -> tuple[str, Usage]`; `prime_model_fn(client, model, box) -> ModelFn`; `prime_reviewer(client, model, box, prompts_dir) -> Reviewer`; `resolve_prime_key(env=None, config_path=...) -> str`; `make_client(key) -> httpx.Client`; `BASE_URL`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_providers.py
from __future__ import annotations
import json
import httpx
import pytest
from prime_pr_review.providers import (
    BASE_URL, BudgetExceeded, CostMeter, MeterBox, ProviderError, Usage, chat,
    fetch_pricing, prime_model_fn, prime_reviewer, resolve_prime_key,
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_providers.py -q --no-cov`
Expected: FAIL with `ModuleNotFoundError: prime_pr_review.providers`

- [ ] **Step 3: Implement providers.py**

```python
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
         sleep: Callable[[float], None] = time.sleep) -> tuple[str, Usage]:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0}
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
        sleep(BACKOFF_BASE_SECONDS * (2 ** attempt) + random.uniform(0, 1))
    raise ProviderError(f"{model}: {last}")


def _extract(payload: dict) -> tuple[str, Usage]:
    try:
        text = str(payload["choices"][0]["message"]["content"])
        usage = payload.get("usage") or {}
        return text, Usage(int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)))
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(f"unexpected response shape: {payload!r}"[:300]) from exc


def prime_model_fn(client: httpx.Client, model: str, box: MeterBox) -> ModelFn:
    def model_fn(prompt: str) -> str:
        text, usage = chat(client, model, prompt)
        box.meter = box.meter.record(model, usage)
        return text
    return model_fn


def prime_reviewer(client: httpx.Client, model: str, box: MeterBox, prompts_dir: Path | str) -> Reviewer:
    model_fn = prime_model_fn(client, model, box)
    def reviewer(pr: PullRequest, payload: str, lane: str) -> str:
        template = (Path(prompts_dir) / f"{lane}_pr.md").read_text(encoding="utf-8")
        return model_fn(build_prompt(template, pr, payload))
    return reviewer
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_providers.py -q --no-cov && .venv/bin/ruff check prime_pr_review/providers.py tests/test_providers.py`
Expected: all PASS, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add prime_pr_review/providers.py tests/test_providers.py
git commit -m "feat: Prime Inference provider with retries and a hard cost meter"
```

---

### Task 3: Corpus-backed gh runner

**Files:**
- Create: `prime_pr_review/evaluation/runner.py`
- Test: `tests/test_eval_runner.py`

**Interfaces:**
- Consumes: `Row` (Task 1), `GhRunner`, `GitHubError`, `diffs.split_by_file`.
- Produces: `HeadFileStore(path)` with `.get(path_in_repo) -> str | None`, `.put(path_in_repo, text | None)`, `.paths()`; `pr_list_json(row) -> str`; `corpus_runner(row, fallback: GhRunner, head_files: HeadFileStore) -> GhRunner`; `replay_runner(store) -> GhRunner`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_eval_runner.py
from __future__ import annotations
import json
import pytest
from prime_pr_review.evaluation.corpus import ReferenceComment, Row
from prime_pr_review.evaluation.runner import HeadFileStore, corpus_runner, pr_list_json, replay_runner
from prime_pr_review.github import GitHubError, _parse_pr_list

def _row() -> Row:
    return Row(instance_id="o__r-7@abc", repo="o/r", language="Python", pull_number=7, title="T", body="B",
               base_commit="base", head_sha="abc123", head_commit_message="msg",
               patch="diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,2 +1,2 @@\n-x\n+y\n+z\n",
               reference_comments=(ReferenceComment("a.py", 1, None, "t"),))

def test_pr_list_json_parses_through_production_parser():
    (pr,) = _parse_pr_list(pr_list_json(_row()))
    assert (pr.number, pr.head_sha, pr.author, pr.base_ref) == (7, "abc123", "swe-care", "main")
    assert (pr.additions, pr.deletions, pr.changed_files) == (2, 1, 1)

def test_runner_answers_list_diff_checks_comments(tmp_path):
    run = corpus_runner(_row(), fallback=lambda a, s: "NEVER", head_files=HeadFileStore(tmp_path / "h.json"))
    assert json.loads(run(["pr", "list", "--repo", "o/r"], None))[0]["number"] == 7
    assert run(["pr", "diff", "7", "--repo", "o/r"], None) == _row().patch
    assert run(["pr", "checks", "7", "--repo", "o/r", "--json", "x"], None) == "[]"
    assert run(["api", "repos/o/r/issues/7/comments"], None) == "[]"

def test_runner_delegates_content_fetch_and_records(tmp_path):
    store = HeadFileStore(tmp_path / "h.json")
    run = corpus_runner(_row(), fallback=lambda a, s: "BASE64", head_files=store)
    assert run(["api", "repos/o/r/contents/a.py?ref=abc123", "--jq", ".content"], None) == "BASE64"
    assert HeadFileStore(tmp_path / "h.json").get("a.py") == "BASE64"

def test_runner_records_failed_fetch_as_none(tmp_path):
    store = HeadFileStore(tmp_path / "h.json")
    def boom(a, s): raise GitHubError("404")
    run = corpus_runner(_row(), fallback=boom, head_files=store)
    with pytest.raises(GitHubError):
        run(["api", "repos/o/r/contents/gone.py?ref=abc123", "--jq", ".content"], None)
    assert store.get("gone.py") is None and "gone.py" in store.paths()

def test_runner_rejects_unexpected_calls(tmp_path):
    run = corpus_runner(_row(), fallback=lambda a, s: "", head_files=HeadFileStore(tmp_path / "h.json"))
    with pytest.raises(GitHubError, match="unexpected gh call"):
        run(["pr", "comment", "7"], "body")

def test_replay_runner_serves_store(tmp_path):
    store = HeadFileStore(tmp_path / "h.json"); store.put("a.py", "B64"); store.put("gone.py", None)
    run = replay_runner(store)
    assert run(["api", "repos/o/r/contents/a.py?ref=abc123", "--jq", ".content"], None) == "B64"
    with pytest.raises(GitHubError):
        run(["api", "repos/o/r/contents/gone.py?ref=abc123", "--jq", ".content"], None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_eval_runner.py -q --no-cov`
Expected: FAIL with `ImportError` on `prime_pr_review.evaluation.runner`

- [ ] **Step 3: Implement runner.py**

```python
"""A `gh` runner backed by one SWE-CARE row, so `sweep_lane` reviews the exact
mid-review diff humans commented on, with no GitHub listing or diff calls.

Only head-file content fetches (citation validation) reach the real `gh`; each
response is persisted so the offline scorer replays it without network."""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

from ..diffs import split_by_file
from ..github import GhRunner, GitHubError
from .corpus import Row

CONTENTS_RE = re.compile(r"^repos/[^/]+/[^/]+/contents/(?P<path>.+?)\?ref=")
EVAL_AUTHOR = "swe-care"
EVAL_BASE_REF = "main"


class HeadFileStore:
    """`{path: raw gh response | None}` persisted as JSON after every put."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._data: dict[str, str | None] = (
            json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        )

    def get(self, path: str) -> str | None:
        return self._data.get(path)

    def paths(self) -> tuple[str, ...]:
        return tuple(self._data)

    def put(self, path: str, text: str | None) -> None:
        self._data = {**self._data, path: text}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data), encoding="utf-8")


def pr_list_json(row: Row) -> str:
    added = deleted = 0
    files = split_by_file(row.patch)
    for f in files:
        for line in f.body.splitlines():
            added += line.startswith("+") and not line.startswith("+++")
            deleted += line.startswith("-") and not line.startswith("---")
    return json.dumps([{
        "number": row.pull_number, "title": row.title, "author": {"login": EVAL_AUTHOR},
        "headRefOid": row.head_sha, "baseRefName": EVAL_BASE_REF,
        "url": f"https://github.com/{row.repo}/pull/{row.pull_number}",
        "additions": added, "deletions": deleted, "changedFiles": len(files), "mergedAt": None,
    }])


def corpus_runner(row: Row, fallback: GhRunner, head_files: HeadFileStore) -> GhRunner:
    listing = pr_list_json(row)

    def runner(args: Sequence[str], stdin: str | None = None) -> str:
        head = tuple(args[:2])
        if head == ("pr", "list"):
            return listing
        if head == ("pr", "diff"):
            return row.patch
        if head == ("pr", "checks") or (len(args) > 1 and args[0] == "api" and args[1].endswith("/comments")):
            return "[]"
        path = _contents_path(args)
        if path is not None:
            return _fetch_and_record(args, stdin, path, fallback, head_files)
        raise GitHubError(f"eval runner: unexpected gh call: {list(args)!r}")

    return runner


def replay_runner(store: HeadFileStore) -> GhRunner:
    def runner(args: Sequence[str], stdin: str | None = None) -> str:
        path = _contents_path(args)
        text = store.get(path) if path is not None else None
        if text is None:
            raise GitHubError(f"eval replay: no recorded head file for {list(args)!r}")
        return text
    return runner


def _contents_path(args: Sequence[str]) -> str | None:
    if len(args) < 2 or args[0] != "api":
        return None
    match = CONTENTS_RE.match(args[1])
    return match.group("path") if match else None


def _fetch_and_record(args, stdin, path, fallback: GhRunner, store: HeadFileStore) -> str:
    try:
        text = fallback(args, stdin)
    except Exception:
        store.put(path, None)
        raise
    store.put(path, text)
    return text
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_eval_runner.py -q --no-cov && .venv/bin/ruff check prime_pr_review/evaluation/runner.py tests/test_eval_runner.py`
Expected: all PASS, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add prime_pr_review/evaluation/runner.py tests/test_eval_runner.py
git commit -m "feat(eval): corpus-backed gh runner with recorded head-file fetches"
```

---

### Task 4: Recording and replay of model calls

**Files:**
- Create: `prime_pr_review/evaluation/recording.py`
- Test: `tests/test_eval_recording.py`

**Interfaces:**
- Consumes: `build_prompt`, `PullRequest`.
- Produces: `Call(seq, role, model, prompt_sha256, prompt, response, prompt_tokens, completion_tokens, seconds)`; `Recorder(dir)` with `.record(role, model, prompt, response, seconds, usage=(0, 0)) -> Call` and `.calls(role=None) -> tuple[Call, ...]`; `recording_model_fn(role, model, inner, recorder) -> ModelFn`; `recording_reviewer(seat_models, make_model_fn, recorder, prompts_dir) -> Reviewer`; `replay_model_fn(recorder, role) -> ModelFn` raising `ReplayMiss`; `replay_reviewer(recorder) -> Reviewer`; `mark_done(dir)`, `is_done(dir)`; `sha256(text)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_eval_recording.py
from __future__ import annotations
import pytest
from prime_pr_review.evaluation.recording import (
    Recorder, ReplayMiss, is_done, mark_done, recording_model_fn, recording_reviewer,
    replay_model_fn, replay_reviewer,
)
from .conftest import make_pr

def test_recorder_persists_calls_in_sequence(tmp_path):
    rec = Recorder(tmp_path)
    fn = recording_model_fn("skeptic", "m/a", lambda p: "R:" + p, rec)
    assert fn("one") == "R:one" and fn("two") == "R:two"
    calls = Recorder(tmp_path).calls("skeptic")
    assert [c.seq for c in calls] == [0, 1] and calls[1].response == "R:two"
    assert calls[0].prompt_sha256 != calls[1].prompt_sha256

def test_recording_reviewer_assigns_seats_round_robin(tmp_path):
    (tmp_path / "open_pr.md").write_text("T", encoding="utf-8")
    rec = Recorder(tmp_path)
    reviewer = recording_reviewer(["m/a", "m/b", "m/c"], lambda model: (lambda p: model), rec, tmp_path)
    pr = make_pr()
    assert [reviewer(pr, "D", "open") for _ in range(3)] == ["m/a", "m/b", "m/c"]
    assert [c.model for c in rec.calls("seat")] == ["m/a", "m/b", "m/c"]
    assert rec.calls("seat")[0].prompt.startswith("T")

def test_replay_model_fn_hits_by_prompt_hash_and_misses_loudly(tmp_path):
    rec = Recorder(tmp_path)
    recording_model_fn("judge", "m/a", lambda p: "J", rec)("the prompt")
    replay = replay_model_fn(Recorder(tmp_path), "judge")
    assert replay("the prompt") == "J"
    with pytest.raises(ReplayMiss):
        replay("a different prompt")

def test_replay_reviewer_returns_seats_in_order(tmp_path):
    (tmp_path / "open_pr.md").write_text("T", encoding="utf-8")
    rec = Recorder(tmp_path)
    reviewer = recording_reviewer(["m/a", "m/b"], lambda model: (lambda p: "out-" + model), rec, tmp_path)
    pr = make_pr(); [reviewer(pr, "D", "open") for _ in range(2)]
    replay = replay_reviewer(Recorder(tmp_path))
    assert [replay(pr, "D", "open"), replay(pr, "D", "open")] == ["out-m/a", "out-m/b"]
    with pytest.raises(ReplayMiss):
        replay(pr, "D", "open")

def test_done_marker(tmp_path):
    assert not is_done(tmp_path)
    mark_done(tmp_path)
    assert is_done(tmp_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_eval_recording.py -q --no-cov`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement recording.py**

```python
"""Record every model call of one live review to disk, and replay them offline.

Replay is exact or it is an error: a judge/skeptic prompt that differs from the
recorded one means the offline arm is not the live pipeline, and the scorer
must know that rather than silently calling a model."""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from ..github import PullRequest
from ..reviewers import build_prompt

ModelFn = Callable[[str], str]
Reviewer = Callable[[PullRequest, str, str], str]
CALLS_DIR = "calls"
DONE_MARKER = "done"


class ReplayMiss(RuntimeError):
    """No recorded response for this prompt (or no seats left)."""


@dataclass(frozen=True)
class Call:
    seq: int
    role: str
    model: str
    prompt_sha256: str
    prompt: str
    response: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Recorder:
    def __init__(self, directory: Path) -> None:
        self._dir = Path(directory) / CALLS_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    def record(self, role: str, model: str, prompt: str, response: str, seconds: float,
               usage: tuple[int, int] = (0, 0)) -> Call:
        seq = len(list(self._dir.glob("*.json")))
        call = Call(seq, role, model, sha256(prompt), prompt, response, usage[0], usage[1], seconds)
        (self._dir / f"{seq:03d}-{role}.json").write_text(json.dumps(asdict(call)), encoding="utf-8")
        return call

    def calls(self, role: str | None = None) -> tuple[Call, ...]:
        loaded = (Call(**json.loads(p.read_text(encoding="utf-8"))) for p in sorted(self._dir.glob("*.json")))
        return tuple(c for c in loaded if role is None or c.role == role)


def recording_model_fn(role: str, model: str, inner: ModelFn, recorder: Recorder) -> ModelFn:
    def model_fn(prompt: str) -> str:
        started = time.monotonic()
        response = inner(prompt)
        recorder.record(role, model, prompt, response, time.monotonic() - started)
        return response
    return model_fn


def recording_reviewer(seat_models: Sequence[str], make_model_fn: Callable[[str], ModelFn],
                       recorder: Recorder, prompts_dir: Path | str) -> Reviewer:
    seats = [recording_model_fn("seat", m, make_model_fn(m), recorder) for m in seat_models]
    counter = {"k": 0}

    def reviewer(pr: PullRequest, payload: str, lane: str) -> str:
        template = (Path(prompts_dir) / f"{lane}_pr.md").read_text(encoding="utf-8")
        seat = seats[counter["k"] % len(seats)]
        counter["k"] += 1
        return seat(build_prompt(template, pr, payload))

    return reviewer


def replay_model_fn(recorder: Recorder, role: str) -> ModelFn:
    by_hash = {c.prompt_sha256: c.response for c in recorder.calls(role)}

    def model_fn(prompt: str) -> str:
        try:
            return by_hash[sha256(prompt)]
        except KeyError as exc:
            raise ReplayMiss(f"no recorded {role} response for this prompt") from exc
    return model_fn


def replay_reviewer(recorder: Recorder) -> Reviewer:
    responses = [c.response for c in recorder.calls("seat")]
    counter = {"k": 0}

    def reviewer(pr: PullRequest, payload: str, lane: str) -> str:
        if counter["k"] >= len(responses):
            raise ReplayMiss("more seat calls than recorded")
        response = responses[counter["k"]]
        counter["k"] += 1
        return response
    return reviewer


def mark_done(directory: Path) -> None:
    (Path(directory) / DONE_MARKER).write_text("ok", encoding="utf-8")


def is_done(directory: Path) -> bool:
    return (Path(directory) / DONE_MARKER).is_file()
```

Token usage per call is metered by `providers.MeterBox`; the `Call` usage fields stay `(0, 0)`. Per-PR cost is still exact from the meter delta in Task 7.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_eval_recording.py -q --no-cov && .venv/bin/ruff check prime_pr_review/evaluation/recording.py tests/test_eval_recording.py`
Expected: all PASS, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add prime_pr_review/evaluation/recording.py tests/test_eval_recording.py
git commit -m "feat(eval): record and replay every model call per PR"
```

---

### Task 5: Offline arm reassembly

**Files:**
- Create: `prime_pr_review/evaluation/arms.py`
- Test: `tests/test_eval_arms.py`

**Interfaces:**
- Consumes: Task 3 `HeadFileStore`, `replay_runner`; Task 4 `Recorder`, `replay_reviewer`, `replay_model_fn`, `ReplayMiss`; `ensemble_review_detailed`, `refute_findings`, `parse_verdict`, `validate_citations`, `head_line_counts`, `paths_needing_head_counts`.
- Produces: `ARMS`, `HEAD_FILES = "head_files.json"`, `ArmResult(arm, verdict, notes=(), replay_miss=False, error="")`, `build_arm(arm, instance_dir, pr, diff, lane, prompts_dir) -> ArmResult`, `apply_citations(verdict, diff, instance_dir, repo_slug, head_sha) -> tuple[Verdict, int]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_eval_arms.py
from __future__ import annotations
import base64
import json
from dataclasses import replace
from pathlib import Path
import pytest
from prime_pr_review.evaluation.arms import ARMS, HEAD_FILES, apply_citations, build_arm
from prime_pr_review.evaluation.recording import Recorder, recording_model_fn, recording_reviewer
from prime_pr_review.evaluation.runner import HeadFileStore
from prime_pr_review.ensemble import ensemble_review_detailed
from prime_pr_review.refute import build_refute_prompt
from prime_pr_review.review import Finding, Severity
from .conftest import make_pr

PROMPTS = Path("skills/pr-review/prompts")
DIFF = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,2 +1,2 @@\n-x\n+y\n+z\n"
def finding(line, claim):
    return {"file": "a.py", "line": line, "severity": "HIGH", "claim": claim, "evidence": "e"}
def verdict(*fs):
    return json.dumps({"introduces": list(fs), "fixes": [], "confidence": 0.9})
SEATS = [verdict(finding(1, "bug one")), verdict(finding(2, "bug one again")), verdict()]

@pytest.fixture
def recorded(tmp_path):
    rec = Recorder(tmp_path)
    outputs = iter(SEATS)
    reviewer = recording_reviewer(["m/a", "m/b", "m/c"], lambda m: (lambda p: next(outputs)), rec, PROMPTS)
    judge = recording_model_fn("judge", "m/a", lambda p: '{"clusters": [[0, 1]]}', rec)
    live, _ = ensemble_review_detailed(make_pr(), DIFF, "open", reviewer, size=3, min_agreement=1,
                                       judge_fn=judge, prompts_dir=PROMPTS)
    skeptic = recording_model_fn("skeptic", "m/a", lambda p: '{"refuted": true, "reasoning": "no"}', rec)
    template = (PROMPTS / "refute.md").read_text(encoding="utf-8")
    for f in live.introduces:
        skeptic(build_refute_prompt(template, f, DIFF))
    return tmp_path, live

def test_all_arms_are_buildable(recorded):
    d, _ = recorded
    for arm in ARMS:
        r = build_arm(arm, d, make_pr(), DIFF, "open", PROMPTS)
        assert r.arm == arm and not r.replay_miss and r.verdict is not None, r.error

def test_seat_and_ensemble_arms_differ_in_grouping(recorded):
    d, _ = recorded
    assert len(build_arm("seat-1", d, make_pr(), DIFF, "open", PROMPTS).verdict.introduces) == 1
    assert len(build_arm("ensemble", d, make_pr(), DIFF, "open", PROMPTS).verdict.introduces) == 2
    assert len(build_arm("ensemble+judge", d, make_pr(), DIFF, "open", PROMPTS).verdict.introduces) == 1

def test_full_arm_marks_refuted_and_matches_live(recorded):
    d, live = recorded
    full = build_arm("full", d, make_pr(), DIFF, "open", PROMPTS).verdict
    assert all(f.refuted for f in full.introduces)
    assert [f.claim for f in full.introduces] == [f.claim for f in live.introduces]

def test_replay_miss_is_reported_not_raised(tmp_path):
    Recorder(tmp_path)  # no calls at all
    r = build_arm("ensemble", tmp_path, make_pr(), DIFF, "open", PROMPTS)
    assert r.replay_miss and r.verdict is None

def test_apply_citations_uses_recorded_head_files(recorded):
    d, live = recorded
    HeadFileStore(d / HEAD_FILES).put("a.py", base64.b64encode(b"y\nz\n").decode())
    bad = Finding(file="a.py", line=99, severity=Severity.LOW, claim="beyond", evidence="e")
    v = replace(live, introduces=(*live.introduces, bad))
    kept, dropped = apply_citations(v, DIFF, d, "o/r", "abc")
    assert dropped == 1 and all(f.line != 99 for f in kept.introduces)
```

If `judge.py` expects a different JSON key than `clusters`, read `judge._parse_clusters` and adjust the literal in the fixture. If the judge merge requires a different line bucket to see the two findings as candidates, put both at line 1 and 2 as above (same file, `LINE_BUCKET=5` puts them in one bucket already, so the judge may not even be consulted; in that case expect `ensemble` to also return 1 and change the second seat's line to 7 so the judge is needed).

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_eval_arms.py -q --no-cov`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement arms.py**

```python
"""Rebuild each ablation arm from one instance's recordings, at zero model cost.

The ladder: seat-i (single pass) -> ensemble (agreement 1, no judge) ->
ensemble+judge -> full (plus skeptic; refuted findings count as not reported).
The live run *is* `full`; the scorer checks the replayed `full` matches it."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from ..citations import head_line_counts, paths_needing_head_counts, validate_citations
from ..ensemble import ensemble_review_detailed
from ..github import PullRequest
from ..refute import refute_findings
from ..review import Verdict, VerdictError, parse_verdict
from .recording import Recorder, ReplayMiss, replay_model_fn, replay_reviewer
from .runner import HeadFileStore, replay_runner

ARMS = ("seat-1", "seat-2", "seat-3", "ensemble", "ensemble+judge", "full")
HEAD_FILES = "head_files.json"
JUDGED_ARMS = frozenset({"ensemble+judge", "full"})


@dataclass(frozen=True)
class ArmResult:
    arm: str
    verdict: Verdict | None
    notes: tuple[str, ...] = ()
    replay_miss: bool = False
    error: str = ""


def build_arm(arm: str, instance_dir: Path, pr: PullRequest, diff: str, lane: str,
              prompts_dir: Path | str) -> ArmResult:
    recorder = Recorder(instance_dir)
    try:
        verdict, notes = _build(arm, recorder, pr, diff, lane, prompts_dir)
    except ReplayMiss as exc:
        return ArmResult(arm, None, replay_miss=True, error=str(exc))
    except (VerdictError, IndexError) as exc:
        return ArmResult(arm, None, error=str(exc))
    return ArmResult(arm, verdict, notes)


def _build(arm, recorder, pr, diff, lane, prompts_dir) -> tuple[Verdict, tuple[str, ...]]:
    if arm.startswith("seat-"):
        seats = recorder.calls("seat")
        if not seats:
            raise ReplayMiss("no recorded seats")
        return parse_verdict(seats[int(arm[5:]) - 1].response), ()
    judge = replay_model_fn(recorder, "judge") if arm in JUDGED_ARMS else None
    verdict, notes = ensemble_review_detailed(
        pr, diff, lane, replay_reviewer(recorder), size=3, min_agreement=1,
        judge_fn=judge, prompts_dir=prompts_dir,
    )
    if arm != "full":
        return verdict, notes
    findings, refute_notes = refute_findings(
        verdict.introduces, diff, replay_model_fn(recorder, "skeptic"), prompts_dir
    )
    return replace(verdict, introduces=findings), notes + refute_notes


def apply_citations(verdict: Verdict, diff: str, instance_dir: Path, repo_slug: str,
                    head_sha: str) -> tuple[Verdict, int]:
    store = HeadFileStore(Path(instance_dir) / HEAD_FILES)
    needed = paths_needing_head_counts(verdict.introduces, diff, None)
    counts = head_line_counts(repo_slug, head_sha, needed, replay_runner(store)) if needed else {}
    kept, _ = validate_citations(verdict.introduces, diff, counts, None)
    return replace(verdict, introduces=kept), len(verdict.introduces) - len(kept)
```

Check `ensemble_review_detailed`'s behavior when the replayed reviewer raises `ReplayMiss` on an empty recording: `_collect_runs` may catch generic exceptions and count a failed run instead of propagating. If `test_replay_miss_is_reported_not_raised` fails because the ensemble swallowed the miss, add a guard at the top of `_build` for non-seat arms: `if not recorder.calls("seat"): raise ReplayMiss("no recorded seats")`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_eval_arms.py -q --no-cov && .venv/bin/ruff check prime_pr_review/evaluation/arms.py tests/test_eval_arms.py`
Expected: all PASS, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add prime_pr_review/evaluation/arms.py tests/test_eval_arms.py
git commit -m "feat(eval): rebuild ablation arms offline from recordings"
```

---

### Task 6: Scoring and report

**Files:**
- Create: `prime_pr_review/evaluation/scoring.py`
- Create: `prime_pr_review/evaluation/report.py`
- Test: `tests/test_eval_scoring.py`, `tests/test_eval_report.py`

**Interfaces:**
- Produces: `match(finding, ref, window=5) -> bool`; `InstanceScore(findings, matched_findings, file_matched_findings, refs, matched_refs, dropped)`; `score_instance(verdict, refs, dropped=0, window=5) -> InstanceScore`; `Aggregate(arm, mode, instances, precision, recall, file_precision, fabrication_rate, findings_per_pr)`; `aggregate(arm, mode, scores, fabricated_total) -> Aggregate`; `render_markdown(run_id, config, aggregates, severity_rows, limitations, cost_usd, seconds, drift, misses) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_eval_scoring.py
from __future__ import annotations
from prime_pr_review.evaluation.corpus import ReferenceComment
from prime_pr_review.evaluation.scoring import InstanceScore, aggregate, match, score_instance
from prime_pr_review.review import Finding, Severity, Verdict

def f(file="a.py", line=10, refuted=False):
    return Finding(file=file, line=line, severity=Severity.HIGH, claim="c", evidence="e", refuted=refuted)
def ref(path="a.py", line=10, start=None):
    return ReferenceComment(path=path, line=line, start_line=start, text="t")

def test_match_window_edges():
    assert match(f(line=15), ref(line=10)) and not match(f(line=16), ref(line=10))
    assert match(f(line=3), ref(line=10, start=8)) and not match(f(line=2), ref(line=10, start=8))
    assert not match(f(file="b.py"), ref())

def test_line_none_matches_file_level_only():
    assert not match(f(line=None), ref())
    s = score_instance(Verdict(introduces=(f(line=None),), fixes=(), confidence=0.9), (ref(),))
    assert (s.matched_findings, s.file_matched_findings) == (0, 1)

def test_score_instance_excludes_refuted_and_counts_refs():
    v = Verdict(introduces=(f(line=10), f(line=40, refuted=True), f(line=90)), fixes=(), confidence=0.9)
    s = score_instance(v, (ref(line=10), ref(line=200)), dropped=1)
    assert s == InstanceScore(findings=2, matched_findings=1, file_matched_findings=2,
                              refs=2, matched_refs=1, dropped=1)

def test_none_verdict_scores_as_no_findings():
    assert score_instance(None, (ref(),)).findings == 0

def test_aggregate_micro_averages():
    a = aggregate("full", "on", [InstanceScore(2, 1, 2, 2, 1, 0), InstanceScore(2, 2, 2, 1, 1, 0)], fabricated_total=1)
    assert (a.precision, a.recall, a.file_precision) == (0.75, 2 / 3, 1.0)
    assert a.fabrication_rate == 0.2 and a.findings_per_pr == 2.0 and a.instances == 2

def test_aggregate_with_zero_findings_is_defined():
    a = aggregate("x", "off", [InstanceScore(0, 0, 0, 1, 0, 0)], fabricated_total=0)
    assert a.precision == 0.0 and a.recall == 0.0 and a.fabrication_rate == 0.0
```

```python
# tests/test_eval_report.py
from prime_pr_review.evaluation.report import render_markdown
from prime_pr_review.evaluation.scoring import Aggregate

def test_report_has_ladder_and_limitations():
    aggs = [Aggregate("seat-1", "off", 3, 0.5, 0.25, 0.6, 0.1, 2.0),
            Aggregate("full", "on", 3, 0.7, 0.3, 0.8, 0.0, 1.5)]
    md = render_markdown("run1", {"seed": 0, "count": 3}, aggs, severity_rows=[("HIGH", 2, 1)],
                         limitations=["diff-only"], cost_usd=1.23, seconds=100.0, drift=0, misses=0)
    assert "| full | on |" in md and "0.70" in md and "diff-only" in md and "$1.23" in md
    assert "prime-review eval run --count 3 --seed 0" in md
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_eval_scoring.py tests/test_eval_report.py -q --no-cov`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement scoring.py**

```python
"""Score one arm's verdict against SWE-CARE reference comments.

A finding matches a reference comment on the same path when its line falls in
the comment's range widened by `window` lines. Refuted findings are treated as
not reported. Metrics are micro-averaged across instances. `fabrication_rate`
is dropped / (kept + dropped): the share of raw findings that pointed nowhere."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..review import Finding, Verdict
from .corpus import ReferenceComment

DEFAULT_WINDOW = 5


@dataclass(frozen=True)
class InstanceScore:
    findings: int
    matched_findings: int
    file_matched_findings: int
    refs: int
    matched_refs: int
    dropped: int


@dataclass(frozen=True)
class Aggregate:
    arm: str
    mode: str
    instances: int
    precision: float
    recall: float
    file_precision: float
    fabrication_rate: float
    findings_per_pr: float


def match(finding: Finding, ref: ReferenceComment, window: int = DEFAULT_WINDOW) -> bool:
    if finding.file != ref.path or finding.line is None or ref.line is None:
        return False
    lo = ref.start_line if ref.start_line is not None else ref.line
    return lo - window <= finding.line <= ref.line + window


def score_instance(verdict: Verdict | None, refs: Sequence[ReferenceComment], dropped: int = 0,
                   window: int = DEFAULT_WINDOW) -> InstanceScore:
    findings = tuple(f for f in (verdict.introduces if verdict else ()) if not f.refuted)
    matched = sum(any(match(f, r, window) for r in refs) for f in findings)
    file_matched = sum(any(f.file == r.path for r in refs) for f in findings)
    matched_refs = sum(any(match(f, r, window) for f in findings) for r in refs)
    return InstanceScore(len(findings), matched, file_matched, len(refs), matched_refs, dropped)


def aggregate(arm: str, mode: str, scores: Sequence[InstanceScore], fabricated_total: int) -> Aggregate:
    findings = sum(s.findings for s in scores)
    refs = sum(s.refs for s in scores)
    return Aggregate(
        arm=arm, mode=mode, instances=len(scores),
        precision=_ratio(sum(s.matched_findings for s in scores), findings),
        recall=_ratio(sum(s.matched_refs for s in scores), refs),
        file_precision=_ratio(sum(s.file_matched_findings for s in scores), findings),
        fabrication_rate=_ratio(fabricated_total, findings + fabricated_total),
        findings_per_pr=_ratio(findings, len(scores)),
    )


def _ratio(num: int, den: int) -> float:
    return num / den if den else 0.0
```

- [ ] **Step 4: Implement report.py**

```python
"""Render the ablation results as markdown for docs/eval/."""
from __future__ import annotations

from collections.abc import Sequence

from .scoring import Aggregate

LADDER = ("seat-1", "seat-2", "seat-3", "ensemble", "ensemble+judge", "full")


def render_markdown(run_id: str, config: dict, aggregates: Sequence[Aggregate],
                    severity_rows: Sequence[tuple[str, int, int]], limitations: Sequence[str],
                    cost_usd: float, seconds: float, drift: int, misses: int) -> str:
    lines = [f"# SWE-CARE ablation — `{run_id}`", "",
             f"Instances: {config.get('count')} · seed {config.get('seed')} · "
             f"cost ${cost_usd:.2f} · wall {seconds / 60:.0f} min · "
             f"live/replay drift {drift} · replay misses {misses}", "",
             "## Ablation ladder", "",
             "| arm | citations | PRs | precision | recall | file-level precision | fabrication | findings/PR |",
             "|---|---|---|---|---|---|---|---|"]
    order = {a: i for i, a in enumerate(LADDER)}
    for a in sorted(aggregates, key=lambda a: (order.get(a.arm, 99), a.mode)):
        lines.append(f"| {a.arm} | {a.mode} | {a.instances} | {a.precision:.2f} | {a.recall:.2f} | "
                     f"{a.file_precision:.2f} | {a.fabrication_rate:.2f} | {a.findings_per_pr:.2f} |")
    lines += ["", "## Findings by severity (full arm, citations on)", "",
              "| severity | findings | matched |", "|---|---|---|"]
    lines += [f"| {sev} | {n} | {m} |" for sev, n, m in severity_rows]
    lines += ["", "## Limitations", ""] + [f"- {item}" for item in limitations]
    lines += ["", "## Reproduce", "", "```",
              f"prime-review eval run --count {config.get('count')} --seed {config.get('seed')}",
              f"prime-review eval score --run-id {run_id}",
              f"prime-review eval report --run-id {run_id}", "```", ""]
    return "\n".join(lines)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_eval_scoring.py tests/test_eval_report.py -q --no-cov && .venv/bin/ruff check prime_pr_review/evaluation tests/test_eval_scoring.py tests/test_eval_report.py`
Expected: all PASS, ruff clean.

- [ ] **Step 6: Commit**

```bash
git add prime_pr_review/evaluation/scoring.py prime_pr_review/evaluation/report.py tests/test_eval_scoring.py tests/test_eval_report.py
git commit -m "feat(eval): reference-comment scoring and markdown report"
```

---

### Task 7: CLI — run, score, report

**Files:**
- Create: `scripts/eval_swecare.py`
- Modify: `prime_pr_review/cli.py` (`_SCRIPT_FOR` and `_USAGE`: add `eval`)
- Modify: `.gitignore` (add `eval/corpus/`, `eval/runs/`)
- Create: `docs/eval/.gitkeep`
- Test: `tests/test_eval_cli.py`

**Interfaces:**
- Consumes: everything above.
- Produces: module-level `Provider(make_reviewer_model_fn, aux_fn, skeptic_fn, judge_fn, box, seat_models=SEATS)`; `build_provider(client, box) -> Provider`; `eval_config(base: Config, row) -> Config`; `run_one(row, config, provider, run_dir, prompts_dir, fallback_runner=github.default_runner) -> dict`; `score_run(run_dir, prompts_dir) -> dict`; `write_report(run_dir, out_dir) -> Path`; `main(argv=None) -> int`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_eval_cli.py
from __future__ import annotations
import importlib.util
import json
import sys
from pathlib import Path
from prime_pr_review.evaluation.corpus import parse_rows
from prime_pr_review.providers import CostMeter, MeterBox
from .conftest import make_config

FIXTURE = Path("tests/fixtures/swecare_rows.json")
PROMPTS = Path("skills/pr-review/prompts")
CLEAN = '{"introduces":[],"fixes":[],"confidence":0.9}'

def _load():
    spec = importlib.util.spec_from_file_location("eval_swecare", Path("scripts/eval_swecare.py"))
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
    return mod

def _fake_provider(mod, response=CLEAN):
    box = MeterBox(CostMeter(cap_usd=10, pricing={"m": (1.0, 1.0)}))
    return mod.Provider(make_reviewer_model_fn=lambda model: (lambda p: response),
                        aux_fn=lambda p: '{"summary":"s","claims":[]}',
                        skeptic_fn=lambda p: '{"refuted": false, "reasoning": "r"}',
                        judge_fn=lambda p: '{"clusters": []}', box=box, seat_models=("m", "m", "m"))

def _fail(a, s):
    raise RuntimeError("no network in tests")

def test_eval_config_locks_down_and_targets_row():
    mod = _load(); (row, *_) = parse_rows(json.loads(FIXTURE.read_text()))
    cfg = mod.eval_config(make_config(), row)
    assert cfg.review.dry_run and cfg.repo.read_only and not cfg.sinks.pr_comment
    assert (cfg.repo.owner, cfg.repo.name) == tuple(row.repo.split("/"))
    assert cfg.review.ensemble_size == 3 and cfg.review.min_agreement == 1 and cfg.review.repo_root == ""

def test_run_one_writes_layout_and_is_resumable(tmp_path):
    mod = _load(); (row, *_) = parse_rows(json.loads(FIXTURE.read_text()))
    run_dir = tmp_path / "run"
    out = mod.run_one(row, mod.eval_config(make_config(), row), _fake_provider(mod), run_dir, PROMPTS, _fail)
    inst = run_dir / row.instance_id
    assert (inst / "outcome.json").is_file() and (inst / "done").is_file() and (inst / "calls").is_dir()
    assert out["error"] is None and len(list((inst / "calls").glob("*-seat.json"))) == 3
    again = mod.run_one(row, mod.eval_config(make_config(), row), _fake_provider(mod), run_dir, PROMPTS, _fail)
    assert again["skipped"] is True

def test_score_and_report_on_fake_run(tmp_path):
    mod = _load(); raw = json.loads(FIXTURE.read_text())["rows"][:2]
    rows = parse_rows({"rows": raw})
    run_dir = tmp_path / "run"
    for row in rows:
        mod.run_one(row, mod.eval_config(make_config(), row), _fake_provider(mod), run_dir, PROMPTS, _fail)
    (run_dir / "rows.json").write_text(json.dumps({"rows": raw}))
    (run_dir / "config.json").write_text(json.dumps({"count": 2, "seed": 0, "run_id": "run"}))
    summary = mod.score_run(run_dir, PROMPTS)
    assert summary["instances"] == 2 and any(a["arm"] == "full" for a in summary["aggregates"])
    path = mod.write_report(run_dir, tmp_path / "docs")
    assert path.read_text().startswith("# SWE-CARE ablation")
```

The intent pass in `sweep_lane` will call `aux_fn`; if `intent.run_intent_check` rejects `{"summary":"s","claims":[]}`, read `intent._parse_statement` and use a minimal valid JSON for pass 1 and pass 2 (both go through the same `aux_fn`, so return one object that satisfies both parsers, or key on `"Pass 2"` in the prompt text). An `IntentError` is caught by the sweep as a note, so the test still passes; just keep `aux_fn` returning quickly.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_eval_cli.py -q --no-cov`
Expected: FAIL (script missing)

- [ ] **Step 3: Implement scripts/eval_swecare.py**

```python
"""SWE-CARE ablation harness: `run` reviews sampled PRs live on Prime Inference
and records every model call; `score` rebuilds each ablation arm offline and
scores it; `report` renders docs/eval/<run_id>.md.
Spec: docs/superpowers/specs/2026-09-11-swecare-ablation-harness-design.md."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import httpx

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from prime_pr_review import github  # noqa: E402
from prime_pr_review.config import Config, load_config  # noqa: E402
from prime_pr_review.evaluation import arms as arms_mod  # noqa: E402
from prime_pr_review.evaluation.corpus import Row, fetch_rows, parse_rows, select  # noqa: E402
from prime_pr_review.evaluation.recording import (  # noqa: E402
    Recorder, is_done, mark_done, recording_model_fn, recording_reviewer,
)
from prime_pr_review.evaluation.report import render_markdown  # noqa: E402
from prime_pr_review.evaluation.runner import HeadFileStore, corpus_runner, pr_list_json  # noqa: E402
from prime_pr_review.evaluation.scoring import Aggregate, aggregate, match, score_instance  # noqa: E402
from prime_pr_review.providers import (  # noqa: E402
    BudgetExceeded, CostMeter, MeterBox, fetch_pricing, make_client, prime_model_fn, resolve_prime_key,
)
from prime_pr_review.review import Verdict  # noqa: E402
from prime_pr_review.state import LANE_OPEN, State  # noqa: E402
from prime_pr_review.sweep import Enrichment, sweep_lane  # noqa: E402

DATASET_URL = "https://datasets-server.huggingface.co/rows"
DATASET = "inclusionAI/SWE-CARE"
SEATS = ("deepseek/deepseek-v4-pro", "qwen/qwen3.8-max", "z-ai/glm-5.2")
AUX_MODEL = "deepseek/deepseek-v4-flash"
SKEPTIC_MODEL = JUDGE_MODEL = "deepseek/deepseek-v4-pro"
PROMPTS_DIR = AGENT_ROOT / "skills" / "pr-review" / "prompts"
EVAL_ROOT = AGENT_ROOT / "eval"
DOCS_EVAL = AGENT_ROOT / "docs" / "eval"
LIMITATIONS = (
    "Diff-only: no local checkout per PR, so context, blast-radius, unwired-export and linter passes are skipped.",
    "Ground truth is human review comments; a real defect humans did not comment on counts against precision.",
    "Intent pass sees the PR title only (PullRequest carries no body/commits in the headless path).",
    "SWE-CARE contains no cosmetic/silent PRs, so the silence rate is not measured.",
    "Offline ladder applies citation validation after refutation; production applies it before.",
)


@dataclass(frozen=True)
class Provider:
    make_reviewer_model_fn: Callable[[str], Callable[[str], str]]
    aux_fn: Callable[[str], str]
    skeptic_fn: Callable[[str], str]
    judge_fn: Callable[[str], str]
    box: MeterBox
    seat_models: tuple[str, ...] = SEATS


def build_provider(client: httpx.Client, box: MeterBox) -> Provider:
    return Provider(
        make_reviewer_model_fn=lambda model: prime_model_fn(client, model, box),
        aux_fn=prime_model_fn(client, AUX_MODEL, box),
        skeptic_fn=prime_model_fn(client, SKEPTIC_MODEL, box),
        judge_fn=prime_model_fn(client, JUDGE_MODEL, box),
        box=box,
    )


def eval_config(base: Config, row: Row) -> Config:
    owner, name = row.repo.split("/", 1)
    return replace(
        base,
        repo=replace(base.repo, owner=owner, name=name, read_only=True),
        review=replace(base.review, dry_run=True, ensemble_size=3, min_agreement=1, judge_merge=True,
                       check_refute=True, validate_citations=True, check_intent=True,
                       repo_root="", graph_path="", bot_login=""),
        sinks=replace(base.sinks, pr_comment=False, webhook=False, local_file=True),
    )


def run_one(row: Row, config: Config, provider: Provider, run_dir: Path, prompts_dir: Path,
            fallback_runner: github.GhRunner = github.default_runner) -> dict:
    inst = run_dir / row.instance_id
    if is_done(inst):
        return {"instance_id": row.instance_id, "skipped": True, "error": None}
    inst.mkdir(parents=True, exist_ok=True)
    recorder = Recorder(inst)
    reviewer = recording_reviewer(provider.seat_models, provider.make_reviewer_model_fn, recorder, prompts_dir)
    enrichment = Enrichment(
        model_fn=recording_model_fn("aux", AUX_MODEL, provider.aux_fn, recorder),
        prompts_dir=prompts_dir,
        skeptic_fn=recording_model_fn("skeptic", SKEPTIC_MODEL, provider.skeptic_fn, recorder),
        judge_fn=recording_model_fn("judge", JUDGE_MODEL, provider.judge_fn, recorder),
    )
    runner = corpus_runner(row, fallback_runner, HeadFileStore(inst / arms_mod.HEAD_FILES))
    started, spent_before = time.monotonic(), provider.box.meter.spent_usd
    report, _ = sweep_lane(config, LANE_OPEN, reviewer, State.empty(), runner=runner,
                           reviews_dir=run_dir / "reviews", enrichment=enrichment)
    outcome = report.outcomes[0] if report.outcomes else None
    result = {
        "instance_id": row.instance_id, "skipped": False,
        "error": outcome.error if outcome else "no outcome",
        "notes": list(outcome.notes) if outcome else [],
        "verdict": _verdict_json(outcome.verdict) if outcome and outcome.verdict else None,
        "seconds": time.monotonic() - started,
        "cost_usd": provider.box.meter.spent_usd - spent_before,
    }
    (inst / "outcome.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    mark_done(inst)
    return result


def _verdict_json(verdict: Verdict) -> dict:
    return json.loads(json.dumps(asdict(verdict), default=str))


def _pr_for(row: Row) -> github.PullRequest:
    return github._parse_pr_list(pr_list_json(row))[0]


def score_run(run_dir: Path, prompts_dir: Path) -> dict:
    rows = {r.instance_id: r for r in parse_rows(json.loads((run_dir / "rows.json").read_text(encoding="utf-8")))}
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    acc = _Accumulator()
    done_dirs = sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name in rows and is_done(p))
    for inst in done_dirs:
        _score_instance_dir(inst, rows[inst.name], prompts_dir, acc)
    aggregates = [aggregate(arm, mode, scores, acc.fabricated.get(arm, 0))
                  for (arm, mode), scores in acc.per_arm.items()]
    summary = {"run_id": config.get("run_id"), "config": config, "instances": len(done_dirs),
               "aggregates": [asdict(a) for a in aggregates], "severity": acc.severity, "drift": acc.drift,
               "replay_misses": acc.misses, "cost_usd": acc.cost, "seconds": acc.seconds,
               "limitations": list(LIMITATIONS)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


class _Accumulator:
    def __init__(self) -> None:
        self.per_arm: dict[tuple[str, str], list] = {}
        self.fabricated: dict[str, int] = {}
        self.severity: dict[str, list[int]] = {}
        self.drift = self.misses = 0
        self.cost = self.seconds = 0.0


def _score_instance_dir(inst: Path, row: Row, prompts_dir: Path, acc: _Accumulator) -> None:
    outcome = json.loads((inst / "outcome.json").read_text(encoding="utf-8"))
    acc.cost += outcome.get("cost_usd", 0.0)
    acc.seconds += outcome.get("seconds", 0.0)
    pr, refs = _pr_for(row), row.reference_comments
    for arm in arms_mod.ARMS:
        built = arms_mod.build_arm(arm, inst, pr, row.patch, LANE_OPEN, prompts_dir)
        acc.misses += built.replay_miss
        if built.verdict is None:
            continue
        acc.per_arm.setdefault((arm, "off"), []).append(score_instance(built.verdict, refs))
        validated, dropped = arms_mod.apply_citations(built.verdict, row.patch, inst, row.repo, row.head_sha)
        acc.fabricated[arm] = acc.fabricated.get(arm, 0) + dropped
        acc.per_arm.setdefault((arm, "on"), []).append(score_instance(validated, refs, dropped))
        if arm == "full":
            acc.drift += _drifted(outcome.get("verdict"), built.verdict)
            for f in validated.introduces:
                if not f.refuted:
                    bucket = acc.severity.setdefault(str(f.severity.value), [0, 0])
                    bucket[0] += 1
                    bucket[1] += any(match(f, r) for r in refs)


def _drifted(live: dict | None, replayed: Verdict) -> int:
    if live is None:
        return 1
    def key(items):
        return sorted((str(i.get("file")), i.get("line"), str(i.get("claim"))) for i in items)
    return int(key(live.get("introduces", [])) != key([asdict(f) for f in replayed.introduces]))


def write_report(run_dir: Path, out_dir: Path) -> Path:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    aggs = [Aggregate(**a) for a in summary["aggregates"]]
    sev = [(k, v[0], v[1]) for k, v in sorted(summary["severity"].items())]
    md = render_markdown(summary["run_id"], summary["config"], aggs, sev, summary["limitations"],
                         summary["cost_usd"], summary["seconds"], summary["drift"], summary["replay_misses"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{summary['run_id']}.md"
    path.write_text(md, encoding="utf-8")
    (out_dir / f"{summary['run_id']}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return path


def _fetch_page(offset: int, length: int) -> dict:
    params = {"dataset": DATASET, "config": "default", "split": "test", "offset": offset, "length": length}
    response = httpx.get(DATASET_URL, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def _row_raw(row: Row) -> dict:
    return {"instance_id": row.instance_id, "repo": row.repo, "language": row.language,
            "pull_number": row.pull_number, "title": row.title, "body": row.body, "base_commit": row.base_commit,
            "commit_to_review": {"head_commit": row.head_sha, "head_commit_message": row.head_commit_message,
                                 "patch_to_review": row.patch},
            "reference_review_comments": [{"path": c.path, "line": c.line, "original_line": c.line,
                                           "start_line": c.start_line, "original_start_line": c.start_line,
                                           "text": c.text, "diff_hunk": ""} for c in row.reference_comments]}


def cmd_run(args: argparse.Namespace) -> int:
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%d-%H%M")
    run_dir = EVAL_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = select(fetch_rows(EVAL_ROOT / "corpus" / "swecare-test.json", _fetch_page), args.count, args.seed)
    (run_dir / "rows.json").write_text(json.dumps({"rows": [{"row": _row_raw(r)} for r in rows]}), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps({"run_id": run_id, "count": args.count, "seed": args.seed,
                                                     "cap_usd": args.cap_usd, "seats": SEATS}), encoding="utf-8")
    client = make_client(resolve_prime_key())
    pricing = fetch_pricing(client, [*SEATS, AUX_MODEL, SKEPTIC_MODEL, JUDGE_MODEL])
    meter_path = run_dir / "meter.json"
    meter = (CostMeter.from_json(meter_path.read_text(encoding="utf-8"), pricing) if meter_path.is_file()
             else CostMeter(cap_usd=args.cap_usd, pricing=pricing))
    provider = build_provider(client, MeterBox(meter))
    base = load_config(AGENT_ROOT / "config.toml")
    for i, row in enumerate(rows, 1):
        try:
            result = run_one(row, eval_config(base, row), provider, run_dir, PROMPTS_DIR)
        except Exception as exc:  # noqa: BLE001 - one PR must never end the run
            result = {"instance_id": row.instance_id, "error": repr(exc), "skipped": False}
        meter_path.write_text(provider.box.meter.to_json(), encoding="utf-8")
        status = "skipped" if result.get("skipped") else (result.get("error") or "ok")
        print(f"[{i}/{len(rows)}] {row.instance_id}: {status} | spent ${provider.box.meter.spent_usd:.2f}")
        try:
            provider.box.meter.check()
        except BudgetExceeded as exc:
            print(f"stopping: {exc}")
            return 3
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="prime-review eval")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--count", type=int, default=200)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--cap-usd", type=float, default=40.0)
    run.add_argument("--run-id", default="")
    run.set_defaults(fn=cmd_run)
    score = sub.add_parser("score")
    score.add_argument("--run-id", required=True)
    score.set_defaults(fn=lambda a: (score_run(EVAL_ROOT / "runs" / a.run_id, PROMPTS_DIR) and 0))
    report = sub.add_parser("report")
    report.add_argument("--run-id", required=True)
    report.set_defaults(fn=lambda a: (print(write_report(EVAL_ROOT / "runs" / a.run_id, DOCS_EVAL)) or 0))
    args = parser.parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
```

`--resume` is implicit: `run` always skips instances that carry a `done` marker, and the meter file is reloaded when present.

- [ ] **Step 4: Wire the CLI, gitignore, docs dir**

In `prime_pr_review/cli.py` add `"eval": "eval_swecare.py"` to `_SCRIPT_FOR` and a usage line `eval run|score|report   SWE-CARE ablation harness (docs/superpowers/specs/2026-09-11-swecare-ablation-harness-design.md)`. Append to `.gitignore`:

```
# Evaluation harness runtime artifacts (results are committed under docs/eval/)
eval/corpus/
eval/runs/
```

Create `docs/eval/.gitkeep` (empty).

- [ ] **Step 5: Run the full suite and lint**

Run: `.venv/bin/python -m pytest -q && .venv/bin/ruff check scripts/eval_swecare.py prime_pr_review/cli.py tests/test_eval_cli.py`
Expected: all PASS, coverage ≥ 80%, ruff clean.

- [ ] **Step 6: Commit**

```bash
git add scripts/eval_swecare.py prime_pr_review/cli.py .gitignore docs/eval/.gitkeep tests/test_eval_cli.py
git commit -m "feat(eval): prime-review eval run/score/report for the SWE-CARE ablation"
```

---

### Task 8: Smoke run, then the real run

**Files:**
- Modify: `README.md` (one "Evaluation" section pointing at `docs/eval/`)
- Create: `docs/eval/<run_id>.md` and `.json` (generated)

- [ ] **Step 1: Preflight**

Run: `.venv/bin/python -c "from prime_pr_review.providers import *; c=make_client(resolve_prime_key()); print(fetch_pricing(c, ['deepseek/deepseek-v4-pro','qwen/qwen3.8-max','z-ai/glm-5.2','deepseek/deepseek-v4-flash']))"` and `gh auth status`.
Expected: four priced models; gh logged in.

- [ ] **Step 2: Smoke run on 3 PRs with a $2 cap**

Run: `.venv/bin/prime-review eval run --count 3 --seed 0 --cap-usd 2 --run-id smoke`
Expected: three `[i/3] … ok` lines, spend well under $1. Inspect `eval/runs/smoke/<instance>/calls/` (3 seat files, a judge or skeptic file when findings exist) and the review markdown under `eval/runs/smoke/reviews/`.

- [ ] **Step 3: Score and report the smoke run**

Run: `.venv/bin/prime-review eval score --run-id smoke && .venv/bin/prime-review eval report --run-id smoke && cat docs/eval/smoke.md`
Expected: a ladder table with 12 rows (6 arms × 2 modes), drift 0, replay misses 0. If drift > 0, stop and diagnose before spending more: compare `outcome.json` `verdict.introduces` with the replayed full arm for that instance.

- [ ] **Step 4: Full run**

Run under `caffeinate -i`: `caffeinate -i .venv/bin/prime-review eval run --count 200 --seed 0 --cap-usd 40 --run-id swecare-200-s0`
Expected: 3–4 hours; re-run the same command to resume after any interruption.

- [ ] **Step 5: Score, report, and commit results**

Run: `.venv/bin/prime-review eval score --run-id swecare-200-s0 && .venv/bin/prime-review eval report --run-id swecare-200-s0`
Delete `docs/eval/smoke.md` and `docs/eval/smoke.json`. Add to `README.md` under a new `## Evaluation` heading: two sentences naming the corpus, the arms, and a link to `docs/eval/swecare-200-s0.md`, plus the headline ladder rows for `seat-1` and `full` with citations on.

```bash
git add docs/eval/swecare-200-s0.md docs/eval/swecare-200-s0.json README.md
git commit -m "docs(eval): SWE-CARE ablation results, 200 PRs, seed 0"
```

---

## Self-review

**Spec coverage:** §2 corpus → Task 1. §3.1 provider/meter → Task 2. §3.3 runner incl. head-file recording → Task 3. §3.4 recording/replay/done marker → Task 4. §3.5 arms incl. citations on/off and drift check → Task 5 + Task 7 `score_run`. §3.6 scoring → Task 6. §3.7 report incl. limitations and reproduce command → Task 6 + Task 7. §3.8 CLI, lockdown config, resume, budget stop → Task 7. §4 error handling: provider retries (Task 2), corpus fetched before the client is built so a corpus failure spends nothing (Task 7 `cmd_run`), replay miss reported (Task 5/7), budget stop (Task 7). §5 tests → each task. §6 smoke/full run → Task 8. Per-call token usage in `Call` stays zero (Task 4 note); per-PR cost is exact from the meter delta.

**Placeholder scan:** none; every step has code or an exact command. Two conditional notes (judge JSON key, intent JSON shape) tell the implementer exactly which function to read and what to change.

**Type consistency:** `Row` fields (`head_sha`, `patch`, `reference_comments`, `repo`) used identically in Tasks 3, 5, 7. `HeadFileStore.put/get/paths` consistent across 3, 5. `Recorder.calls(role)` returns `Call` with `.response/.model/.prompt_sha256` as used in Task 5. `Provider` fields match `_fake_provider` in Task 7's test. `apply_citations -> (Verdict, int)` and `score_instance(..., dropped=int)` match between Tasks 5, 6, 7. `Aggregate` constructor order matches `test_eval_report.py`.
