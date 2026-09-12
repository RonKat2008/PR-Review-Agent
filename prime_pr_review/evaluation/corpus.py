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
    eligible = _eligible(rows, max_patch_bytes)
    if len(eligible) <= count:
        return tuple(eligible)
    return tuple(random.Random(seed).sample(eligible, count))


def select_with_counts(
    rows: tuple[Row, ...], count: int, seed: int, max_patch_bytes: int = DEFAULT_MAX_PATCH_BYTES
) -> tuple[tuple[Row, ...], dict[str, int]]:
    """`select` plus how many rows each filter removed, so the run's config can
    record what the corpus looked like before sampling."""
    python = [r for r in rows if r.language == "Python"]
    sized = [r for r in python if len(r.patch.encode("utf-8")) <= max_patch_bytes]
    eligible = _eligible(rows, max_patch_bytes)
    selected = select(rows, count, seed, max_patch_bytes)
    counts = {
        "total": len(rows),
        "not_python": len(rows) - len(python),
        "oversize": len(python) - len(sized),
        "unscorable": len(sized) - len(eligible),
        "eligible": len(eligible),
        "selected": len(selected),
    }
    return selected, counts


def _eligible(rows: tuple[Row, ...], max_patch_bytes: int) -> list[Row]:
    return [
        r for r in rows
        if r.language == "Python" and len(r.patch.encode("utf-8")) <= max_patch_bytes and r.scorable
    ]


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
