# First Sweep — Results

First end-to-end run of the review agent against a controlled repo with planted defects.

- **Repo:** [RonKat2008/prime-agent-review-demo](https://github.com/RonKat2008/prime-agent-review-demo) (private, throwaway)
- **Model:** `gemini-flash-latest` (Google AI Studio, 1M token input limit)
- **Lane:** `open` · **Mode:** dry-run (nothing posted to GitHub)
- **Result:** `4 considered | 4 reviewed | 0 posted | 0 skipped | 0 errors`

## Score: 4 / 4

The demo repo was seeded so every PR has a known correct answer. Nothing was tuned
between writing the answer key and running the sweep.

| PR | Planted | Expected | Agent returned | |
|---|---|---|---|---|
| [#3](https://github.com/RonKat2008/prime-agent-review-demo/pull/3) | off-by-one + swallowed exception | HIGH, 2 bugs | HIGH — 2 introduced | ✅ |
| [#4](https://github.com/RonKat2008/prime-agent-review-demo/pull/4) | null-deref guard added | fixes only, no bugs | 0 introduced, 1 fixed | ✅ |
| [#5](https://github.com/RonKat2008/prime-agent-review-demo/pull/5) | SQL injection | CRITICAL | CRITICAL — 1 introduced | ✅ |
| [#6](https://github.com/RonKat2008/prime-agent-review-demo/pull/6) | rename + docstring only | **silence** | 0 introduced, 0 fixed | ✅ |

**PR #6 is the one that matters.** A reviewer that flags everything is worse than no
reviewer — it trains you to ignore it. Staying quiet on a behavior-preserving change is
the harder test, and the one that decides whether this is tolerable on a real repo.

---

## Full reviews

### PR #3 — "Speed up total_price"

> **PR #3** · `36a79435` · confidence 95%
>
> #### Potential bugs introduced (2)
>
> - **HIGH** `shop/orders.py:6` — Slicing `items[:-1]` excludes the last item from the order total calculation.
>   - Line 6 changes iteration over `items` to `items[:-1]`, dropping the final element in the list and causing incorrect total price sums.
> - **MEDIUM** `shop/orders.py:13` — Exceptions during discount calculation are silently caught, causing `apply_discount` to return `None` on error.
>   - Lines 11-14 catch broad `Exception` with `pass`, swallowing errors and silently returning `None` instead of raising or returning a valid numerical result.

Both planted defects found, correctly located, correctly ranked. It caught that the
swallowed exception causes an implicit `None` return — a consequence, not just the
`except: pass` pattern.

### PR #4 — "Guard against a missing order"

> **PR #4** · `1da8a4ea` · confidence 95%
>
> #### Bugs fixed (1)
>
> - Fixes a potential `TypeError` when an order is not found.
>   - Adds an `if order is None:` guard check in `order_summary` to return a missing order message instead of attempting to subscript `None`.

Correctly identified as a **fix** rather than a change. Named the exact exception the
old code would raise. Reported no false positives.

### PR #5 — "Add customer search by name"

> **PR #5** · `62c8015f` · confidence 95%
>
> #### Potential bugs introduced (1)
>
> - **CRITICAL** `shop/customers.py:15` — Introduces a SQL injection vulnerability by directly interpolating user input into a database query string.
>   - Line 15 constructs the SQL query using an f-string (`f"SELECT ... WHERE name LIKE '%{name}%'"`) instead of using parameterized query placeholders.

CRITICAL severity, correct line, and it noted the file's *existing* code already used
parameterized queries — so the PR is inconsistent with its own module.

### PR #6 — "Clarify parameter naming in apply_discount"

> **PR #6** · `7c7f4c98` · confidence 95%

No findings. Correct. The diff renames a parameter and expands a docstring; there is
nothing to report, and it reported nothing.

---

## What this run establishes

**The pipeline works end to end.** Candidate selection → head-SHA dedupe → diff fetch →
lockfile filtering → per-PR model call → verdict parsing → gate evaluation → local
write → watermark advance.

**Error containment works.** The first attempt used a retired model ID and returned 404
on all four PRs. Every failure was caught per-PR, recorded on the report, and the sweep
completed cleanly — `4 considered | 0 reviewed | 4 errors` — instead of crashing. The
PRs were left unmarked so they retried on the next run.

**Dry-run works.** All four verdicts were held with reason `dry_run enabled`. Nothing
reached GitHub, and every review still landed in `reviews/` as the audit trail.

**Flash-tier is sufficient.** A cheap model scored identically to what a frontier model
would on these four cases. Worth knowing before committing to per-token spend on a busy
repo.

## Caveats

- **Four PRs is a small sample**, and the defects were deliberately planted. Real PRs are
  longer, subtler, and more ambiguous. Expect a lower hit rate.
- **Confidence was 95% on all four**, including the empty verdict. The model is not
  meaningfully discriminating on confidence yet, so `min_confidence` is not currently
  doing real work as a filter.
- **The merged lane is untested.** Only the `open` lane has run.
- **Nothing has posted to GitHub yet.** The comment path is exercised by unit tests but
  has never run live.

## Reproducing

```bash
python scripts/run_sweep.py --lane open --fresh --model gemini-flash-latest
```

`--fresh` ignores the watermark and re-reviews everything. Without it, unchanged PRs are
skipped before any model call.

To exercise live posting on the demo repo — safe, since it is private and yours:

```toml
# config.toml
[review]
dry_run = false
```
```bash
rm state/watermark.json    # held reviews are marked reviewed; clear to re-run
python scripts/run_sweep.py --lane open
```
