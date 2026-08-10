# Blast radius — "what else does this change break?" (P9)

You are judging one changed symbol against the real, complete list of places
elsewhere in the repository that reference it. The list below was produced by
`git grep` against files this diff does not touch — it is not a guess, and it
is not partial. Your only job is judgment: for each call site, does this
specific change actually break it, or does the call site not even reach the
part of the symbol that changed?

**Do not invent call sites.** If you believe there are other callers not shown
below, you are wrong — the search already covered the whole repository. Judge
only the call sites given. Every one of them must end up counted: either in
`breaks`, with a one-sentence reason, or in `unbroken_callers`. A call site
you cannot see does not exist for this exercise.

## Input

- The changed symbol: name, file, what changed, and its signature before/after
- Every real call site outside the diff that references this symbol's name,
  each as `file:line — the line's text`

## What counts as "breaks"

Judge against the actual change, not against the symbol's name in general:

- **signature_change** — a caller passes the old argument count/order, or
  omits a newly required parameter, or passes a now-removed one
- **return_shape_change** — a caller uses the return value in a way the new
  shape no longer supports (wrong type, missing key, unexpected `None`)
- **exception_change** — a caller's `except` no longer matches what is raised,
  or an unguarded caller now receives an exception it never handled before
- **semantic_change** — same signature, but the call site clearly depended on
  the old behavior (e.g. an inverted default, a changed rounding rule)
- **removal_or_rename** — the caller references a name that no longer exists
- **constant_change** — a caller's logic assumed the old constant value
- **schema_change** — a caller reads a field/column this change altered or
  dropped

A call site is **unbroken** when the part of the symbol it exercises did not
change (e.g. it only uses parameters that kept their position and meaning), or
when it already passes something compatible with the new form.

## What does NOT count

- A call site that merely mentions the symbol's name in a comment or string,
  with no actual call/reference — count it as unbroken, do not fabricate a break
- Speculation about callers not in the given list
- Style or naming opinions about the call site itself

## Calibration

**Precision over recall, but never silence.** Only put a call site in `breaks`
when you can point at exactly why the call, as written, stops working. When you
are unsure whether a specific call site breaks, prefer `unbroken_callers` and a
lower severity elsewhere over a confident wrong claim — but every call site
must be accounted for one way or the other. Do not drop one silently.

## Severity

- `CRITICAL` — breaks a caller in a way that loses or corrupts data, or a
  security-relevant path (auth, permissions, payments)
- `HIGH` — breaks a caller with a guaranteed crash or wrong result on a common path
- `MEDIUM` — breaks a caller only on an edge case or rarely-exercised branch
- `LOW` — technically incompatible but practically inert (e.g. dead code, a
  parameter the caller always passed as the new default anyway)

## Output

Return **only** this JSON object — one entry, for this one symbol. No prose
before or after, no code fence.

```
{
  "symbol": "total_price",
  "kind": "signature_change",
  "change": "One sentence describing what changed about this symbol.",
  "breaks": [
    {
      "file": "shop/invoice.py",
      "line": 44,
      "severity": "HIGH",
      "claim": "Calls total_price(items) with one argument; the new required tax_rate parameter is missing, so this will raise TypeError."
    }
  ],
  "unbroken_callers": 3
}
```

`len(breaks) + unbroken_callers` must equal the number of call sites you were
given. `kind` must be one of: `signature_change`, `return_shape_change`,
`exception_change`, `semantic_change`, `removal_or_rename`, `constant_change`,
`schema_change`.
