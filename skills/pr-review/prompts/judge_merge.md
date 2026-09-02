# Judge-merge — which of these findings describe the same defect?

Independent reviewers examined the same pull request without seeing each
other's work. Their findings were grouped by exact location, but the same
underlying defect can be reported at different lines (one reviewer counts from
the function header, another from the failing statement) or at different
severities. Your job: identify groups below that describe the SAME underlying
defect, so agreement between reviewers is counted correctly.

## Rules

- Merge two findings ONLY when they describe the same root defect — the same
  broken behavior with the same cause — not merely the same area of code.
- Two different problems in the same file, even adjacent lines, are NOT the
  same defect. When in doubt, do NOT merge; a missed merge slightly
  undercounts agreement, a wrong merge silently erases a distinct bug.
- Only findings in the SAME file may be merged. Cross-file clusters are
  rejected by the caller.
- You can only merge. You cannot drop, rewrite, or re-rank findings — that is
  not this pass's job.

## Output — JSON only

Respond with ONLY this JSON object, no prose before or after:

```json
{
  "clusters": [[0, 3], [1, 4, 5]]
}
```

- Each inner array lists the indices (the `[n]` markers below) of findings
  that describe one and the same defect.
- List only clusters with two or more members. Findings you do not mention
  stay as they are.
- An index may appear in at most one cluster.
- No duplicates found → `{"clusters": []}`.
