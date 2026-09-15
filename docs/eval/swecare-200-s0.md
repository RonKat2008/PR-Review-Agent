# SWE-CARE ablation — `swecare-200-s0`

Instances: 197 · seed 0 · cost $39.26 · wall 1568 min · live/replay drift 40 · replay misses 21

Corpus: total 671 · not Python 0 · oversize patch 35 · no usable comment 0 · eligible 636 · sampled 200

## Ablation ladder

| arm | citations | PRs | precision | recall | file-level precision | fabrication | findings/PR | cost $/PR | wall s/PR |
|---|---|---|---|---|---|---|---|---|---|
| seat-1 | off | 192 | 0.24 | 0.24 | 0.62 | — | 1.80 | 0.084 | 271.3 |
| seat-1 | on | 192 | 0.24 | 0.24 | 0.62 | 0.00 | 1.79 | 0.084 | 271.3 |
| seat-2 | off | 185 | 0.23 | 0.15 | 0.64 | — | 1.08 | 0.035 | 69.8 |
| seat-2 | on | 185 | 0.23 | 0.15 | 0.64 | 0.00 | 1.08 | 0.035 | 69.8 |
| seat-3 | off | 190 | 0.29 | 0.22 | 0.68 | — | 1.29 | 0.032 | 75.4 |
| seat-3 | on | 190 | 0.29 | 0.22 | 0.68 | 0.00 | 1.29 | 0.032 | 75.4 |
| single | off | 185 | 0.25 | 0.20 | 0.65 | — | 1.39 | 0.050 | 138.9 |
| single | on | 185 | 0.25 | 0.20 | 0.65 | 0.00 | 1.39 | 0.050 | 138.9 |
| ensemble | off | 192 | 0.24 | 0.34 | 0.65 | — | 3.46 | 0.144 | 359.6 |
| ensemble | on | 192 | 0.24 | 0.34 | 0.65 | 0.00 | 3.45 | 0.144 | 359.6 |
| ensemble+judge | off | 192 | 0.21 | 0.30 | 0.62 | — | 2.59 | 0.150 | 379.4 |
| ensemble+judge | on | 192 | 0.22 | 0.30 | 0.62 | 0.00 | 2.59 | 0.150 | 379.4 |
| full | off | 192 | 0.23 | 0.27 | 0.63 | — | 2.17 | 0.212 | 527.7 |
| full | on | 192 | 0.23 | 0.27 | 0.63 | 0.00 | 2.17 | 0.212 | 527.7 |

## Findings by severity (full arm, citations on)

| severity | findings | matched |
|---|---|---|
| CRITICAL | 17 | 6 |
| HIGH | 125 | 29 |
| LOW | 87 | 21 |
| MEDIUM | 187 | 41 |

## Limitations

- Diff-only: no local checkout per PR, so context, blast-radius, unwired-export and linter passes are skipped.
- Ground truth is human review comments; a real defect humans did not comment on counts against precision.
- Intent pass sees the PR title only (PullRequest carries no body/commits in the headless path).
- SWE-CARE contains no cosmetic/silent PRs, so the silence rate is not measured.
- The full arm replays production order (citation validation, then refutation); the other arms are scored raw and again with validation applied afterwards.
- Reference comments without a line number are excluded from the recall denominator; they still count for file-level matching.
- Head-file line counts were fetched live only for paths the ensemble+judge verdict needed, so seat arms may keep an out-of-hunk citation as unverified where the full arm would drop it; seat-arm fabrication rates are therefore lower bounds.
- Evaluation seat 2 is openai/gpt-5.4-mini (reasoning effort medium) instead of the production lineup's qwen/qwen3.8-max, which reasons without bound (~7 min and $0.10 per PR).
- Instances repaired after the live run (missing seats re-issued) had their judge and skeptic calls re-issued offline against the rebuilt verdict; those calls are recorded and replayed like live ones but were not part of the original review's wall time. Their live verdicts were produced with a seat missing, so they no longer match the replay and are what the live/replay drift count measures.
- PRs excluded from every arm: no line-anchored reference comments 3.
- Arm-instance pairs excluded: replay_miss 21, error 18 (each PR counts once per affected arm).

## Reproduce

```
prime-review eval run --count 200 --seed 0
prime-review eval score --run-id swecare-200-s0
prime-review eval report --run-id swecare-200-s0
```
