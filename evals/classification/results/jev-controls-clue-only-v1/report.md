# content-clue-only-v1 — jev-1.13.0

Cases: 68; successful requests: 68; errors: 0; threshold: 0.5.

Labels are source-specific references, not automatically ground truth. Historical feedback never enters model state. Unlabeled cases contribute only to flag counts; a missing label is never treated as clean. Threshold sweeps are exploratory, not held-out validation. Repeat metrics use one mean probability per case.

Label sources: `{'agent-authored-control': 68}`

Estimated cost: 0.001406034; latency p50/p95: 0.3381/0.5161 seconds.

| Question | Scored | Flagged | Labels (+/-) | Accuracy | Recall | FPR | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| current_state | 68 | 11 | 3/3 | 0.833 | 0.667 | 0.000 | 1.000 |
| firearms_angle | 68 | 4 | 3/3 | 0.833 | 1.000 | 0.333 | 1.000 |
| alcohol_angle | 68 | 3 | 3/3 | 1.000 | 1.000 | 0.000 | 1.000 |
| drug_use_angle | 68 | 4 | 2/4 | 1.000 | 1.000 | 0.000 | 1.000 |

See `review.csv` for all predictions and label disagreements.
