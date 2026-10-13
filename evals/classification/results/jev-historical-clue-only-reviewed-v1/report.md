# content-clue-only-v1 — jev-1.13.0

Cases: 85; successful requests: 85; errors: 0; threshold: 0.5.

Labels are source-specific references, not automatically ground truth. Historical feedback never enters model state. Unlabeled cases contribute only to flag counts; a missing label is never treated as clean. Threshold sweeps are exploratory, not held-out validation. Repeat metrics use one mean probability per case.

Label sources: `{'agent-reviewed-historical': 85}`

Estimated cost: 0.0017644620000000003; latency p50/p95: 0.3395/0.4916 seconds.

| Question | Scored | Flagged | Labels (+/-) | Accuracy | Recall | FPR | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| current_state | 85 | 13 | 2/83 | 0.871 | 1.000 | 0.133 | 0.934 |
| firearms_angle | 85 | 0 | 0/84 | 1.000 | — | 0.000 | — |
| alcohol_angle | 85 | 1 | 1/83 | 1.000 | 1.000 | 0.000 | 1.000 |
| drug_use_angle | 85 | 0 | 0/85 | 1.000 | — | 0.000 | — |

See `review.csv` for all predictions and label disagreements.
