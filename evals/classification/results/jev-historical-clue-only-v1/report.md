# content-clue-only-v1 — jev-1.13.0

Cases: 500; successful requests: 500; errors: 0; threshold: 0.5.

Labels are source-specific references, not automatically ground truth. Historical feedback never enters model state. Unlabeled cases contribute only to flag counts; a missing label is never treated as clean. Threshold sweeps are exploratory, not held-out validation. Repeat metrics use one mean probability per case.

Label sources: `{'historical-model-unverified': 500}`

Estimated cost: 0.010384416; latency p50/p95: 0.333/0.4916 seconds.

| Question | Scored | Flagged | Labels (+/-) | Accuracy | Recall | FPR | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| current_state | 500 | 57 | 0/0 | — | — | — | — |
| firearms_angle | 500 | 1 | 0/0 | — | — | — | — |
| alcohol_angle | 500 | 8 | 0/0 | — | — | — | — |
| drug_use_angle | 500 | 1 | 0/0 | — | — | — | — |

See `review.csv` for all predictions and label disagreements.
