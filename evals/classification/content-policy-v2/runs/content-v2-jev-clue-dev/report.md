# content-policy-v2-clue-only — jev-1.13.0

Cases: 68; successful requests: 68; errors: 0; threshold: 0.5.

Labels are source-specific references, not automatically ground truth. Historical feedback never enters model state. Unlabeled cases contribute only to flag counts; a missing label is never treated as clean. Threshold sweeps are exploratory, not held-out validation. Repeat metrics use one mean probability per case.

Label sources: `{'agent-authored-policy-case': 60, 'agent-reviewed-policy-history': 8}`

Estimated cost: 0.0016482900000000002; latency p50/p95: 0.3236/0.5025 seconds.

| Question | Scored | Flagged | Labels (+/-) | Accuracy | Recall | FPR | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| firearms_angle | 68 | 9 | 10/58 | 0.985 | 0.900 | 0.000 | 1.000 |
| alcohol_angle | 68 | 7 | 7/61 | 1.000 | 1.000 | 0.000 | 1.000 |
| drug_use_angle | 68 | 13 | 12/56 | 0.985 | 1.000 | 0.018 | 1.000 |

See `review.csv` for all predictions and label disagreements.
