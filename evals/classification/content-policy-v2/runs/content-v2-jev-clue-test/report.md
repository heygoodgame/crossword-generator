# content-policy-v2-clue-only — jev-1.13.0

Cases: 132; successful requests: 132; errors: 0; threshold: 0.5.

Labels are source-specific references, not automatically ground truth. Historical feedback never enters model state. Unlabeled cases contribute only to flag counts; a missing label is never treated as clean. Threshold sweeps are exploratory, not held-out validation. Repeat metrics use one mean probability per case.

Label sources: `{'agent-authored-policy-case': 100, 'agent-reviewed-policy-history': 32}`

Estimated cost: 0.003199182; latency p50/p95: 0.32184999999999997/0.4179 seconds.

| Question | Scored | Flagged | Labels (+/-) | Accuracy | Recall | FPR | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| firearms_angle | 132 | 18 | 17/113 | 0.985 | 0.941 | 0.009 | 0.998 |
| alcohol_angle | 132 | 22 | 20/110 | 0.992 | 1.000 | 0.009 | 0.999 |
| drug_use_angle | 132 | 21 | 15/117 | 0.955 | 1.000 | 0.051 | 1.000 |

See `review.csv` for all predictions and label disagreements.
