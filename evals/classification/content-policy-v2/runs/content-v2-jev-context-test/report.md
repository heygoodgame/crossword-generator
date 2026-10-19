# content-policy-v2-context — jev-1.13.0

Cases: 132; successful requests: 132; errors: 0; threshold: 0.5.

Labels are source-specific references, not automatically ground truth. Historical feedback never enters model state. Unlabeled cases contribute only to flag counts; a missing label is never treated as clean. Threshold sweeps are exploratory, not held-out validation. Repeat metrics use one mean probability per case.

Label sources: `{'agent-authored-policy-case': 100, 'agent-reviewed-policy-history': 32}`

Estimated cost: 0.003249078; latency p50/p95: 0.324/0.4206 seconds.

| Question | Scored | Flagged | Labels (+/-) | Accuracy | Recall | FPR | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| firearms_angle | 132 | 18 | 17/113 | 0.992 | 1.000 | 0.009 | 1.000 |
| alcohol_angle | 132 | 22 | 20/110 | 0.969 | 0.950 | 0.027 | 0.993 |
| drug_use_angle | 132 | 19 | 15/117 | 0.970 | 1.000 | 0.034 | 1.000 |

See `review.csv` for all predictions and label disagreements.
