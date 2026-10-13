# content-policy-v2-context — claude-sonnet-5

Cases: 200; successful requests: 200; errors: 0; threshold: 0.5.

Labels are source-specific references, not automatically ground truth. Historical feedback never enters model state. Unlabeled cases contribute only to flag counts; a missing label is never treated as clean. Threshold sweeps are exploratory, not held-out validation. Repeat metrics use one mean probability per case.

Label sources: `{'agent-authored-policy-case': 160, 'agent-reviewed-policy-history': 40}`

Estimated cost: 0.37159; latency p50/p95: 1.34175/2.5585 seconds.

| Question | Scored | Flagged | Labels (+/-) | Accuracy | Recall | FPR | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| firearms_angle | 200 | 29 | 27/171 | 0.995 | 1.000 | 0.006 | 0.999 |
| alcohol_angle | 200 | 31 | 27/171 | 0.985 | 1.000 | 0.018 | 0.997 |
| drug_use_angle | 200 | 28 | 27/173 | 0.995 | 1.000 | 0.006 | 1.000 |

See `review.csv` for all predictions and label disagreements.
