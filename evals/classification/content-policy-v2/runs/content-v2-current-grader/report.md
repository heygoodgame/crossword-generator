# content-policy-v2-current-grader — claude-sonnet-5

Cases: 200; successful requests: 200; errors: 0; threshold: 0.5.

Labels are source-specific references, not automatically ground truth. Historical feedback never enters model state. Unlabeled cases contribute only to flag counts; a missing label is never treated as clean. Threshold sweeps are exploratory, not held-out validation. Repeat metrics use one mean probability per case.

Label sources: `{'agent-authored-policy-case': 160, 'agent-reviewed-policy-history': 40}`

Estimated cost: 0.8020482; latency p50/p95: 3.3382/9.0799 seconds.

| Question | Scored | Flagged | Labels (+/-) | Accuracy | Recall | FPR | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| fairness_repair | 200 | 91 | 0/0 | — | — | — | — |

See `review.csv` for all predictions and label disagreements.
