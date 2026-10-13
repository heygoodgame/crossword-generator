# editorial-v1 — jev-1.13.0

Cases: 85; successful requests: 85; errors: 0; threshold: 0.5.

Labels are source-specific references, not automatically ground truth. Historical feedback never enters model state. Unlabeled cases contribute only to flag counts; a missing label is never treated as clean. Threshold sweeps are exploratory, not held-out validation. Repeat metrics use one mean probability per case.

Label sources: `{'agent-reviewed-historical': 85}`

Estimated cost: 0.002966544; latency p50/p95: 0.3225/0.4963 seconds.

| Question | Scored | Flagged | Labels (+/-) | Accuracy | Recall | FPR | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| current_state | 85 | 5 | 2/83 | 0.965 | 1.000 | 0.036 | 0.964 |
| number_mismatch | 85 | 0 | 0/85 | 1.000 | — | 0.000 | — |
| tense_mismatch | 85 | 0 | 0/85 | 1.000 | — | 0.000 | — |
| part_of_speech_mismatch | 85 | 10 | 0/85 | 0.882 | — | 0.118 | — |
| firearms_angle | 85 | 1 | 0/84 | 1.000 | — | 0.000 | — |
| alcohol_angle | 85 | 1 | 1/83 | 1.000 | 1.000 | 0.000 | 1.000 |
| drug_use_angle | 85 | 0 | 0/85 | 1.000 | — | 0.000 | — |
| plain_definition | 85 | 29 | 35/43 | 0.756 | 0.629 | 0.140 | 0.838 |
| specialist_knowledge | 85 | 11 | 0/0 | — | — | — | — |
| unnatural_wording | 85 | 1 | 0/0 | — | — | — | — |
| hint_easier | 0 | 0 | 0/0 | — | — | — | — |
| hint_new_angle | 0 | 0 | 0/0 | — | — | — | — |
| same_clue_angle | 0 | 0 | 0/0 | — | — | — | — |

See `review.csv` for all predictions and label disagreements.
