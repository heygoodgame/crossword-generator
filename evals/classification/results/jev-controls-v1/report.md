# editorial-v1 — jev-1.13.0

Cases: 68; successful requests: 204; errors: 0; threshold: 0.5.

Labels are source-specific references, not automatically ground truth. Historical feedback never enters model state. Unlabeled cases contribute only to flag counts; a missing label is never treated as clean. Threshold sweeps are exploratory, not held-out validation. Repeat metrics use one mean probability per case.

Label sources: `{'agent-authored-control': 68}`

Estimated cost: 0.00719271; latency p50/p95: 0.3307/0.4735 seconds.

| Question | Scored | Flagged | Labels (+/-) | Accuracy | Recall | FPR | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| current_state | 68 | 2 | 3/3 | 0.833 | 0.667 | 0.000 | 1.000 |
| number_mismatch | 68 | 2 | 3/5 | 0.875 | 0.667 | 0.000 | 1.000 |
| tense_mismatch | 68 | 0 | 3/3 | 0.500 | 0.000 | 0.000 | 1.000 |
| part_of_speech_mismatch | 68 | 7 | 2/4 | 0.667 | 0.500 | 0.250 | 0.625 |
| firearms_angle | 68 | 3 | 3/3 | 1.000 | 1.000 | 0.000 | 1.000 |
| alcohol_angle | 68 | 3 | 3/3 | 1.000 | 1.000 | 0.000 | 1.000 |
| drug_use_angle | 68 | 3 | 2/4 | 1.000 | 1.000 | 0.000 | 1.000 |
| plain_definition | 68 | 36 | 3/3 | 1.000 | 1.000 | 0.000 | 1.000 |
| specialist_knowledge | 68 | 11 | 3/3 | 1.000 | 1.000 | 0.000 | 1.000 |
| unnatural_wording | 68 | 2 | 2/2 | 0.750 | 0.500 | 0.000 | 1.000 |
| hint_easier | 4 | 2 | 2/2 | 1.000 | 1.000 | 0.000 | 1.000 |
| hint_new_angle | 4 | 3 | 2/2 | 0.750 | 1.000 | 0.500 | 1.000 |
| same_clue_angle | 4 | 2 | 2/2 | 1.000 | 1.000 | 0.000 | 1.000 |

See `review.csv` for all predictions and label disagreements.
