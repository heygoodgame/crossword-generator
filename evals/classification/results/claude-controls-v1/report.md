# editorial-v1 — claude-sonnet-5

Cases: 68; successful requests: 67; errors: 1; threshold: 0.5.

Labels are source-specific references, not automatically ground truth. Historical feedback never enters model state. Unlabeled cases contribute only to flag counts; a missing label is never treated as clean. Threshold sweeps are exploratory, not held-out validation. Repeat metrics use one mean probability per case.

Label sources: `{'agent-authored-control': 68}`

Estimated cost: None; latency p50/p95: 4.0101/7.1602 seconds.

| Question | Scored | Flagged | Labels (+/-) | Accuracy | Recall | FPR | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| current_state | 67 | 3 | 3/3 | 1.000 | 1.000 | 0.000 | 1.000 |
| number_mismatch | 67 | 4 | 3/5 | 1.000 | 1.000 | 0.000 | 1.000 |
| tense_mismatch | 67 | 2 | 2/3 | 1.000 | 1.000 | 0.000 | 1.000 |
| part_of_speech_mismatch | 67 | 1 | 2/4 | 0.667 | 0.000 | 0.000 | 1.000 |
| firearms_angle | 67 | 3 | 3/3 | 1.000 | 1.000 | 0.000 | 1.000 |
| alcohol_angle | 67 | 3 | 3/3 | 1.000 | 1.000 | 0.000 | 1.000 |
| drug_use_angle | 67 | 2 | 2/4 | 1.000 | 1.000 | 0.000 | 1.000 |
| plain_definition | 67 | 51 | 3/3 | 0.833 | 1.000 | 0.333 | 1.000 |
| specialist_knowledge | 67 | 6 | 3/3 | 0.833 | 0.667 | 0.000 | 1.000 |
| unnatural_wording | 67 | 1 | 2/2 | 0.750 | 0.500 | 0.000 | 1.000 |
| hint_easier | 4 | 2 | 2/2 | 1.000 | 1.000 | 0.000 | 1.000 |
| hint_new_angle | 4 | 2 | 2/2 | 1.000 | 1.000 | 0.000 | 1.000 |
| same_clue_angle | 4 | 2 | 2/2 | 1.000 | 1.000 | 0.000 | 1.000 |

See `review.csv` for all predictions and label disagreements.
