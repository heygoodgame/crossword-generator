# editorial-v1 — claude-sonnet-5

Cases: 1; successful requests: 1; errors: 0; threshold: 0.5.

Labels are source-specific references, not automatically ground truth. Historical feedback never enters model state. Unlabeled cases contribute only to flag counts; a missing label is never treated as clean. Threshold sweeps are exploratory, not held-out validation. Repeat metrics use one mean probability per case.

Label sources: `{'agent-authored-control': 1}`

Estimated cost: 0.008368; latency p50/p95: 7.07/7.07 seconds.

| Question | Scored | Flagged | Labels (+/-) | Accuracy | Recall | FPR | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| current_state | 1 | 0 | 0/0 | — | — | — | — |
| number_mismatch | 1 | 0 | 0/0 | — | — | — | — |
| tense_mismatch | 1 | 1 | 1/0 | 1.000 | 1.000 | — | — |
| part_of_speech_mismatch | 1 | 0 | 0/0 | — | — | — | — |
| firearms_angle | 1 | 0 | 0/0 | — | — | — | — |
| alcohol_angle | 1 | 0 | 0/0 | — | — | — | — |
| drug_use_angle | 1 | 0 | 0/0 | — | — | — | — |
| plain_definition | 1 | 1 | 0/0 | — | — | — | — |
| specialist_knowledge | 1 | 0 | 0/0 | — | — | — | — |
| unnatural_wording | 1 | 0 | 0/0 | — | — | — | — |
| hint_easier | 0 | 0 | 0/0 | — | — | — | — |
| hint_new_angle | 0 | 0 | 0/0 | — | — | — | — |
| same_clue_angle | 0 | 0 | 0/0 | — | — | — | — |

See `review.csv` for all predictions and label disagreements.
