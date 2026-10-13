# editorial-v1 — jev-1.13.0

Cases: 500; successful requests: 500; errors: 0; threshold: 0.5.

Labels are source-specific references, not automatically ground truth. Historical feedback never enters model state. Unlabeled cases contribute only to flag counts; a missing label is never treated as clean. Threshold sweeps are exploratory, not held-out validation. Repeat metrics use one mean probability per case.

Label sources: `{'historical-model-unverified': 500}`

Estimated cost: 0.01745772; latency p50/p95: 0.32805/0.4556 seconds.

| Question | Scored | Flagged | Labels (+/-) | Accuracy | Recall | FPR | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| current_state | 500 | 24 | 0/0 | — | — | — | — |
| number_mismatch | 500 | 0 | 0/0 | — | — | — | — |
| tense_mismatch | 500 | 2 | 0/0 | — | — | — | — |
| part_of_speech_mismatch | 500 | 39 | 0/0 | — | — | — | — |
| firearms_angle | 500 | 4 | 0/0 | — | — | — | — |
| alcohol_angle | 500 | 9 | 0/0 | — | — | — | — |
| drug_use_angle | 500 | 0 | 0/0 | — | — | — | — |
| plain_definition | 500 | 222 | 0/0 | — | — | — | — |
| specialist_knowledge | 500 | 56 | 0/0 | — | — | — | — |
| unnatural_wording | 500 | 10 | 0/0 | — | — | — | — |
| hint_easier | 0 | 0 | 0/0 | — | — | — | — |
| hint_new_angle | 0 | 0 | 0/0 | — | — | — | — |
| same_clue_angle | 0 | 0 | 0/0 | — | — | — | — |

See `review.csv` for all predictions and label disagreements.
