# Content policy v2: measured results

Thresholds and state were selected using only the 68 development cases.
Test set: 132 cases; four ambiguous category labels excluded where applicable.

| Check | State | Threshold | TP | FN | FP | TN |
|---|---|---:|---:|---:|---:|---:|
| firearms_angle | context | 0.78 | 17 | 0 | 0 | 113 |
| alcohol_angle | clue-only | 0.82 | 16 | 4 | 0 | 110 |
| drug_use_angle | context | 0.95 | 14 | 1 | 0 | 117 |

Combined: 47/52 violations caught; 0/76 content-clean cases flagged.

## Missed violations

- SAUCE / Liquor, in the phrase on the ___: alcohol_angle = 0.72 (threshold 0.82).
- MALT / Grain prepared for making whisky: alcohol_angle = 0.72 (threshold 0.82).
- AGE / Mature gracefully, like fine wine: alcohol_angle = 0.55 (threshold 0.82).
- BARS / Where a bartender works: alcohol_angle = 0.62 (threshold 0.82).
- USER / Person who takes illicit drugs: drug_use_angle = 0.92 (threshold 0.95).

## Existing grader overlap

Raw fairness <15 flagged 45/52 content-positive test cases. Jev added 5 positive flags.
Combined: 50/52 content-positive test cases flagged.

It also flagged 14/76 content-negative cases for fairness. These may be valid flags for other defects; they are not automatically content false positives.

Raw fairness <15 also flags leaks and other defects. Content-negative does not mean the overall clue is clean. No crossings; Easy Mini rubric, one clue per call.

## Blind Claude classifier baseline at 0.5

Claude's probabilities were not threshold-tuned; this is a reference, not an equal-operating-point ranking.

| Check | TP | FN | FP | TN |
|---|---:|---:|---:|---:|
| firearms_angle | 17 | 0 | 1 | 112 |
| alcohol_angle | 20 | 0 | 2 | 108 |
| drug_use_angle | 15 | 0 | 1 | 116 |

Blind Claude review disagreed with 5 of the 596 known category labels across the full corpus. Original labels were retained, not revised to match Claude or Jev. See baseline-comparison.json.

Labels are provisional agent judgments. Cases are deliberately enriched with boundary examples and are not a production-prevalence sample. Historical cases were explored previously. No production decision policy was changed.
