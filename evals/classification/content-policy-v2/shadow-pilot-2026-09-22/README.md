# First real generation shadow pilot

Four locally generated 5x5 puzzles: two Easy and two Hard. All completed without
pipeline errors. Claude generated, graded, fact-checked, repaired, named, and wrote
hints; Jev only observed main-clue revisions. No uploads or scheduling occurred.

| Measure | Result |
|---|---:|
| Final clues | 40 |
| Distinct clue revisions observed | 49 |
| Jev requests | 98 |
| Request errors / incomplete predictions | 0 / 0 |
| Jev threshold flags | 0 |

The selected state/threshold configuration is unchanged from the preceding
development/test evaluation. This sample provides no observed incremental catches.

**Open editorial boundary:** NORM / `Barstool regular on "Cheers"` received 0.56
for alcohol, below 0.82. The grader gave fairness 21/25 with no feedback. The review
CSV contains an explicitly labeled agent-triage note; no editor verdict is assumed.
This could expose a recall limitation if bar-setting references are disallowed.

[Review CSV](content-shadow-review.csv) · [Summary](content-shadow-summary.json) ·
[Manifest](manifest.json) · [Observer documentation](../../../../docs/jev-content-shadow.md)

All four raw sidecar JSONs and final IPUZ files are preserved in the same relative
layout as the original run. The manifest retains original output/log paths for
provenance; detailed generation logs remain under
`output/classification-evals/shadow-pilot-2026-09-22/`. Regenerate this archive's
report with `uv run python -m crossword_generator.content_shadow` followed by this
directory. It resolves shadow artifacts relative to the copied manifest.

The original invocation used a randomly selected seed start of **461288725**.
For a comparable new experiment (a new output directory avoids overwriting notes):

```bash
uv run crossword-generator generate-pilot-batch \
  --output-root output/classification-evals/shadow-pilot-NEXT \
  --batch-id jev-shadow-pilot-NEXT --buckets easy/5,hard/5 --count 2 \
  --llm claude --jev-content-shadow --max-workers 2 \
  --no-refresh-dictionaries --no-avoid-existing-clues \
  --no-exclude-recent-answers --no-exclude-scheduled-sixty \
  --no-intra-batch-dedup --no-unlimited-answer-novelty --no-open-day-guard
```

Those exclusion/history flags are specifically for a local unscheduled experiment.
Use the normal batch rules for candidates intended for production. Model sampling
means reusing the seed does not reproduce identical clues.
