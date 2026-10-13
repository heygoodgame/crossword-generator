# Content-policy experiment, September 22, 2026

**Jev looks useful as an additional content check.** On the held-out portion,
the frozen configuration flagged 47/52 labeled violations and 0/76 content-clean
cases. It added five alcohol flags missed by the existing grading rubric;
together they caught 50/52. Keep the broad grader: Jev missed five violations
on its own. This experiment changes no production behavior.

See [measured results](results.md), [every test prediction](test-results.json),
and [grader comparison and label disagreements](baseline-comparison.json).

Follow-up: [real generation shadow pilot](shadow-pilot-2026-09-22/README.md) and
[enabling the observer](../../../docs/jev-content-shadow.md).

## Protocol and limitations

- **200 fixed cases:** 150 positive/negative cases in `authored.tsv`, ten additional
  blank/medical boundary cases, and 40 selected historical clues. `build.py`
  reproduces them from that TSV and the preserved historical corpus.
- **Labels:** drafted and reviewed by the coding agent before inference. Claude
  received only answer/clue and the frozen questions as a blind second opinion.
  It disagreed with five of 596 known category labels; none were changed to agree
  with either model. This is model-assisted review, **not independent human gold**.
  Four ambiguous historical labels are null and excluded from their metrics.
- **Scope:** firearms/ammunition, alcohol, and recreational/illicit drug angles.
  Non-firearm violence and other editorial problems are outside these three
  labels. Ordinary prescribed medical treatment is excluded. A content-negative
  clue can still have leaks, factual errors, or inappropriate wording.
- **Split:** deterministic answer grouping, keeping paired senses and plural
  variants together: 68 development, 132 test. This blocks exact-answer overlap,
  not all shared language or semantic themes. Historical examples were explored
  previously; the authored cases were designed from known failure categories.
  This is a diagnostic stress test, not a representative production sample.
- **State ablation:** identical instructions and question count in both suites;
  only the `answer` field differs. One sample per case per model/state.
- **Threshold selection:** maximize development recall subject to empirical
  false-positive rate <=2%, per category. Ties favor lower FPR, then higher
  threshold, then the first variant. Threshold 1.01 means a disabled gate.
  State and thresholds were frozen in `selection.json` before Jev test calls.
  Development positives are only 10 firearms / 7 alcohol / 12 drugs. The resulting
  thresholds are provisional; 0/76 clean-case flags does not establish a population
  false-positive rate below 2%. No thresholds were revised after seeing test results.
- **Claude baseline:** `claude-sonnet-5`, same three questions with answer/clue,
  temperature 0; its default 0.5 threshold was not tuned. Do not read this as a
  calibrated model ranking.
- **Current grader:** separate 200-call replay using the actual
  `build_clue_evaluation_messages` prompt and configured grading model,
  `claude-sonnet-5`. Easy Mini context, no crossings, one clue per call. Flags use
  raw fairness <15, matching the default repair threshold. This does not replay
  full puzzle context, deterministic penalties, all repair triggers, or repair
  generations. Its 14 fairness flags on content-negative test cases often concern
  valid leaks/other defects and are not counted as content false positives.

## What the gain looks like

The five additional flags were all authored alcohol cases:

| Answer | Clue | Existing raw fairness / 25 |
|---|---|---:|
| ALE | Pub drink served by the pint | 20 |
| BREW | Make beer by fermentation | 24 |
| AGED | Matured in oak, as whisky | 22 |
| BOURBON | Whiskey associated with Kentucky | 23 |
| RUM | Spirit used in a mojito | 22 |

Jev's conservative thresholds missed SAUCE/liquor, MALT/whisky, AGE/wine,
BARS/bartender, and USER/illicit drugs. The combination still missed MALT and AGE.
The next useful validation is a shadow run on actual generation batches, recording
the marginal flags and editor decisions before enabling automatic repairs.

## Reproduce and extend

```bash
# Deterministic corpus/suites. Preserve originals when starting a new version.
uv run python evals/classification/content-policy-v2/build.py

# Example development run; use a fresh output directory for new model versions.
uv run python -m crossword_generator.classification_eval run \
  --corpus evals/classification/content-policy-v2/dev.json \
  --suite evals/classification/content-policy-v2/suite-context.json \
  --model evals/classification/models/jev-1.13.json \
  --output output/classification-evals/content-v2-jev-context-dev --workers 4
```

Repeat with `suite-clue-only.json` into `content-v2-jev-clue-dev`, then freeze
selection **before** any test inference:

```bash
uv run python evals/classification/content-policy-v2/analyze.py fit \
  --runs-root output/classification-evals
```

Run both variants on `test.json` into corresponding `*-test` directories. Run
the classifier baseline on `corpus.json` with `claude-example.json` and context
suite into `content-v2-claude`. Run the production-rubric adapter with
`model-current-grader.json` and `suite-current-grader.json` into
`content-v2-current-grader`. The adapter records source/rubric hashes in its config;
refresh those hashes if versioning its implementation.

```bash
uv run python evals/classification/content-policy-v2/analyze.py report \
  --runs-root output/classification-evals

# Recreate reports entirely offline from the included archives:
uv run python evals/classification/content-policy-v2/analyze.py report
```

`fit` refuses to overwrite a different existing selection. A future experiment
should have a new version/directory, new held-out cases, and independent review.
The reusable calibration module rejects incomplete runs, non-development inputs
for fitting, overlapping dev/test groups, and changed model/question configurations.

All six scored runs are archived under `runs/`, including raw responses, usage,
latency, model IDs, corpus, and question manifests. No credentials are stored.
Live JSONL and the one-case connection probe remain under the ignored
`output/classification-evals/`. The probe is excluded from scored results.

## Observed runtime and estimated cost

All 800 scored requests returned valid results: 400 Jev calls across two state
variants, 200 Claude classifier calls, and 200 current-grader calls.

| Run | Median seconds per request | Estimated USD |
|---|---:|---:|
| Jev, all development/test variants | 0.31–0.32 | 0.00977 |
| Claude focused classifier | 1.34 | 0.37159 |
| Existing grading rubric replay | 3.34 | 0.80205 |

Costs use the frozen Jev configuration's September 18 token rates and the repo
provider's reported estimates for Claude, not billing receipts. The connection
probe is excluded. A combined Jev policy requires both state variants; its per-clue
latency depends on whether the two requests run concurrently. The grader replay
uses individual calls, so these figures are not production batch speedup estimates.
