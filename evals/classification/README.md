# Classification model evaluations

Offline experiments, separate from production grading and repair. Compares binary
probability judgments on frozen historical inputs, with versioned question suites
and configurable model adapters. No production clues or records are changed.

Follow-up: [200-case content-policy evaluation](content-policy-v2/README.md), with
development-only state/threshold selection and a held-out grader comparison.

## Run

```bash
uv run python -m crossword_generator.classification_eval run \
  --corpus evals/classification/controls.json \
  --suite evals/classification/suites/editorial-v1.json \
  --model evals/classification/models/jev-1.13.json \
  --output output/classification-evals/my-jev-run --workers 4 --repeats 3
```

Use `TYPESAFE_API_KEY` from the environment. `--dry-run` validates without API
calls. `--limit N --seed NAME` selects a reproducible hash-ordered sample.
Each request contains one clue and all applicable questions. Missing `hint` or
`previous_clue` fields skip dependent questions; they do not produce false labels.

Rerun the identical command to resume successes and retry failures. Use a new
output directory if inputs or evaluator source changes. Requests flush immediately;
there are no hidden retries in the Jev adapter (existing generative providers retain
their retry policies). A `.running` lock prevents concurrent writers. After a killed
process, confirm it is stopped before removing a stale lock. Malformed result-file
tails are rejected rather than silently discarded.

## Preserved data and label caveats

- `historical-shadow.json`: original 500 cases and seven legacy Jev probabilities.
  Historical scores, feedback, and `opus_defect` are under `metadata.reference`,
  **not labels**. The old field name does not prove the original model: those
  per-row model IDs were not preserved. Some feedback invents nonexistent leaks.
- `historical-reviewed.json`: 85 unchanged historical states with narrow labels
  reviewed by the coding agent before the historical Jev run. Selected, mostly
  negative examples; not human gold or a representative accuracy benchmark.
- `controls.json`: 68 agent-authored diagnostic cases, labeled only for their target
  questions. Not an unbiased test set. The questions and controls were authored
  together, so strong performance requires independent validation.
- `legacy/`: preserved final/structured leak prompts, 360 leak cases, and the prior
  minimal-state results. The original morpheme-label builder was not saved;
  `mech` remains legacy evidence, not a newly verified oracle. The saved results
  predate `q_final.json` and must not be attributed to that later tuned prompt.
- `suites/legacy-shadow-v2.json`: the exact earlier questions, including flawed
  examples, retained for reproducing the negative result.
- `suites/legacy-leak-{minimal,full}.json`: reusable state variants of the saved
  final leak question. Use the same imported leak corpus for each variant.

Fixtures use `.json` because this repo ignores JSONL globally. They are now outside
`/private/tmp` and ready for version control. All original scratch files also remain
locally in `output/classification-evals/imported/`.

Canonical corpus:

```json
[
  {
    "id": "stable-id",
    "state": {"answer": "CATS", "clue": "A household feline"},
    "labels": {"number_mismatch": true},
    "metadata": {"label_source": "human-reviewed", "split": "test"}
  }
]
```

Only suite-allowlisted state fields enter inference. Labels and metadata never do.
Missing/null labels mean unknown, not clean. Do not infer negative category labels
from the absence of a complaint in historical model feedback.

## Rebuild historical inputs

```bash
uv run python -m crossword_generator.classification_eval import-logs \
  output/batches output/classification-evals/rebuilt.json

uv run python -m crossword_generator.classification_eval import-legacy \
  old-corpus.jsonl output/classification-evals/imported.json
```

The log importer joins inputs and grades by number/direction, supports quoted titles,
saves file/line/request/model provenance, deduplicates identical states, and preserves
all grading references. Its `.import-report.json` lists malformed or unsupported
records. It does not invent historical runtime repair decisions from scores without
the run's exact configuration, or regenerate the lost morpheme-labeling algorithm.

## New questions and models

Copy a suite, change its ID, and add entries with `required_fields` and `question`
(`type: noul`, instructions, optional criteria). Add needed fields to `state_fields`.
This version supports binary probabilities, not Choice/Score conversion.

Model configs accept an arbitrary model ID and one of:

- `typesafe`: native Nouls, configurable endpoint, credential environment variable,
  timeout, and explicitly dated token prices.
- `claude`, `openai`, `ollama`: existing repo providers, credentials, and optional
  dependencies. Provider configuration goes in `options`; per-call overrides go in
  `generation_kwargs`. Their probabilities are prompted estimates, not native Jev
  Nouls or guaranteed calibration. Examples are included for Claude and Ollama.
- `plugin`: a trusted local `factory: your_package.module:factory` receives the
  config and returns an adapter exposing `evaluate(state, questions)` with
  `{probabilities, raw, resolved_model, usage}` and optional `cost`. Adapters must
  be thread-safe, or run with `--workers 1`.

Keep API secrets in environment variables, never configs. Provider exception text
is suppressed to avoid leaking headers. Invalid generative probability responses
retain their model output for debugging. Cost is estimated from configured rates
or complete provider estimates; unknown usage, including failed calls, may still
be billed. Summary JSON reports usage coverage.

## Reports, comparison, and revised labels

Runs contain a manifest with model/suite/source hashes, frozen corpus, append-only
responses, summary JSON, Markdown report, and `review.csv`. Responses preserve the
actual resolved model, probabilities, usage, latency, and raw output.

```bash
uv run python -m crossword_generator.classification_eval report \
  output/classification-evals/my-jev-run --threshold 0.5

uv run python -m crossword_generator.classification_eval compare \
  output/classification-evals/my-jev-run \
  output/classification-evals/other-model-run \
  --output output/classification-evals/comparison.json

uv run python -m crossword_generator.classification_eval rescore \
  output/classification-evals/my-jev-run \
  --labels reviewed-corpus.json \
  --output output/classification-evals/reviewed-report

uv run python -m crossword_generator.classification_eval archive \
  output/classification-evals/my-jev-run evals/classification/results/my-jev-run
```

Comparison requires identical corpus and suite and reports missing predictions.
Different repeat counts are allowed: means are compared per case. Matching numerical
thresholds across models is not calibration. Rescoring accepts an unchanged subset
of original states with revised labels, records the parent run, and calls no API.
Its reported cost describes the reused subset, not new spending.

Archives work with `report`, `compare`, and `rescore`, but cannot be resumed for
inference. They retain every attempt and raw response in `results.json`; repeated
request objects are reconstructed from the frozen corpus and suite. A new archive
directory is required. This avoids the repository's global JSONL ignore rule.

Reports include positive/negative label counts, precision, recall, false-positive
rate, AUC with ties, Brier score, exploratory threshold sweeps, and repeat ranges.
Undefined metrics stay null. Repeats never inflate labeled sample counts. Different
label sources also receive separate metrics in JSON. For actual model selection,
split by answer or puzzle, tune on development data, then use a separate frozen test
set. Current fixtures are diagnostic; no production threshold is selected here.

See [initial findings](findings-2026-09-18.md) and compact snapshots in `results/`.
Full request/response records and review CSVs are in `output/classification-evals/`.
