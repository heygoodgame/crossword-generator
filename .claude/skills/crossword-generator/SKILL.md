---
name: crossword-generator
description: Use when working in this repo to generate crossword puzzle batches, update dictionaries/configs, validate fill and clue quality, prepare or upload generated puzzles to the HeyGG admin data store, replace existing generated records, or debug generator pipeline behavior.
---

# Crossword Generator

Use this skill for any task in this repository involving crossword generation,
dictionary preparation, fill/clue quality, batch manifests, or generated-puzzle
uploads.

## Operating Rules

- Keep generated-puzzle uploads on the authenticated HeyGG admin API. Do not
  write directly to the `hey-you` database.
- Use `uv run crossword-generator ...` for CLI commands.
- Use Claude for production-quality clue generation unless explicitly testing
  Ollama: pass `--llm claude`.
- Treat tokens as secrets. Pass `HEYGG_ADMIN_API_TOKEN` through the environment
  and do not print it. To obtain or refresh a token — or on a `401`/`403` —
  use the `heygg-admin-auth` skill / `hgg-auth` (pick the right profile, e.g.
  `hgg-auth exec prod -- ...`); never copy JWTs from DevTools.
- Keep generated records in `status=draft` with `metadata.review_status=unreviewed`.
- If replacing an already-uploaded generated candidate, use `--replace-existing`
  with the same deterministic batch/difficulty/size/seed key.
- Before uploading, run a dry run and scan answers for known disallowed
  patterns.
- When asked for a generated batch across Mini Crossword and Midi Crossword
  without explicit size counts, default to a rough 5:2:7 ratio for 5x5, 7x7,
  and 9x9 puzzles. Midi Crossword always uses 9x9; Mini Crossword dailies are
  five 5x5 puzzles and two 7x7 puzzles per week. Use `--bucket-counts` for this
  instead of equal per-bucket `--count`.

## Core Workflow

Read [references/generator-workflow.md](references/generator-workflow.md) before
doing non-trivial work. It documents:

- Pipeline architecture and ownership map
- Config and dictionary conventions
- Word list management end-to-end (UI → consolidate-list → committed .txt → generator)
- Single-puzzle and batch commands
- Data-store upload contract and replacement workflow
- Quality guardrails, including terminal-S duplicate variants
- Verification commands and common failure modes

## Common Commands

Generate a clean cross-site Easy batch using the default 5x5:7x7:9x9 ratio:

```bash
uv run crossword-generator generate-pilot-batch \
  --output-root output/batches/<batch-id> \
  --batch-id <batch-id> \
  --buckets easy/5,easy/7,easy/9 \
  --bucket-counts 5=5,7=2,9=7 \
  --seed-start 1 \
  --llm claude
```

Generate a full Easy/Hard batch using the same cross-site size ratio:

```bash
uv run crossword-generator generate-pilot-batch \
  --output-root output/batches/<batch-id> \
  --batch-id <batch-id> \
  --bucket-counts 5=5,7=2,9=7 \
  --seed-start 1 \
  --llm claude
```

Upload generated candidates:

```bash
export HEYGG_API_BASE_URL=https://play.hey.gg/api
export HEYGG_ADMIN_API_TOKEN=<token>

uv run crossword-generator save-generated-puzzles \
  --manifest output/batches/<batch-id>/manifest.json \
  --dry-run

uv run crossword-generator save-generated-puzzles \
  --manifest output/batches/<batch-id>/manifest.json
```

Replace an existing uploaded candidate:

```bash
uv run crossword-generator save-generated-puzzles \
  --manifest output/batches/<replacement-batch>/manifest.json \
  --replace-existing
```

Focused validation:

```bash
uv run pytest tests/test_fill_grader.py tests/test_fill_with_grading_step.py -q
uv run pytest tests/test_dictionary_prep.py tests/test_config.py tests/test_data_store.py tests/test_cli_batch.py -q
uv run ruff check src/crossword_generator/ tests/
```
