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
  and do not print it.
- Keep generated records in `status=draft` with `metadata.review_status=unreviewed`.
- If replacing an already-uploaded generated candidate, use `--replace-existing`
  with the same deterministic batch/difficulty/size/seed key.
- Before uploading, run a dry run and scan answers for known disallowed
  patterns.

## Core Workflow

Read [references/generator-workflow.md](references/generator-workflow.md) before
doing non-trivial work. It documents:

- Pipeline architecture and ownership map
- Config and dictionary conventions
- Single-puzzle and batch commands
- Data-store upload contract and replacement workflow
- Quality guardrails, including terminal-S duplicate variants
- Verification commands and common failure modes

## Common Commands

Generate a clean Easy batch using the prevalent 8/9-letter Easy dictionary:

```bash
uv run crossword-generator generate-pilot-batch \
  --output-root output/batches/<batch-id> \
  --batch-id <batch-id> \
  --buckets easy/5,easy/7,easy/9 \
  --count 5 \
  --seed-start 1 \
  --llm claude
```

Upload generated candidates:

```bash
export HEYGG_API_BASE_URL=https://id-beta.hey.gg/api
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
