# Crossword Generator Workflow Reference

## Goal

This repo produces reviewable mini and midi crossword candidates for HeyGG.
The immediate operating goal is not direct publication; it is to generate
draft IPUZ records, upload them to `crosswords/generated-puzzles`, and let the
admin/editor workflow review, edit, approve, or reject them.

Current emphasis:

- Easy puzzles should favor accessible, one-word fill.
- Easy 9x9 generation uses Jeff Chen's prevalent 8/9-letter Easy attachment
  merged with the prior Easy 3-7 list.
- 9x9 midi generation uses expanded Jeff-feedback mirror-style and
  regular-symmetry patterns with safe top-to-bottom flips and conservative
  cheater-square variants, while avoiding three-black-square perimeter runs
  that press into a corner and procedural rotational windmill patterns that
  can read as swastika-like.
- Easy 9x9 generation skips grids with more than three 8-/9-letter slots.
- Avoid unsuitable or controversial fill before clue generation and upload.
- Preserve reproducible batch manifests and deterministic data-store keys.

## Architecture

The CLI entrypoint is `crossword_generator.cli:main`.

The generation pipeline is assembled in `src/crossword_generator/pipeline.py`.
It passes a `PuzzleEnvelope` through these steps:

1. Optional theme generation, mostly off for current batch work.
2. Grid fill via `CSPFiller`.
3. Fill grading via `FillGrader`.
4. Clue generation via Ollama or Claude.
5. Clue grading and repair via the configured LLM.
6. Puzzle naming.
7. IPUZ export.

Important files:

- `src/crossword_generator/cli.py`: commands, batch runner, upload command.
- `src/crossword_generator/config.py`: YAML config model.
- `src/crossword_generator/fillers/csp.py`: CSP fill engine.
- `src/crossword_generator/steps/fill_step.py`: grid variant iteration, fill retries.
- `src/crossword_generator/graders/fill_grader.py`: fill quality and hard board-level rejections.
- `src/crossword_generator/graders/clue_grader.py`: LLM clue scoring.
- `src/crossword_generator/data_store.py`: HeyGG admin API record contract.
- `src/crossword_generator/exporters/ipuz_exporter.py`: IPUZ output.

## Configs and Dictionaries

Use committed difficulty configs:

- `config.easy.yaml`
- `config.hard.yaml`

Current Easy config points both `dictionary.path` and
`fill.csp.dictionary_path` at:

```text
dictionaries/hgg-easy-prevalent-flat-55.txt
```

Current Hard config points both paths at:

```text
dictionaries/hgg-hard-flat-55.txt
```

The hard flat dictionary is length-mixed: 3-, 4-, and 5-letter entries come
from the prepared Easy/prevalent list, while 6+ entries come from
`dictionaries/HggCuratedCrosswordList.txt`. This keeps short fill accessible
and avoids leaning on crosswordese-heavy hard-list glue.

Flat dictionaries use `WORD;55` rows and `quality_tiers: [55]`.

Easy dictionary preparation:

```bash
uv run crossword-generator prepare-dictionaries \
  --easy-source dictionaries/hgg-easy-flat-55.txt \
  --easy-extra-source dictionaries/Wordplete-PrevalentCulled-8-9-length.txt \
  --easy-exclude-source dictionaries/XwiJeffChenList-NotFamilyFriendly.txt \
  --easy-exclude-source dictionaries/Wordplete-PrevalentCulled-8-9-length-Removed.txt \
  --easy-output dictionaries/hgg-easy-prevalent-flat-55.txt \
  --hard-source dictionaries/HggCuratedCrosswordList.txt \
  --hard-output dictionaries/hgg-hard-flat-55.txt
```

By default, preparation filters true scored 7-, 8-, and 9-letter source rows
below `60` before flattening accepted entries to `;55`. Previously flattened
`WORD;55` Easy inputs are treated as flat dictionaries, not original source
scores. Use
`--long-word-min-source-score 0` only when intentionally disabling that
long-word source-score floor.

The May 14 prevalent Easy merge produced 18,593 rows after excluding 146 entries
while preserving the prior flat 3-7-letter Easy source. The May 15 hard
dictionary run produced 128,758 rows after taking short entries from Easy and
longer entries from the hard source, while filtering 67,257 scored long rows
below the source-score floor.

## Fill Quality Rules

`FillGrader` scores words against the active dictionary and applies board-level
penalties.

Current hard guardrails:

- Exact duplicate answers are penalized.
- Answers that only differ by a terminal `S` are a hard fail, e.g.
  `OPAH`/`OPAHS`.
- The terminal-S rule is deliberately simple. It does not try to catch
  irregular morphology like `EAT`/`ATE`.
- Unknown-heavy grids are penalized.
- Short-glue penalties were removed because 3-letter entries are structurally
  unavoidable in 5x5-11x11 grids.

When adding new fill-quality rules, prefer the fill grader if a board should
be rejected before clue generation. Add focused tests in `tests/test_fill_grader.py`
and, where relevant, `tests/test_fill_with_grading_step.py`.

## Batch Generation

`generate-pilot-batch` creates manifest-driven batches. Despite the name, it is
the current production batch runner.

Grid selection notes:

- 5x5 and 7x7 minis use weighted pattern catalogs from `grid_specs.py`.
- 9x9 midis use a Jeff-feedback catalog with mirror-style and regular-symmetry
  examples, top-to-bottom flips, and validated cheater-square variants, not the
  procedural rotational generator. Catalog validation rejects patterns with
  three consecutive black squares pressed into any corner along a perimeter
  edge; non-corner perimeter triples remain allowed to match Jeff's examples.
- `config.easy.yaml` sets `fill.max_long_entries_8_9: 3`, so Easy 9x9 variants
  with more than three long slots are skipped before filling.
- 10x10 and 11x11 midis still fall back to procedural pattern generation.

Recommended Easy batch:

```bash
uv run crossword-generator generate-pilot-batch \
  --output-root output/batches/<batch-id> \
  --batch-id <batch-id> \
  --buckets easy/5,easy/7,easy/9 \
  --count 5 \
  --seed-start 1 \
  --llm claude
```

Recommended full Easy/Hard pilot:

```bash
uv run crossword-generator generate-pilot-batch \
  --output-root output/batches/<batch-id> \
  --batch-id <batch-id> \
  --count 5 \
  --seed-start 1 \
  --llm claude
```

Useful targeted replacement run:

```bash
uv run crossword-generator generate-pilot-batch \
  --output-root output/batches/<replacement-output> \
  --batch-id <original-batch-id> \
  --buckets hard/9 \
  --count 1 \
  --seed-start 1 \
  --llm claude
```

The batch runner records:

- `manifest.json`
- per-puzzle IPUZ files
- per-puzzle logs
- `grid_variants`
- `fill_attempts`
- `skipped_incompatible_variants`
- `fill_seconds`
- `clue_seconds`
- `total_seconds`
- `failure_category`

Batch fill controls are intentionally stricter than normal single-puzzle
generation:

- `per_pattern_attempts=1`
- `max_grid_variants=200`
- timeout defaults: 5x5 15s, 7x7 30s, 9x9 120s

## Upload Contract

Generated candidates upload through:

```text
POST /admin/data-store/records/bulk
```

Default API base:

```text
https://id-beta.hey.gg/api
```

Record contract:

- `namespace`: `crosswords`
- `collection`: `generated-puzzles`
- `status`: `draft`
- `metadata.review_status`: `unreviewed`
- `metadata.publication_status`: `draft`
- `metadata.author`: `crossword-generator`

Game keys:

- 5x5 and 7x7: `minicrossword`
- 9x9: `midicrossword`

Deterministic key shape:

```text
generated:<game_key>:<batch_id>:<difficulty>:<size>x<size>:seed-<seed>
```

Example:

```text
generated:midicrossword:phase-2b-pilot:hard:9x9:seed-1
```

Dry run first:

```bash
uv run crossword-generator save-generated-puzzles \
  --manifest output/batches/<batch-id>/manifest.json \
  --dry-run
```

Live upload:

```bash
export HEYGG_API_BASE_URL=https://id-beta.hey.gg/api
export HEYGG_ADMIN_API_TOKEN=<token>

uv run crossword-generator save-generated-puzzles \
  --manifest output/batches/<batch-id>/manifest.json
```

Replace existing uploaded records:

```bash
uv run crossword-generator save-generated-puzzles \
  --manifest output/batches/<replacement-output>/manifest.json \
  --replace-existing
```

Do not echo tokens or commit them. If a token returns 401, report that the
token is visible but unauthenticated; do not retry blindly.

If `uv` hits a sandbox cache permission error under `/Users/neil/.cache/uv`,
rerun the same `uv run ...` command with elevated permissions rather than
changing the command.

## Answer Scans Before Upload

Before upload, scan generated IPUZ answers for:

- Hits against `dictionaries/XwiJeffChenList-NotFamilyFriendly.txt`
- Hits against `dictionaries/Wordplete-PrevalentCulled-8-9-length-Removed.txt`
- Terminal-S pairs such as `OPAH`/`OPAHS`
- Obvious disease, violence, sexual, drug, or tough geography terms in new Easy
  attachments

One-off scans are acceptable, but if a rule should persist, implement it in
`FillGrader` or dictionary preparation and add tests.

## Verification

Focused tests for dictionary/config/upload work:

```bash
uv run pytest tests/test_dictionary_prep.py tests/test_config.py tests/test_data_store.py tests/test_cli_batch.py -q
```

Focused tests for fill-quality rules:

```bash
uv run pytest tests/test_fill_grader.py tests/test_fill_with_grading_step.py -q
```

Lint touched files:

```bash
uv run ruff check <paths>
```

Broader checks:

```bash
uv run pytest -q
uv run ruff check src/ tests/
```

## Known Generated Batch Context

Recent clean Easy prevalent batch:

```text
output/batches/phase-2c-easy-prevalent-8-9-clean/manifest.json
```

It generated 15/15 Easy candidates and uploaded 15 created records under batch
id `phase-2c-easy-prevalent-8-9-clean`.

Older pilot issue:

```text
output/batches/phase-2b-pilot/hard/9x9/seed-001.ipuz
```

That local output contained `OPAH`/`OPAHS`, which motivated the terminal-S
variant rule. Regenerate and replace that deterministic record if asked to
repair the old uploaded pilot batch.
