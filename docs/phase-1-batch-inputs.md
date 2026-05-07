# Phase 1 Batch Inputs

Phase 1 prepares data for easy/hard batch generation experiments. It does not
build the review UI, add play.hey.gg APIs, or deploy puzzles into
crossword-midi-and-mini.

## Dictionaries

The generator dictionary loader expects one entry per line as `WORD;SCORE`.
Phase 1 uses a flat score of `55` so the fill solver tests the curated word
sets rather than old score gradients.

Prepare both dictionaries from the source inputs:

```bash
uv run crossword-generator prepare-dictionaries \
  --easy-source /private/tmp/crossword-generator-thread/19dd6724cd3c4827_ANGjdJ8p_WordpleteCulledJYC.txt
```

Default outputs:

- `dictionaries/hgg-easy-flat-55.txt` from Jeff's `WordpleteCulledJYC.txt`
- `dictionaries/hgg-hard-flat-55.txt` from `dictionaries/HggCuratedCrosswordList.txt`

The command logs input rows, output rows, malformed rows, invalid words,
duplicates, and the flat score used.

The initial run produced 8,967 easy rows and 201,978 hard rows. The hard source
had one invalid alphanumeric entry, `catch22;50`, which was skipped because the
current generator word format uses letters only.

## Configs

Use these committed configs for difficulty-specific runs:

- `config.easy.yaml`
- `config.hard.yaml`

Both configs point `dictionary.path` and `fill.csp.dictionary_path` at the same
flat-score dictionary. Their score thresholds are `55`, and `quality_tiers` is
`[55]` so a flat `55` dictionary is not filtered out by the old tier ladder.

Ollama remains the repo default for clue generation. Phase 1 prep and
validation do not require an LLM. For future generation runs, Claude remains
available through the existing CLI override:

```bash
uv run crossword-generator generate --config config.easy.yaml --llm claude
```

## Validation

Validate the generated dictionaries load through the existing loader:

```bash
uv run python - <<'PY'
from crossword_generator.dictionary import Dictionary

for path in (
    "dictionaries/hgg-easy-flat-55.txt",
    "dictionaries/hgg-hard-flat-55.txt",
):
    d = Dictionary.load(path, min_word_score=55, min_2letter_score=55)
    print(path, len(d), d.score(next(iter(d.words_by_length(5)))))
PY
```

Validate catalogued mini grid patterns:

```bash
uv run crossword-generator validate-mini-patterns
```

Expected pattern inventory:

- 5x5 Mini: 34 weighted patterns, total weight 95
- 7x7 Mini: 50 weighted patterns, total weight 86

The validation checks dimensions, positive weights, connected white cells,
no slots shorter than three letters, and symmetry status. It also reports
which pattern indexes are asymmetric so future generation can filter or
down-rank them.

Initial validation found all catalogued patterns valid. Symmetry split:

- 5x5 Mini: 9 symmetric, 25 asymmetric
- 7x7 Mini: 18 symmetric, 32 asymmetric

Run the targeted test subset:

```bash
uv run pytest tests/test_dictionary.py tests/test_dictionary_prep.py \
  tests/test_grid_specs.py tests/test_config.py
```

## Open Questions

- Whether the flat easy dictionary is large enough for acceptable 7x7 and 9x9
  fill rates needs to be measured during batch generation.
- Asymmetric mini patterns are preserved for now, but Jeff prefers unthemed
  mini symmetry. Future batch scripts can use the structured pattern symmetry
  flag to filter or down-rank asymmetric patterns.

## Easy 9x9 Smoke Finding

The initial easy 9x9 smoke pass failed for most seeds even with 50 retries.
The easy dictionary only contains word lengths 3-7, while generated 9x9 midi
patterns often contain 8- or 9-letter slots. Seed 26 succeeded because its
9x9 slot lengths were only 3, 4, 5, and 7.

Generation now checks the selected grid pattern against the active dictionary
before invoking the filler. Unsupported patterns are skipped with a log line
like:

```text
Grid variant N skipped: slot lengths [8, 9] unsupported by dictionary
```

For non-themed generation, `fill.max_grid_variants` now allows the direct fill
path to walk forward through later grid seeds until it finds a compatible
pattern or exhausts the variant budget. This avoids spending CSP time on grids
that cannot possibly be filled by a constrained dictionary.
