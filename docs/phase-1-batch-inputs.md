# Phase 1 Batch Inputs

Phase 1 prepares data for easy/hard batch generation experiments. It does not
build the review UI, add play.hey.gg APIs, or deploy puzzles into
crossword-midi-and-mini.

## Dictionaries

The generator dictionary loader expects one entry per line as `WORD;SCORE`.
The current Jeff-reviewed split is explicit:

- `dictionaries/hgg-easy.txt`: 3-9 letter HGG Easy entries, all `;50`, with
  known source-score 60 entries removed.
- `dictionaries/hgg-60.txt`: 7-9 letter HGG 60 entries, all `;60`.

Easy 9x9 and Hard 7x7 do not have their own dictionary files; those pools are
composed at generation time from the two effective master lists.

The legacy hard dictionary still filters true scored 6-, 7-, 8-, and 9-letter
source rows below `60` before flattening accepted entries. Previously flattened
`;55` Easy sources are treated as flat inputs, not original source-scored rows.

Prepare the HGG Easy and HGG 60 dictionaries from the source inputs. The command
also refreshes the legacy hard dictionary while `config.hard.yaml` still exists:

```bash
uv run crossword-generator prepare-dictionaries \
  --easy-source dictionaries/hgg-easy-flat-55.txt \
  --easy-extra-source dictionaries/Wordplete-PrevalentCulled-8-9-length.txt \
  --easy-exclude-source dictionaries/XwiJeffChenList-NotFamilyFriendly.txt \
  --easy-exclude-source dictionaries/Wordplete-PrevalentCulled-8-9-length-Removed.txt \
  --easy-exclude-source dictionaries/HggGeneratedSafetyExclude.txt \
  --easy-output dictionaries/hgg-easy.txt \
  --sixty-output dictionaries/hgg-60.txt \
  --hard-source dictionaries/HggCuratedCrosswordList.txt \
  --hard-output dictionaries/hgg-hard-flat-55.txt
```

`dictionaries/HggThumbsDownEasy.txt` and `HggThumbsDownHard.txt` (written
by `crossword-generator consolidate-list`) are picked up automatically
if present — no flag needed. Easy thumbs-down rows are excluded only
from the easy dictionary; hard thumbs-down rows are excluded only from
the hard dictionaries.

Default outputs:

- `dictionaries/hgg-easy.txt` from the prior Easy flat dictionary plus Jeff's
  prevalent 8-9-letter list, after removing known 60-pointers.
- `dictionaries/hgg-60.txt` from source-score 60 entries in the curated source,
  length 7-9.
- `dictionaries/hgg-hard-flat-55.txt` remains the legacy hard/midi dictionary
  until Jeff's HGG Hard list lands.

The command logs input rows, output rows, malformed rows, invalid words,
excluded words, rows below the source-score floor, duplicates, and the flat
score used.

The May 22 beta-backed dictionary run produced 18,320 HGG Easy rows, 5,881 HGG
60 rows, and 116,840 legacy Hard rows. The hard source had one invalid alphanumeric entry,
`catch22;50`, which was skipped because the current generator word format uses
letters only.

To publish the effective dictionaries for admin UI read-only views, use the
single-step publish command. It builds HGG Easy and HGG 60 into a temp directory,
validates their invariants, then sends both lists in one compact payload to the
admin API publish endpoint:

```bash
uv run crossword-generator publish-effective-dictionaries --dry-run
uv run crossword-generator publish-effective-dictionaries
```

The endpoint is `POST /admin/crossword-effective-dictionaries/publish`.
Backend storage should activate the received snapshot transactionally, so the
admin UI cannot observe new HGG Easy with old HGG 60. Effective dictionaries
are shared by the mini and midi crossword games, so the snapshot and runtime
recipes are game-agnostic. The command writes local `dictionaries/hgg-easy.txt`
and `dictionaries/hgg-60.txt` only after the publish request succeeds; pass
`--no-write-local` to skip that.

The prevalent 8-9-letter Easy merge excludes the high-confidence unsuitable
entries listed in `dictionaries/Wordplete-PrevalentCulled-8-9-length-Removed.txt`.
The May 13 merge produced 18,593 Easy rows after excluding 146 entries via the
existing family-unfriendly list plus the high-confidence removals from the new
attachment. The May 14 source-score floor intentionally leaves the prior flat
Easy 3-7 source intact because those `;55` values are normalized fill scores,
not original source quality scores.

## Configs

Use these committed configs for difficulty-specific runs:

- `config.easy.yaml`
- `config.easy9.yaml`
- `config.hard.yaml`
- `config.hard7.yaml`

Each config points `dictionary.path` and `fill.csp.dictionary_path` at the same
primary dictionary. HGG Easy configs use score threshold `50` and
`quality_tiers: [50]`. `config.easy9.yaml` and `config.hard7.yaml` merge
`dictionaries/hgg-60.txt` through `additional_paths` and
`additional_dictionary_paths`. `config.easy9.yaml` also sets
`fill.csp.min_score_by_length` for 8 and 9 to `60` so 9x9 Easy long slots cannot
use 50-point entries.

The batch runner uses `config.easy9.yaml` for `easy/9`, `config.easy.yaml` for
`easy/5`, `easy/7`, and `hard/5`, and `config.hard7.yaml` only for `hard/7`.
That matches Jeff's table while leaving `config.hard.yaml` available for the
legacy hard/midi path until the HGG Hard list arrives.

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
    "dictionaries/hgg-easy.txt",
    "dictionaries/hgg-60.txt",
    "dictionaries/hgg-hard-flat-55.txt",
):
    d = Dictionary.load(path, min_word_score=50, min_2letter_score=50)
    lengths = sorted(d.supported_lengths())
    print(path, len(d), lengths)

merged = Dictionary.load_many(
    ["dictionaries/hgg-easy.txt", "dictionaries/hgg-60.txt"],
    min_word_score=50,
    min_2letter_score=50,
)
print("runtime merged", len(merged), sorted(merged.supported_lengths()))
PY
```

Validate catalogued mini grid patterns:

```bash
uv run crossword-generator validate-mini-patterns
```

Expected pattern inventory:

- 5x5 Mini: 12 weighted patterns, total weight 58
- 7x7 Mini: 18 weighted patterns, total weight 50

The validation checks dimensions, positive weights, connected white cells,
no slots shorter than three letters, and symmetry status. It also reports any
pattern indexes that do not have regular 180-degree rotational symmetry or
left-right mirror symmetry.

Current validation found all catalogued patterns valid. Symmetry split:

- 5x5 Mini: 9 regular, 3 mirror-only, 0 unsupported
- 7x7 Mini: 12 regular, 6 mirror-only, 0 unsupported

The 9x9 midi catalog uses curated mirror-style patterns from Jeff's feedback
instead of procedural rotational patterns. This intentionally avoids
windmill-like rotational symmetry and keeps long 8-/9-letter slots sparse for
Easy midi generation.

Run the targeted test subset:

```bash
uv run pytest tests/test_dictionary.py tests/test_dictionary_prep.py \
  tests/test_grid_specs.py tests/test_config.py
```

## Open Questions

- Whether the flat easy dictionary is large enough for acceptable 7x7 and 9x9
  fill rates needs to be measured during batch generation.
- If Jeff sends more grid-pattern feedback, keep the mini catalog restricted to
  regular 180-degree rotational symmetry or left-right mirror symmetry.
- 7x7 patterns with more than four 7-letter slots are excluded after
  center-square normalization because the recent fill logs showed a much lower
  accepted-fill rate.

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

## Phase 2B Pilot Data Store Save

Generated pilot candidates can be saved to the authenticated HeyGG admin data
store without writing directly to the hey-you database. The prod API base
(`https://play.hey.gg/api`) is the default; override `HEYGG_API_BASE_URL` to
target staging (`https://id-beta.hey.gg/api`):

```bash
export HEYGG_API_BASE_URL=https://play.hey.gg/api
export HEYGG_CROSSWORD_GENERATOR_TOKEN=<crossword-generator service-account token>

uv run crossword-generator save-generated-puzzles \
  --manifest output/batches/phase-2b-pilot/manifest.json
```

The command reads successful IPUZ files from the manifest and creates records
in `crosswords/generated-puzzles` with `status=draft`,
`metadata.review_status=unreviewed`, and
`metadata.publication_status=draft`. By default, 5x5 and 7x7 records use
`game_key=minicrossword`, while 9x9 records use `game_key=midicrossword`.

Keys are deterministic and include game, batch, difficulty, size, and seed, for
example:

```text
generated:minicrossword:phase-2b-pilot:easy:5x5:seed-1
```

Reruns do not create second records for duplicate keys. If a duplicate is
reported by the API, the command queries the existing record and skips it. To
intentionally replace existing draft candidates, pass `--replace-existing`,
which PATCHes the existing record instead.

Validate records without API calls:

```bash
uv run crossword-generator save-generated-puzzles \
  --manifest output/batches/phase-2b-pilot/manifest.json \
  --dry-run
```
