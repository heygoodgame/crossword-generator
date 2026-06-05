# Crossword Generator Workflow Reference

## Goal

This repo produces reviewable mini and midi crossword candidates for HeyGG.
The immediate operating goal is not direct publication; it is to generate
draft IPUZ records, upload them to `crosswords/generated-puzzles`, and let the
admin/editor workflow review, edit, approve, or reject them.

Current emphasis:

- Easy puzzles should favor accessible, one-word fill.
- Easy clue generation should be easier than NYT Monday: direct definitions,
  obvious fill-in-the-blanks, and broad casual-audience accessibility.
- Hard clue generation should target NYT Tuesday/Wednesday: fair but more
  oblique definitions, mild wordplay, and occasional misdirection. Avoid
  forced difficulty, strained pop-culture references, ultra-current slang, and
  clues that need a long explanation to be fair. For pop-culture, celebrity,
  entertainment, sports, brand, and historical references, the older or more
  niche the reference is, the more broadly iconic it must be. Accuracy and
  exact answer fit beat cleverness: if a proper noun, song, quote, team name,
  idiom, or fill-in-the-blank angle is not certain, use a clean direct clue
  instead.
- Clues should avoid unpleasant wording such as "death" and "undocumented
  immigrant"; if dying must be referenced, use gentle wording such as
  "passed on."
- Clue prompts omit word-count tags such as `(two words)` until the pipeline has
  explicit word-boundary metadata. Explanatory clue tags should be
  parenthetical, not comma/colon appendages.
- The clue grader applies deterministic penalties for answer leakage, and the
  repair pass also fixes low individual clue scores or explicit evaluator
  "major issue" / "factual error" feedback even when aggregate clue score
  passes. Hard configs use a higher individual clue repair threshold so
  borderline Hard clues are more likely to be surgically regenerated.
- A second-pass fact-risk checker runs after normal clue grading/repair. It
  pre-screens risky clue forms such as proper-noun trivia, titles, quotes,
  dates, superlatives, and fill-in-the-blank phrases, then rewrites clues the
  checker marks `uncertain` or `incorrect`.
- HGG Easy is a single 3-9 letter effective list, scored `;50`, with known
  source-score 60 entries removed.
- HGG 60 is a single 7-9 letter effective list, scored `;60`.
- Easy 9x9 generation uses HGG Easy only. Hard 9x9 uses HGG Easy for 3-7
  letter fill and HGG 60 for all 8-/9-letter fill. Seven-letter HGG 60 entries
  are reserved for Hard 7x7.
- Hard 5x5 uses the same HGG Easy config as Easy minis.
- Hard 7x7 generation uses HGG Easy plus a board-level rule requiring exactly
  one 7-letter HGG 60 entry.
- Mini generation applies Jeff's pattern feedback: keep only regular
  180-degree rotational symmetry or left-right mirror symmetry; discard
  asymmetric, diagonal-only, and up-down-only patterns. For 7x7, add the center
  black square whenever doing so does not create short slots. Jeff's Utah-block
  attachment is the explicit exception: it is kept both with and without the
  center square.
- 7x7 grids with more than four 7-letter slots are excluded. Recent fill logs
  showed those variants produced accepted fills at a much lower rate than the
  rest of the 7x7 catalog.
- 9x9 midi generation uses expanded Jeff-feedback mirror-style and
  regular-symmetry patterns with safe top-to-bottom flips for mirror patterns,
  left-right flips for regular-symmetry patterns, and conservative
  cheater-square variants, while avoiding three-black-square perimeter runs
  that press into a corner and procedural rotational windmill patterns that
  can read as swastika-like.
- Easy 9x9 generation skips grids with more than three 8-/9-letter slots.
- Avoid unsuitable or controversial fill before clue generation and upload.
- For known dated or too-niche answer entries, add them to
  `dictionaries/HggThumbsDownHard.txt` or `dictionaries/HggThumbsDownEasy.txt`
  and rebuild effective dictionaries. Prompt guidance can handle subjective
  clue angles, but the thumbs-down lists are the durable mechanism for answer
  entries Jeff does not want selected at all.
- Preserve reproducible batch manifests and deterministic data-store keys.
- When asked for a generated batch across Mini Crossword and Midi Crossword
  without explicit size counts, default to a rough 5:2:7 ratio for 5x5, 7x7,
  and 9x9 puzzles. Midi Crossword always uses 9x9; Mini Crossword dailies are
  five 5x5 puzzles and two 7x7 puzzles per week.

## Architecture

The CLI entrypoint is `crossword_generator.cli:main`.

The generation pipeline is assembled in `src/crossword_generator/pipeline.py`.
It passes a `PuzzleEnvelope` through these steps:

1. Optional theme generation, mostly off for current batch work.
2. Grid fill via `CSPFiller`.
3. Fill grading via `FillGrader`.
4. Clue generation via Ollama or Claude.
5. Clue grading, repair, and fact-risk checking via the configured LLMs.
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
- `config.easy9.yaml`
- `config.hard.yaml`
- `config.hard7.yaml`
- `config.hard9.yaml`

Current Easy config points both `dictionary.path` and
`fill.csp.dictionary_path` at:

```text
dictionaries/hgg-easy.txt
```

Current Easy 9x9 config points both primary paths at:

```text
dictionaries/hgg-easy.txt
```

Current Hard 9x9 config points both primary paths at `hgg-easy.txt` and merges
only the 8-/9-letter slice of `hgg-60.txt` through `additional_paths` /
`additional_dictionary_paths`:

```text
dictionaries/hgg-easy.txt + dictionaries/hgg-60.txt
```

Current Hard config points both paths at:

```text
dictionaries/hgg-hard-flat-55.txt
```

Current Hard 7x7 config points both primary paths at `hgg-easy.txt` and merges
`hgg-60.txt` through `additional_paths` / `additional_dictionary_paths`:

```text
dictionaries/hgg-easy.txt + dictionaries/hgg-60.txt
```

The master effective lists are:

- `dictionaries/hgg-easy.txt`: 3-9 letter HGG Easy entries, all written as
  `WORD;50`.
- `dictionaries/hgg-60.txt`: 7-9 letter source-score 60 entries, all written
  as `WORD;60`.

No derived 9x9 or Hard 7x7 dictionary files are committed. Those pools are
composed at generation time from the two effective master lists. Hard 9x9 sets
`fill.csp.min_score_by_length: {8: 60, 9: 60}` and length-filters the additional
HGG 60 source to 8-/9-letter entries, so long slots cannot fall back to 50-point
entries and 7-letter HGG 60 entries remain reserved for Hard 7x7. Hard 7x7 uses
the fill grader to require exactly one 7-letter entry with score 60.

The hard flat dictionary is length-mixed: 3-, 4-, and 5-letter entries come
from the prepared Easy/prevalent list, while 6+ entries come from
`dictionaries/HggCuratedCrosswordList.txt`. This keeps short fill accessible
and avoids leaning on crosswordese-heavy hard-list glue.

`generate-pilot-batch` selects `config.easy9.yaml` for the `easy/9` bucket,
`config.hard9.yaml` for the `hard/9` bucket, `config.easy.yaml` for `easy/5`,
`easy/7`, and `hard/5`, and `config.hard7.yaml` for `hard/7`.

The HGG Easy configs use `quality_tiers: [50]`; the 60-aware configs merge
`hgg-60.txt` so slot and board-level rules can enforce Jeff's table.

Dictionary preparation:

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

`prepare-dictionaries` also auto-discovers two files if they exist on
disk: `dictionaries/HggThumbsDownEasy.txt` (unioned into easy-only
excludes) and `dictionaries/HggThumbsDownHard.txt` (unioned into
hard-only excludes). These are written by `consolidate-list` — see
the Word List Management section below. No flag required; the
operator's reference command above is unchanged.

By default, preparation filters true scored 6-, 7-, 8-, and 9-letter hard-source
rows below `60` for the legacy hard dictionary. HGG Easy separately removes
known source-score 60 entries before writing `;50`. Previously flattened
`WORD;55` Easy inputs are treated as flat dictionaries, not original source
scores.

The May 22 dictionary run produced:

- HGG Easy: 18,320 rows, all `;50`.
- HGG 60: 5,881 rows, all `;60` and length 7-9.
- Hard: 116,840 rows after taking 3-5 letter entries from HGG Easy and
  6+ entries from the hard source.

The regular Hard output still filters scored 6-9-letter rows below the
source-score floor. Hard 7x7 composes HGG Easy and HGG 60 at runtime; the
exact-one-60 requirement is enforced during fill grading.

Effective dictionary publish:

```bash
uv run crossword-generator publish-effective-dictionaries --dry-run
uv run crossword-generator publish-effective-dictionaries
```

`publish-effective-dictionaries` builds HGG Easy and HGG 60 into a temp
directory, validates both lists, then posts one compact payload to
`POST /admin/crossword-effective-dictionaries/publish`. Backend storage should
activate that snapshot transactionally so the admin UI gets an all-or-nothing
view; it cannot observe a newly published HGG Easy with an old HGG 60.
Effective dictionaries are shared by the mini and midi crossword games, so the
snapshot and runtime recipes are game-agnostic. After the API write succeeds,
the command writes local `dictionaries/hgg-easy.txt` and `dictionaries/hgg-60.txt`
unless `--no-write-local` is passed.

As of the HGGXW cutover (2026-06), the source lists are Jeff's three
consolidated lists, registered on prod and pulled via `consolidate-list`:

- `--easy-source dictionaries/HGGXW-Easy.txt` — plain Easy fill (no scores)
- `--hard-source dictionaries/HGGXW-Hard.txt` — plain Hard fill (no scores)
- `--sixty-source dictionaries/XwiJeffChenList.txt` — scored master; the
  source-score-60 entries become HGG 60 and are excluded from HGG Easy.

Because `HGGXW-Easy/Hard` are plain (unscored), the 60-pointers can no
longer be read from the hard list. `--sixty-source` (the scored master
XWordInfo list) supplies them.

`publish-effective-dictionaries` builds three local dictionaries:

- `hgg-easy.txt` — HGG Easy fill (flat 50, source-score-60 entries removed)
- `hgg-hard.txt` — **HGG Easy ∪ HGG Hard** fill (flat 50, same exclusions).
  This is Jeff's "hard puzzles draw from both lists." It is a derived
  generation input, deterministically rebuilt from the source lists every
  run, so it is written locally only and is NOT part of the published
  snapshot (hey-you's snapshot allowlist is `hgg-easy` + `hgg-60`).
- `hgg-60.txt` — the source-score-60 entries (lengths 7–9) from the master.

Composition by bucket:

- Easy puzzles draw from `hgg-easy.txt` (`config.easy.yaml`, `config.easy9.yaml`).
- Hard puzzles draw from `hgg-hard.txt` as the base fill:
  - `config.hard5.yaml` — hard 5×5, no 60-pointers (per Jeff).
  - `config.hard7.yaml` — hard 7×7, base `hgg-hard.txt` + `hgg-60.txt`;
    grading requires exactly one 7-letter 60.
  - `config.hard9.yaml` — hard 9×9, base `hgg-hard.txt` + `hgg-60.txt`
    (8–9 letter slots), `min_score_by_length {8:60,9:60}`.

The per-size 60-point requirements are enforced by those config grading
rules against the flat-`;60` `hgg-60.txt`.

## Word List Management

Master word lists are first-class entities in hey-you's MySQL
(`crossword_lists`, `crossword_list_words`, `crossword_list_word_audits`
— migrated 2026-05-21). Jeff manages them from two UIs that share the
same backend:

- `midicrossword.com/admin/lists` for setters (Jeff lives here)
- `hey-you/admin/crossword-lists` for full admins

The on-disk `dictionaries/*.txt` files in this repo remain authoritative
for the generator. They are kept in sync via a deliberate operator step.

### End-to-end flow

```
Jeff edits in /admin/lists  →  rows live in hey-you's MySQL
   (add word, remove, rescore, or thumbs-down during puzzle review)

Operator runs `crossword-generator consolidate-list [slug]`
   → GET /api/admin/crossword-lists/{slug}/download for each list
   → write to dictionaries/<file_path> (per registered file_path)
   → POST /mark-consolidated so /admin/lists shows freshness

Operator reviews `git diff dictionaries/`, commits, pushes

Next `publish-effective-dictionaries` run picks up the new state automatically
   (including the auto-discovered HggThumbsDown*.txt files)
   → build/validate HGG Easy + HGG 60 together
   → POST `/admin/crossword-effective-dictionaries/publish`

Run `prepare-dictionaries` only when you also need to refresh the committed
local dictionary artifacts, including the legacy hard flat dictionary.
```

### Registered lists (seeded 2026-05-21)

| Slug | Format | Scope | Exclude? | What it is |
|---|---|---|---|---|
| `hgg-easy-flat-55` | scored | easy | no | 3-7 letter Easy entries flattened to 55 |
| `wordplete-prevalent-8-9` | plain | easy | no | Prevalent 8-9 letter Wordplete entries |
| `wordplete-prevalent-8-9-removed` | word_with_reason | easy | yes | Removed entries from the above |
| `xwi-jeff-chen-not-family-friendly` | scored | easy | yes | XWI Jeff Chen entries flagged unsuitable |
| `hgg-safety-exclude` | word_with_reason | easy | yes | HGG-generated safety exclude list |
| `hgg-curated` | scored | hard | no | Main curated hard list (the bulk) |
| `hgg-thumbs-down-easy` | word_with_reason | easy | yes | Auto-created on first thumbs-down |
| `hgg-thumbs-down-hard` | word_with_reason | hard | yes | Auto-created on first thumbs-down |

Easy-scoped exclude lists are passed as `--easy-exclude-source` flags to
`prepare-dictionaries`. `hgg-thumbs-down-easy` / `hgg-thumbs-down-hard`
are auto-discovered (no flag).

Thumbs-down semantics:

- **Easy puzzle thumbs-down** → row in `hgg-thumbs-down-easy`. Blocks
  from the easy dictionary only; hard puzzles can still use the word.
- **Hard puzzle thumbs-down** → row in `hgg-thumbs-down-hard`. Blocks
  from the hard dictionaries only; easy puzzles can still use the word.

### `consolidate-list` command

```bash
# Default: walk every registered list, write each .txt file, mark consolidated
HEYGG_ADMIN_TOKEN=<token> uv run crossword-generator consolidate-list

# Single slug
uv run crossword-generator consolidate-list hgg-curated

# Preview only — fetches, diffs, prints; does not write or ack
uv run crossword-generator consolidate-list --dry-run

# Prod vs beta (defaults to play.hey.gg)
uv run crossword-generator consolidate-list --api-base https://id-beta.hey.gg/api
```

Diff is computed on the leading WORD-before-semicolon, so per-word score
or note changes are not flagged as add/remove — they show up as a
byte-level change in `git diff` if the operator wants the detail. The
HTTP layer lives in `src/crossword_generator/consolidate_list.py`.

### When NOT to use `consolidate-list`

If you're experimenting locally with hand-edited `.txt` files (e.g.
testing a new exclude rule), don't run `consolidate-list` against the
shared environment — it will overwrite your local edits with the
committed-from-UI state. Use `--dry-run` first if uncertain.

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
- Hard puzzles (all sizes) enforce two Jeff Hard-list rules (Jeff, 2026-06),
  both keyed off the same `hard_word_set` the grader receives. The Hard list
  (`dictionaries/HGGXW-Hard.txt`) is disjoint from Easy, so a "Hard-list entry"
  is simply any answer in that file. The 60-point scored-master pool
  (`hgg-60.txt`) is NOT treated as Hard-list for either rule. The set is
  configured per size with `grading.fill.hard_cross_words_path` (set on
  `config.hard5/7/9.yaml`); Easy puzzles pass no set and skip both rules.
  - `no_hard_entry` hard fail: a Hard board must contain at least one
    Hard-list entry, else it is really an Easy puzzle. This mostly affects
    5x5 (where ~45% of unconstrained boards came out all-Easy); 7x7/9x9
    effectively always already include a hard entry.
  - `hard_cross` hard fail: no two Hard-list entries may cross each other.
    Many Hard-list entries are proper names; crossing two of them can force an
    unsatisfying total guess. Without it, ~80% of hard 9x9 boards crossed two
    Hard-list entries.
  The large Easy pool lets the filler satisfy both rules on essentially every
  seed.

When adding new fill-quality rules, prefer the fill grader if a board should
be rejected before clue generation. Add focused tests in `tests/test_fill_grader.py`
and, where relevant, `tests/test_fill_with_grading_step.py`.

## Batch Generation

`generate-pilot-batch` creates manifest-driven batches. Despite the name, it is
the current production batch runner.

Grid selection notes:

- 5x5 and 7x7 minis use weighted pattern catalogs from `grid_specs.py`.
- Mini catalogs keep only regular 180-degree rotational symmetry or left-right
  mirror symmetry; asymmetric, diagonal-only, and up-down-only patterns are
  excluded.
- 7x7 minis start from the raw weighted catalog, then apply Jeff's feedback:
  add the central black square when the resulting grid remains structurally
  valid. Two late attachment examples are also included; the Utah-block example
  appears both centered and uncentered.
- 7x7 grids with more than four 7-letter slots are excluded after center-square
  normalization, based on Jeff's May 27 decision and the recent fill-log
  success-rate split.
- 9x9 midis use a Jeff-feedback catalog with mirror-style and regular-symmetry
  examples, top-to-bottom flips for mirror patterns, left-right flips for
  regular-symmetry patterns, and validated cheater-square variants, not the
  procedural rotational generator. Catalog validation rejects patterns with
  three consecutive black squares pressed into any corner along a perimeter
  edge; non-corner perimeter triples remain allowed to match Jeff's examples.
- `config.easy.yaml` sets `fill.max_long_entries_8_9: 3`, so Easy 9x9 variants
  with more than three long slots are skipped before filling.
- 10x10 and 11x11 midis still fall back to procedural pattern generation.

Recommended cross-site Easy batch:

```bash
uv run crossword-generator generate-pilot-batch \
  --output-root output/batches/<batch-id> \
  --batch-id <batch-id> \
  --buckets easy/5,easy/7,easy/9 \
  --bucket-counts 5=5,7=2,9=7 \
  --seed-start 1 \
  --llm claude
```

Recommended full Easy/Hard pilot using the same cross-site size ratio:

```bash
uv run crossword-generator generate-pilot-batch \
  --output-root output/batches/<batch-id> \
  --batch-id <batch-id> \
  --bucket-counts 5=5,7=2,9=7 \
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
https://play.hey.gg/api
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
export HEYGG_API_BASE_URL=https://play.hey.gg/api
export HEYGG_ADMIN_API_TOKEN=<token>

uv run crossword-generator save-generated-puzzles \
  --manifest output/batches/<batch-id>/manifest.json
```

Get the token from `hgg-auth` (the `heygg-admin-auth` skill) rather than
copying a JWT from DevTools. Pick the profile that matches the upload target —
`prod` for `play.hey.gg`, `beta` for `id-beta.hey.gg` — and inject it for the
single command, e.g. uploading to prod:

```bash
hgg-auth exec prod -- bash -c '
  export HEYGG_API_BASE_URL="$HGG_ADMIN_BASE_URL/api"
  uv run crossword-generator save-generated-puzzles \
    --manifest output/batches/<batch-id>/manifest.json'
```

`hgg-auth exec <profile>` exports `HGG_ADMIN_BASE_URL` and `HEYGG_ADMIN_TOKEN`;
the uploader reads `HEYGG_API_BASE_URL` and `HEYGG_ADMIN_TOKEN`, so set the
former from the latter as shown. The uploader defaults the base URL to
play.hey.gg (prod), so set `HEYGG_API_BASE_URL` explicitly when targeting beta.

Replace existing uploaded records:

```bash
uv run crossword-generator save-generated-puzzles \
  --manifest output/batches/<replacement-output>/manifest.json \
  --replace-existing
```

Do not echo tokens or commit them. A `401 Unauthenticated` means the token is
missing, expired, or for the wrong environment (e.g. a beta token against
prod); a `403` means the token is valid but the user is not an admin there.
Refresh via `hgg-auth login <profile>` (the `heygg-admin-auth` skill) and
re-run with the matching profile — do not retry blindly with the same token.

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
