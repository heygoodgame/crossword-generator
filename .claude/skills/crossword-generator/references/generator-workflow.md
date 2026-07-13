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
- Hard clue generation should target solid NYT Tuesday, with Wednesday as the
  ceiling rather than the target: fair but more oblique definitions, mild
  wordplay, and occasional misdirection. Per Jeff's June 2026 feedback, an
  instantly-solvable Easy-style clue (e.g. "One more than two" for THREE) is a
  defect on a Hard puzzle: the generation prompt defaults Hard clues to a
  harder fair angle (secondary meanings, specific examples, fair trivia, mild
  misdirection) and reserves plain direct definitions for glue entries
  (roughly a quarter of clues at most). Equally (per Jeff's follow-up after
  the first harder-Hard batch overshot), do not strain for difficulty:
  one fair twist per clue, never obscurity, convoluted phrasing, or stacked
  tricks — when in doubt between two fair angles, pick the easier one. The
  evaluator flags both "too easy" and "too hard" clues for repair. Avoid
  forced difficulty, strained pop-culture references, ultra-current slang, and
  clues that need a long explanation to be fair. For pop-culture, celebrity,
  entertainment, sports, brand, and historical references, the older or more
  niche the reference is, the more broadly iconic it must be. Accuracy and
  exact answer fit beat cleverness: if a proper noun, song, quote, team name,
  idiom, or fill-in-the-blank angle is not certain, drop that angle — on Hard,
  switch to a different solid hard angle rather than an instantly-solvable
  clue.
- Clues must be timeless: no dependence on current employers, broadcast
  rights, rosters, reigning champions, or current hosts (per Jeff's June 2026
  feedback on stale TNT/NBA and Ari Shapiro/NPR clues). Titles of movies, TV
  shows, songs, albums, books, and plays take quotation marks, including in
  fill-in-the-blanks ("Better Call ___", quoted).
- Clues should avoid unpleasant wording such as "death" and "undocumented
  immigrant"; if dying must be referenced, use gentle wording such as
  "passed on."
- Puzzle titles should lean punny (Jeff, June 2026): prefer puns, double
  meanings, and playful twists on familiar phrases over plain evocative
  titles. The naming prompt (`llm/prompts/puzzle_naming.py`) encodes this;
  the existing sensitivity guardrails (no identity-based, suggestive, or
  at-someone's-expense wordplay) still win over cleverness.
- Clue prompts omit word-count tags such as `(two words)` until the pipeline has
  explicit word-boundary metadata. Explanatory clue tags should be
  parenthetical, not comma/colon appendages.
- The clue grader applies deterministic penalties for answer leakage, and the
  repair pass also fixes low individual clue scores or explicit evaluator
  "major issue" / "factual error" feedback even when aggregate clue score
  passes. Hard configs use a higher individual clue repair threshold so
  borderline Hard clues are more likely to be surgically regenerated. On Hard
  puzzles, the evaluator scores too-easy clues 0-9 on freshness with "too
  easy" in the feedback; Hard configs set
  `grading.clue.freshness_repair_threshold: 10` so those clues are surgically
  repaired with a harder fair angle.
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
- `proper_noun_cap` hard fail (all difficulties/sizes, Jeff, 2026-07): at most
  `max(2, floor(0.15 × answer_count))` proper-noun answers per grid — 2 for a
  typical 5x5/7x7, 3 for a typical 9x9. "Too many names makes it a trivia
  contest, not a word puzzle." Additionally, `proper_noun_first_across`
  hard fail: 1-Across may never be a proper noun (solver's first
  impression), regardless of the cap. A word counts only if it is viable SOLELY as a
  proper noun (OPRAH, ERIE, OREO, NBA); dual-reading words (AMBER, CHINA,
  MARK) do not count. Classifications live in
  `dictionaries/HggProperNounClassifications.txt` (`WORD;P|C`), configured via
  `grading.fill.proper_nouns_path` on every batch config. Theme seed entries
  are not in the file, so intentional themed names never count.
  **After any dictionary refresh (`consolidate-list` /
  `prepare-dictionaries`), run `uv run crossword-generator
  classify-proper-nouns` — it is incremental and only classifies words not
  yet in the file — then commit the updated classification file.**

When adding new fill-quality rules, prefer the fill grader if a board should
be rejected before clue generation. Add focused tests in `tests/test_fill_grader.py`
and, where relevant, `tests/test_fill_with_grading_step.py`.

## Scheduling-Time Answer-Repeat Windows

hey-you enforces rolling no-repeat windows when dailies are scheduled,
reassigned, or edited (`CrosswordPuzzleController`), cross-game and
cross-track:

- Regular answers: ±6 days (the original rule).
- HGG 60 answers: ±180 days (Jeff's "feature entries" rule, 2026-06-09).
  Membership is checked against the active published `hgg-60` effective
  dictionary snapshot, so rescoring a word in/out of the 60 list changes
  future checks automatically.

Because a 60-pt collision blocks a candidate for months (not days),
`generate-pilot-batch` excludes already-scheduled 60s at fill time — see the
`--exclude-scheduled-sixty` notes below. The scheduling check remains the
source of truth; the generator exclusion just keeps candidates schedulable.

`GET /api/admin/crossword-puzzles/daily-answers/recent-sixty` returns the
distinct scheduled HGG 60 answers from `window_days` (default 180) ago through
all scheduled future days.

Regular answers get the same treatment (Jeff, June 2026): because batch
candidates are scheduled at the first unscheduled daily slot or later,
`GET /api/admin/crossword-puzzles/daily-answers/recent` returns the distinct
scheduled answers within a bounded window around the first unscheduled slot —
`window_days` (default 7) back through `forward_days` (default 13) ahead,
cross-game and cross-track. A 7-day lookback fully covers the ±6-day regular
window for every placement at or after the slot; the forward bound covers a
weekly batch scheduled within ~7 days of the slot plus the ±6 margin. The
server computes the first unscheduled slot as the earliest day from today
where any game/track combination has no scheduled slot.

The forward bound is load-bearing (June 2026 incident): an unbounded future
sweep returned 2,564 answers off a ~6-week schedule and gutted the short-word
fill pools (3-letter: 604 → 127), which collapsed 9x9 fill diversity so badly
that 28 puzzles produced 125 cross-puzzle duplicate answers. Never widen
`forward_days` past what the batch being generated actually needs.

## Batch Generation

`generate-pilot-batch` creates manifest-driven batches. Despite the name, it is
the current production batch runner.

When the batch includes a `hard/7` or `hard/9` bucket, the runner fetches
scheduled HGG 60 answers from the admin API (requires `HEYGG_ADMIN_TOKEN` or
`HEYGG_ADMIN_API_TOKEN`) and writes `hgg-60-scheduled-filtered.txt` into the
output root; every `hgg-60.txt` config reference is pointed at that filtered
copy for the run. This is on by default; pass `--no-exclude-scheduled-sixty`
for offline/experimental runs. The manifest records the exclusion under
`exclude_scheduled_sixty`.

Every batch also fetches the recent daily answers list (7 days before the
first unscheduled daily slot plus all scheduled future days) and writes a
`<dictionary>-recent-filtered.txt` copy of each fill dictionary the active
buckets reference (`hgg-easy.txt`, `hgg-hard.txt`; recent answers are also
unioned into the hgg-60 filter). All dictionary references in the loaded
configs are pointed at the filtered copies — primary, themed, and CSP slots
alike; `grading.fill.hard_cross_words_path` is a grading membership set, not
a fill pool, and is left untouched. This is on by default; pass
`--no-exclude-recent-answers` for offline/experimental runs. The manifest
records the exclusion under `exclude_recent_answers`, including the window,
the first unscheduled date, and per-dictionary removed-row counts.

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
  --llm claude
```

Recommended full Easy/Hard pilot using the same cross-site size ratio:

```bash
uv run crossword-generator generate-pilot-batch \
  --output-root output/batches/<batch-id> \
  --batch-id <batch-id> \
  --bucket-counts 5=5,7=2,9=7 \
  --llm claude
```

Useful targeted replacement run:

```bash
uv run crossword-generator generate-pilot-batch \
  --output-root output/batches/<replacement-output> \
  --batch-id <original-batch-id> \
  --buckets hard/9 \
  --count 1 \
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

## Intra-Batch Duplicate-Answer Gate (before upload)

A weekly batch is scheduled across consecutive days, so any answer shared by
two puzzles in the same batch trips the scheduling-time no-repeat windows.

Two structural facts make intra-batch duplicates likely rather than rare
(diagnosed June 2026 on the first recent-answer-exclusion batch, which came
out with 125 cross-puzzle duplicate answers in 28 puzzles):

- The weighted 9x9 grid catalog funnels many seeds onto the same popular
  patterns (six of seven easy 9x9s drew one pattern).
- The CSP fill is heavily biased on a given grid: two different fill seeds
  on the same grid shared 19/30 answers in a controlled test, and flexible
  glue like AMEN/GEN/SAFE/USER appears in nearly every fill. Random value
  ordering does not overcome the constraint structure's preferred solutions.

The batch runner therefore threads a shared used-answer set through the run
(`_UsedAnswerSet` in `cli.py` → `create_pipeline(excluded_fill_words=...)` →
`Dictionary.remove_words`): each batch item's fill dictionary drops every
answer used by already-completed batch-mates. With `--max-workers 1` this
guarantees intra-batch uniqueness by construction; with parallel workers,
concurrent items can't see each other, so a residue of duplicates remains
possible and the post-batch gate below catches it.

After every batch — and before uploading — run:

```bash
uv run crossword-generator check-batch-answers \
  --manifest output/batches/<batch-id>/manifest.json \
  --write-answers-file output/batches/<batch-id>/batch-answers.txt
```

It exits non-zero when any cross-puzzle duplicate exists, and labels each as
either **blocking** or **short-window**:

- **`short-window` (acceptable — do NOT regenerate):** 3-letter answers
  confined to 9x9 puzzles. The scheduler spaces these out (it allows 3-letter
  9x9 repeats 3+ days apart), so a week with only short-window dupes uploads
  fine and schedules without manual care. This is the common case for hard 9x9
  weeks — the weighted grids + biased CSP fill collide on short glue (ADS, IPO,
  EAR, AND…), and intra-batch dedup deliberately ignores <=3-letter answers
  (`--exclude-answers-min-length` default 4) because excluding them makes 9x9
  grids unfillable. Operator decision (Jeff, June 2026): a non-zero gate exit
  whose dupes are ALL short-window is NOT a blocker — proceed to upload. (This
  has been re-litigated repeatedly; it is settled.)
- **`blocking` (must fix):** any duplicate that is NOT short-window — i.e. a
  >=4-letter answer shared by two puzzles, or any shared answer the scheduler's
  windows would actually collide on. The gate prints a `N blocking` count;
  if it is `0 blocking`, upload regardless of the non-zero exit.

So: read the summary line. `0 blocking, K short-window` → upload as-is. Any
blocking dupes → regenerate the affected puzzles (same batch id, same seeds,
separate `-replace-*` output root, the affected `--buckets`), passing the
batch answers file so the refill cannot reuse any answer already in the
batch:

```bash
uv run crossword-generator generate-pilot-batch \
  --output-root output/batches/<batch-id>-replace-<bucket> \
  --batch-id <batch-id> \
  --buckets <difficulty>/<size> \
  --count 1 \
  --seed-start <original-seed> \
  --exclude-answers-file output/batches/<batch-id>/batch-answers.txt \
  --llm claude
```

The same seed refills differently because the pool changed. Re-run
`check-batch-answers` across the combined set (kept + replacements) until
clean, then upload the main manifest and the replacement manifests (the
replacement upload overwrites the deterministic keys when the originals were
already uploaded; for a fresh batch, upload main first, then replacements
with `--replace-existing`).

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

## Daily vs Unlimited Batches — generation flags (READ FIRST)

The single most-repeated decision: a batch is either **dated dailies** (placed
on consecutive calendar slots) or an **unlimited pool** (pulled by `public_id`,
never schedule-adjacent). They need OPPOSITE generator flags. Decide which one
the request is, then use the matching column. When unsure, ask — getting it
wrong either ships an unschedulable daily week or needlessly starves an
unlimited batch's fill pool.

| Concern | DAILY (dated week) | UNLIMITED (pool) |
|---|---|---|
| `--intra-batch-dedup` | ON (default) — week is scheduled across consecutive days, so answers MUST be disjoint or they trip the no-repeat windows | `--no-intra-batch-dedup` — never adjacent; disjointness only hurts fill |
| `--exclude-recent-answers` | ON (default) — avoid answers already scheduled around the first open slot | `--no-exclude-recent-answers` — no schedule to collide with |
| `--exclude-scheduled-sixty` | ON (default) — for hard 7x7/9x9, respect the 180-day HGG-60 window | `--no-exclude-scheduled-sixty` |
| `--avoid-existing-clues` | ON (default) | ON (default) — same either way |
| `check-batch-answers` gate | RUN IT — must be all-unique before upload; regenerate dups | SKIP — intra-batch overlap is expected by design |
| `--max-workers` | 1 (guarantees intra-batch uniqueness by construction) | 6+ (no shared-answer constraint, so parallel is safe) |
| Answer scans (nsfw / removed / terminal-S) | run both | run both |
| Token needed at generation time | yes (exclusions hit the admin API) | yes (only for `--avoid-existing-clues`) |

After upload, the destination also differs — see the two subsections below
(daily → leave as draft candidates for the admin to schedule, or
`schedule-daily`; unlimited → `publish-unlimited`).

### Daily (dated) batches

Daily batches use the DEFAULT flags — do NOT pass any of the `--no-*`
exclusion/dedup flags — **when the batch targets ~30 puzzles or fewer**
(Neil, 2026-07-10). Above that, pass `--no-intra-batch-dedup` (parallel
`--max-workers` then OK; keep `--exclude-recent-answers` /
`--exclude-scheduled-sixty` ON) and rely on the scheduling-time no-repeat
windows to space shared answers. Rationale: the shared used-answer set grows
with every completed puzzle, and on a 200-puzzle run it (a) degraded hard/9
fills to ~1h each and (b) caused 21 of 32 hard/9 boards to exhaust all grid
variants and export their best FAILING board (hard_cross/proper_noun_cap,
score 0.0) while the batch reported "ok" — removing that much easy fill
forces the CSP onto Hard-list words, making hard-cross violations
unavoidable. The runner does NOT mark fill-threshold exhaustion as failure
and `save-generated-puzzles` only auto-blocks `LEAK:`/`DUPLICATE:`, so after
any slow batch verify `fill.grade_report.passing` in the final envelopes (or
grep manifest `error_message` for "Fill quality below threshold") before
upload. Example: a week (7) of midi hard 9x9 dailies:

```bash
uv run crossword-generator generate-pilot-batch \
  --output-root output/batches/daily-midi-hard9-<date> \
  --batch-id daily-midi-hard9-<date> \
  --buckets hard/9 \
  --count 7 \
  --max-workers 1 \
  --llm claude
```

Then run the duplicate-answer gate (see "Intra-Batch Duplicate-Answer Gate")
and the answer scans, regenerating any flagged puzzles, BEFORE upload.

Upload paths for a daily batch:
- **Draft only (default ask):** `save-generated-puzzles` writes draft
  candidates to `crosswords/generated-puzzles`; the admin/editor then schedules
  them onto calendar slots from the review UI. The generator has NO scheduling
  command. "Upload to the proposed/daily bucket" usually means exactly this —
  there is no "proposed" status in hey-you; daily slots are created as
  `status=scheduled` only when actually scheduled.
- **Direct schedule (only if explicitly asked):** the admin "Schedule" action
  calls `POST /admin/crossword-puzzles/{record_id}/schedule-daily`
  (`{"track":"easy"|"hard","date":"YYYY-MM-DD"}` — date optional; omit to let
  the server place it at the first open slot for that game/track). This writes
  the LIVE daily calendar and deletes the candidate, same shape as
  `publish-unlimited`. Do not do this without explicit instruction.

The generator cannot pin the recent-answer window to an arbitrary date; it
auto-detects the first unscheduled slot server-side. If the operator says "the
first open slot is <date>", the default detection already lands there.

## Unlimited-Pool Batches (non-scheduled)

Unlimited puzzles are NOT placed on dated daily slots — players pull them from
a published pool by `public_id`. This changes the batch workflow in four ways
versus a dated weekly batch:

1. **Disable date-based exclusions.** There is no schedule to avoid colliding
   with, so pass `--no-exclude-recent-answers --no-exclude-scheduled-sixty`.
   (Scheduled-sixty only matters for hard 7x7/9x9 anyway.)
2. **Disable intra-batch answer dedup** with `--no-intra-batch-dedup`. For
   dated dailies the run forces every puzzle's answers to be disjoint so a
   week scheduled across consecutive days can't trip the no-repeat windows.
   Unlimited puzzles are never schedule-adjacent, so forcing 50 minis to have
   disjoint answer sets only starves the fill pool and hurts quality. With this
   off, the `check-batch-answers` gate is NOT meaningful (intra-batch answer
   overlap is expected) — skip it. Still run the disallowed-answer / terminal-S
   scans (Answer Scans Before Upload).
3. **Keep answer-novelty weighting on.** `generate-pilot-batch` defaults
   `--unlimited-answer-novelty` on, but it only activates when
   `--no-intra-batch-dedup` is passed. The runner fetches active
   `crosswords/unlimited-pool` records for the requested size/difficulty,
   counts existing answers, collects multiple passing fill candidates
   (`--answer-novelty-candidates`, default 8), and picks the board with the
   lowest frequency-weighted answer reuse. As each puzzle in the batch
   completes, its answers increment the same in-memory weights for later
   puzzles. With parallel workers, in-flight puzzles cannot see each other;
   completed batch-mates are still counted by later workers.
4. **Keep `--avoid-existing-clues` on.** Clue-angle variety vs. the live corpus
   is still wanted and is unrelated to scheduling. Requires a prod admin token.

Generate (example: 50 easy 5x5 for the mini unlimited pool):

```bash
uv run crossword-generator generate-pilot-batch \
  --output-root output/batches/unlimited-easy5-<date> \
  --batch-id unlimited-easy5-<date> \
  --buckets easy/5 \
  --count 50 \
  --no-intra-batch-dedup \
  --no-exclude-recent-answers \
  --no-exclude-scheduled-sixty \
  --avoid-existing-clues \
  --answer-novelty-candidates 8 \
  --max-workers 6 \
  --llm claude
```

Hard 5x5 is the same with `--buckets hard/5` (selects `config.hard5.yaml`:
hard clue difficulty, `hgg-hard.txt` fill, the hard-cross board rules). Hard
clue scores run lower than Easy (different grading bar) — that is expected,
not a defect.

### Two-step publish: upload candidates, then promote

Uploading with `save-generated-puzzles` only writes DRAFT candidates to
`crosswords/generated-puzzles` — it does NOT publish to the unlimited pool.
The pool is the data-store collection `unlimited-pool` with `status=active`;
records land there only via the promote endpoint the admin "Save to Unlimited"
button calls.

Upload guard reminder: `save-generated-puzzles` refuses puzzles only for
surviving `LEAK:`/`DUPLICATE:` issues. A "clue quality below threshold"
`error_message` does NOT block upload — those puzzles upload unless you exclude
them. To hold back specific seeds (e.g. low-scoring hards), write a filtered
manifest copy dropping those `results` entries and upload that copy:

```bash
# filtered manifest dropping seeds 9 and 20, then upload the 48 that remain
python3 - <<'PY'
import json
m=json.load(open("output/batches/<batch>/manifest.json"))
m["results"]=[r for r in m["results"] if r["seed"] not in {9,20}]
json.dump(m, open("output/batches/<batch>/manifest.publish.json","w"), indent=2)
PY
uv run crossword-generator save-generated-puzzles \
  --manifest output/batches/<batch>/manifest.publish.json --dry-run
# (then live upload with hgg-auth exec prod, see Upload Contract)
```

Then promote each uploaded candidate to the unlimited pool. There is no
generator CLI for this; call the hey-you admin API directly (per-record, by the
candidate's data-store record ID — NOT the `generated:...:seed-N` key):

```
POST /admin/crossword-puzzles/{record_id}/publish-unlimited
body: {"difficulty": "easy"}   # or "hard"
```

The server creates an active `unlimited-pool` publication, assigns the next
`public_id` for that route scope (e.g. `unlimited:5x5:N`), archives the
candidate, and DELETES it from `generated-puzzles`. So the authoritative
success check is that the batch's candidate count in `generated-puzzles` drains
to 0.

Collect the record IDs by listing `generated-puzzles` filtered to the batch id,
then loop the promote call:

```bash
hgg-auth exec prod -- bash -c '
  BASE="$HGG_ADMIN_BASE_URL/api"
  # 1. collect candidate record IDs for this batch (paginate)
  # GET $BASE/admin/data-store/records?namespace=crosswords&collection=generated-puzzles&game_key=minicrossword&per_page=100&page=N
  #    keep rows whose "key" contains the batch id; capture each row "id"
  # 2. for each id: POST $BASE/admin/crossword-puzzles/<urlencoded id>/publish-unlimited  -d {"difficulty":"easy"}
  # 3. verify: re-list and confirm 0 candidates remain for the batch id
'
```

**Byline / author.** The server stamps a personal byline only when the
publishing user's id is in hey-you's `BYLINE_USER_IDS` (currently just Jeff
Chen). Anyone else publishes with NO byline, and the client falls back to the
house default "Hey Good Game". So to publish under the Hey Good Game byline,
just publish as any admin who is not a credited constructor — no flag needed.

**Game keys.** 5x5 and 7x7 -> `minicrossword`; 9x9 -> `midicrossword`. Mini
unlimited is nested by size (`unlimited:5x5:N`); midi unlimited is flat.
