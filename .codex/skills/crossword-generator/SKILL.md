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
- `generate-pilot-batch` refreshes word lists by default before filling:
  it pulls the latest admin-managed lists, rebuilds local `hgg-easy.txt`,
  `hgg-hard.txt`, and `hgg-60.txt`, then continues generation. Do not pass
  `--no-refresh-dictionaries` unless explicitly doing an offline/local-file
  experiment.
- Before uploading, run a dry run and scan answers for known disallowed
  patterns.
- FIRST decide whether a batch is DATED DAILIES or an UNLIMITED pool — they
  need OPPOSITE generator flags, and this is the most-repeated decision. Ask if
  the request doesn't make it clear. See the "Daily vs Unlimited Batches"
  decision table in references/generator-workflow.md. In short:
  - DAILY (consecutive calendar slots): use the DEFAULT flags — do NOT pass any
    `--no-*` flag. Intra-batch dedup, recent-answer exclusion, and
    scheduled-sixty exclusion all stay ON; use `--max-workers 1`.
    SIZE EXCEPTION (Neil, 2026-07-10): intra-batch dedup only scales to
    batches of ~30 puzzles or fewer. For larger daily batches pass
    `--no-intra-batch-dedup` (parallel `--max-workers` then OK; keep the
    recent/sixty exclusions ON) and let the scheduler's no-repeat windows
    space any shared answers. Full disjointness on a big batch starves the
    fill pool: a 200-puzzle run degraded hard/9 fills to ~1h each and
    silently exported 21 failing boards (hard_cross/proper_noun_cap,
    score 0.0) marked "ok" — dedup had removed so many easy words that
    hard-list crossings became unavoidable. The runner exports best-effort
    FAILING boards on exhaustion and upload only auto-blocks
    LEAK:/DUPLICATE:, so also verify `fill.grade_report.passing` (or
    manifest `error_message` fill-threshold text) before uploading any
    batch that ran slow. Run
    `check-batch-answers` BEFORE upload and read its summary. It judges
    3-letter 9x9 repeats against the scheduler's +/-2-day window ASSUMING
    THE SET IS SCHEDULED IN SEED ORDER (each puzzle's day = its seed rank in
    its bucket, easy/hard tracks aligned by day): `short-window` = the two
    puzzles are 3+ days apart in that order (fine — upload as-is);
    `blocking` = a >=4-letter shared answer OR a 3-letter repeat within 2
    days. Regenerate only blocking offenders, as a continuation run with
    `--prior-batch-manifest <manifest>` (seeds the used-answer counts so the
    new puzzles take the next days) — never with `--exclude-answers-min-length
    3`, which starves 9x9 fill (2026-09-01: puzzle 5 of 7 ran >4h and failed).
    The generator enforces the same window while filling
    (`--intra-batch-short-window 2`, soft `--intra-batch-short-penalty`,
    backstop `--intra-batch-short-cap`), so a clean run usually gates clean.
    Tell the scheduler to place each bucket in seed order. Jeff (2026-09-01):
    a set where ALL/TED/OWE each hit 3-4 puzzles could only fill 2 of 7 open
    slots, so multiplicity/spacing — not pairwise repeats — is the constraint.
    Upload as draft candidates
    with `save-generated-puzzles`; the admin
    schedules them (there is no "proposed" status — daily slots become
    `scheduled` only when actually scheduled, via the admin `schedule-daily`
    action, which the generator does NOT do). Only call `schedule-daily` if
    explicitly told to write the live calendar.
  - OPEN-DAY HOLES (the schedule already runs months ahead — the normal
    state since Sept 2026 — and the ask is "a week of dailies" / "fill the
    open days"): run `fill-open-days --through <date>` (all four game/track
    runs, chained, gated, dry-run; add `--upload` to upload), or per track
    `--target-game <game> --target-track <track> --target-through <date>`
    (or `--target-dates`) instead of `--buckets/--count`. The default daily
    path now refuses to start (open-day guard) when the holes are scattered. One puzzle per open day, each excluding
    the answers the scheduler would reject on its own date, tagged with
    `target_date`/`publish_slot` so the reviewer schedules it on that day.
    The default first-unscheduled-slot window mis-targets scattered holes
    (Sept 2026: 14 easy midis walked into 2027). See "Open-day targeting"
    in references/generator-workflow.md.
  - UNLIMITED (pool, never schedule-adjacent): pass
    `--no-intra-batch-dedup --no-exclude-recent-answers --no-exclude-scheduled-sixty`
    (`--max-workers 6`+ fine); keep default `--unlimited-answer-novelty`
    on so active unlimited-pool answers and already-completed batch answers
    lower candidate-board priority; SKIP `check-batch-answers` (some overlap
    is still expected and not a scheduling blocker); after uploading draft
    candidates, PROMOTE each via
    `POST /admin/crossword-puzzles/{record_id}/publish-unlimited`
    (`{"difficulty":"easy"|"hard"}`). `save-generated-puzzles` alone does NOT
    publish to unlimited.
  - Either way: keep `--avoid-existing-clues` ON and run the answer scans
    (nsfw / prevalent-removed / terminal-S) before upload.
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
- Unlimited-pool (non-scheduled) batch generation and the two-step
  upload-then-`publish-unlimited` promotion flow
- Quality guardrails, including terminal-S duplicate variants
- Verification commands and common failure modes

## Common Commands

Batch generation refreshes dictionaries and loads prior-clue history from the
admin API by default (`--refresh-dictionaries`, `--avoid-existing-clues`), so
`HEYGG_CROSSWORD_GENERATOR_TOKEN` (service account; falls back to `HEYGG_ADMIN_TOKEN`, then `HEYGG_ADMIN_API_TOKEN`) must be set in the environment
before generating — not just before uploading. The dictionary refresh pulls
Jeff's latest Easy/Hard moves from the admin list UI into local effective
generator files before any fill starts. The clue history feeds already-used
clues into the generation prompt so the model avoids exact repeats and varies
its clue angles per answer. Only pass `--no-refresh-dictionaries` or
`--no-avoid-existing-clues` for throwaway local experiments.

Refresh dictionaries without generating:

```bash
uv run crossword-generator refresh-dictionaries
```

Seeds default to a RANDOM start per run (echoed in the output and recorded
in the manifest) so repeated batches explore different grid patterns and
fills. Only pass `--seed-start` to reproduce a specific prior run.

Generate a clean cross-site Easy batch using the default 5x5:7x7:9x9 ratio:

```bash
uv run crossword-generator generate-pilot-batch \
  --output-root output/batches/<batch-id> \
  --batch-id <batch-id> \
  --buckets easy/5,easy/7,easy/9 \
  --bucket-counts 5=5,7=2,9=7 \
  --llm claude
```

Generate a full Easy/Hard batch using the same cross-site size ratio:

```bash
uv run crossword-generator generate-pilot-batch \
  --output-root output/batches/<batch-id> \
  --batch-id <batch-id> \
  --bucket-counts 5=5,7=2,9=7 \
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
