# Proposal: 6-Month Rolling No-Repeat Window for 60-Point Entries

**Status:** Implemented 2026-06-09 (hey-you controller + endpoint, generator
exclusion); pending hey-you deploy
**Author:** Neil Berget (with Claude)
**Date:** 2026-06-09
**Requested by:** Jeff Chen (Discord, 2026-06-08): keep a list of used 60-pt
entries so "feature" answers don't repeat for ~6 months; at least a couple of
months is acceptable.
**Related:** `hey-you/app/Http/Controllers/Api/CrosswordPuzzleController.php`,
`dictionaries/hgg-60.txt`, hey-you `crossword_daily_answers` /
`crossword_effective_dictionary_entries`

## TL;DR

Extend the existing scheduling-time answer-repeat check in hey-you from a
single ±6-day window to a **two-tier window**: ±6 days for ordinary answers,
**±180 days for answers in the active HGG 60 list**. The infrastructure is
already in place — every scheduled daily's answers are stored in
`crossword_daily_answers` keyed by `day_number`, and the active HGG 60 list is
already stored server-side in `crossword_effective_dictionary_entries`
(`dictionary_slug = 'hgg-60'`). This is a small, contained change to one
controller plus a generator-side exclusion so hard candidates don't arrive
pre-conflicted.

**Feasibility:** comfortably. The HGG 60 pool is ~8,100 entries. Dailies burn
~20 sixties/week (two hard 7×7 minis at exactly one each, seven hard 9×9 midis
at ≤3 each). A 180-day window locks up ~520 entries — **~6% of the pool**. Even
a 12-month window would only lock ~13%.

## Current state

### The 7-day rule (what Jeff calls "our current rule")

Lives entirely in hey-you's `CrosswordPuzzleController`:

- `insertDailyAnswers()` writes every normalized answer (across + down, len > 1)
  of every scheduled daily slot into `crossword_daily_answers` with
  `day_number`, `game_key`, `track`.
- `dailyAnswerConflict()` rejects a placement if any answer already appears in
  a scheduled slot `whereBetween('day_number', [$day - 6, $day + 6])` —
  **cross-game and cross-track** (no game/track filter, intentionally).
- `validatePlannedDailyAnswers()` enforces the same `abs(diff) <= 6` rule
  pairwise during bulk slot reassignment.
- `findFirstAvailableDay()` walks forward up to 730 days to find a
  conflict-free slot.

### Where 60-pointers appear

Only hard puzzles contain them, by construction:

- Hard 7×7: exactly one 7-letter HGG 60 entry (fill-grader rule).
- Hard 9×9: all 8/9-letter slots are 60s (`min_score_by_length {8:60, 9:60}`),
  capped at 3 long slots (`max_long_entries_8_9: 3`).
- Easy puzzles and hard 5×5: none (source-score-60 entries are excluded from
  HGG Easy; hard 5×5 has no 60s per Jeff).

So the rule only bites on the hard track: ~2 sixties/week from minis +
~14–21/week from midis.

### The 60-pt list is already server-side

`publish-effective-dictionaries` posts the HGG Easy + HGG 60 snapshot to
hey-you, stored per-word in `crossword_effective_dictionary_entries` with
`dictionary_slug = 'hgg-60'` on the active batch. Membership lookup at
scheduling time is a single indexed query (or a cached set — ~8k words).

## Proposed change

### 1. hey-you: two-tier conflict window (the core)

In `CrosswordPuzzleController`:

```php
private const ANSWER_WINDOW_DAYS = 6;        // existing rule
private const SIXTY_ANSWER_WINDOW_DAYS = 180; // Jeff's feature-entry rule
```

- `dailyAnswerConflict()`: split incoming `$answers` into sixty / regular by
  checking against the active `hgg-60` entry set; run the existing
  `whereBetween` once per tier with the matching window. Return the same
  conflict shape (include which window triggered, for the error message).
- `validatePlannedDailyAnswers()`: same split; threshold per answer is 180 if
  the answer is in the sixty set, else 6.
- `updateOfficial()` (the edit-time check) inherits the fix via
  `dailyAnswerConflict()` automatically.
- Error message should say which rule fired, e.g.
  `"Assignment would repeat MOONWALK within 180 days (60-pt entry) ..."` —
  reviewers need to know why a slot six weeks out is refused.

Membership uses the **active snapshot at check time**. When Jeff rescores a
word out of (or into) the 60 list, future checks follow the new list; history
is not re-litigated.

Notes:

- `crossword_daily_answers` needs an index on `(normalized_answer, day_number)`
  if one doesn't already exist — the 180-day scan covers ~360 days × ~40
  answers/day ≈ 14k rows, trivial either way.
- `findFirstAvailableDay()` needs no change — it already retries forward and
  will naturally skip past 60-pt conflicts. But see §2: with a 180-day window a
  conflicted candidate isn't "pushed a few days," it's **dead for months**.

### 2. Generator: don't produce pre-conflicted candidates

With a ±6-day window, a colliding candidate just lands a week later. With
±180 days, a hard candidate whose 60-pointer was used last month is
unschedulable until the window expires. At ~520 locked entries the per-puzzle
collision odds are ~6% for hard 7×7 and ~15–18% for hard 9×9 (1−(1−0.064)³) —
enough to be a steady annoyance in review.

Fix it at fill time: subtract scheduled-recently 60s from the HGG 60 pool
before composing the hard 7×7 / 9×9 dictionaries.

- New hey-you endpoint (admin, read):
  `GET /api/admin/crossword-daily-answers/recent-sixty?window_days=180`
  → distinct `normalized_answer` values from `crossword_daily_answers` in
  scheduled slots within ±window of today, intersected with the active
  `hgg-60` set. (Cheap; could also be a generic recent-answers endpoint that
  the generator intersects locally with its own `hgg-60.txt`.)
- `generate-pilot-batch` fetches that list when the batch includes `hard/7` or
  `hard/9` buckets (flag `--exclude-scheduled-sixty/--no-exclude-scheduled-sixty`,
  default on, requires the admin token that batch runs already use for clue
  history) and removes those words from the `hgg-60.txt` slice merged via
  `additional_paths`. Log the count removed.
- Pool impact is negligible: 8,100 → ~7,600 effective. The filler won't notice.

The scheduling-time check (§1) remains the source of truth — the generator
exclusion is an optimization so candidates arrive clean, not the enforcement
point. The window only moves forward between generation and scheduling, and
batches are scheduled within days of generation, so drift is not a practical
concern.

### 3. Knobs we are NOT turning yet

- **General midi window 6 → 4–5 days:** Jeff offered this relaxation if the
  midi answer volume makes scheduling tight. Nothing observed suggests we need
  it; it stays a one-constant change (`ANSWER_WINDOW_DAYS`) if we ever do.
- **Unlimited pool:** unlimited puzzles aren't date-scheduled, so no rolling
  window applies. If we generate "thousands of unlimiteds," 60-pt variety
  there is a different problem (cross-puzzle dedup within the pool, not a time
  window) — out of scope here, worth a one-line mention to Jeff.
- **Window > 6 months:** Jeff said "as large as possible." 180 days locks ~6%
  of the pool; 365 locks ~13%. Both fine. Recommend launching at 180 and
  revisiting after we see real scheduling friction, since the constant is
  trivially adjustable.

## Feasibility math

| | per week | per 180 days |
|---|---|---|
| Hard 7×7 minis (2/wk × 1 sixty) | 2 | 52 |
| Hard 9×9 midis (7/wk × ≤3 sixties) | ≤21 | ≤546 |
| **Total locked** | ~20 | **~520 of 8,138 (~6%)** |

Pool: `dictionaries/hgg-60.txt`, 8,138 entries (lengths 7–9), mirrored
server-side per publish snapshot.

## Work plan

1. **hey-you** — two-tier window in `dailyAnswerConflict()` +
   `validatePlannedDailyAnswers()`, constants, sixty-set lookup (cached per
   request), index check, feature tests (conflict inside 180d, clean outside,
   regular answers still 6d, reassignment path, edit path). ~half day.
2. **hey-you** — `recent-sixty` endpoint + test. ~1–2 hours.
3. **crossword-generator** — fetch + subtract in `generate-pilot-batch`
   dictionary composition for hard buckets, flag, manifest note, tests
   (`test_cli_batch.py`, dictionary composition). ~half day.
4. Update `.claude/skills/crossword-generator/references/generator-workflow.md`
   (fill-quality/scheduling rules section) once shipped.

Order matters only in that §1 can ship alone and immediately satisfies Jeff's
request; §2/§3 remove the review-time friction it introduces.
