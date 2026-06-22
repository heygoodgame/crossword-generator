# First-Class Hint System — Implementation Plan

**Status:** Draft for review
**Author:** (drafted with Claude Code)
**Scope:** `crossword-generator`, `hey-you`, `heygg-common`, `crossword-midi-and-mini`

## Summary

Make "hints" a real, first-class feature. A **hint is one easier alternate
clue** per entry. When a solver asks for a hint on a clue, they first see this
easier clue; the existing **letter-reveal** stays as the next escalation step.

```
HINT pressed  →  show easier clue (the hint)  →  still optionally Reveal letters
```

Decisions locked with the team:

- **Hint model:** single easier clue per entry (not progressive tiers).
- **Source of truth:** inline in the ipuz clue tuple (the 3rd element).
- **Coverage:** every clue gets a hint.
- **QA:** hints reuse the existing leak detector + Opus fact-checker.
- **Backfill:** all puzzles, via the Anthropic Batches API + prompt caching.
- **Published-puzzle backfill:** through a hey-you admin endpoint (validated).
- **Game UI:** cherry-pick Nate's `hint-improvements` hint commits.

---

## Background: what already exists

### Two layers of "hint" in the game

1. **Primitive reveal (on `main`).** `CluePanel.tsx` shows a **HINT** button on
   the active clue. Pressing it marks the clue "hinted" and unlocks a
   `WordProgress` row with a **Reveal** button that fills one letter at a time
   (`revealNextLetter` in `GameContext`). Today "hint" only means "let me
   reveal letters."

2. **Nate's richer hint (branch `origin/hint-improvements`, commit `8520d92`).**
   Threads an *easier-clue string* through the stack:
   - `Clue.hint?: string` added to `src/types/puzzle.ts` (**already on main**).
   - ipuz clue tuples extended `[number, text]` → `[number, text, hint?]`,
     parsed in `puzzle-loader.ts`.
   - When a clue is hinted, `CluePanel`/`ActiveClue` render the easier clue
     *above* the Reveal row.
   - A **dev-only fixture** (`src/dev/dev-hints.ts`) injects hints in DEV only,
     because the API doesn't serve them yet. This is preview scaffolding.

Nate built the **UI and the data shape** but stopped before real persistence,
admin editing, generation, and backfill. This plan finishes those.

### How a puzzle reaches a player

`game → hey-you API (/api/crosswords/...) → ipuz JSON`

- Generator produces ipuz and uploads records to the **data store**
  (`namespace=crosswords`, `collection=generated-puzzles`).
- Admin (in `crossword-midi-and-mini`) reviews/edits and promotes/publishes.
- hey-you stores published puzzles in `crossword_published_puzzles.ipuz` and
  serves them at `/api/crosswords/{gameKey}/daily/...` and `/unlimited/...`.

---

## Data format

The hint rides as the **optional 3rd element** of each ipuz clue tuple — the
shape Nate's UI already reads:

```json
"clues": {
  "Across": [[1, "Capital of France", "This city has the Eiffel Tower"]],
  "Down":   [[2, "Feline pet", "Says 'meow'"]]
}
```

### Why inline tuple element (vs an `hgg.*` extension)

This repo's existing convention (`docs/ipuz-extensions.md`) namespaces custom
data under `hgg.*` (e.g. `hgg.references`). A hint *could* live there as
`hgg.hints`. We are choosing the **inline 3rd element** instead because:

- It is valid ipuz; other readers ignore extra tuple elements.
- Nate's game UI **already parses and renders it** — least new code.
- It keeps the hint physically next to the clue it belongs to.

Trade-off: it is less self-documenting than an `hgg.*` block. We will document
it in `docs/ipuz-extensions.md` so the convention is explicit.

### Verified: the format survives every hop unchanged

- ✅ **Game** already parses the 3rd element (`puzzle-loader.ts`).
- ✅ **hey-you** `validateOfficialIpuz` only checks ipuz *shape*; it returns and
  stores the ipuz verbatim, so the 3rd element passes through serve/store.
  **No serving-path change required** for data to flow.
- ⚠️ **Admin** `buildEditedIpuz` (`dataStoreAdapter.ts`) rebuilds clue rows as
  `[number, text]` and **silently drops the hint**. This is the one real bug to
  fix for admin round-tripping.

---

## Workstreams

### W1 — Generator: new puzzles ship with hints
**Repo:** `crossword-generator`

1. `models.py` — add `hint: str = ""` to `ClueEntry`.
2. New step `steps/hint_step.py` (mirrors `clue_step.py`). Runs **after** clue
   generation so it can see the real clue and take a different angle. Chunked +
   cache-warmed like `clue_step` for cost.
3. New prompt `llm/prompts/hint_generation.py`. Explicit instructions:
   - "Write a **very simple, very easy** hint — a beginner or child could get it."
   - "The hint must **never contain or reveal the answer**" (reuse the strict
     abbreviation/morphological/leak rules already in `clue_generation.py`).
   - "Take a **different angle** from the main clue; don't just reword it."
4. **QA reuse:** run hints through `graders/leak_detector.py` and the Opus
   `clue_fact_check` pass; regenerate any hint that leaks or is wrong.
5. `exporters/ipuz_exporter.py` — append `clue.hint` as the 3rd tuple element
   when non-empty (the only change: extend `pair`).
6. Wire `hint_step` into `pipeline.py` + `config.yaml`; update the repo-local
   skill (`.claude/skills/crossword-generator/`).

### W2 — Backfill all existing puzzles
**Repo:** `crossword-generator` — new `scripts/backfill_hints.py`

- Reuses `data_store.py` (`list_generated_puzzle_records`, patch via
  `save_data_store_record`) and `claude_provider.py`.
- **Anthropic Batches API + prompt caching:** one cached system prompt (hint
  rules) shared across requests; batch all clues. Latency-tolerant (hours),
  cheapest at scale.
- For each record: collect clues missing a hint → batch request → write hints
  into the ipuz 3rd element in **both** `data.ipuz` and (if present)
  `data.hgg_admin_edit.puzzle` → patch record. Same leak/fact-check QA as W1.
- **Idempotent:** skip clues that already have a hint; safe to re-run.
- **Published puzzles** live in hey-you's `crossword_published_puzzles` table,
  not just the data store. The script patches those through a new hey-you admin
  endpoint (see W5) so **live** puzzles get hints without a re-publish.

### W3 — Admin: view + edit hints (published and unpublished)
**Repo:** `crossword-midi-and-mini`

1. **Fix the drop:** `dataStoreAdapter.ts` — preserve the 3rd element in
   `clueRowsFromPuzzle`/`buildEditedIpuz`; add a `hintOverrides` map parallel to
   `clueOverrides`, persisted in the `hgg_admin_edit` overlay.
2. `EditPuzzle.tsx` + `PublishedEditPuzzle.tsx` — add an inline **hint field**
   under each clue editor (same pattern as clue text). Optional later:
   "regenerate hint" button.
3. Published edits flow through hey-you `updateOfficial`, which already passes
   ipuz verbatim — edited hints persist once the client sends the 3rd element.

### W4 — Game UI: surface it
**Repo:** `crossword-midi-and-mini`

- **Cherry-pick `8520d92`** (the hint-display commit) onto a fresh branch off
  `main`; leave Nate's unrelated SEO/color/scrollbar commits out of this PR.
- Once the API serves real 3rd elements, retire the dev fixture
  (`src/dev/dev-hints.ts`) and the `applyDevHints` shim in `puzzle-loader.ts`.

### W5 — hey-you: published-puzzle hint endpoint
**Repo:** `hey-you` (worktree feature branch — main tree gets reset by parallel
sessions)

- Add an admin route (or extend `updateOfficial`) to patch hints into a
  published puzzle's ipuz, validated by `validateOfficialIpuz`. Used by the W2
  backfill for live puzzles. No change needed to the *serving* path — it already
  returns ipuz verbatim.

### W6 — Shared types
**Repo:** `heygg-common`

- If the ipuz / `CrosswordPuzzleMetadata` types are shared here, widen the clue
  tuple type to `[number, string, string?]`.

---

## Suggested sequencing

Land in this order so the loop is **visible early**:

1. **W3 + W4** — fix the admin hint drop, add the admin editor, ship Nate's UI.
   Hand-author a couple of hints in admin and watch them render in-game.
2. **W1** — generator emits hints for new puzzles.
3. **W5 + W2** — hey-you endpoint, then the batch backfill for the archive.

---

## Risks / open items

- **Hint quality at scale.** Reusing leak + fact-check mitigates leaks and
  factual errors, but "is this actually *easier*?" is subjective. Spot-check a
  sample from the first backfill batch before running the full archive.
- **Backfill cost.** Every clue, every puzzle. Batches API + caching keeps it
  low, but estimate token volume before launching the full run.
- **Two storage homes.** Data-store candidates and hey-you published puzzles
  are separate; the backfill must hit both (W2 + W5) or live puzzles stay
  hint-less.
- **Convention drift.** Inline 3rd element vs `hgg.*` — document the choice in
  `docs/ipuz-extensions.md` so future readers aren't surprised.
