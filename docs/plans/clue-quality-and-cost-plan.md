# Clue Quality & Cost Optimization Plan

**Status:** Approved — decisions locked 2026-06-07
**Author:** Audit of batch `weekly-1w-20260607-145850-s624518`
**Date:** 2026-06-07
**Owner:** Neil

### Locked decisions

1. **Stuck leaks → soft error + block upload.** A leak surviving all repair
   attempts is appended to `envelope.errors`; the puzzle still saves as a draft,
   the batch continues, but the upload guard refuses to push it. (Not a hard
   raise — one stuck clue must not lose the whole puzzle.)
2. **Morphology = lightweight stemmer + curated map that grows from misses.**
   A Porter/Snowball-style stemmer catches regular inflections
   (`teaches`/`teacher`, `trims`/`trim`) automatically; a curated high-precision
   map handles irregular plurals (`wife`/`wives`) and abbreviation expansions
   (`est`→`eastern standard time`). The map is seeded small and grown from real
   misses observed in batches.
3. **Opus generation effort = A/B medium vs. high.** Phase 3 runs a small batch
   at each effort level, compares pre-filter defect rates via the Phase 1
   detector, and picks the winner before locking the default.
4. **`clue_grading` → Sonnet 4.6 now.** Grading is the leak gate that's been
   failing; upgrade the judge immediately for safety (~+$0.3/batch) rather than
   deferring to data.

---

## 0. Background & motivation

We audited all 82 LLM calls across the 14 puzzles in batch
`output/batches/weekly-1w-20260607-145850-s624518/` (5×5 ×5, 7×7 ×2, 9×9 ×7),
plus the model-assignment and caching code. Two questions drove it: **are we
spending model budget in the right places**, and **why do obvious clue defects
(answer word appearing in its own clue, abbreviation-expansion leaks) still slip
through** despite the system prompt forbidding them.

### What we found

**Cost is not the constraint.** The batch cost **$1.27 total (~$0.09/puzzle,
~$66/yr at this cadence)**. Even tripling spend is ~$200/yr. This reframes
everything below: we optimize for **quality first**, and treat the cost levers
as cleanup, not priorities.

| Step | Model | Calls | Cost | % of batch |
|---|---|---|---|---|
| `clue_generation` | Sonnet 4.6 | 27 | $0.736 | 58% |
| `clue_grading` | Haiku 4.5 | 27 | $0.308 | 24% |
| `puzzle_naming` | Sonnet 4.6 | 14 | $0.134 | 11% |
| `clue_fact_check` | Sonnet 4.6 | 14 | $0.090 | 7% |
| **Total** | | **82** | **$1.27** | |

**The pipeline already has surgical per-clue repair.** Contrary to the initial
read, `ClueWithGradingStep` (`src/crossword_generator/steps/clue_grading_step.py`)
already does targeted, per-clue repair via `_repair_entries` +
`build_clue_repair_messages`, plus a fact-check repair pass and a duplicate-clue
repair pass. So "one call per clue" is **partly already built** — the gap is not
the repair mechanism, it's **what triggers it**.

**The leak gate is fragile.** Leaks are only repaired when:
1. the Haiku grader's *fairness* sub-score drops low enough, **or**
2. the grader's free-text `feedback` happens to contain one of a hardcoded list
   of marker strings (`"leaks"`, `"contains the answer"`, `"exact answer"`, …)
   matched in `_should_repair_grade`.

Both are LLM-judgment-dependent. When Haiku grades ~30 clues in one bulk call, a
single bad clue (e.g. `SOY` → `"___ sauce"`, which scored fairness **5/100** in
this batch) can pass the *puzzle-level* threshold and only gets repaired if the
keyword match fires. There is **no deterministic, mechanical leak check** —
even though the leaks we care about (answer-in-clue, shared root,
abbreviation-expansion) are mechanically detectable with zero LLM cost.

**Caching is configured but mostly inert.** The code sets
`cache_control: ephemeral` on the system block
(`claude_provider.py:89-96`), but the prompts fall under the **model-specific
minimum cacheable prefix**, so Anthropic silently skips caching:

| Step | Model | Min to cache | Sys prompt size | Caches? |
|---|---|---:|---:|---|
| `clue_generation` (midi 9×9) | Sonnet 4.6 | 2048 tok | ~1,250 tok | ✅ partial |
| `clue_generation` (mini 5/7) | Sonnet 4.6 | 2048 tok | ~700 tok | ❌ |
| `clue_grading` | **Haiku 4.5** | **4096 tok** | ~1,110 tok | ❌ never |
| `clue_fact_check` | Sonnet 4.6 | 2048 tok | ~460 tok | ❌ |
| `puzzle_naming` | Sonnet 4.6 | 2048 tok | ~620 tok | ❌ |

The 34% hit rate measured on `clue_generation` comes **only** from midi puzzles;
every mini puzzle and every Haiku grading call misses. Because output tokens
dominate cost, fixing caching saves only ~$0.10–0.20/batch — real, but minor.

### Model wiring (for reference)

- Per-step models resolved by `ClaudeConfig.model_for()` (`config.py:145`),
  wired in `pipeline.py:226-245` via `_claude_for(step)`.
- `clue_generation` gets adaptive thinking + `effort=medium`
  (`config.py:140-141`, `pipeline.py:231-234`).
- **`puzzle_naming` has no model field** — it reuses `clue_gen_llm` (Sonnet),
  see `pipeline.py:270`. That's why naming runs on Sonnet.

---

## Guiding principles

1. **Quality over cost.** At ~$66/yr, a visible answer-in-clue leak in a daily
   product is far more expensive (reputationally) than any token spend.
2. **Mechanical before model.** Anything we can detect with deterministic Python
   should be caught there — it's free, instant, and 100% reliable. Reserve the
   LLM for judgment that genuinely requires it.
3. **Repair, don't regenerate wholesale.** Per-clue repair already exists and is
   cheaper and less churny than whole-puzzle regeneration. Feed it better
   triggers rather than regenerating all 30 clues.
4. **Each phase ships independently** and is validated against this exact batch
   (the 586 graded clues in the audited run) so we can show before/after.

---

## Phase 1 — Deterministic leak filter (highest leverage) ✅ DONE

**Status:** Implemented on branch `clue-quality-leak-filter`. 718 tests green.

**Shipped:**
- `src/crossword_generator/graders/leak_detector.py` — `detect_leak` /
  `detect_leaks` with rules: `exact`, `shared_root` (stemmer + morpheme-boundary
  affix check + agent-noun/verb unification), `irregular` (curated map),
  `abbrev_expansion` (expansion map + initialism). 40 unit tests.
- Wiring in `clue_grading_step.py`: `_run_leak_repair` runs after fact-check
  repair, repairs via the existing per-clue path, re-checks up to
  `leak_repair_attempts` (default 3). Surviving leaks → `LEAK:` soft error.
- Upload guard in `data_store.records_from_manifest` refuses any puzzle with a
  `LEAK:` error; `--allow-leaks` overrides. 2 guard tests + 2 end-to-end
  repair tests.
- `snowballstemmer` added as a dependency.

**Key finding — two leak classes:** validating against the audited batch
revealed the leaks split into two kinds:
1. **Direct echo** (answer/root/abbrev appears in clue text) — what this
   detector catches. 0 false positives across 329 real generated clues.
2. **Collocation fill-in-the-blank** (`"___ sauce"` → SOY, `"Shopping ___"` →
   LIST) — the blank plus a partner word forms a common phrase that gives away
   the answer. The answer text never appears, so this is **semantic, not
   mechanical**, and the detector cannot catch it. The `SOY`/`LIST` cases from
   the audit are this class. **Carried into Phase 2** as an LLM-judgment target
   (the grader/fact-check should flag give-away FITB collocations), since no
   deterministic rule covers it without a phrase lexicon.

**Scope limit (documented in the module):** morphological rules require answers
of length ≥ 4. Three-letter answers get only exact-match + abbreviation checks —
substring/root matching on short strings produces too many coincidental false
positives (CAT/category, EAR/early).

---

### Original design (for reference)

**Goal:** Catch answer-in-clue, shared-root, and abbreviation-expansion leaks
mechanically, with zero LLM cost, and route them into the existing repair path.

**Why first:** This directly fixes the complaint ("still see the answer word in
the clue all the time") and does not depend on an LLM noticing. It is
self-contained and testable against the audited batch.

### Design

New module `src/crossword_generator/graders/leak_detector.py` exposing:

```python
@dataclass(frozen=True)
class LeakFinding:
    number: int
    direction: str
    answer: str
    clue: str
    kind: str        # "exact" | "substring" | "shared_root" | "abbrev_expansion"
    detail: str      # the offending clue word / expansion, for the repair prompt

def detect_leaks(clues: Iterable[ClueEntry]) -> list[LeakFinding]: ...
```

Detection rules (all case-insensitive, word-boundary aware):

1. **Exact answer:** answer appears as a whole word in the clue.
2. **Substring/root (stemmer-based):** flag when the **stem** of the answer
   equals the stem of any clue word — catches regular inflections like
   `TEACHER`/`teaches`, `TRIMS`/`trim`, `BAKING`/`baked` automatically via a
   lightweight Porter/Snowball stemmer. Backstop with a substring check (answer
   len ≥ 4 appearing inside a clue word) for cases the stemmer normalizes apart.
3. **Curated map (irregulars + abbreviations):** a high-precision
   `LEAK_MAP: dict[str, set[str]]` for cases the stemmer can't catch — irregular
   plurals/forms (`wife`↔`wives`, `child`↔`children`) and abbreviation
   expansions (`est`→`eastern standard time`, `ceo`→`chief executive officer`,
   `ops`→`operations`, `pol`→`politician/political`). Plus an initialism check
   (clue words whose leading letters spell the answer). **The map starts small
   and grows from observed misses** — Phase 1 ships with a seed set, and a
   documented process adds entries when a real leak slips through in a later
   batch.

A stopword list and a min-answer-length guard avoid false positives on tiny
fills (`A`, `AN`, `IT`).

**Dependency note:** the stemmer needs a stemming lib (e.g. `snowballstemmer`
or `nltk`'s PorterStemmer). Prefer the smallest pure-Python option; add via uv.

### Wiring

In `ClueWithGradingStep.run`, **before** the surgical repair pass
(`clue_grading_step.py:134`), run `detect_leaks` on `best_envelope.clues`.
Convert each `LeakFinding` into a forced `(ClueEntry, ClueGrade)` repair entry
(score 0, `fairness` low, `feedback` naming the exact offending word so the
repair prompt has a concrete negative example), and pass them through the
existing `_run_clue_repair(..., forced_entries=...)` path — the same mechanism
duplicate-clue repair already uses (`clue_grading_step.py:356-394`). After
repair, **re-run `detect_leaks`** up to N attempts (default 3). If leaks remain,
**append a `LEAK: <answer> in clue "<clue>"` entry to `envelope.errors`**
(soft error — decision 1). The puzzle still saves as a draft and the batch
continues; the **upload guard must treat any `LEAK:`-prefixed error as a block**
so it cannot be pushed. This is the locked behavior — *not* a hard raise (unlike
duplicate-clue repair, which does raise).

**Upload guard change:** locate the pre-upload answer/pattern scan referenced in
the skill ("scan answers for known disallowed patterns") and extend it to refuse
any candidate whose `envelope.errors` contains a `LEAK:` entry.

### Validation

Offline harness over the audited batch: parse the final clues from
`output/batches/weekly-1w-.../logs/*.llm.jsonl`, run `detect_leaks`, and report
the catch list. Target: catches the `SOY`/`SHAW`/`LIST` cases and any
answer-in-clue instances, with a manually-reviewed false-positive rate near 0.

### Deliverables

- `graders/leak_detector.py` + unit tests in `tests/`.
- Wiring in `clue_grading_step.py`.
- An offline validation script (can live under `scripts/` or `tests/`).

### Acceptance

- All known leak categories from the audited batch are caught.
- No new false-positive repairs introduced on the batch's clean clues.
- `make test` / `make lint` green.

---

## Phase 2 — Per-clue targeted repair hardening ✅ DONE

**Status:** Implemented on branch `clue-quality-leak-filter`. 722 tests green.

**Shipped:**
- **Structured repair thresholds** — `_should_repair_grade` now repairs on
  explicit per-dimension sub-scores (`accuracy < 12`, `fairness < 15`,
  `craft < 8`, `score < 65`), keeping the keyword list only as a backstop. New
  config: `fairness_repair_threshold`, `craft_repair_threshold`.
- **Mostly-good shortcut** — `_is_mostly_good`: when ≥ `surgical_repair_pass_ratio`
  (default 0.8) of clues already pass, the whole-puzzle regeneration loop breaks
  early and surgical repair fixes the few bad clues instead — avoids the
  "fix one, break another" churn. New config: `surgical_repair_pass_ratio`.
- **Verify-after-repair** — `_run_clue_repair` now re-checks re-graded clues and
  repairs any still-flagged ones up to `repair_verify_attempts` (default 2)
  extra rounds. Duplicate-repair path uses the single-pass variant to avoid
  nested loops. New config: `repair_verify_attempts`.
- **Collocation-FITB penalty** — the grader's FAIRNESS rubric now explicitly
  fails (fairness 0-8) fill-in-the-blank clues whose blank + partner word form a
  give-away collocation (`"___ sauce"` → SOY, `"Shopping ___"` → LIST). Because
  fairness 0-8 < the new `fairness_repair_threshold`, these route automatically
  into surgical repair — closing the loop on the semantic leak class Phase 1
  could not catch mechanically.

New tests: mostly-good shortcut (skip + regen branches), verify-after-repair,
low-fairness→repair routing.

---

### Original design (for reference)

**Goal:** Make repair the default response to *any* flagged clue (from grading,
fact-check, or the Phase 1 leak filter) and stop the "fix one, break another"
churn from whole-puzzle regeneration.

**Current state:** Whole-puzzle regeneration happens in the
`generation → grading → generation` loop (`clue_grading_step.py:76-127`). The
batch step-sequence showed 9/14 puzzles did at least one full regen, some two.
Surgical per-clue repair already exists *after* that loop.

### Changes

1. **Lower the bar for entering surgical repair.** Replace the fragile keyword
   list in `_should_repair_grade` (`clue_grading_step.py:396-435`) with explicit
   structured signals: repair if `fairness < F`, `accuracy < A`, `craft < C`, or
   `score < S` (all configurable). Keep keyword matching only as a backstop. The
   leak filter (Phase 1) already covers the mechanical fairness cases, so this
   focuses the LLM-judgment triggers on accuracy/craft.
2. **Cap whole-puzzle regeneration earlier when surgical repair can carry it.**
   If a graded attempt is "mostly good" (e.g. ≥ K clues passing) prefer going
   straight to surgical repair of the few bad clues rather than regenerating all
   30 and risking new leaks. Make the "regenerate vs. repair" decision explicit
   and configurable.
3. **Verify-after-repair loop.** After `_repair_entries`, re-grade only the
   repaired clues (the grader currently re-grades the whole puzzle —
   `clue_grading_step.py:328`); if a repaired clue is *still* flagged, repair it
   again up to a small cap before giving up. Mirrors the duplicate-repair
   attempt loop.
4. **Catch collocation FITB leaks (from Phase 1 finding).** The mechanical
   detector cannot catch give-away fill-in-the-blank clues where the answer
   never appears as text (`"___ sauce"` → SOY, `"Shopping ___"` → LIST). Tighten
   the grader/fact-check prompts to explicitly penalize fill-in-the-blank clues
   whose blank + partner word form a common collocation that uniquely gives away
   the answer, and route those flags into surgical repair. This is the LLM-
   judgment complement to the deterministic filter.

### Non-goal

**Do NOT switch to one-LLM-call-per-clue generation from scratch.** It's ~30×
the calls, loses cross-clue variety/dupe control, and produces worse results for
more money than targeted repair. The win is per-clue *repair*, not per-clue
*generation*. This is recorded here so it isn't revisited.

### Acceptance

- On a fresh batch, flagged clues are repaired individually; whole-puzzle regen
  count drops vs. the audited batch.
- No regression in mean clue score (was 92.3/100).

---

## Phase 3 — Upgrade `clue_generation` to Opus 4.8

**Goal:** Fewer defects reach the filters in the first place by using the
strongest model for the one quality-critical generative step.

### Changes

- Set `clue_generation_model = "claude-opus-4-8"` in `ClaudeConfig`
  (`config.py:132`) — or override via `config.yaml` so it's reversible without a
  code change.
- **Thinking/effort:** Opus 4.8 is adaptive-thinking-only. The current code sets
  `thinking_enabled=True` for clue_generation (`pipeline.py:231`). Confirm the
  provider sends `thinking: {type: "adaptive"}` and **not** `budget_tokens`
  (removed on Opus 4.7/4.8 — would 400). Audit `claude_provider.py:82-86`:
  today it sends `{"type": self._config.thinking_type}` where `thinking_type`
  defaults to `"adaptive"` (`config.py:136`) — good, but verify no
  `budget_tokens`/`temperature`/`top_p` leak through for Opus (those 400 on
  4.7/4.8).
- **Effort A/B (decision 3):** run one small batch at `effort=medium` and one at
  `effort=high`, compare **pre-filter** defect rates using the Phase 1 detector
  on raw generation output (before repair), then lock the winner as the default.
  This A/B is part of Phase 3's deliverables, not a follow-up.
- **Sampling params:** Opus 4.8 rejects `temperature`/`top_p`/`top_k`. The
  provider currently always sets `temperature` (`claude_provider.py:79`). Gate
  it off for Opus models (only send sampling params for Sonnet/Haiku/Ollama).
  **This is a prerequisite for Phase 3 to not hard-error.**

### Cost impact

Generation output is ~45K tokens/batch. At Opus rates this is roughly +$1–2/
batch over Sonnet. Negligible at our volume.

### Acceptance

- A batch runs end-to-end on Opus 4.8 with no 400s.
- Pre-filter leak/accuracy defect rate (measured by Phase 1 detector on raw
  generation output, before repair) drops vs. the Sonnet baseline.

---

## Phase 4 — Fix prompt caching to actually fire

**Goal:** Get real cache hits on the repeated, near-static system prompts.
Minor cost win, but free once done and it reduces latency.

### Root cause

Anthropic silently skips caching when the cacheable prefix is below the
**model-specific minimum**: Sonnet 4.6 = 2048 tok, Haiku 4.5 = 4096 tok,
Opus = 4096 tok (per `claude-api` skill / prompt-caching reference). Our system
prompts are smaller than these thresholds for every step except midi
clue_generation.

### Options (pick per step)

1. **Move volatile content out of the cached prefix.** Confirm per-puzzle data
   (entries, crossing words, prior clues) lives in the **user** message, not the
   system block. (Spot-check showed it does — keep it that way.) The system
   block must be byte-identical across calls of the same step+difficulty to be a
   cache prefix.
2. **For steps below the minimum:** either (a) accept no caching (fact_check,
   naming are cheap and short), or (b) if we want the hit, consolidate stable
   few-shot examples / rubric text into the system block to push it over the
   threshold *only where the read volume justifies the write premium*. Don't pad
   artificially just to "turn caching on" — break-even needs ≥ 2 reads
   (5-min TTL).
3. **`clue_grading` (Haiku, 4096 min):** this is the highest-volume sub-min
   step (27 calls). If we keep grading on Haiku, its ~1,110-tok prompt will
   never cache. If Phase 5 moves grading to Sonnet (2048 min), it *still*
   wouldn't cache at 1,110 tok — so caching grading requires either a larger
   stable rubric block or accepting the miss. **Recommendation: accept the miss;
   the win isn't worth padding the prompt.**
4. **`clue_generation` mini prompts (~700 tok):** below 2048. Moving generation
   to Opus (Phase 3) *raises* the minimum to 4096, so mini generation won't
   cache on Opus either. Document this tradeoff: Opus improves quality but
   forfeits the partial caching midi currently gets. Given cost is negligible,
   quality wins.

### Verification

Add a post-batch assertion/report that sums
`cache_creation + cache_read + input` per step and logs the realized hit rate,
so caching regressions are visible (the audit script already does this — graduate
it into the repo).

### Acceptance

- Per-step cache hit rate reported in batch summary.
- Any step we *intend* to cache shows `cache_read > 0` on the 2nd+ call.

---

## Phase 5 — Right-size the cheap/expensive steps

**Goal:** Stop using Sonnet where Haiku suffices; keep judges strong.

### Changes

1. **`puzzle_naming` → Haiku 4.5.** Naming is a trivial creative task currently
   on Sonnet at 11% of batch cost. Add a `puzzle_naming_model` field to
   `ClaudeConfig`, wire it in `pipeline.py` (currently `puzzle_naming_llm`
   reuses `clue_gen_llm` at `pipeline.py:270` — give it its own `_claude_for`/
   logging), default `claude-haiku-4-5`. Saves ~$0.10/batch.
2. **`clue_grading` → Sonnet 4.6 now (decision 4).** Grading is the leak gate
   that's been failing; upgrade the judge immediately for safety rather than
   deferring to data. Set `clue_grading_model = "claude-sonnet-4-6"`
   (`config.py:133`). Cost: ~$0.3 → ~$0.6/batch. (A later Haiku-vs-Sonnet
   re-evaluation can revisit this once Phase 1 is offloading mechanical leaks,
   but the default ships as Sonnet.)
3. **`clue_fact_check` — keep on Sonnet.** It's a correctness gate at only 7% of
   cost; no reason to downgrade.

### Acceptance

- Naming runs on Haiku with no quality complaints on a sample batch.
- Grading runs on Sonnet 4.6; batch completes with no judge regressions.

---

## Sequencing & rationale

| Phase | What | Why this order |
|---|---|---|
| 1 | Deterministic leak filter | Biggest quality win, zero ongoing cost, fixes the exact complaint, fully testable against the audited batch. |
| 2 | Per-clue repair hardening | Makes the leak findings (and grader/fact-check flags) reliably actionable; reduces regen churn. Depends on the repair path Phase 1 reuses. |
| 3 | Opus 4.8 for generation | Fewer defects upstream. Requires the provider sampling-param fix; lands after the filter so we can measure the pre-filter improvement. |
| 4 | Caching fix | Free cleanup + visibility; informed by Phase 3's model-minimum change. |
| 5 | Right-size models | Lowest urgency; naming→Haiku and grading→Sonnet are independent of the others. |

**Net cost of all phases:** roughly +$2–4/batch (~$150–200/yr) in the worst
case — dominated by Phase 3 (Opus) and the grading→Sonnet upgrade. Acceptable
given the product context.

---

## Decisions (locked 2026-06-07)

All four pre-build questions are resolved — see **Locked decisions** at the top:

1. Stuck leaks → **soft error + upload block** (Phase 1 wiring).
2. Morphology → **stemmer + curated map that grows from misses** (Phase 1 design).
3. Opus generation effort → **A/B medium vs. high**, lock the winner (Phase 3).
4. `clue_grading` model → **Sonnet 4.6 now** (Phase 5.2).

---

## Validation assets

- Audited batch: `output/batches/weekly-1w-20260607-145850-s624518/`
  (logs in `logs/*.llm.jsonl`, 82 calls, 586 graded clues, mean 92.3/100).
- Known leak cases to regression-test against: `SOY` → `"___ sauce"` (fairness
  5), `SHAW` proper-noun FITB (14), `LIST` → `"Shopping ___"` (14).
- The audit aggregation script (models/cost/cache per step) should be promoted
  into `scripts/` as a reusable batch-quality report.
