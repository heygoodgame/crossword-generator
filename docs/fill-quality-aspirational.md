# Fill Quality — Aspirational Goals vs. Current Implementation

## What Great Fill Looks Like

### Word Quality

- **Familiar, everyday vocabulary.** Every entry should be a word or phrase a typical solver recognizes without needing crosses. If you'd hesitate to use it in conversation, it probably doesn't belong in the grid.
- **Lively, contemporary language.** Fill should feel current. "STREAM" over "BROOK," "TEXT" over "WIRE." The grid shouldn't read like a 1990s puzzle archive.
- **Real multi-word phrases.** Entries like NO WAY, AS IF, LET'S GO, ON ICE add sparkle. They feel like things people actually say, not dictionary artifacts.
- **Broadly known proper nouns.** Names and places that cross demographics and generations. BEYONCE and EINSTEIN are fair game; the deputy mayor of a mid-size city in 1973 is not.
- **Interesting letter combinations.** J, K, X, Z, Q entries that feel earned — JAZZ, QUIXOTIC, KAYAK — rather than forced into a corner to escape a tight fill.
- **Entries with multiple clue angles.** The best fill words can be clued in several interesting ways. PITCH can be baseball, music, sales, or tar. That flexibility makes the puzzle more fun to write and solve.

### Word Quality Anti-Patterns

- **No crosswordese.** ESNE, ANOA, OLEO, ETUI — words that exist in puzzles but nowhere else. Even one of these in a mini is a blemish.
- **No junk abbreviations.** SSE, ENE, ETAL as a single entry. If it wouldn't appear on a road sign or in normal writing, skip it.
- **No partial phrases.** "I'M A," "TO A," "IS IT" — fragments that aren't real expressions.
- **No Roman numeral filler.** III, VII, DCI. These are construction shortcuts, not entries.
- **No standalone affixes.** RE-, -EST, -IER standing alone as grid entries.
- **No plural/tense padding.** Adding S or ED to a 4-letter word to fill a 5-letter slot is lazy. The plural should be a natural word in its own right.
- **No variant or archaic spellings.** Using OLDE or COLOUR (in an American-style grid) to escape a corner is a red flag.

### Grid-Level Quality

- **No duplicate entries.** The same word appearing twice is an automatic failure.
- **Minimal short glue.** Every grid needs some 3-letter connective tissue, but it should be common, clean words (THE, AND, FOR) rather than obscure fragments (AAL, OOT, EEE).
- **No Natick crossings.** Two obscure entries crossing at a letter the solver can't infer is unfair. Every crossing should have at least one accessible entry.
- **No clusters of weak fill.** One marginal entry is forgivable. Three adjacent ones signal a structural problem.
- **Varied word lengths.** A natural rhythm of short and long entries, not a grid dominated by 3s and 4s.
- **Clean corners.** Corners connected to the rest of the grid by only one or two entries are especially hard to fill well. Quality fill navigates these constraints without resorting to junk.

### Structural Awareness

- **Grid density balance.** More black squares give the filler more room, but too many make the puzzle trivial. The filler should succeed within the density constraints the grid pattern sets.
- **Symmetry as a given, not a compromise.** Rotational symmetry is a requirement of the format, not an excuse for bad fill. The filler should find clean solutions within symmetry constraints.
- **Stacking tolerance.** Grids with stacked long entries (common in themeless puzzles) put enormous pressure on short crossings. The filler should handle this gracefully or signal that the pattern is infeasible.

---

## Current CSP Filler: What's Implemented

### Word Selection

| Aspirational Goal | Current State | Gap |
|---|---|---|
| Familiar, everyday words | Dictionary score filtering (min 50 for 3+ letters, min 30 for 2-letter). Higher-scoring words tried first via tiered shuffling. | Score is a proxy for word quality, not a direct measure of familiarity. A word scoring 55 might still be crosswordese. No explicit crosswordese blacklist. |
| Lively, contemporary language | No mechanism. Word selection is score-based only. | No recency signal in the dictionary. ETAPE and EMAIL score similarly despite very different solver recognition. |
| Multi-word phrases | Supported if present in the dictionary. | Dictionary coverage of phrases is limited. No special handling to prefer phrases. |
| Broadly known proper nouns | Supported if present in the dictionary. | No demographic breadth check. An obscure opera singer and a pop star might score the same. |
| Interesting letter combos | No mechanism. | No bonus for uncommon letters; no penalty for dull letter patterns. |
| Multiple clue angles | No mechanism. | This is inherently hard to measure at fill time, but could be approximated by word frequency or sense count. |

### Word Quality Anti-Patterns

| Anti-Pattern | Current State | Gap |
|---|---|---|
| Crosswordese | Partially handled by dictionary scores (low-scoring words filtered out). | No explicit blacklist. Some crosswordese scores above the threshold. |
| Junk abbreviations | Partially handled by dictionary scores. | Same issue — some abbreviations score above 50. |
| Partial phrases | No detection. | Would need a classifier or curated list. |
| Roman numeral filler | No detection. | Could be caught with a simple pattern match. |
| Standalone affixes | No detection. | Could be caught with a curated list. |
| Plural/tense padding | No detection. | Could penalize entries that are just a known word + S/ED. |
| Variant/archaic spellings | No detection. | Hard to detect programmatically without a curated list. |

### Grid-Level Quality

| Aspirational Goal | Current State | Gap |
|---|---|---|
| No duplicate entries | Enforced during search (used-word set). | Fully implemented. |
| Minimal short glue | Fill grader penalizes weak 3-letter words (score < 55) with -5 per word, and applies -5 grid penalty if > 30% of entries are weak 3-letter words. | Detection exists but is relatively lenient. No adjacency check for clusters of weak fill. |
| No Natick crossings | No detection. | Would require analyzing crossing pairs for mutual obscurity. |
| No clusters of weak fill | No detection. | Grader scores words independently, not by grid position or adjacency. |
| Varied word lengths | No mechanism. | Word length distribution is determined by grid pattern, not the filler. |
| Clean corners | No special handling. | MRV heuristic helps (constrained corners get filled first) but there's no explicit corner quality check. |

### Structural Awareness

| Aspirational Goal | Current State | Gap |
|---|---|---|
| Grid density balance | Grid patterns are generated separately; filler works within them. | No feedback loop between pattern generation and fill quality. |
| Symmetry | Grid patterns enforce symmetry before filling. | Handled upstream, not by the filler. |
| Stacking tolerance | Backtracking + restart handles difficult regions. | No explicit detection of stacking pressure or proactive slot ordering for stacked regions. |

### Scoring & Grading

The fill grader computes a length-weighted mean of per-word scores, then applies grid-level penalties:

- **Per-word**: base dictionary score, -5 for 2-letter words, -5 for weak 3-letter glue (score < 55)
- **Grid-level**: -30 per duplicate word, -10 if > 20% unknown words, -5 if > 30% weak 3-letter glue
- **Pass threshold**: 60 (configurable, default 70)

This captures basic quality but misses most of the aspirational goals around crosswordese detection, Natick crossings, cluster analysis, and contemporary vocabulary preference.

### Search Strategy

The CSP filler's search is well-optimized for speed and correctness:

- **Multi-tier quality passes**: tries words scoring 60+ first, then falls back to 50+
- **MRV + degree ordering**: fills the most constrained slots first
- **Score-tiered shuffling**: within 10-point score bands, candidates are shuffled for variety while preferring higher-scoring words
- **AC-3 + prefix pruning + forward checking**: efficient constraint propagation catches dead ends early
- **Bitset domains**: O(1) constraint operations
- **Backtrack limit + random restarts**: 10,000 backtracks per attempt, reseeded restarts for diverse search trajectories

The search infrastructure is solid. The primary gaps are in what signals it optimizes for (dictionary score alone) rather than how it searches.
