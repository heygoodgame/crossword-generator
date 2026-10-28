# Fill Quality

**Positive indicators (high fill quality):**

-   Common, everyday words that most solvers recognize without needing crosses
-   Lively, contemporary vocabulary that feels current rather than dusty
-   Multi-word phrases that people actually say ("NO WAY," "AS IF," "LET'S GO")
-   Pop culture entries that are broadly known, not deeply niche
-   Proper nouns that are widely recognizable across demographics
-   Varied word lengths that create a natural rhythm across the grid
-   Interesting letter combinations — entries with J, K, X, Z, Q that feel earned rather than forced
-   Long down and across entries that are genuinely fun to uncover
-   Entries that could themselves be the seed of a clue with multiple interesting angles
-   Consistent register — the fill doesn't wildly swing between academic jargon and slang without reason
-   Entries that allow for creative, misdirecting clues rather than only having one obvious clue path

**Negative indicators (low fill quality):**

-   Crosswordese — obscure short words that exist only because of convenient letter patterns (ESNE, ANOA, OLEO, ETUI, EPEE)
-   Abbreviations that aren't commonly used in real life (SSE, ENE, ETAL as one word)
-   Partial phrases that feel incomplete ("I'M A," "TO A," "IS IT")
-   Roman numerals used as filler (III, VII, DCI)
-   Obscure proper nouns that only crossword solvers would know
-   Foreign words that aren't common English loanwords
-   Variant spellings or archaic forms used to escape a tight corner
-   Repeated reliance on the same tired entries across puzzles (the "crossword greatest hits" problem)
-   Three-letter entries that are essentially glue (EEE, AAL, OOT)
-   Entries that are technically real words but feel contrived or dictionary-only
-   Proper nouns that skew heavily toward one demographic, era, or knowledge domain without balance
-   Prefixes or suffixes standing alone as entries (RE-, -EST, -IER)
-   Plural or past-tense padding — taking a good 4-letter word and adding S or ED to fill a 5- or 6-letter slot
-   Adjacent short crosswordese — one obscure entry is forgivable, but a cluster of them signals a construction problem
-   "Natick" crossings — where two obscure entries cross at a letter the solver has no way to infer, creating an unfair guess

**Proper-noun cap (Jeff, 2026-07):**

Too many names turns a word puzzle into a trivia contest — a common
complaint about mainstream crosswords. The fill grader hard-fails any grid
with more than `max(2, floor(0.15 × answer_count))` proper-noun answers:
2 for a typical 5x5 or 7x7, 3 for a typical 9x9. Independently of the
cap, 1-Across may never be a proper noun — it is the solver's first
impression, and opening with a name reads as trivia.

A "proper-noun answer" is a word only viable in a crossword as a proper
noun — any reasonable clue must reference a specific named entity (OPRAH,
ERIE, OREO, NBA, ELSA). Words with a common-English reading (AMBER, CHINA,
MARK) do not count, because they can be clued without trivia. The
classification lives in `dictionaries/HggProperNounClassifications.txt`
(`WORD;P|C` per line), built and incrementally refreshed by
`crossword-generator classify-proper-nouns` after dictionary updates.
Theme seed entries are not in the file, so intentional themed names never
count against the cap.

**Abbreviation cap (Jeff, 2026-10-27):** at most 1 abbreviation answer in a
5x5, 2 in a 7x7, 3 in a 9x9. An abbreviation is an answer every fair clue must
tag as shortened ("(abbr.)", ", for short"): ETA, NBA, OTS, IDS, SIS. Clipped
forms that are words now (MEMO, INFO, PRO) and words with an ordinary reading
(AMP, ORE) do not count. The classification lives in
`dictionaries/HggAbbreviationClassifications.txt` (`WORD;A|W` per line),
built and incrementally refreshed by `crossword-generator
classify-abbreviations` after dictionary updates.

**Demoted fill (Jeff, 2026-10-27):** words Jeff wants used much less often
(HOLT, AMES, LENA, FRO) or somewhat less often (OTS, SIS, DOO) carry a draw
weight in `dictionaries/HggFillWeights.txt`. They stay in the lists and can
still fill a slot when nothing else fits.

**One word family per grid (Jeff, 2026-10):**

Two answers from the same root in one grid read as the same word twice,
so the fill grader hard-fails them (`shared_root` penalty). Jeff hand-edited
these out of the 2026-09 midi batch, and his replacement letters pushed two
puzzles off their target days. A pair counts when both answers share a
lemma that is a real word in the root lexicon (`hgg-easy` + `hgg-hard` +
`hgg-starter`, `grading.fill.root_lexicon_paths`):

-   Inflections, including three-letter stems: OWE/OWED, ICE/ICED, HIRED/HIRING, PLAY/PLAYED, STOP/STOPPED
-   Irregular forms from a curated table: ATE/EAT, MEET/MET, SAID/SAYS, GONE/WENT, TEETH/TOOTH
-   -Y adjectives of short stems, also curated: ICE/ICY, FOG/FOGGY, BONE/BONY (Jeff edited ICE out of a grid with ICY, 2026-10)
-   Derivatives, up to two steps: STINKY/STINKS, COMEDIC/COMEDIANS (both from COMEDY)

Coincidental overlaps do not count: START/STAR, RID/RIDES, BUS/BUSY,
BET/BETTER, BAR/BARED, ERIC/ERICA, RED/REDO. Plain plurals (ART/ARTS) are
reported by the separate `terminal_s_variants` rule. The logic lives in
`src/crossword_generator/word_families.py`; a backtest on the 947
generated puzzles Jeff has reviewed caught all seven same-root grid edits
he made and flagged 16 approved grids (mostly real pairs he let through,
such as AWE/AWED and SAW/SEE).

**Structural/holistic factors:**

-   Grid density — how many theme entries and black squares constrain the fill; more openness generally allows cleaner fill
-   Word list quality — the scoring database used to rank candidate entries fundamentally caps how good autofill output can be
-   Symmetry tax — maintaining rotational symmetry limits the constructor's options, sometimes forcing compromises
-   Corner isolation — corners connected to the rest of the grid by only one or two entries are especially hard to fill cleanly
-   Stacking — grids with stacked long entries (common in themeless puzzles) demand exceptional fill skill or computation to avoid junk in the short crossings

