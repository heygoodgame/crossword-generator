# Clue Quality

**HGG difficulty calibration:**

-   Easy clues should be easier than an NYT Monday. Favor direct definitions, obvious fill-in-the-blank clues, familiar meanings, and fast recognition for a broad casual audience. For Easy, plain and instantly solvable is a feature, not a flaw. No question-mark clues on any difficulty (see the house rules below).
-   Hard clues should land around NYT Monday/Tuesday (Jeff, 2026-10-27: "Generally make both Easy and Hard puzzles easier"). Clue the familiar meaning of the answer and make it a little less direct through phrasing: a specific example, a fresh angle, or light misdirection. Never make a clue harder by switching to a rare or secondary sense (DELL as "Small wooded valley"). A clue a solver gets right away is fine on Hard.

**House rules (Jeff, 2026-10-27):**

-   Everyday meanings only, on every difficulty: never a rare, technical, or archaic sense such as ARE as a metric unit of area.
-   Abbreviation answers are tagged "(abbr.)" or ", for short". Never "briefly" or "in brief"; the generator rewrites those tags automatically.
-   Fill-in-the-blank clues are welcome when the phrase is common and colorful, one people actually say. A made-up or stitched-together phrase is a defect.
-   Names: at most 2 per Mini and 5 per Midi. An entry counts when the answer is a proper name or the clue reaches an ordinary word through one ("Actor Hanks" for TOM). The repair loop enforces this from the grader's per-clue name flag.
-   "Opposite of ___" is good on Easy: about one per puzzle, never more than one.
-   Drinks are fine as plain things (RUM: "Spirit in a mojito"). Drunkenness, hangovers, and drug use are not.
-   Hints are paste-ready clues: Jeff swaps them in when a clue feels too hard.
-   Hidden words: etymology matters, letter sequences do not. STAR/START is not a duplicate; ICE/ICY is (the same-root fill rule).

**House rules (Jeff, 2026-10-29):**

-   No question-mark clues, on any difficulty: never a pun or wordplay clue ending in `?` ("Plant manager?" for GARDENER). A literal question in quotes or a fill-in-the-blank, like `"___ you okay?"` for ARE, is fine. A deterministic `question` detector sends any `?` clue to repair; one that survives is reported as a `QUESTION:` soft error.
-   Write "for example", never "e.g." ("Sedan, for example"). Many solvers don't know "e.g."; `clue_style.py` rewrites it in clues, repairs and hints. All published puzzles and drafts were rewritten on 2026-10-29.

**Positive indicators (high clue quality):**

-   Difficulty-fit — Easy clues are direct and obvious; Hard clues can be cleverer while staying fair
-   Misdirection — for Hard clues, the clue points the solver's mind in the wrong direction before the answer clicks, without a question mark
-   Wordplay — exploiting double meanings, homophones, or alternate readings of a word or phrase when the target difficulty allows it
-   Precise, clean surface reading — the clue reads naturally as an English phrase or sentence, not as an awkward construction forced to fit
-   Multiple viable interpretations that narrow to exactly one answer with crossing letters
-   Cultural references that are accessible but not dumbed down
-   Clues that teach the solver something interesting even when they don't know the answer
-   Appropriate difficulty calibration — HGG Easy should be easier than Monday; HGG Hard should feel Monday/Tuesday
-   Varied clue styles within a single puzzle — a mix of definitional, wordplay, trivia, fill-in-the-blank, and lateral thinking
-   Clues that reward a solver's general knowledge across domains (science, art, sports, food, geography) rather than leaning on one area
-   Contemporary references that make the puzzle feel alive and timely
-   Economy of language — saying exactly what's needed, no filler words
-   Parenthetical tags — explanatory fill-in-the-blank tags should use parentheses, e.g. "To the \_\_\_ (in the extreme)" rather than colon/comma appendages
-   Clever use of parts of speech ambiguity ("Lead" as a noun vs. verb vs. adjective)
-   Clues that make the solver smile or groan after the answer clicks
-   Fresh angles on common entries — finding a new way to clue OREO for the thousandth time
-   Fill-in-the-blank clues on common, colorful phrases that solvers actually know ("\_\_\_ of the Tiger", "Easy as \_\_\_")
-   Thematic clue consistency — when theme entries have clues that subtly echo each other or build toward the revealer

**Negative indicators (low clue quality):**

-   Dictionary-definition clues that feel like they were copy-pasted from Merriam-Webster ("A small nocturnal mammal" for BAT); concise everyday definitions are fine for Easy when they sound natural
-   Overly vague clues that could apply to dozens of answers without crossings ("Thing" for ITEM)
-   Obscure trivia that feels like a gotcha rather than a fair challenge ("17th-century Flemish cartographer" for someone only crossword databases know)
-   Clues that rely on the same tired angles every time (ERIE is always "Great Lake," never anything fresher)
-   Forced or strained wordplay where the surface reading doesn't hold together naturally
-   Hard clues that try too hard to be difficult, especially via strained pop-culture references or debatable slang
-   Clues that are technically accurate but misleading in an unfair rather than clever way
-   Gratuitous use of "Abbr." or "Var." tags that signal the entry itself is compromised
-   Clues requiring hyper-specific knowledge with no gettable path from crossings
-   Inconsistent difficulty within a puzzle — one corner plays like a Monday, another like a Friday
-   Clues that are longer and more complex than the answer warrants (a 15-word clue for a 3-letter entry that isn't doing any wordplay)
-   More than one "Opposite of" clue in a puzzle (one is welcome on Easy)
-   Dated references that haven't been culturally relevant in decades without being classic enough to endure
-   Clues with redundant or circular phrasing
-   Trivia clues that are binary — you either know it or you don't, with no way to reason toward the answer
-   Cross-reference clues ("See 17-Across") used excessively or unnecessarily
-   Fill-in-the-blank clues for obscure quotes nobody recognizes, or for made-up phrases nobody says
-   Demographic blind spots — clues that assume a narrow cultural lens (all sports clues from one league, all music clues from one era)
-   Inconsistent word-count tags such as "(two words)." Until explicit word-boundary metadata exists, omit these tags rather than applying them sporadically.

**Craft and editorial dimensions:**

-   Voice and personality — the best-edited puzzles have a consistent editorial sensibility that solvers can feel, almost like recognizing an author's writing style
-   Clue-answer fairness — every clue should have a defensible, unambiguous path to exactly one answer given the letter pattern
-   Balancing solver types — some solvers love trivia, others love wordplay, others love lateral thinking; a good puzzle serves all of them
-   Clue-pair awareness — when two clues in the same puzzle reference similar domains, they shouldn't feel redundant or create unintentional confusion
-   Revealer clue craft — the clue for a theme revealer has a special job; it should work both as a standalone clue and as the "aha" trigger for the theme mechanic
-   Difficulty curve within a week — for daily puzzles, clue style and difficulty should match the solver's expectations for that day of the week
-   Avoiding unintentional offense — clues should be reviewed for insensitivity, double meanings the constructor didn't intend, or outdated terminology
-   Clue-entry echo avoidance — the clue shouldn't contain the answer or a close variant of it, and shouldn't share roots with other nearby answers
-   Deterministic repair triggers — individual clues with answer leakage, very low score, or evaluator feedback such as "major issue" / "factual error" should be surgically repaired even if the aggregate clue set passes
-   Freshness over time — across a run of daily puzzles, the clue pool shouldn't feel repetitive; solvers notice when they see the same clue for ALOE three weeks in a row
-   Solver experience pacing — within a single puzzle, the clue set should create a satisfying arc where the solver builds momentum, hits a wall, and then breaks through
