"""Prompt template for LLM-powered clue quality evaluation."""

from __future__ import annotations

import json

from crossword_generator.models import (
    ClueEntry,
    PuzzleDifficulty,
    PuzzleType,
    ThemeConcept,
)

_ROLE = (
    "You are an expert crossword puzzle editor evaluating clue quality. "
    "Score each clue on four rubric dimensions (0-25 points each):"
)

_RUBRIC = (
    "SCORING RUBRIC (each dimension 0-25 points):\n\n"
    "1. ACCURACY (0-25):\n"
    "   20-25: Factually correct, exactly one defensible answer, "
    "grammar and part of speech match perfectly.\n"
    "   10-19: Minor ambiguity (two plausible answers) or debatable "
    "accuracy that a solver could argue.\n"
    "    0-9:  Factually wrong, clearly multiple valid answers, or "
    "grammar/part-of-speech mismatch.\n"
    "   IMPORTANT: If your feedback identifies a factual error in the "
    "clue, the accuracy score MUST be 9 or below. Do not give 20+ "
    "accuracy to a clue you describe as incorrect.\n\n"
    "2. FRESHNESS (0-25): Does the clue style fit the target difficulty? "
    "For Easy clues, obvious definitions and fill-in-the-blank clues can "
    "score highly when they are clean and welcoming. For Hard clues, reward "
    "fair misdirection, wordplay, or creative angles that make the clue "
    "interesting without becoming obscure.\n\n"
    "3. CRAFT (0-25): Is the language economical and the surface reading clean? "
    "Does the clue read naturally as English? Is the difficulty appropriate "
    "for the puzzle type?\n\n"
    "4. FAIRNESS (0-25): Does the clue avoid echoing the answer word, answer "
    "word-parts, roots, or related morphological variants? Penalize "
    "singular/plural leaks, verb-form leaks, compound-part leaks, and "
    "famous-title fill-in-the-blanks that point at a different form of the "
    "answer, e.g. HOUSEWIFE clued via \"Desperate ___wives.\" Does it avoid "
    "using any crossing words in the clue text? "
    "Is the clue culturally accessible without being obscure? Penalize "
    "strained pop-culture references, ultra-current slang, and Hard clues "
    "that try to create Friday/Saturday-level difficulty. Penalize "
    "word-count tags like \"(two words)\" unless explicit word-boundary "
    "metadata was provided, and reward explanatory tags only when they are "
    "formatted parenthetically, e.g. \"To the ___ (in the extreme).\" "
    "Penalize "
    "body-part, underwear, anatomy, sexuality, appearance, identity, "
    "or demographic-based jokes, especially when a neutral clue would "
    "be more welcoming."
)

_THEME_EVAL_RULES = (
    "THEME CLUE EVALUATION:\n"
    "- [THEME ENTRY] clues: reward well-crafted standalone clues "
    "(high CRAFT, ACCURACY). A theme entry clue does NOT need to "
    "reference the revealer to score well.\n"
    "- FRESHNESS penalty: if ALL theme entry clues use the same "
    "style (e.g., all cross-reference the revealer, or all use "
    "the identical formula), deduct FRESHNESS for lack of variety.\n"
    "- FRESHNESS penalty: formulaic phrasing like \"one of "
    "[REVEALER ANSWER]\" should be scored low — it is "
    "grammatically unnatural for most theme types.\n"
    "- ACCURACY penalty: if the [REVEALER] clue claims the full "
    "revealer word has a relationship to theme entries that only "
    "a *component* of the word has (e.g., saying GOLDMINES "
    "precedes BAR when only GOLD does), score ACCURACY low. The "
    "connecting element may be a sub-part of the revealer, not "
    "the full answer.\n"
    "- CRAFT penalty: a bare \"See X-Across\" or \"See X-Down\" "
    "cross-reference with no standalone definition should score "
    "low on CRAFT. Cross-references must integrate the revealer "
    "position naturally into a real, solvable clue. Awkward "
    "appendages like \"per X-Across\" tacked onto an otherwise "
    "complete clue should also be penalized.\n"
    "- [REVEALER] clue: reward clues that elegantly explain the "
    "theme connection using natural language (e.g., \"a hint to "
    "some other answers in this puzzle\"). Penalize FRESHNESS if "
    "the revealer clue uses the phrase \"theme entries.\""
)

_EXAMPLE_OUTPUT = json.dumps(
    [
        {
            "number": 1,
            "direction": "across",
            "accuracy": 22,
            "freshness": 18,
            "craft": 20,
            "fairness": 15,
            "feedback": "Good misdirection but slightly vague.",
        },
    ],
    indent=2,
)

_OUTPUT_SECTION = (
    "OUTPUT FORMAT:\n"
    "Return ONLY a JSON array with one object per clue. "
    "Return the four sub-scores (accuracy, freshness, craft, fairness — "
    "each 0-25), NOT a total score. "
    "No other text before or after.\n"
    f"\n{_EXAMPLE_OUTPUT}\n"
    "\nEvaluate every clue listed in the user message. "
    "Return ONLY the JSON array."
)


def _difficulty_note(
    puzzle_type: PuzzleType, difficulty: PuzzleDifficulty
) -> str:
    if difficulty == PuzzleDifficulty.EASY:
        base = (
            "This is an HGG Easy crossword. Reward clues that are easier than "
            "NYT Monday: direct, obvious, familiar, and instantly solvable. "
            "Do not penalize plain definitions or obvious fill-in-the-blank "
            "clues for lack of cleverness. Penalize oblique definitions, "
            "tricky wordplay, niche trivia, or lateral-thinking clues as too "
            "hard for this audience."
        )
    else:
        base = (
            "This is an HGG Hard crossword. Reward NYT Tuesday/Wednesday-level "
            "clues: fair, broadly accessible, and a notch more challenging. "
            "Mild misdirection, common secondary meanings, wordplay, and "
            "occasional question-mark clues are appropriate. Penalize "
            "Saturday-level obscurity, forced cleverness, ultra-current slang, "
            "or trivia solvers cannot reason toward."
        )
    if puzzle_type == PuzzleType.MINI:
        return f"{base} MINI clues should remain especially concise."
    return (
        f"{base} MIDI clues can have a little more surface variety, within "
        "the target difficulty."
    )


def build_clue_evaluation_messages(
    clues: list[ClueEntry],
    crossing_words: dict[tuple[int, str], list[str]],
    puzzle_type: PuzzleType,
    theme: ThemeConcept | None = None,
    difficulty: PuzzleDifficulty = PuzzleDifficulty.EASY,
) -> tuple[str, str]:
    """Build (system, user) messages for clue quality evaluation."""
    themed = bool(theme and theme.topic)

    system_parts = [_ROLE, _RUBRIC, _difficulty_note(puzzle_type, difficulty)]
    if themed:
        system_parts.append(_THEME_EVAL_RULES)
    system_parts.append(_OUTPUT_SECTION)
    system_text = "\n\n".join(system_parts)

    revealer_answer = theme.revealer.upper() if themed else ""
    seed_answers = (
        {s.upper() for s in theme.seed_entries} if themed else set()
    )

    clue_lines: list[str] = []
    for clue in clues:
        key = (clue.number, clue.direction)
        crossings = crossing_words.get(key, [])
        crossing_str = ", ".join(crossings) if crossings else "none"
        tag = ""
        if clue.answer.upper() in seed_answers:
            tag = " [THEME ENTRY]"
        elif clue.answer.upper() == revealer_answer:
            tag = " [REVEALER]"
        clue_lines.append(
            f"- {clue.number}-{clue.direction.upper()}: "
            f"Answer={clue.answer}{tag}, Clue=\"{clue.clue}\" "
            f"(crossing words: {crossing_str})"
        )
    clues_block = "\n".join(clue_lines)

    theme_context_block = ""
    if themed:
        theme_context_block = (
            "\nTHEME CONTEXT:\n"
            f"- Topic: {theme.topic}\n"
            f"- Wordplay type: {theme.wordplay_type}\n"
            f"- Revealer: {theme.revealer}\n"
        )

    user_text = (
        f"{theme_context_block}"
        f"CLUES TO EVALUATE:\n{clues_block}\n\n"
        f"Now evaluate all {len(clues)} clues above."
    )

    return system_text, user_text


def build_clue_evaluation_prompt(
    clues: list[ClueEntry],
    crossing_words: dict[tuple[int, str], list[str]],
    puzzle_type: PuzzleType,
    theme: ThemeConcept | None = None,
    difficulty: PuzzleDifficulty = PuzzleDifficulty.EASY,
) -> str:
    """Build a single-string prompt (system+user concatenated).

    Kept for backward compatibility. Prefer
    ``build_clue_evaluation_messages`` to enable prompt caching.
    """
    system_text, user_text = build_clue_evaluation_messages(
        clues, crossing_words, puzzle_type, theme, difficulty
    )
    return f"{system_text}\n\n{user_text}"
