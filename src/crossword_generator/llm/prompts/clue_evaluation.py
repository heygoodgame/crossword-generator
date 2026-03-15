"""Prompt template for LLM-powered clue quality evaluation."""

from __future__ import annotations

import json

from crossword_generator.models import ClueEntry, PuzzleType, ThemeConcept


def build_clue_evaluation_prompt(
    clues: list[ClueEntry],
    crossing_words: dict[tuple[int, str], list[str]],
    puzzle_type: PuzzleType,
    theme: ThemeConcept | None = None,
) -> str:
    """Build a prompt that asks the LLM to evaluate clue quality.

    Args:
        clues: The clue entries to evaluate.
        crossing_words: Maps (number, direction) to crossing answer words.
        puzzle_type: Mini or midi — affects difficulty expectations.
        theme: Optional theme concept for midi puzzles.

    Returns:
        A prompt string ready to send to the LLM.
    """
    # Identify theme entries and revealer for annotation
    revealer_answer = ""
    seed_answers: set[str] = set()
    if theme and theme.topic:
        revealer_answer = theme.revealer.upper()
        seed_answers = {s.upper() for s in theme.seed_entries}

    # Build the clue list section
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

    # Difficulty expectations
    if puzzle_type == PuzzleType.MINI:
        difficulty_note = (
            "This is a MINI crossword (Monday-level). Clues should be "
            "accessible and straightforward but still clever."
        )
    else:
        difficulty_note = (
            "This is a MIDI crossword (Tuesday/Wednesday-level). "
            "Clues can use more wordplay and misdirection."
        )

    # Theme section
    theme_block = ""
    if theme and theme.topic:
        theme_block = (
            "\nTHEME CONTEXT:\n"
            f"- Topic: {theme.topic}\n"
            f"- Wordplay type: {theme.wordplay_type}\n"
            f"- Revealer: {theme.revealer}\n"
            "\n[THEME ENTRY] clues should tie back to the revealer — "
            "either by cross-referencing the revealer's clue number or "
            "by connecting to the theme concept. Score FRESHNESS higher "
            "when theme entries make this connection, lower when they "
            "ignore the theme entirely.\n"
        )

    # JSON format example — sub-scores only, we compute the total
    example_output = json.dumps(
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

    role = (
        "You are an expert crossword puzzle editor evaluating clue quality. "
        "Score each clue on four rubric dimensions (0-25 points each):"
    )

    rubric = (
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
        "2. FRESHNESS (0-25): Does the clue avoid dictionary-definition staleness? "
        "Does it use misdirection, wordplay, or a creative angle? "
        "Would a solver find this clue interesting rather than rote?\n\n"
        "3. CRAFT (0-25): Is the language economical and the surface reading clean? "
        "Does the clue read naturally as English? Is the difficulty appropriate "
        "for the puzzle type?\n\n"
        "4. FAIRNESS (0-25): Does the clue avoid echoing the answer word or its "
        "roots? Does it avoid using any crossing words in the clue text? "
        "Is the clue culturally accessible without being obscure?"
    )

    output_section = (
        "OUTPUT FORMAT:\n"
        "Return ONLY a JSON array with one object per clue. "
        "Return the four sub-scores (accuracy, freshness, craft, fairness — "
        "each 0-25), NOT a total score. "
        "No other text before or after.\n"
        f"\n{example_output}\n"
        f"\nNow evaluate all {len(clues)} clues listed above. "
        "Return ONLY the JSON array."
    )

    return (
        f"{role}\n\n{rubric}\n\n{difficulty_note}\n{theme_block}\n"
        f"CLUES TO EVALUATE:\n{clues_block}\n\n{output_section}"
    )
