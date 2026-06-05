"""Prompt template for LLM-powered clue fact-risk checks."""

from __future__ import annotations

import json

from crossword_generator.models import ClueEntry, PuzzleDifficulty, PuzzleType

_ROLE = (
    "You are a cautious crossword fact checker. Your job is to catch clues "
    "that may be factually wrong or may not uniquely and exactly clue the "
    "given answer. Prefer conservative rewrites over risky trivia."
)

_RULES = (
    "FACT CHECK RULES:\n"
    "- Evaluate whether each clue is factually correct for the provided answer.\n"
    "- Check proper nouns, titles, quotes, songs, books, films, teams, dates, "
    "places, people, brands, and superlative claims especially carefully.\n"
    "- Check fill-in-the-blank clues for an exact answer fit, including "
    "number, tense, spacing, contractions, and idiomatic phrasing.\n"
    "- Check that the clue has exactly one defensible answer and does not rely "
    "on a debatable or obscure fact.\n"
    "- If you are not confident the factual claim is airtight, classify the "
    "clue as uncertain. Do not classify uncertain trivia as safe just because "
    "it sounds plausible.\n"
    "- Use status \"safe\" only when the clue is factually sound, exact, and "
    "has one defensible answer.\n"
    "- Use status \"uncertain\" when the clue may be right but should be "
    "rewritten as a plain, safer clue.\n"
    "- Use status \"incorrect\" when the clue is factually wrong, points to "
    "a different answer form, or has a clear grammar/number/tense mismatch."
)

_EXAMPLE_OUTPUT = json.dumps(
    [
        {
            "number": 1,
            "direction": "across",
            "status": "uncertain",
            "reason": (
                "Specific trivia claim is not reliable enough; use a direct "
                "definition."
            ),
        },
    ],
    indent=2,
)

_OUTPUT_SECTION = (
    "OUTPUT FORMAT:\n"
    "Return ONLY a JSON array with one object per clue. "
    "Each object must contain number, direction, status, and reason. "
    "status must be one of: safe, uncertain, incorrect. "
    "No other text before or after.\n"
    f"\n{_EXAMPLE_OUTPUT}\n"
)


def _difficulty_note(
    puzzle_type: PuzzleType, difficulty: PuzzleDifficulty
) -> str:
    if difficulty == PuzzleDifficulty.EASY:
        base = (
            "This is an HGG Easy crossword. Favor direct, familiar clues over "
            "trivia or clever factual angles."
        )
    else:
        base = (
            "This is an HGG Hard crossword. Mild trivia and misdirection are "
            "allowed only when the clue remains factually airtight and broadly "
            "inferable."
        )
    if puzzle_type == PuzzleType.MINI:
        return f"{base} MINI clues should be especially clean and concise."
    return f"{base} MIDI clues can have more variety but must stay accurate."


def build_clue_fact_check_messages(
    clues: list[tuple[ClueEntry, str]],
    puzzle_type: PuzzleType,
    difficulty: PuzzleDifficulty = PuzzleDifficulty.EASY,
) -> tuple[str, str]:
    """Build (system, user) messages for fact-risk checking.

    Args:
        clues: Pairs of (clue, risk reason) selected by deterministic
            pre-screening.
        puzzle_type: Mini or midi.
        difficulty: Easy or hard.
    """
    system_text = "\n\n".join(
        [_ROLE, _difficulty_note(puzzle_type, difficulty), _RULES, _OUTPUT_SECTION]
    )

    clue_lines = []
    for clue, risk_reason in clues:
        clue_lines.append(
            f"- {clue.number}-{clue.direction.upper()}: "
            f"Answer={clue.answer}, Clue=\"{clue.clue}\" "
            f"(risk reason: {risk_reason})"
        )
    clues_block = "\n".join(clue_lines)
    user_text = (
        f"CLUES TO FACT CHECK:\n{clues_block}\n\n"
        f"Fact-check all {len(clues)} clues above."
    )
    return system_text, user_text
