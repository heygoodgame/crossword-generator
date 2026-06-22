"""Prompt template for LLM-powered hint generation.

A "hint" is an *easier alternate clue* surfaced to a solver who is stuck and
asks for help. It must point at the same answer as the real clue, but from the
simplest, most obvious angle possible — and, like a clue, it must never leak
the answer. We reuse the clue rubric's leak/abbreviation rules verbatim.
"""

from __future__ import annotations

import json

from crossword_generator.exporters.numbering import NumberedEntry
from crossword_generator.llm.prompts.clue_generation import (
    _GUIDELINES,
    _LEAK_EXAMPLES,
)

_ROLE = (
    "You are an expert crossword constructor writing HINTS for a finished "
    "puzzle. A hint is a SECOND, much easier clue shown to a solver who is "
    "stuck on an entry and has asked for help."
)

_HINT_GUIDANCE = (
    "WHAT A HINT IS:\n"
    "- A hint points at the SAME answer as the original clue, but is as easy "
    "as possible — the kind of clue a young child or total beginner would get "
    "instantly, with no crossing letters.\n"
    "- Always use the single most common, most obvious meaning of the answer. "
    "Never reach for a secondary, technical, tricky, or niche sense.\n"
    "- Take a DIFFERENT angle from the original clue — do not just reword it. "
    "If the clue is a fill-in-the-blank, the hint should usually be a plain "
    "definition or a familiar everyday example, and vice versa.\n"
    "- Reach for: a plain dictionary definition (\"Frozen water\" for ICE); a "
    "totally obvious example or category (\"A color\" for BLUE); a simple "
    "synonym (\"Happy\" for GLAD); a clear antonym (\"Opposite of up\" for "
    "DOWN); or a household-familiar association (\"Says 'meow'\" for CAT).\n"
    "- Keep it short and plain. No wordplay, no \"?\"-trick clues, no "
    "lateral-thinking, no obscure trivia — ever, regardless of how hard the "
    "original puzzle is. The whole point of a hint is to rescue a stuck "
    "solver.\n"
    "- A hint that an average adult could not answer instantly is a FAILURE. "
    "When in doubt, make it even easier and more direct.\n"
)

_EXAMPLE_OUTPUT = json.dumps(
    [
        {"number": 1, "direction": "across", "hint": "Frozen water"},
        {"number": 5, "direction": "down", "hint": "Opposite of yes"},
    ],
    indent=2,
)

_OUTPUT_SECTION = (
    "OUTPUT FORMAT:\n"
    "Return ONLY a JSON array with one object per entry, each having "
    '"number", "direction", and "hint". No other text before or after.\n'
    f"\n{_EXAMPLE_OUTPUT}\n"
    "\nWrite a hint for every entry listed in the user message. "
    "Return ONLY the JSON array."
)


def build_hint_generation_messages(
    entries: list[NumberedEntry],
    clues_by_key: dict[tuple[int, str], str],
) -> tuple[str, str]:
    """Build (system, user) messages for hint generation.

    The system text bundles the role, hint guidance, and the leak rules
    (shared verbatim with clue generation) plus the output format — all of
    which is identical across puzzles and caches well. The user text carries
    the per-puzzle entries with their answers and existing clues.

    Args:
        entries: The numbered grid entries to write hints for.
        clues_by_key: Map of (number, direction) -> the entry's real clue, so
            the model can deliberately take a different, easier angle.

    Returns:
        Tuple of (system_text, user_text).
    """
    system_text = "\n\n".join(
        [
            _ROLE,
            _HINT_GUIDANCE,
            # Reuse the clue rubric's leak rules verbatim — a hint that leaks
            # the answer is just as broken as a clue that does.
            _GUIDELINES,
            _LEAK_EXAMPLES,
            _OUTPUT_SECTION,
        ]
    )

    entry_lines: list[str] = []
    for entry in entries:
        clue = clues_by_key.get((entry.number, entry.direction), "")
        entry_lines.append(
            f"- {entry.number}-{entry.direction.upper()}: {entry.answer}\n"
            f'    Existing clue: "{clue}"'
        )
    entries_block = "\n".join(entry_lines)

    user_text = (
        "ENTRIES TO WRITE HINTS FOR (answer + the existing clue to make "
        "easier):\n"
        f"{entries_block}\n\n"
        f"Now write one easy hint for all {len(entries)} entries above."
    )

    return system_text, user_text


def parse_hint_response(
    raw_response: str, entries: list[NumberedEntry]
) -> dict[tuple[int, str], str]:
    """Parse the LLM's JSON hint response into a {(number, direction): hint} map.

    Mirrors the clue parser's leniency: strips preamble/markdown around the
    JSON array and auto-corrects a wrong direction when the number is
    unambiguous.

    Raises:
        json.JSONDecodeError: If no JSON array can be extracted.
        ValueError: If the response shape is wrong or a hint targets an entry
            not in the grid.
    """
    text = raw_response.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise json.JSONDecodeError("No JSON array found in response", text, 0)
    parsed = json.loads(text[start : end + 1])

    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON array, got {type(parsed).__name__}")

    valid_keys = {(e.number, e.direction) for e in entries}
    number_dirs: dict[int, list[str]] = {}
    for e in entries:
        number_dirs.setdefault(e.number, []).append(e.direction)

    hints: dict[tuple[int, str], str] = {}
    for item in parsed:
        number = int(item["number"])
        direction = str(item["direction"]).lower()
        hint = str(item["hint"]).strip()

        key = (number, direction)
        if key not in valid_keys:
            dirs = number_dirs.get(number, [])
            if len(dirs) == 1 and dirs[0] != direction:
                key = (number, dirs[0])
            else:
                raise ValueError(
                    f"LLM returned hint for {number}-{direction} "
                    "which is not in the grid"
                )
        if hint:
            hints[key] = hint

    return hints
