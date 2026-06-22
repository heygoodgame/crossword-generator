"""Prompt template for LLM-powered hint generation.

A "hint" is an *easier alternate clue* surfaced to a solver who is stuck and
asks for help. It must point at the same answer as the real clue, but from the
simplest, most obvious angle possible — and, like a clue, it must never leak
the answer. We reuse the clue rubric's leak/abbreviation rules verbatim.
"""

from __future__ import annotations

import json

from crossword_generator.exporters.numbering import NumberedEntry

_ROLE = (
    "You are writing HINTS for a finished crossword puzzle. A hint is a "
    "near-giveaway shown to a solver who is stuck and has tapped 'hint'. Your "
    "ONE job is to make the answer pop into their head immediately. Err on the "
    "side of TOO easy — a hint that leaves them still guessing has failed."
)

_HINT_GUIDANCE = (
    "WHAT A HINT IS:\n"
    "- A hint points at the SAME answer as the original clue, but should be a "
    "DEAD GIVEAWAY — so easy that almost anyone gets it instantly with zero "
    "crossing letters. When in doubt, make it even more obvious.\n"
    "- Always use the single most common, most obvious meaning of the answer, "
    "and state it as plainly and directly as possible. Never reach for a "
    "secondary, technical, tricky, clever, or niche sense.\n"
    "- The most giveaway angle wins. Reach for: the bluntest dictionary "
    'definition ("Frozen water" for ICE); the single most famous example '
    '("The yellow ___" for SUN, or "Our home planet" for EARTH); a plain '
    'one-word synonym ("Happy" for GLAD); a clear antonym ("Opposite of up" '
    'for DOWN); the most familiar fill-in-the-blank ("Peanut butter and ___" '
    'for JELLY); or a household association a child knows ("Says \'meow\'" '
    "for CAT). Pick whichever makes the answer most unmistakable.\n"
    "- For a famous name, point straight at who/what it is in the plainest "
    'terms ("First name of pop star ___ Swift" for TAYLOR). For a place, name '
    "the most famous fact about it. For an abbreviation, say what it stands "
    "for in plain words WITHOUT using those exact words as the clue's "
    "give-away (see the hard rule below).\n"
    "- Take a DIFFERENT angle from the original clue — do not just reword it. "
    "If the clue is a fill-in-the-blank, prefer a flat definition, and vice "
    "versa.\n"
    "- Keep it short and plain. No wordplay, no \"?\"-trick clues, no "
    "lateral-thinking, no trivia deeper than common knowledge — ever, no "
    "matter how hard the original puzzle is. The whole point is to rescue a "
    "stuck solver, not to challenge them.\n"
    "- A hint that an average adult could not answer instantly is a FAILURE.\n"
)

# The ONE boundary on a giveaway hint: it may not literally hand over the
# answer string. Everything else is fair game — be as obvious as you like.
_LEAK_RULE = (
    "THE ONLY HARD RULE (a hint that breaks this is rejected automatically):\n"
    "- The hint must NOT contain the answer word itself, or a simple "
    "morphological variant of it (singular/plural, verb tense, -er/-ing/-ed "
    "forms). E.g. for TEACHER do not write \"teaches\"; for RUNNING do not "
    "write \"run\".\n"
    "- If the answer is an abbreviation/acronym/initialism, the hint must NOT "
    "contain ANY of the exact words its letters stand for. E.g. for ETA do "
    "not use \"estimated\", \"time\", or \"arrival\"; for PDF do not use "
    "\"portable\", \"document\", or \"format\"; for APR do not use \"annual\", "
    "\"percentage\", or \"rate\". Instead point at what it IS in everyday "
    "terms: ETA -> \"When your driving app says you'll get there\"; PDF -> "
    "\"File type you can't easily edit, often used for forms\"; APR -> \"The "
    "interest figure on a credit card offer.\"\n"
    "- That single rule aside, be as much of a giveaway as you can. Do not "
    "shy away from the most obvious wording for fear of being 'too easy' — "
    "too easy is the GOAL.\n"
)

_EXAMPLE_OUTPUT = json.dumps(
    [
        {"number": 1, "direction": "across", "hint": "Frozen water"},
        {"number": 5, "direction": "down", "hint": "Opposite of yes"},
        {"number": 8, "direction": "across", "hint": "Our home planet"},
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
            # A hint-specific leak rule: keep the one boundary (don't hand over
            # the answer string) without the clue rubric's "avoid even generic
            # words" tone, which fights the dead-giveaway goal.
            _LEAK_RULE,
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
