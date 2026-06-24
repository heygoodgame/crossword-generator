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
    "\n"
    "A HINT IS OPTIONAL — ONLY GIVE ONE IF IT IS CLEARLY EASIER:\n"
    "- A hint exists to RESCUE a stuck solver, so it is only worth giving when "
    "you can make the entry MEANINGFULLY easier than the real clue. If the "
    "real clue is already about as easy and direct as it gets — a short, "
    "common word with a plain definition, where any beginner would already get "
    "it — there is nothing to add. In that case, return an EMPTY hint (\"\") "
    "for that entry. Do NOT invent a barely-different reword just to fill the "
    "slot.\n"
    "- Skipping is the right call whenever your best honest hint would be no "
    "easier (or no clearer) than the clue already shown. A missing hint is far "
    "better than a useless or redundant one.\n"
    "- Only give a hint when there is a genuinely simpler angle: a more obvious "
    "meaning, a more familiar example, or a plainer definition than the clue "
    "uses. If that easier angle exists, give it. If it does not, return \"\".\n"
    "- Never leave a hint empty merely because the entry is hard. A hard entry "
    "with an easier angle available is EXACTLY where a hint helps most — write "
    "the easy hint. Empty is only for entries that are already maximally easy.\n"
)

_EXAMPLE_OUTPUT = json.dumps(
    [
        {"number": 1, "direction": "across", "hint": "Frozen water"},
        {"number": 3, "direction": "across", "hint": ""},
        {"number": 5, "direction": "down", "hint": "Opposite of yes"},
    ],
    indent=2,
)

_OUTPUT_SECTION = (
    "OUTPUT FORMAT:\n"
    "Return ONLY a JSON array with one object per entry, each having "
    '"number", "direction", and "hint". No other text before or after.\n'
    "Include an object for EVERY entry listed in the user message. For an "
    "entry whose real clue is already as easy as it can be, set its \"hint\" "
    'to the empty string "" (as 3-across is below) rather than inventing a '
    "redundant one.\n"
    f"\n{_EXAMPLE_OUTPUT}\n"
    "\nReturn ONLY the JSON array."
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
        f"Now write an easy hint for each of the {len(entries)} entries above, "
        'using "" for any entry whose clue is already as easy as it can get.'
    )

    return system_text, user_text


_REPAIR_GUIDANCE = (
    "YOU ARE REPAIRING HINTS:\n"
    "- Each entry below already has a hint that FAILED a check (it leaked its "
    "answer, or stated something factually risky). Rewrite ONLY these hints.\n"
    "- A rewritten hint must obey every rule above: maximally easy, the single "
    "most common meaning, a different angle from the real clue, and it must "
    "NEVER contain or hint at the answer's letters, roots, or word-parts.\n"
    "- Fix the stated problem. For a leak, drop the offending word entirely "
    "and use a plain definition. For a fact risk, switch to a plainer, "
    "timeless definition or category that does not depend on a specific date, "
    "title, quote, or trivia fact.\n"
    "- If, after fixing the problem, you cannot write a hint that is genuinely "
    'easier than the real clue, return "" for that entry. A missing hint is '
    "always better than a broken one.\n"
)


def build_hint_repair_messages(
    entries: list[NumberedEntry],
    clues_by_key: dict[tuple[int, str], str],
    current_hints: dict[tuple[int, str], str],
    reasons_by_key: dict[tuple[int, str], list[str]],
) -> tuple[str, str]:
    """Build (system, user) messages for repairing flagged hints.

    Shares the generation system rules (role, easy/optional guidance, leak
    rules, output format) and appends repair-specific framing. The user text
    lists each flagged entry with its answer, real clue, the current (broken)
    hint, and the reasons it failed.

    Args:
        entries: The numbered grid entries whose hints need repair.
        clues_by_key: Map of (number, direction) -> the entry's real clue.
        current_hints: Map of (number, direction) -> the current broken hint.
        reasons_by_key: Map of (number, direction) -> human-readable reasons
            the hint was flagged (leak / fact risk).

    Returns:
        Tuple of (system_text, user_text).
    """
    system_text = "\n\n".join(
        [
            _ROLE,
            _HINT_GUIDANCE,
            _REPAIR_GUIDANCE,
            _GUIDELINES,
            _LEAK_EXAMPLES,
            _OUTPUT_SECTION,
        ]
    )

    entry_lines: list[str] = []
    for entry in entries:
        key = (entry.number, entry.direction)
        clue = clues_by_key.get(key, "")
        hint = current_hints.get(key, "")
        reasons = reasons_by_key.get(key, [])
        reason_str = "; ".join(reasons) if reasons else "flagged"
        entry_lines.append(
            f"- {entry.number}-{entry.direction.upper()}: {entry.answer}\n"
            f'    Real clue: "{clue}"\n'
            f'    Current hint: "{hint}"\n'
            f"    Problem: {reason_str}"
        )
    entries_block = "\n".join(entry_lines)

    user_text = (
        "HINTS TO REPAIR (answer + real clue + the broken hint + why it "
        "failed):\n"
        f"{entries_block}\n\n"
        f"Now rewrite a fixed, easy hint for each of the {len(entries)} "
        'entries above, using "" if no hint easier than the real clue is '
        "possible without reintroducing the problem."
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
