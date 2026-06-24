"""Fact-check hints by reusing the clue fact-checker on hint text.

A hint is just a second, easier clue, so the same factual risks apply (a hint
like "Capital of Australia" can be wrong exactly as a clue can). Rather than
duplicate the fact-check engine, this adapter runs the existing
``ClueFactChecker`` against a shadow envelope whose ``ClueEntry.clue`` field
holds the *hint* text, then maps the results back by ``(number, direction)``.

Empty hints (entries the model deliberately skipped) are not fact-checked —
there is nothing to verify, and they must not be turned into findings.
"""

from __future__ import annotations

from crossword_generator.graders.clue_fact_checker import (
    ClueFactChecker,
    ClueFactCheckResult,
)
from crossword_generator.models import ClueEntry, PuzzleEnvelope


def fact_check_hints(
    fact_checker: ClueFactChecker,
    envelope: PuzzleEnvelope,
    hints_by_key: dict[tuple[int, str], str],
) -> dict[tuple[int, str], ClueFactCheckResult]:
    """Fact-check the given hints, returning results keyed by entry.

    Args:
        fact_checker: The shared clue fact-checker (unchanged; used as-is).
        envelope: The puzzle envelope, used for puzzle_type/difficulty context
            and to look up each entry's answer.
        hints_by_key: Map of (number, direction) -> hint text. Empty hints are
            skipped.

    Returns:
        Map of (number, direction) -> ClueFactCheckResult for every hint the
        fact-checker actually evaluated (its deterministic pre-screen may skip
        non-risky hints, which simply do not appear in the result).
    """
    answer_by_key = {
        (c.number, c.direction): c.answer for c in envelope.clues
    }

    # Build shadow clue entries whose `clue` field is the hint text. Only
    # non-empty hints for entries we know the answer to are checkable.
    shadow_clues: list[ClueEntry] = []
    for (number, direction), hint in hints_by_key.items():
        if not hint:
            continue
        answer = answer_by_key.get((number, direction))
        if answer is None:
            continue
        shadow_clues.append(
            ClueEntry(
                number=number,
                direction=direction,
                answer=answer,
                clue=hint,
            )
        )

    if not shadow_clues:
        return {}

    shadow_envelope = envelope.model_copy(update={"clues": shadow_clues})
    results = fact_checker.check(shadow_envelope)
    return {(r.number, r.direction): r for r in results}
