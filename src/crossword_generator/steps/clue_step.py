"""Clue generation pipeline step."""

from __future__ import annotations

import json
import logging

from crossword_generator.exporters.numbering import (
    NumberedEntry,
    compute_crossing_words,
    compute_numbering,
)
from crossword_generator.llm.base import LLMProvider
from crossword_generator.llm.prompts.clue_generation import (
    build_clue_generation_prompt,
)
from crossword_generator.models import ClueEntry, PuzzleEnvelope
from crossword_generator.steps.base import PipelineStep

logger = logging.getLogger(__name__)


class ClueGenerationStep(PipelineStep):
    """Pipeline step that generates clues for a filled grid using an LLM."""

    def __init__(self, llm: LLMProvider, *, max_retries: int = 3) -> None:
        self._llm = llm
        self._max_retries = max_retries

    @property
    def name(self) -> str:
        return "clue-generation"

    def run(self, envelope: PuzzleEnvelope) -> PuzzleEnvelope:
        """Generate clues for all entries in the filled grid."""
        errors = self.validate_input(envelope)
        if errors:
            raise ValueError(
                f"ClueGenerationStep validation failed: {'; '.join(errors)}"
            )

        assert envelope.fill is not None
        grid = envelope.fill.grid

        # Compute numbering and crossing words
        entries = compute_numbering(grid)
        crossing_words = compute_crossing_words(entries, grid)

        # Build prompt
        prompt = build_clue_generation_prompt(
            entries=entries,
            crossing_words=crossing_words,
            puzzle_type=envelope.puzzle_type,
            theme=envelope.theme,
        )

        # Call LLM with retries on parse failure
        clue_entries: list[ClueEntry] | None = None
        last_error = ""

        for attempt in range(1, self._max_retries + 1):
            logger.info(
                "Clue generation attempt %d/%d using %s",
                attempt,
                self._max_retries,
                self._llm.name,
            )
            raw_response = self._llm.generate(prompt)
            logger.debug(
                "Raw LLM response (%d chars): %.200s",
                len(raw_response),
                raw_response,
            )
            try:
                clue_entries = _parse_clue_response(raw_response, entries)
                break
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                last_error = str(exc)
                logger.warning(
                    "Attempt %d: failed to parse LLM response: %s",
                    attempt,
                    last_error,
                )

        if clue_entries is None:
            raise ValueError(
                f"Failed to parse clue response after {self._max_retries} "
                f"attempts. Last error: {last_error}"
            )

        return envelope.model_copy(
            update={
                "clues": clue_entries,
                "step_history": [*envelope.step_history, self.name],
            }
        )

    def validate_input(self, envelope: PuzzleEnvelope) -> list[str]:
        errors: list[str] = []
        if envelope.fill is None:
            errors.append("Envelope has no fill result — run fill step first")
        if envelope.clues:
            errors.append("Envelope already has clues")
        return errors


def _parse_clue_response(
    raw_response: str, entries: list[NumberedEntry]
) -> list[ClueEntry]:
    """Parse the LLM's JSON response into a list of ClueEntry objects.

    Raises:
        json.JSONDecodeError: If the response is not valid JSON.
        ValueError: If the response doesn't contain the expected entries.
        KeyError: If required fields are missing.
    """
    # Extract the JSON array from the response.
    # LLMs often add preamble text or markdown fences around the JSON.
    text = raw_response.strip()

    # Find the first '[' and last ']' to extract the JSON array
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise json.JSONDecodeError(
            "No JSON array found in response",
            text,
            0,
        )
    text = text[start : end + 1]

    parsed = json.loads(text)

    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON array, got {type(parsed).__name__}")

    # Build lookup from entries for answer matching
    entry_lookup: dict[tuple[int, str], str] = {
        (e.number, e.direction): e.answer for e in entries
    }
    # Also build a number → directions map for auto-correction
    number_dirs: dict[int, list[str]] = {}
    for e in entries:
        number_dirs.setdefault(e.number, []).append(e.direction)

    clue_entries: list[ClueEntry] = []
    for item in parsed:
        number = int(item["number"])
        direction = item["direction"].lower()
        clue_text = str(item["clue"])

        key = (number, direction)
        if key not in entry_lookup:
            # Auto-correct: if this number exists in only one
            # direction, the LLM just got the direction wrong
            dirs = number_dirs.get(number, [])
            if len(dirs) == 1 and dirs[0] != direction:
                direction = dirs[0]
                key = (number, direction)
                logger.debug(
                    "Auto-corrected %d-%s → %d-%s",
                    number,
                    item["direction"].lower(),
                    number,
                    direction,
                )
            else:
                raise ValueError(
                    f"LLM returned clue for {number}-"
                    f"{item['direction'].lower()} "
                    f"which is not in the grid"
                )

        clue_entries.append(
            ClueEntry(
                number=number,
                direction=direction,
                answer=entry_lookup[key],
                clue=clue_text,
            )
        )

    # Check we got clues for all entries
    if len(clue_entries) != len(entries):
        raise ValueError(
            f"Expected {len(entries)} clues, got {len(clue_entries)}"
        )

    return clue_entries
