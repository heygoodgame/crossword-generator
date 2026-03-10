"""Clue generation pipeline step."""

from __future__ import annotations

import json
import logging

from crossword_generator.exporters.numbering import NumberedEntry, compute_numbering
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
        crossing_words = _compute_crossing_words(entries, grid)

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


def _compute_crossing_words(
    entries: list[NumberedEntry], grid: list[list[str]]
) -> dict[tuple[int, str], list[str]]:
    """Compute which words cross each entry in the grid.

    For each entry, find all entries that share at least one cell.
    Returns a mapping of (number, direction) → list of crossing answer words.
    """
    # Build a cell → entry index for fast lookup
    cell_to_entries: dict[tuple[int, int], list[int]] = {}
    for idx, entry in enumerate(entries):
        if entry.direction == "across":
            for offset in range(entry.length):
                cell = (entry.row, entry.col + offset)
                cell_to_entries.setdefault(cell, []).append(idx)
        else:  # down
            for offset in range(entry.length):
                cell = (entry.row + offset, entry.col)
                cell_to_entries.setdefault(cell, []).append(idx)

    # For each entry, collect crossing entry answers
    crossing_words: dict[tuple[int, str], list[str]] = {}
    for idx, entry in enumerate(entries):
        crossings: list[str] = []
        if entry.direction == "across":
            cells = [(entry.row, entry.col + offset) for offset in range(entry.length)]
        else:
            cells = [(entry.row + offset, entry.col) for offset in range(entry.length)]

        seen: set[int] = set()
        for cell in cells:
            for other_idx in cell_to_entries.get(cell, []):
                if other_idx != idx and other_idx not in seen:
                    seen.add(other_idx)
                    crossings.append(entries[other_idx].answer)

        crossing_words[(entry.number, entry.direction)] = crossings

    return crossing_words


def _parse_clue_response(
    raw_response: str, entries: list[NumberedEntry]
) -> list[ClueEntry]:
    """Parse the LLM's JSON response into a list of ClueEntry objects.

    Raises:
        json.JSONDecodeError: If the response is not valid JSON.
        ValueError: If the response doesn't contain the expected entries.
        KeyError: If required fields are missing.
    """
    # Try to extract JSON from the response (LLM may wrap it in markdown)
    text = raw_response.strip()
    if text.startswith("```"):
        # Strip markdown code fence
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines)

    parsed = json.loads(text)

    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON array, got {type(parsed).__name__}")

    # Build lookup from entries for answer matching
    entry_lookup: dict[tuple[int, str], str] = {
        (e.number, e.direction): e.answer for e in entries
    }

    clue_entries: list[ClueEntry] = []
    for item in parsed:
        number = int(item["number"])
        direction = item["direction"].lower()
        clue_text = str(item["clue"])

        key = (number, direction)
        if key not in entry_lookup:
            raise ValueError(
                f"LLM returned clue for {number}-{direction} which is not in the grid"
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
