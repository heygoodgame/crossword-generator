"""Puzzle naming pipeline step."""

from __future__ import annotations

import json
import logging

from crossword_generator.llm.base import LLMProvider
from crossword_generator.llm.prompts.puzzle_naming import (
    build_puzzle_naming_prompt,
)
from crossword_generator.models import PuzzleEnvelope
from crossword_generator.steps.base import PipelineStep

logger = logging.getLogger(__name__)


class PuzzleNamingStep(PipelineStep):
    """Pipeline step that generates a creative puzzle title using an LLM."""

    def __init__(self, llm: LLMProvider, *, max_retries: int = 3) -> None:
        self._llm = llm
        self._max_retries = max_retries

    @property
    def name(self) -> str:
        return "puzzle-naming"

    def run(self, envelope: PuzzleEnvelope) -> PuzzleEnvelope:
        """Generate a title for the completed puzzle."""
        errors = self.validate_input(envelope)
        if errors:
            raise ValueError(
                f"PuzzleNamingStep validation failed: {'; '.join(errors)}"
            )

        assert envelope.fill is not None

        prompt = build_puzzle_naming_prompt(
            puzzle_type=envelope.puzzle_type,
            grid_size=envelope.grid_size,
            clues=envelope.clues,
            grid=envelope.fill.grid,
            theme=envelope.theme,
        )

        title: str | None = None
        last_error = ""

        for attempt in range(1, self._max_retries + 1):
            logger.info(
                "Puzzle naming attempt %d/%d using %s",
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
                title = _parse_title_response(raw_response)
                break
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                last_error = str(exc)
                logger.warning(
                    "Attempt %d: failed to parse LLM response: %s",
                    attempt,
                    last_error,
                )

        if title is None:
            # Fall back to generic title — don't block the pipeline
            fallback = f"{envelope.puzzle_type.value.title()} Crossword"
            logger.warning(
                "Failed to generate title after %d attempts "
                "(last error: %s). Falling back to: %s",
                self._max_retries,
                last_error,
                fallback,
            )
            title = fallback

        return envelope.model_copy(
            update={
                "title": title,
                "step_history": [*envelope.step_history, self.name],
            }
        )

    def validate_input(self, envelope: PuzzleEnvelope) -> list[str]:
        errors: list[str] = []
        if envelope.fill is None:
            errors.append("Envelope has no fill result — run fill step first")
        if not envelope.clues:
            errors.append("Envelope has no clues — run clue step first")
        return errors


def _parse_title_response(raw_response: str) -> str:
    """Parse the LLM's JSON response to extract the title.

    Raises:
        json.JSONDecodeError: If the response is not valid JSON.
        ValueError: If the title is empty or missing.
        KeyError: If the 'title' key is missing.
    """
    text = raw_response.strip()

    # Find the JSON object in the response
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise json.JSONDecodeError(
            "No JSON object found in response",
            text,
            0,
        )
    text = text[start : end + 1]

    parsed = json.loads(text)

    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")

    title = parsed["title"]
    if not isinstance(title, str) or not title.strip():
        raise ValueError("Title is empty or not a string")

    return title.strip()
