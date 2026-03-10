"""Theme generation pipeline step for midi puzzles."""

from __future__ import annotations

import json
import logging

from crossword_generator.dictionary import Dictionary
from crossword_generator.fillers.csp import extract_slots
from crossword_generator.grid_specs import get_grid_spec
from crossword_generator.llm.base import LLMProvider
from crossword_generator.llm.prompts.theme_generation import (
    build_theme_generation_prompt,
)
from crossword_generator.models import PuzzleEnvelope, PuzzleType, ThemeConcept
from crossword_generator.steps.base import PipelineStep

logger = logging.getLogger(__name__)


class ThemeGenerationStep(PipelineStep):
    """Pipeline step that generates a theme concept for midi puzzles."""

    def __init__(
        self,
        llm: LLMProvider,
        dictionary: Dictionary,
        *,
        grid_size: int = 9,
        max_retries: int = 3,
        num_seed_entries: int = 3,
    ) -> None:
        self._llm = llm
        self._dictionary = dictionary
        self._grid_size = grid_size
        self._max_retries = max_retries
        self._num_seed_entries = num_seed_entries

    @property
    def name(self) -> str:
        return "theme-generation"

    def run(self, envelope: PuzzleEnvelope) -> PuzzleEnvelope:
        """Generate a theme concept and return an updated envelope."""
        errors = self.validate_input(envelope)
        if errors:
            raise ValueError(
                f"ThemeGenerationStep validation failed: {'; '.join(errors)}"
            )

        # Preview the grid to learn available slot lengths
        seed = envelope.metadata.get("seed")
        spec = get_grid_spec(envelope.puzzle_type, envelope.grid_size, seed=seed)
        black = set(spec.black_cells)
        slots = extract_slots(spec.rows, spec.cols, black)
        available_lengths = sorted({s.length for s in slots})

        prompt = build_theme_generation_prompt(
            grid_size=self._grid_size,
            available_slot_lengths=available_lengths,
            num_seed_entries=self._num_seed_entries,
        )

        theme: ThemeConcept | None = None
        last_error = ""

        for attempt in range(1, self._max_retries + 1):
            logger.info(
                "Theme generation attempt %d/%d using %s",
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
                theme = _parse_theme_response(raw_response)
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                last_error = str(exc)
                logger.warning(
                    "Attempt %d: failed to parse theme response: %s",
                    attempt,
                    last_error,
                )
                continue

            validation_errors = _validate_theme_entries(
                theme, self._dictionary, self._grid_size, available_lengths,
            )
            if validation_errors:
                last_error = "; ".join(validation_errors)
                logger.warning(
                    "Attempt %d: theme validation failed: %s",
                    attempt,
                    last_error,
                )
                theme = None
                continue

            break

        if theme is None:
            raise ValueError(
                f"Failed to generate valid theme after {self._max_retries} "
                f"attempts. Last error: {last_error}"
            )

        logger.info(
            "Theme generated: topic=%r, %d seed entries, revealer=%r",
            theme.topic,
            len(theme.seed_entries),
            theme.revealer,
        )

        return envelope.model_copy(
            update={
                "theme": theme,
                "step_history": [*envelope.step_history, self.name],
            }
        )

    def validate_input(self, envelope: PuzzleEnvelope) -> list[str]:
        """Validate that the envelope is ready for theme generation."""
        errors: list[str] = []
        if envelope.theme is not None:
            errors.append("Envelope already has a theme")
        if envelope.puzzle_type != PuzzleType.MIDI:
            errors.append(
                f"Theme generation is only for midi puzzles, "
                f"got {envelope.puzzle_type.value}"
            )
        return errors


def _parse_theme_response(raw_response: str) -> ThemeConcept:
    """Parse the LLM's JSON response into a ThemeConcept.

    Raises:
        json.JSONDecodeError: If the response is not valid JSON.
        ValueError: If required fields are missing or invalid.
        KeyError: If required fields are missing.
    """
    text = raw_response.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove last ``` line
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Find JSON object boundaries
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise json.JSONDecodeError(
            "No JSON object found in response", text, 0
        )
    text = text[start : end + 1]

    parsed = json.loads(text)

    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")

    topic = str(parsed["topic"])
    wordplay_type = str(parsed.get("wordplay_type", "literal"))
    seed_entries = [str(e).upper() for e in parsed["seed_entries"]]
    revealer = str(parsed["revealer"]).upper()
    revealer_clue = str(parsed.get("revealer_clue", ""))

    if not topic:
        raise ValueError("Theme topic is empty")
    if not seed_entries:
        raise ValueError("No seed entries provided")
    if not revealer:
        raise ValueError("Revealer is empty")

    return ThemeConcept(
        topic=topic,
        wordplay_type=wordplay_type,
        seed_entries=seed_entries,
        revealer=revealer,
        revealer_clue=revealer_clue,
    )


def _validate_theme_entries(
    theme: ThemeConcept,
    dictionary: Dictionary,
    grid_size: int,
    available_lengths: list[int],
) -> list[str]:
    """Validate that all theme entries meet requirements.

    Returns:
        A list of validation error messages. Empty list means valid.
    """
    errors: list[str] = []
    all_words = list(theme.seed_entries) + [theme.revealer]

    for word in all_words:
        if len(word) < 3 or len(word) > grid_size:
            errors.append(
                f"{word!r} length {len(word)} is outside range 3-{grid_size}"
            )
        if not dictionary.contains(word):
            errors.append(f"{word!r} is not in the dictionary")
        if len(word) not in available_lengths:
            errors.append(
                f"{word!r} length {len(word)} doesn't match any available "
                f"slot length ({available_lengths})"
            )

    # Check for duplicates
    seen: set[str] = set()
    for word in all_words:
        if word in seen:
            errors.append(f"Duplicate entry: {word!r}")
        seen.add(word)

    return errors
