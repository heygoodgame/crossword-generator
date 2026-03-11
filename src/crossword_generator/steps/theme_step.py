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


def generate_single_theme(
    llm: LLMProvider,
    dictionary: Dictionary,
    grid_size: int,
    seed: int | None = None,
    max_retries: int = 5,
    num_seed_entries: int = 3,
    num_candidates: int = 12,
    avoid_topics: list[str] | None = None,
) -> ThemeConcept:
    """Generate a single theme concept using the LLM.

    This is the core theme generation logic, extracted for reuse by both
    the pipeline step and the standalone generate-themes CLI command.

    Args:
        llm: LLM provider to use for generation.
        dictionary: Word dictionary for validation.
        grid_size: Grid dimension (e.g., 9 for 9x9).
        seed: Optional seed for grid preview (slot length discovery).
        max_retries: Maximum number of generation attempts.
        num_seed_entries: Minimum number of valid entries needed.
        num_candidates: Total number of candidate entries to request.
        avoid_topics: List of topic strings to avoid (for dedup).

    Returns:
        A validated ThemeConcept.

    Raises:
        ValueError: If all retries are exhausted.
    """
    # Preview the grid to learn available slot lengths
    spec = get_grid_spec(PuzzleType.MIDI, grid_size, seed=seed)
    black = set(spec.black_cells)
    slots = extract_slots(spec.rows, spec.cols, black)
    available_lengths = sorted({s.length for s in slots})

    prompt = build_theme_generation_prompt(
        grid_size=grid_size,
        available_slot_lengths=available_lengths,
        num_seed_entries=num_seed_entries,
        num_candidates=num_candidates,
        avoid_topics=avoid_topics,
    )

    theme: ThemeConcept | None = None
    last_error = ""
    current_prompt = prompt
    use_surplus = num_candidates > num_seed_entries

    for attempt in range(1, max_retries + 1):
        logger.info(
            "Theme generation attempt %d/%d using %s",
            attempt,
            max_retries,
            llm.name,
        )
        raw_response = llm.generate(current_prompt)
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
            current_prompt = _retry_prompt(prompt, last_error)
            continue

        validation_errors = _validate_theme_entries(
            theme,
            dictionary,
            grid_size,
            available_lengths,
            min_valid_entries=num_seed_entries if use_surplus else None,
        )
        if validation_errors:
            last_error = "; ".join(validation_errors)
            logger.warning(
                "Attempt %d: theme validation failed: %s",
                attempt,
                last_error,
            )
            theme = None
            current_prompt = _retry_prompt(prompt, last_error)
            continue

        # When generating surplus, move seed_entries → candidate_entries
        if use_surplus:
            theme = theme.model_copy(
                update={
                    "candidate_entries": list(theme.seed_entries),
                    "seed_entries": [],
                }
            )

        break

    if theme is None:
        raise ValueError(
            f"Failed to generate valid theme after {max_retries} "
            f"attempts. Last error: {last_error}"
        )

    candidate_count = len(theme.candidate_entries)
    seed_count = len(theme.seed_entries)
    logger.info(
        "Theme generated: topic=%r, %d candidates, %d seeds, revealer=%r",
        theme.topic,
        candidate_count,
        seed_count,
        theme.revealer,
    )

    return theme


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
        num_candidates: int = 6,
    ) -> None:
        self._llm = llm
        self._dictionary = dictionary
        self._grid_size = grid_size
        self._max_retries = max_retries
        self._num_seed_entries = num_seed_entries
        self._num_candidates = num_candidates

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

        seed = envelope.metadata.get("seed")
        theme = generate_single_theme(
            llm=self._llm,
            dictionary=self._dictionary,
            grid_size=self._grid_size,
            seed=seed,
            max_retries=self._max_retries,
            num_seed_entries=self._num_seed_entries,
            num_candidates=self._num_candidates,
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

    # Find JSON object start and parse only the first object
    start = text.find("{")
    if start == -1:
        raise json.JSONDecodeError(
            "No JSON object found in response", text, 0
        )

    decoder = json.JSONDecoder()
    parsed, _ = decoder.raw_decode(text, start)

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


def _retry_prompt(original_prompt: str, error: str) -> str:
    """Build a retry prompt that includes the error from the previous attempt."""
    return (
        f"{original_prompt}\n\n"
        f"IMPORTANT: Your previous attempt was rejected because: {error}\n"
        f"Please fix these issues. Count the letters in each word carefully."
    )


def _validate_theme_entries(
    theme: ThemeConcept,
    dictionary: Dictionary,
    grid_size: int,
    available_lengths: list[int],
    *,
    min_valid_entries: int | None = None,
) -> list[str]:
    """Validate that theme entries meet requirements.

    When min_valid_entries is None, ALL seed entries must be valid (strict).
    When min_valid_entries is set, at least that many seed entries must be
    valid (relaxed — for surplus candidate generation). Invalid entries are
    filtered out rather than causing an error.

    Returns:
        A list of validation error messages. Empty list means valid.
    """
    errors: list[str] = []

    # Always validate revealer strictly
    revealer = theme.revealer
    if len(revealer) < 3 or len(revealer) > grid_size:
        errors.append(
            f"{revealer!r} length {len(revealer)} is outside range "
            f"3-{grid_size}"
        )
    if not dictionary.contains(revealer):
        errors.append(f"{revealer!r} is not in the dictionary")
    if len(revealer) not in available_lengths:
        errors.append(
            f"{revealer!r} length {len(revealer)} doesn't fit any grid slot"
        )

    # Validate seed entries
    valid_entries: list[str] = []
    seen: set[str] = {revealer}
    for word in theme.seed_entries:
        word_errors: list[str] = []
        if len(word) < 3 or len(word) > grid_size:
            word_errors.append(
                f"{word!r} length {len(word)} is outside range 3-{grid_size}"
            )
        if not dictionary.contains(word):
            word_errors.append(f"{word!r} is not in the dictionary")
        # In relaxed/surplus mode, skip slot-length check — the fill step
        # handles filtering candidates by what fits each grid pattern.
        if min_valid_entries is None and len(word) not in available_lengths:
            word_errors.append(
                f"{word!r} length {len(word)} doesn't fit any grid slot"
            )
        if word in seen:
            word_errors.append(f"Duplicate entry: {word!r}")
        seen.add(word)

        if word_errors:
            if min_valid_entries is not None:
                # Relaxed mode: log but don't block on individual entries
                logger.debug(
                    "Candidate %s invalid (will be filtered): %s",
                    word,
                    "; ".join(word_errors),
                )
            else:
                errors.extend(word_errors)
        else:
            valid_entries.append(word)

    # In relaxed mode, check we have enough valid entries
    if min_valid_entries is not None:
        if len(valid_entries) < min_valid_entries:
            errors.append(
                f"Only {len(valid_entries)} valid candidates, need at "
                f"least {min_valid_entries}"
            )
        else:
            # Replace seed_entries with only the valid ones
            theme.seed_entries[:] = valid_entries

    return errors
