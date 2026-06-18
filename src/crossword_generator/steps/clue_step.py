"""Clue generation pipeline step."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor

from crossword_generator.clue_history import ClueHistoryIndex
from crossword_generator.exporters.numbering import (
    NumberedEntry,
    compute_crossing_words,
    compute_numbering,
)
from crossword_generator.llm.base import LLMProvider
from crossword_generator.llm.prompts.clue_generation import (
    build_clue_generation_messages,
)
from crossword_generator.models import ClueEntry, PuzzleEnvelope
from crossword_generator.steps.base import PipelineStep

logger = logging.getLogger(__name__)


class ClueGenerationStep(PipelineStep):
    """Pipeline step that generates clues for a filled grid using an LLM."""

    def __init__(
        self,
        llm: LLMProvider,
        *,
        max_retries: int = 3,
        clue_history: ClueHistoryIndex | None = None,
        chunk_size: int = 0,
        parallel_chunks: bool = False,
        parallel_chunk_workers: int = 4,
    ) -> None:
        self._llm = llm
        self._max_retries = max_retries
        self._clue_history = clue_history
        # 0 (or None) => one call for the whole puzzle. >0 => split into
        # chunks of at most this many entries per LLM call. Smaller chunks
        # let the model follow the rules more reliably across a long puzzle;
        # the (cacheable) system prompt is shared across all chunks.
        self._chunk_size = chunk_size or 0
        # When chunking, generate chunks concurrently after warming the cache
        # with the first chunk (see _generate_chunks). Off => serial.
        self._parallel_chunks = parallel_chunks
        self._parallel_chunk_workers = max(1, parallel_chunk_workers)

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
        prior_clues_by_answer = (
            self._clue_history.avoid_clues_for_answers(
                entry.answer for entry in entries
            )
            if self._clue_history is not None
            else None
        )

        # Generate in one call, or in chunks of entries (each chunk a separate
        # LLM call sharing the cacheable system prompt).
        if self._chunk_size and self._chunk_size < len(entries):
            chunks = [
                entries[i : i + self._chunk_size]
                for i in range(0, len(entries), self._chunk_size)
            ]
        else:
            chunks = [entries]

        clue_entries = self._generate_chunks(
            chunks, crossing_words, envelope, prior_clues_by_answer
        )

        return envelope.model_copy(
            update={
                "clues": clue_entries,
                "step_history": [*envelope.step_history, self.name],
            }
        )

    def _generate_chunks(
        self,
        chunks: list[list[NumberedEntry]],
        crossing_words: dict[tuple[int, str], list[str]],
        envelope: PuzzleEnvelope,
        prior_clues_by_answer: dict[str, list[str]] | None,
    ) -> list[ClueEntry]:
        """Generate clues for every chunk and concatenate them in order.

        Serial unless ``parallel_chunks`` is on and there is more than one
        chunk. In the parallel case we deliberately run the FIRST chunk alone
        before fanning out the rest: every chunk of a puzzle sends the same
        cache-eligible system prompt, but an Anthropic cache block is only
        readable once the request that created it returns. Firing all chunks at
        once from a cold cache makes each one pay full price to (re)create the
        same block. Warming with chunk 0 first lets chunks 1..N read the hot
        cache instead. Output order always follows chunk (entry) order so the
        result is identical to the serial path and manifests stay reproducible.
        """
        n = len(chunks)

        def gen(chunk: list[NumberedEntry]) -> list[ClueEntry]:
            return self._generate_for_entries(
                chunk, crossing_words, envelope, prior_clues_by_answer
            )

        if n <= 1 or not self._parallel_chunks:
            clue_entries: list[ClueEntry] = []
            for idx, chunk in enumerate(chunks, start=1):
                if n > 1:
                    logger.info(
                        "Clue generation chunk %d/%d (%d entries)",
                        idx,
                        n,
                        len(chunk),
                    )
                clue_entries.extend(gen(chunk))
            return clue_entries

        # Warm the cache with chunk 0, then fan the rest out concurrently.
        logger.info(
            "Clue generation: warming cache with chunk 1/%d, then %d in "
            "parallel (max %d workers)",
            n,
            n - 1,
            self._parallel_chunk_workers,
        )
        results: list[list[ClueEntry]] = [[] for _ in range(n)]
        results[0] = gen(chunks[0])

        workers = min(self._parallel_chunk_workers, n - 1)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(gen, chunks[i]): i for i in range(1, n)
            }
            for future, i in futures.items():
                # .result() re-raises any exception from the worker, so a chunk
                # that fails all its retries fails the whole puzzle, exactly as
                # in the serial path.
                results[i] = future.result()

        return [clue for chunk_result in results for clue in chunk_result]

    def _generate_for_entries(
        self,
        entries: list[NumberedEntry],
        crossing_words: dict[tuple[int, str], list[str]],
        envelope: PuzzleEnvelope,
        prior_clues_by_answer: dict[str, list[str]] | None,
    ) -> list[ClueEntry]:
        """Generate clues for one set of entries, retrying on parse failure."""
        system_text, user_text = build_clue_generation_messages(
            entries=entries,
            crossing_words=crossing_words,
            puzzle_type=envelope.puzzle_type,
            theme=envelope.theme,
            difficulty=envelope.difficulty,
            prior_clues_by_answer=prior_clues_by_answer,
        )

        last_error = ""
        for attempt in range(1, self._max_retries + 1):
            logger.info(
                "Clue generation attempt %d/%d using %s",
                attempt,
                self._max_retries,
                self._llm.name,
            )
            raw_response = self._llm.generate(user_text, system=system_text)
            logger.debug(
                "Raw LLM response (%d chars): %.200s",
                len(raw_response),
                raw_response,
            )
            try:
                return _parse_clue_response(raw_response, entries)
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                last_error = str(exc)
                logger.warning(
                    "Attempt %d: failed to parse LLM response: %s",
                    attempt,
                    last_error,
                )

        raise ValueError(
            f"Failed to parse clue response after {self._max_retries} "
            f"attempts. Last error: {last_error}"
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
