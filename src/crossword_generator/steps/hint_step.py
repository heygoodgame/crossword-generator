"""Hint generation pipeline step.

Writes an easier alternate clue ("hint") for every entry, to be surfaced to
solvers who ask for help. Runs AFTER clue generation so each hint can see the
real clue and deliberately take a simpler, different angle. Generated hints are
screened with the same mechanical leak detector the clues use; any hint that
leaks its answer is regenerated.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor

from crossword_generator.dictionary import Dictionary
from crossword_generator.exporters.numbering import NumberedEntry, compute_numbering
from crossword_generator.graders.leak_detector import detect_leak
from crossword_generator.llm.base import LLMProvider
from crossword_generator.llm.prompts.hint_generation import (
    build_hint_generation_messages,
    parse_hint_response,
)
from crossword_generator.models import PuzzleEnvelope
from crossword_generator.steps.base import PipelineStep

logger = logging.getLogger(__name__)


class HintGenerationStep(PipelineStep):
    """Generate an easy hint for every clue and attach it to the envelope."""

    def __init__(
        self,
        llm: LLMProvider,
        *,
        max_retries: int = 3,
        chunk_size: int = 0,
        parallel_chunks: bool = False,
        parallel_chunk_workers: int = 4,
        dictionary: Dictionary | None = None,
    ) -> None:
        self._llm = llm
        self._max_retries = max_retries
        self._chunk_size = chunk_size or 0
        self._parallel_chunks = parallel_chunks
        self._parallel_chunk_workers = max(1, parallel_chunk_workers)
        self._dictionary = dictionary

    @property
    def name(self) -> str:
        return "hint-generation"

    def run(self, envelope: PuzzleEnvelope) -> PuzzleEnvelope:
        errors = self.validate_input(envelope)
        if errors:
            raise ValueError(
                f"HintGenerationStep validation failed: {'; '.join(errors)}"
            )

        assert envelope.fill is not None
        entries = compute_numbering(envelope.fill.grid)
        clues_by_key = {
            (c.number, c.direction): c.clue for c in envelope.clues
        }

        if self._chunk_size and self._chunk_size < len(entries):
            chunks = [
                entries[i : i + self._chunk_size]
                for i in range(0, len(entries), self._chunk_size)
            ]
        else:
            chunks = [entries]

        hints = self._generate_chunks(chunks, clues_by_key)

        # Attach hints to the matching clue entries.
        updated_clues = [
            clue.model_copy(
                update={"hint": hints.get((clue.number, clue.direction), "")}
            )
            for clue in envelope.clues
        ]

        missing = [
            f"{c.number}-{c.direction}"
            for c in updated_clues
            if not c.hint
        ]
        if missing:
            logger.warning(
                "Hint generation: %d/%d entries have no hint: %s",
                len(missing),
                len(updated_clues),
                ", ".join(missing),
            )

        return envelope.model_copy(
            update={
                "clues": updated_clues,
                "step_history": [*envelope.step_history, self.name],
            }
        )

    def _generate_chunks(
        self,
        chunks: list[list[NumberedEntry]],
        clues_by_key: dict[tuple[int, str], str],
    ) -> dict[tuple[int, str], str]:
        """Generate hints for every chunk and merge them.

        Mirrors clue generation's cache strategy: when running in parallel,
        warm the shared (cacheable) system prompt with chunk 0 before fanning
        out the rest so later chunks read the hot cache instead of each paying
        to recreate the same block.
        """
        n = len(chunks)

        def gen(chunk: list[NumberedEntry]) -> dict[tuple[int, str], str]:
            return self._generate_for_entries(chunk, clues_by_key)

        merged: dict[tuple[int, str], str] = {}
        if n <= 1 or not self._parallel_chunks:
            for idx, chunk in enumerate(chunks, start=1):
                if n > 1:
                    logger.info(
                        "Hint generation chunk %d/%d (%d entries)",
                        idx,
                        n,
                        len(chunk),
                    )
                merged.update(gen(chunk))
            return merged

        logger.info(
            "Hint generation: warming cache with chunk 1/%d, then %d in "
            "parallel (max %d workers)",
            n,
            n - 1,
            self._parallel_chunk_workers,
        )
        merged.update(gen(chunks[0]))
        workers = min(self._parallel_chunk_workers, n - 1)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(gen, chunks[i]): i for i in range(1, n)}
            for future in futures:
                merged.update(future.result())
        return merged

    def _generate_for_entries(
        self,
        entries: list[NumberedEntry],
        clues_by_key: dict[tuple[int, str], str],
    ) -> dict[tuple[int, str], str]:
        """Generate (and leak-screen) hints for one chunk of entries.

        Regenerates only the entries whose hint leaked or is missing, up to
        ``max_retries`` passes. Returns whatever hints survived screening; a
        persistently-leaking entry is dropped (left hint-less) rather than
        shipping a bad hint.
        """
        pending = list(entries)
        answers = {(e.number, e.direction): e.answer for e in entries}
        result: dict[tuple[int, str], str] = {}
        last_error = ""

        for attempt in range(1, self._max_retries + 1):
            if not pending:
                break
            logger.info(
                "Hint generation attempt %d/%d (%d entries) using %s",
                attempt,
                self._max_retries,
                len(pending),
                self._llm.name,
            )
            system_text, user_text = build_hint_generation_messages(
                pending, clues_by_key
            )
            raw = self._llm.generate(user_text, system=system_text)
            try:
                hints = parse_hint_response(raw, pending)
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                last_error = str(exc)
                logger.warning(
                    "Attempt %d: failed to parse hint response: %s",
                    attempt,
                    last_error,
                )
                continue

            still_pending: list[NumberedEntry] = []
            for entry in pending:
                key = (entry.number, entry.direction)
                hint = hints.get(key, "")
                if not hint:
                    still_pending.append(entry)
                    continue
                leak = detect_leak(
                    answers[key], hint, self._dictionary, for_hint=True
                )
                if leak is not None:
                    logger.warning(
                        "Hint for %d-%s (%s) leaked (%s); regenerating",
                        entry.number,
                        entry.direction,
                        answers[key],
                        leak.kind,
                    )
                    still_pending.append(entry)
                    continue
                result[key] = hint
            pending = still_pending

        if pending:
            logger.warning(
                "Hint generation gave up on %d entries after %d attempts "
                "(last error: %s)",
                len(pending),
                self._max_retries,
                last_error or "leak",
            )
        return result

    def validate_input(self, envelope: PuzzleEnvelope) -> list[str]:
        errors: list[str] = []
        if envelope.fill is None:
            errors.append("Envelope has no fill result — run fill step first")
        if not envelope.clues:
            errors.append("Envelope has no clues — run clue step first")
        return errors
