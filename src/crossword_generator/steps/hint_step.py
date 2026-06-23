"""Hint generation pipeline step.

Writes an easier alternate clue ("hint") for every entry, to be surfaced to
solvers who ask for help. Runs AFTER clue generation so each hint can see the
real clue and deliberately take a simpler, different angle. Hints are OPTIONAL:
an entry whose clue is already maximally easy gets no hint (empty), rather than
a redundant one.

Each chunk of entries converges independently: generated hints are screened
with the same mechanical leak detector the clues use and fact-checked with the
shared clue fact-checker; any hint that leaks or is factually flagged is
repaired and re-screened, until a clean sweep or the round budget is spent. A
hint that cannot be made clean is dropped (the entry ships without a hint).
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor

from crossword_generator.dictionary import Dictionary
from crossword_generator.exporters.numbering import NumberedEntry, compute_numbering
from crossword_generator.graders.clue_fact_checker import ClueFactChecker
from crossword_generator.graders.hint_fact_checker import fact_check_hints
from crossword_generator.graders.leak_detector import detect_leak
from crossword_generator.llm.base import LLMProvider
from crossword_generator.llm.prompts.hint_generation import (
    build_hint_generation_messages,
    build_hint_repair_messages,
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
        fact_checker: ClueFactChecker | None = None,
        repair_llm: LLMProvider | None = None,
        max_repair_rounds: int = 3,
    ) -> None:
        self._llm = llm
        self._repair_llm = repair_llm or llm
        self._max_retries = max_retries
        self._chunk_size = chunk_size or 0
        self._parallel_chunks = parallel_chunks
        self._parallel_chunk_workers = max(1, parallel_chunk_workers)
        self._dictionary = dictionary
        self._fact_checker = fact_checker
        self._max_repair_rounds = max(1, max_repair_rounds)

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

        hints = self._generate_chunks(chunks, clues_by_key, envelope)

        # Attach hints to the matching clue entries.
        updated_clues = [
            clue.model_copy(
                update={"hint": hints.get((clue.number, clue.direction), "")}
            )
            for clue in envelope.clues
        ]

        # Hints are optional: an entry whose clue is already maximally easy
        # (or whose only candidate hint kept failing screening) gets none. Log
        # coverage at info level for visibility, not as a warning.
        with_hint = sum(1 for c in updated_clues if c.hint)
        logger.info(
            "Hint generation: %d/%d entries have a hint (%d intentionally or "
            "unavoidably skipped)",
            with_hint,
            len(updated_clues),
            len(updated_clues) - with_hint,
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
        envelope: PuzzleEnvelope,
    ) -> dict[tuple[int, str], str]:
        """Generate (and converge) hints for every chunk and merge them.

        Mirrors clue generation's cache strategy: when running in parallel,
        warm the shared (cacheable) system prompt with chunk 0 before fanning
        out the rest so later chunks read the hot cache instead of each paying
        to recreate the same block.
        """
        n = len(chunks)

        def gen(chunk: list[NumberedEntry]) -> dict[tuple[int, str], str]:
            return self._converge_chunk(chunk, clues_by_key, envelope)

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

    def _converge_chunk(
        self,
        entries: list[NumberedEntry],
        clues_by_key: dict[tuple[int, str], str],
        envelope: PuzzleEnvelope,
    ) -> dict[tuple[int, str], str]:
        """Generate, screen, and repair hints for one chunk until clean.

        First generates a candidate hint per entry (an empty hint is a valid
        "deliberately skipped" result, not a failure). Then runs a convergence
        loop: each round leak-screens and fact-checks the current non-empty
        hints, repairs every flagged hint in one call, and re-screens. The loop
        exits on a clean sweep, when a repair changes nothing (stuck), or when
        the round budget is spent. A hint that never comes clean is dropped, so
        the entry simply ships without a hint rather than with a bad one.
        """
        answers = {(e.number, e.direction): e.answer for e in entries}
        entry_by_key = {(e.number, e.direction): e for e in entries}

        hints = self._generate_candidates(entries, clues_by_key)

        for round_num in range(1, self._max_repair_rounds + 1):
            flagged = self._collect_findings(hints, answers, envelope)
            if not flagged:
                logger.info(
                    "Hint convergence: round %d clean — converged", round_num
                )
                break

            repair_entries = [entry_by_key[key] for key in flagged]
            self._log_findings(round_num, flagged)
            before = {k: hints.get(k, "") for k in flagged}
            repaired = self._repair_hints(
                repair_entries, clues_by_key, hints, flagged
            )
            changed = False
            for key in flagged:
                new_hint = repaired.get(key, "")
                if new_hint != before[key]:
                    changed = True
                hints[key] = new_hint
            if not changed:
                logger.info(
                    "Hint convergence: round %d made no change — stopping "
                    "(stuck); dropping %d still-flagged hint(s)",
                    round_num,
                    len(flagged),
                )
                for key in flagged:
                    hints[key] = ""
                break
        else:
            # Budget spent with findings remaining: drop whatever is still
            # flagged rather than ship a leaking/false hint.
            flagged = self._collect_findings(hints, answers, envelope)
            if flagged:
                logger.warning(
                    "Hint convergence: hit round budget (%d) with %d hint(s) "
                    "still flagged — dropping them",
                    self._max_repair_rounds,
                    len(flagged),
                )
                for key in flagged:
                    hints[key] = ""

        # Drop empties so callers treat the entry as hint-less.
        return {key: hint for key, hint in hints.items() if hint}

    def _generate_candidates(
        self,
        entries: list[NumberedEntry],
        clues_by_key: dict[tuple[int, str], str],
    ) -> dict[tuple[int, str], str]:
        """Generate one candidate hint per entry (empty == deliberate skip).

        Retries only entries the model omitted entirely or that failed to
        parse, up to ``max_retries`` passes. An explicit empty hint is accepted
        as-is; screening/repair happens in the convergence loop.
        """
        pending = list(entries)
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

            # parse_hint_response drops empty hints, so a key absent from the
            # response means the model deliberately skipped it: accept "" and
            # stop retrying that entry. Only entries the model omitted on a
            # parse-failed attempt remain pending.
            for entry in pending:
                key = (entry.number, entry.direction)
                result[key] = hints.get(key, "")
            pending = []

        if pending:
            logger.warning(
                "Hint generation gave up on %d entries after %d attempts "
                "(last error: %s)",
                len(pending),
                self._max_retries,
                last_error or "parse failure",
            )
            for entry in pending:
                result.setdefault((entry.number, entry.direction), "")
        return result

    def _collect_findings(
        self,
        hints: dict[tuple[int, str], str],
        answers: dict[tuple[int, str], str],
        envelope: PuzzleEnvelope,
    ) -> dict[tuple[int, str], list[str]]:
        """Screen current hints; return {key: [reasons]} for flagged ones.

        Empty hints are valid skips and never flagged. A non-empty hint is
        flagged if it leaks its answer (mechanical detector) or the fact-checker
        marks it uncertain/incorrect.
        """
        flagged: dict[tuple[int, str], list[str]] = {}

        for key, hint in hints.items():
            if not hint:
                continue
            answer = answers.get(key)
            if answer is None:
                continue
            leak = detect_leak(answer, hint, self._dictionary)
            if leak is not None:
                flagged.setdefault(key, []).append(
                    f"the hint leaks the answer ({leak.kind}: "
                    f'"{leak.detail}")'
                )

        if self._fact_checker is not None:
            non_empty = {k: h for k, h in hints.items() if h}
            fact_results = fact_check_hints(
                self._fact_checker, envelope, non_empty
            )
            for key, result in fact_results.items():
                if result.needs_repair:
                    flagged.setdefault(key, []).append(
                        f"factually {result.status}: {result.reason}"
                    )

        return flagged

    def _repair_hints(
        self,
        entries: list[NumberedEntry],
        clues_by_key: dict[tuple[int, str], str],
        current_hints: dict[tuple[int, str], str],
        reasons_by_key: dict[tuple[int, str], list[str]],
    ) -> dict[tuple[int, str], str]:
        """Repair flagged hints in one call; return new hints by key.

        A key missing from the response (or with an empty hint) means the
        repair produced no usable replacement — the caller treats that as a
        dropped hint.
        """
        system_text, user_text = build_hint_repair_messages(
            entries, clues_by_key, current_hints, reasons_by_key
        )
        raw = self._repair_llm.generate(user_text, system=system_text)
        try:
            return parse_hint_response(raw, entries)
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.warning("Failed to parse hint repair response: %s", exc)
            return {}

    def _log_findings(
        self,
        round_num: int,
        flagged: dict[tuple[int, str], list[str]],
    ) -> None:
        summary = ", ".join(
            f"{num}-{direction} ({'; '.join(reasons)})"
            for (num, direction), reasons in flagged.items()
        )
        logger.info(
            "Hint convergence: round %d repairing %d hint(s): %s",
            round_num,
            len(flagged),
            summary,
        )

    def validate_input(self, envelope: PuzzleEnvelope) -> list[str]:
        errors: list[str] = []
        if envelope.fill is None:
            errors.append("Envelope has no fill result — run fill step first")
        if not envelope.clues:
            errors.append("Envelope has no clues — run clue step first")
        return errors
