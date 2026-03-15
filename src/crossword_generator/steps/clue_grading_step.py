"""Composite clue generation + quality grading pipeline step."""

from __future__ import annotations

import json
import logging

from crossword_generator.exporters.numbering import (
    compute_crossing_words,
    compute_numbering,
)
from crossword_generator.graders.clue_grader import ClueGrader
from crossword_generator.llm.base import LLMProvider
from crossword_generator.llm.prompts.clue_generation import build_clue_repair_prompt
from crossword_generator.models import ClueEntry, ClueGrade, PuzzleEnvelope
from crossword_generator.steps.base import PipelineStep
from crossword_generator.steps.clue_step import ClueGenerationStep

logger = logging.getLogger(__name__)


class ClueWithGradingStep(PipelineStep):
    """Composite step: generate clues, grade quality, regenerate if below threshold.

    Wraps ClueGenerationStep and ClueGrader, looping up to max_retries attempts.
    Keeps the best-scoring clue set across all attempts.
    After the main loop, runs a single accuracy-repair pass for any clue
    with a dangerously low accuracy sub-score.
    """

    def __init__(
        self,
        llm: LLMProvider,
        grader: ClueGrader,
        *,
        max_retries: int = 3,
        regenerate_on_fail: bool = True,
        accuracy_repair_threshold: float = 12,
    ) -> None:
        self._llm = llm
        self._grader = grader
        self._max_retries = max_retries
        self._regenerate_on_fail = regenerate_on_fail
        self._accuracy_repair_threshold = accuracy_repair_threshold
        self._clue_step = ClueGenerationStep(llm)

    @property
    def name(self) -> str:
        return "clue-generation-with-grading"

    def run(self, envelope: PuzzleEnvelope) -> PuzzleEnvelope:
        """Generate and grade clues, retrying on low scores."""
        errors = self.validate_input(envelope)
        if errors:
            raise ValueError(
                f"ClueWithGradingStep validation failed: {'; '.join(errors)}"
            )

        max_attempts = self._max_retries if self._regenerate_on_fail else 1
        best_envelope: PuzzleEnvelope | None = None
        best_score: float = -1.0

        for attempt in range(1, max_attempts + 1):
            logger.info(
                "Clue generation+grading attempt %d/%d",
                attempt,
                max_attempts,
            )

            # Generate clues (clue_step expects empty clues)
            try:
                clued = self._clue_step.run(envelope)
            except ValueError:
                logger.warning(
                    "Attempt %d: clue generation failed", attempt
                )
                continue

            # Grade the clues
            report = self._grader.grade(clued)

            logger.info(
                "Attempt %d clue score: %.1f/100 (%s)",
                attempt,
                report.overall_score,
                "PASS" if report.passing else "FAIL",
            )

            # Populate per-clue quality scores from the grade report
            scored_clues = list(clued.clues)
            grade_lookup = {
                (g.number, g.direction): g.score
                for g in report.clue_grades
            }
            for i, clue in enumerate(scored_clues):
                key = (clue.number, clue.direction)
                if key in grade_lookup:
                    scored_clues[i] = clue.model_copy(
                        update={"quality_score": grade_lookup[key]}
                    )

            candidate = clued.model_copy(
                update={
                    "clues": scored_clues,
                    "clue_grade_report": report,
                }
            )

            if report.overall_score > best_score:
                best_score = report.overall_score
                best_envelope = candidate

            if report.passing:
                break

        if best_envelope is None:
            raise ValueError(
                f"Clue generation failed on all {max_attempts} attempt(s)"
            )

        # --- Accuracy repair pass ---
        best_envelope = self._run_accuracy_repair(best_envelope)

        # Fix step_history: clue_step already added "clue-generation",
        # replace with our composite name
        step_history = [
            s for s in best_envelope.step_history if s != "clue-generation"
        ]
        step_history.append(self.name)

        new_errors = list(best_envelope.errors)
        report = best_envelope.clue_grade_report
        if not report or not report.passing:
            new_errors.append(
                f"Clue quality below threshold after {max_attempts} attempt(s): "
                f"best score {best_score:.1f}"
            )

        return best_envelope.model_copy(
            update={
                "step_history": step_history,
                "errors": new_errors,
            }
        )

    def _run_accuracy_repair(
        self, envelope: PuzzleEnvelope
    ) -> PuzzleEnvelope:
        """Single repair pass: regenerate clues with low accuracy sub-scores."""
        report = envelope.clue_grade_report
        if not report or not report.clue_grades:
            return envelope

        # Find clues with accuracy below threshold
        grade_lookup: dict[tuple[int, str], ClueGrade] = {
            (g.number, g.direction): g for g in report.clue_grades
        }
        entries_to_repair: list[tuple[ClueEntry, ClueGrade]] = []
        for clue in envelope.clues:
            key = (clue.number, clue.direction)
            grade = grade_lookup.get(key)
            if (
                grade
                and grade.accuracy is not None
                and grade.accuracy < self._accuracy_repair_threshold
            ):
                entries_to_repair.append((clue, grade))

        if not entries_to_repair:
            return envelope

        repair_answers = [c.answer for c, _ in entries_to_repair]
        logger.info(
            "Accuracy repair: %d clue(s) below threshold (%.0f): %s",
            len(entries_to_repair),
            self._accuracy_repair_threshold,
            ", ".join(repair_answers),
        )

        # Build repair prompt
        assert envelope.fill is not None
        grid = envelope.fill.grid
        entries = compute_numbering(grid)
        crossing_words = compute_crossing_words(entries, grid)

        prompt = build_clue_repair_prompt(
            entries_to_repair=entries_to_repair,
            all_clues=envelope.clues,
            crossing_words=crossing_words,
            puzzle_type=envelope.puzzle_type,
            theme=envelope.theme,
        )

        # Call LLM for repair
        try:
            raw_response = self._llm.generate(prompt)
            repaired_clues = _parse_repair_response(
                raw_response, entries_to_repair
            )
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.warning(
                "Accuracy repair parse failed, keeping original clues: %s",
                exc,
            )
            return envelope

        # Swap repaired clues into the clue list
        repair_lookup = {
            (c.number, c.direction): c for c in repaired_clues
        }
        updated_clues = []
        for clue in envelope.clues:
            key = (clue.number, clue.direction)
            if key in repair_lookup:
                updated_clues.append(repair_lookup[key])
            else:
                updated_clues.append(clue)

        repaired_envelope = envelope.model_copy(
            update={"clues": updated_clues}
        )

        # Re-grade to get updated scores
        new_report = self._grader.grade(repaired_envelope)

        logger.info(
            "Post-repair score: %.1f/100 (was %.1f/100)",
            new_report.overall_score,
            report.overall_score,
        )

        # Update quality scores on clues
        scored_clues = list(repaired_envelope.clues)
        new_grade_lookup = {
            (g.number, g.direction): g.score
            for g in new_report.clue_grades
        }
        for i, clue in enumerate(scored_clues):
            key = (clue.number, clue.direction)
            if key in new_grade_lookup:
                scored_clues[i] = clue.model_copy(
                    update={"quality_score": new_grade_lookup[key]}
                )

        return repaired_envelope.model_copy(
            update={
                "clues": scored_clues,
                "clue_grade_report": new_report,
            }
        )

    def validate_input(self, envelope: PuzzleEnvelope) -> list[str]:
        errors: list[str] = []
        if envelope.fill is None:
            errors.append("Envelope has no fill result — run fill step first")
        if envelope.clues:
            errors.append("Envelope already has clues")
        return errors


def _parse_repair_response(
    raw_response: str,
    entries_to_repair: list[tuple[ClueEntry, ClueGrade]],
) -> list[ClueEntry]:
    """Parse the LLM's repair response into ClueEntry objects.

    Raises:
        json.JSONDecodeError: If the response is not valid JSON.
        ValueError: If the response structure is unexpected.
    """
    text = raw_response.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise json.JSONDecodeError(
            "No JSON array found in repair response", text, 0
        )
    text = text[start : end + 1]
    parsed = json.loads(text)

    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON array, got {type(parsed).__name__}")

    # Build lookup of expected entries
    expected = {
        (c.number, c.direction): c for c, _ in entries_to_repair
    }

    repaired: list[ClueEntry] = []
    for item in parsed:
        number = int(item["number"])
        direction = item["direction"].lower()
        key = (number, direction)
        if key not in expected:
            raise ValueError(
                f"Repair response contains unexpected entry {number}-{direction}"
            )
        original = expected[key]
        repaired.append(
            ClueEntry(
                number=number,
                direction=direction,
                answer=original.answer,
                clue=str(item["clue"]),
            )
        )

    if len(repaired) != len(entries_to_repair):
        raise ValueError(
            f"Expected {len(entries_to_repair)} repaired clues, "
            f"got {len(repaired)}"
        )

    return repaired
