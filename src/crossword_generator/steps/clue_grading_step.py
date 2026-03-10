"""Composite clue generation + quality grading pipeline step."""

from __future__ import annotations

import logging

from crossword_generator.graders.clue_grader import ClueGrader
from crossword_generator.llm.base import LLMProvider
from crossword_generator.models import PuzzleEnvelope
from crossword_generator.steps.base import PipelineStep
from crossword_generator.steps.clue_step import ClueGenerationStep

logger = logging.getLogger(__name__)


class ClueWithGradingStep(PipelineStep):
    """Composite step: generate clues, grade quality, regenerate if below threshold.

    Wraps ClueGenerationStep and ClueGrader, looping up to max_retries attempts.
    Keeps the best-scoring clue set across all attempts.
    """

    def __init__(
        self,
        llm: LLMProvider,
        grader: ClueGrader,
        *,
        max_retries: int = 3,
        regenerate_on_fail: bool = True,
    ) -> None:
        self._llm = llm
        self._grader = grader
        self._max_retries = max_retries
        self._regenerate_on_fail = regenerate_on_fail
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

    def validate_input(self, envelope: PuzzleEnvelope) -> list[str]:
        errors: list[str] = []
        if envelope.fill is None:
            errors.append("Envelope has no fill result — run fill step first")
        if envelope.clues:
            errors.append("Envelope already has clues")
        return errors
