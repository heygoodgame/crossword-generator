"""LLM-based clue quality grader."""

from __future__ import annotations

import json
import logging

from crossword_generator.exporters.numbering import (
    compute_crossing_words,
    compute_numbering,
)
from crossword_generator.llm.base import LLMProvider
from crossword_generator.llm.prompts.clue_evaluation import (
    build_clue_evaluation_prompt,
)
from crossword_generator.models import ClueGrade, ClueGradeReport, PuzzleEnvelope

logger = logging.getLogger(__name__)


class ClueGrader:
    """Scores crossword clues using LLM-as-judge evaluation.

    Sends all clues to the LLM with a scoring rubric and parses
    per-clue scores and feedback from the JSON response.
    """

    def __init__(
        self,
        llm: LLMProvider,
        *,
        min_passing_score: int = 70,
        max_parse_retries: int = 3,
    ) -> None:
        self._llm = llm
        self._min_passing_score = min_passing_score
        self._max_parse_retries = max_parse_retries

    def grade(self, envelope: PuzzleEnvelope) -> ClueGradeReport:
        """Grade all clues in the envelope and return a report."""
        if not envelope.clues:
            return ClueGradeReport(
                overall_score=0.0,
                clue_count=0,
                passing=False,
                summary="No clues to grade.",
            )

        assert envelope.fill is not None
        grid = envelope.fill.grid

        # Compute crossing words for the evaluation prompt
        entries = compute_numbering(grid)
        crossing_words = compute_crossing_words(entries, grid)

        # Build evaluation prompt
        prompt = build_clue_evaluation_prompt(
            clues=envelope.clues,
            crossing_words=crossing_words,
            puzzle_type=envelope.puzzle_type,
            theme=envelope.theme,
        )

        # Call LLM with retries on parse failure
        clue_grades: list[ClueGrade] | None = None
        last_error = ""

        for attempt in range(1, self._max_parse_retries + 1):
            logger.info(
                "Clue evaluation attempt %d/%d using %s",
                attempt,
                self._max_parse_retries,
                self._llm.name,
            )
            raw_response = self._llm.generate(prompt)
            logger.debug(
                "Raw evaluation response (%d chars): %.200s",
                len(raw_response),
                raw_response,
            )
            try:
                clue_grades = _parse_evaluation_response(
                    raw_response, envelope.clues
                )
                break
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                last_error = str(exc)
                logger.warning(
                    "Attempt %d: failed to parse evaluation response: %s",
                    attempt,
                    last_error,
                )

        if clue_grades is None:
            logger.error(
                "Failed to parse clue evaluation after %d attempts: %s",
                self._max_parse_retries,
                last_error,
            )
            return ClueGradeReport(
                overall_score=0.0,
                clue_count=len(envelope.clues),
                passing=False,
                summary=(
                    f"Evaluation parse failed after "
                    f"{self._max_parse_retries} attempts."
                ),
            )

        # Compute aggregate score
        if clue_grades:
            overall_score = sum(g.score for g in clue_grades) / len(clue_grades)
        else:
            overall_score = 0.0

        passing = overall_score >= self._min_passing_score

        summary = (
            f"{len(clue_grades)} clues, "
            f"score {overall_score:.1f}/100 "
            f"({'PASS' if passing else 'FAIL'})"
        )

        return ClueGradeReport(
            overall_score=overall_score,
            clue_count=len(clue_grades),
            passing=passing,
            clue_grades=clue_grades,
            summary=summary,
        )


def _parse_evaluation_response(
    raw_response: str, clues: list[object]
) -> list[ClueGrade]:
    """Parse the LLM's JSON evaluation response into ClueGrade objects.

    Raises:
        json.JSONDecodeError: If the response is not valid JSON.
        ValueError: If the response structure is unexpected.
    """
    text = raw_response.strip()

    # Extract JSON array from potential markdown/preamble
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

    grades: list[ClueGrade] = []
    for item in parsed:
        # Extract per-dimension sub-scores (0-25 each)
        accuracy = max(0.0, min(25.0, float(item.get("accuracy", 0))))
        freshness = max(0.0, min(25.0, float(item.get("freshness", 0))))
        craft = max(0.0, min(25.0, float(item.get("craft", 0))))
        fairness = max(0.0, min(25.0, float(item.get("fairness", 0))))
        score = accuracy + freshness + craft + fairness

        grades.append(
            ClueGrade(
                number=int(item["number"]),
                direction=item["direction"].lower(),
                answer=item.get("answer", ""),
                score=score,
                feedback=str(item.get("feedback", "")),
                accuracy=accuracy,
                freshness=freshness,
                craft=craft,
                fairness=fairness,
            )
        )

    if len(grades) != len(clues):
        raise ValueError(
            f"Expected {len(clues)} evaluations, got {len(grades)}"
        )

    return grades
