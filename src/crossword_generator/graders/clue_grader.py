"""LLM-based clue quality grader."""

from __future__ import annotations

import json
import logging
import re

from crossword_generator.exporters.numbering import (
    compute_crossing_words,
    compute_numbering,
)
from crossword_generator.llm.base import LLMProvider
from crossword_generator.llm.prompts.clue_evaluation import (
    build_clue_evaluation_messages,
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
        system_text, user_text = build_clue_evaluation_messages(
            clues=envelope.clues,
            crossing_words=crossing_words,
            puzzle_type=envelope.puzzle_type,
            theme=envelope.theme,
            difficulty=envelope.difficulty,
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
            raw_response = self._llm.generate(user_text, system=system_text)
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

        clue_grades = _apply_deterministic_clue_penalties(
            clue_grades, envelope.clues
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


def _apply_deterministic_clue_penalties(
    grades: list[ClueGrade],
    clues: list[object],
) -> list[ClueGrade]:
    clue_lookup = {
        (getattr(c, "number"), getattr(c, "direction")): c for c in clues
    }
    adjusted: list[ClueGrade] = []

    for grade in grades:
        clue = clue_lookup.get((grade.number, grade.direction))
        feedback_parts: list[str] = []
        has_answer_leak = False
        has_editorial_issue = False
        if clue is not None:
            leak_feedback = _answer_leak_feedback(
                getattr(clue, "answer", ""),
                getattr(clue, "clue", ""),
            )
            if leak_feedback:
                feedback_parts.append(leak_feedback)
                has_answer_leak = True
            editorial_feedback = _editorial_clue_feedback(
                getattr(clue, "clue", ""),
            )
            if editorial_feedback:
                feedback_parts.append(editorial_feedback)
                has_editorial_issue = True

        if feedback_parts:
            feedback = grade.feedback
            deterministic_feedback = " ".join(feedback_parts)
            feedback = (
                f"{feedback} {deterministic_feedback}"
                if feedback
                else deterministic_feedback
            )
            accuracy = grade.accuracy or 0.0
            freshness = grade.freshness or 0.0
            craft = grade.craft or 0.0
            fairness = grade.fairness or 0.0
            if has_answer_leak:
                accuracy = min(accuracy, 8.0)
                fairness = min(fairness, 8.0)
            if has_editorial_issue:
                craft = min(craft, 8.0)
                fairness = min(fairness, 8.0)
            adjusted.append(
                grade.model_copy(
                    update={
                        "accuracy": accuracy,
                        "freshness": freshness,
                        "craft": craft,
                        "fairness": fairness,
                        "score": accuracy + freshness + craft + fairness,
                        "feedback": feedback,
                    }
                )
            )
        else:
            adjusted.append(grade)

    return adjusted


def _answer_leak_feedback(answer: str, clue: str) -> str | None:
    answer_norm = _normalize(answer)
    if len(answer_norm) < 3:
        return None

    tokens = re.findall(r"[A-Z0-9]+", clue.upper())
    for token in tokens:
        token_norm = _normalize(token)
        token_forms = _morphological_forms(token_norm)
        if answer_norm in token_forms:
            return "Deterministic check: clue contains the answer."
        if (
            len(answer_norm) == 3
            and len(token_norm) >= len(answer_norm) + 4
            and token_norm.startswith(answer_norm)
        ):
            return (
                "Deterministic check: clue contains a short answer root "
                "inside a longer word."
            )
        if len(token_norm) < 4:
            continue
        if len(answer_norm) >= 5:
            for token_form in token_forms:
                if len(token_form) < 4:
                    continue
                if (
                    answer_norm.startswith(token_form)
                    or answer_norm.endswith(token_form)
                    or (len(token_form) >= 5 and token_form in answer_norm)
                ):
                    return (
                        "Deterministic check: clue contains a significant "
                        "part of the answer or a morphological variant."
                    )

    return None


_UNPLEASANT_CLUE_PATTERNS = (
    re.compile(r"\bDEATH\b", re.IGNORECASE),
    re.compile(r"\bUNDOCUMENTED\s+IMMIGRANT(?:S)?\b", re.IGNORECASE),
    re.compile(r"\bILLEGAL\s+IMMIGRANT(?:S)?\b", re.IGNORECASE),
)


def _editorial_clue_feedback(clue: str) -> str | None:
    for pattern in _UNPLEASANT_CLUE_PATTERNS:
        if pattern.search(clue):
            return (
                "Deterministic check: clue uses unpleasant wording; use a "
                "neutral or gentle phrasing instead."
            )
    return None


def _normalize(text: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", text.upper())


def _morphological_forms(token: str) -> set[str]:
    """Return simple forms that catch clue/answer echo leaks.

    This is intentionally conservative and only handles high-confidence
    English endings that commonly show up in clue text, like wives/wife,
    teaches/teach, teaching/teach, and plural -s/-es.
    """
    forms = {token}
    if len(token) <= 3:
        return forms

    if token.endswith("IES") and len(token) > 4:
        forms.add(token[:-3] + "Y")
    if token.endswith("VES") and len(token) > 4:
        forms.add(token[:-3] + "F")
        forms.add(token[:-3] + "FE")
    if token.endswith("ING") and len(token) > 5:
        stem = token[:-3]
        forms.add(stem)
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            forms.add(stem[:-1])
    if token.endswith("ED") and len(token) > 4:
        stem = token[:-2]
        forms.add(stem)
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            forms.add(stem[:-1])
    if token.endswith("ES") and len(token) > 4:
        forms.add(token[:-2])
    if token.endswith("S") and not token.endswith("SS") and len(token) > 4:
        forms.add(token[:-1])

    return forms
