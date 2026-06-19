"""LLM-based clue quality grader."""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from importlib import resources

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
        return self._grade_clues(envelope, envelope.clues)

    def grade_subset(
        self,
        envelope: PuzzleEnvelope,
        keys: set[tuple[int, str]],
    ) -> ClueGradeReport:
        """Grade only the clues at ``keys`` (number, direction); merge-ready.

        Used after a repair pass: only the repaired clues actually changed, so
        re-scoring the whole puzzle wastes the bulk of the grader's (output-
        dominated) cost. Crossing words are still computed from the FULL grid,
        so each graded clue sees the same context as a full grade. The returned
        report's ``clue_grades`` covers only ``keys`` — callers merge it into
        the prior report (see ``_merge_grade_report``) and recompute the
        aggregate over the full clue set.
        """
        subset = [
            c for c in envelope.clues if (c.number, c.direction) in keys
        ]
        return self._grade_clues(envelope, subset)

    def _grade_clues(
        self,
        envelope: PuzzleEnvelope,
        clues_to_grade: list,
    ) -> ClueGradeReport:
        """Grade ``clues_to_grade`` (a subset of, or all of, the envelope)."""
        if not clues_to_grade:
            return ClueGradeReport(
                overall_score=0.0,
                clue_count=0,
                passing=False,
                summary="No clues to grade.",
            )

        assert envelope.fill is not None
        grid = envelope.fill.grid

        # Compute crossing words from the FULL grid so a subset clue still sees
        # the same crossing context it would in a full grade.
        entries = compute_numbering(grid)
        crossing_words = compute_crossing_words(entries, grid)

        # Build evaluation prompt
        system_text, user_text = build_clue_evaluation_messages(
            clues=clues_to_grade,
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
                    raw_response, clues_to_grade
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
                clue_count=len(clues_to_grade),
                passing=False,
                summary=(
                    f"Evaluation parse failed after "
                    f"{self._max_parse_retries} attempts."
                ),
            )

        clue_grades = _apply_deterministic_clue_penalties(
            clue_grades, clues_to_grade
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

    answer_forms = _lexical_forms(answer_norm)
    answer_related_forms = _related_lexical_forms(answer_forms)
    tokens = [_normalize(token) for token in re.findall(r"[A-Z0-9]+", clue.upper())]
    if _has_initialism_expansion(answer_norm, tokens):
        return (
            "Deterministic check: clue contains an abbreviation expansion "
            "for the answer."
        )

    clue_candidates = _clue_lexical_candidates(tokens)
    for candidate in clue_candidates:
        if candidate in answer_related_forms:
            return (
                "Deterministic check: clue contains a related answer root "
                "or abbreviation expansion."
            )

    for token in tokens:
        token_norm = token
        token_forms = _lexical_forms(token_norm)
        if answer_norm == token_norm:
            return "Deterministic check: clue contains the answer."
        if answer_norm in token_forms or token_norm in answer_forms:
            return (
                "Deterministic check: clue contains a morphological variant "
                "of the answer."
            )
        if _significant_form_overlap(
            answer_forms,
            token_forms,
        ):
            return (
                "Deterministic check: clue contains a significant "
                "answer root or morphological variant."
            )
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


def _lexical_forms(token: str) -> set[str]:
    """Return conservative morphological forms for answer/clue leakage checks."""
    return _morphological_forms(token)


def _related_lexical_forms(answer_forms: set[str]) -> set[str]:
    """Return curated related roots/expansions for answer forms."""
    related: set[str] = set()
    family_index = _related_lexical_family_index()
    for form in answer_forms:
        related.update(family_index.get(form, set()))
    return related - answer_forms


def _clue_lexical_candidates(tokens: list[str]) -> set[str]:
    """Return clue token and phrase forms to compare against lexical families."""
    candidates: set[str] = set()
    max_phrase_words = _max_related_lexical_family_words()
    for index, token in enumerate(tokens):
        candidates.update(_lexical_forms(token))
        phrase = token
        for end_index in range(index + 1, min(len(tokens), index + max_phrase_words)):
            phrase += tokens[end_index]
            candidates.add(phrase)
    return candidates


_ABBREVIATION_SIGNAL_TOKENS = {
    "ABBR",
    "ABBREVIATED",
    "ABBREVIATION",
    "BRIEF",
    "BRIEFLY",
    "INITIAL",
    "INITIALLY",
    "INITIALS",
    "SHORT",
}


def _has_initialism_expansion(answer: str, tokens: list[str]) -> bool:
    """Return true when a clue spells out a signaled initialism answer."""
    if not (2 <= len(answer) <= 5 and answer.isalpha()):
        return False
    if not _has_abbreviation_signal(tokens):
        return False

    phrase_tokens = [
        token for token in tokens if token not in _ABBREVIATION_SIGNAL_TOKENS
    ]
    for index in range(0, len(phrase_tokens) - len(answer) + 1):
        window = phrase_tokens[index : index + len(answer)]
        if any(len(token) <= 1 for token in window):
            continue
        if "".join(token[0] for token in window) == answer:
            return True
    return False


def _has_abbreviation_signal(tokens: list[str]) -> bool:
    token_set = set(tokens)
    if token_set & _ABBREVIATION_SIGNAL_TOKENS:
        return True
    joined = " ".join(tokens)
    return "FOR SHORT" in joined or "IN BRIEF" in joined


@lru_cache(maxsize=1)
def _related_lexical_family_index() -> dict[str, set[str]]:
    """Load editable lexical leakage families from package data."""
    index: dict[str, set[str]] = {}
    text = resources.files("crossword_generator.resources").joinpath(
        "clue_leakage_families.txt"
    ).read_text(encoding="utf-8")
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" in line:
            line = line.split(":", 1)[1]
        forms = {
            normalized
            for part in line.split("|")
            if (normalized := _normalize(part))
        }
        for form in tuple(forms):
            forms.update(_morphological_forms(form))
        for form in forms:
            index.setdefault(form, set()).update(forms)
    return index


@lru_cache(maxsize=1)
def _max_related_lexical_family_words() -> int:
    max_words = 1
    text = resources.files("crossword_generator.resources").joinpath(
        "clue_leakage_families.txt"
    ).read_text(encoding="utf-8")
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" in line:
            line = line.split(":", 1)[1]
        for part in line.split("|"):
            words = re.findall(r"[A-Z0-9]+", part.upper())
            max_words = max(max_words, len(words))
    return max_words


def _significant_form_overlap(
    answer_forms: set[str],
    token_forms: set[str],
) -> bool:
    """Return true when answer and clue token share a meaningful root form."""
    return any(
        len(form) >= 4
        for form in answer_forms & token_forms
    )


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
    if token.endswith("LY") and len(token) > 4:
        forms.add(token[:-2])
    if token.endswith("ES") and len(token) > 4:
        forms.add(token[:-2])
    if token.endswith("S") and not token.endswith("SS") and len(token) > 4:
        forms.add(token[:-1])

    return forms
