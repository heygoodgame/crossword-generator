"""Tests for the ClueWithGradingStep composite pipeline step."""

from __future__ import annotations

import json

import pytest

from crossword_generator.graders.clue_grader import ClueGrader
from crossword_generator.llm.base import LLMProvider
from crossword_generator.models import (
    ClueEntry,
    FillResult,
    PuzzleEnvelope,
    PuzzleType,
)
from crossword_generator.steps.clue_grading_step import ClueWithGradingStep

# Simple 3x3 grid for testing
MOCK_GRID = [
    ["C", "A", "T"],
    ["A", "R", "E"],
    ["B", "E", "D"],
]

# Expected entries from compute_numbering for this grid:
# 1A: CAT, 2A: ARE, 3A: BED, 1D: CAB, 2D: ARE, 3D: TED
EXPECTED_ENTRIES = [
    {"number": 1, "direction": "across", "answer": "CAT"},
    {"number": 1, "direction": "down", "answer": "CAB"},
    {"number": 2, "direction": "across", "answer": "ARE"},
    {"number": 2, "direction": "down", "answer": "ARE"},
    {"number": 3, "direction": "across", "answer": "BED"},
    {"number": 3, "direction": "down", "answer": "TED"},
]


def _build_clue_json() -> str:
    """Build a valid clue generation JSON response."""
    return json.dumps(
        [
            {
                "number": e["number"],
                "direction": e["direction"],
                "clue": f"Clue for {e['answer']}",
            }
            for e in EXPECTED_ENTRIES
        ]
    )


def _build_eval_json(score: float = 80.0) -> str:
    """Build a valid clue evaluation JSON response."""
    return json.dumps(
        [
            {
                "number": e["number"],
                "direction": e["direction"],
                "score": score,
                "feedback": "Good",
            }
            for e in EXPECTED_ENTRIES
        ]
    )


class SequentialMockLLM(LLMProvider):
    """Mock LLM that returns responses in sequence."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    @property
    def name(self) -> str:
        return "sequential-mock-llm"

    def generate(self, prompt: str, **kwargs: object) -> str:
        self._call_count += 1
        if self._responses:
            return self._responses.pop(0)
        return ""

    def is_available(self) -> bool:
        return True

    @property
    def call_count(self) -> int:
        return self._call_count


def _make_envelope(
    *,
    grid: list[list[str]] | None = MOCK_GRID,
    clues: list[ClueEntry] | None = None,
) -> PuzzleEnvelope:
    fill = None
    if grid is not None:
        fill = FillResult(grid=grid, filler_used="mock")
    return PuzzleEnvelope(
        puzzle_type=PuzzleType.MINI,
        grid_size=3,
        fill=fill,
        clues=clues or [],
    )


class TestHappyPath:
    def test_generate_and_grade_passes(self) -> None:
        """Generate clues → grade → pass on first attempt."""
        clue_json = _build_clue_json()
        eval_json = _build_eval_json(score=85.0)
        llm = SequentialMockLLM([clue_json, eval_json])

        grader = ClueGrader(llm, min_passing_score=70)
        step = ClueWithGradingStep(llm, grader, max_retries=3)
        envelope = _make_envelope()
        result = step.run(envelope)

        assert len(result.clues) == len(EXPECTED_ENTRIES)
        assert result.clue_grade_report is not None
        assert result.clue_grade_report.passing is True
        assert result.clue_grade_report.overall_score == 85.0

    def test_step_name(self) -> None:
        llm = SequentialMockLLM([])
        grader = ClueGrader(llm)
        step = ClueWithGradingStep(llm, grader)
        assert step.name == "clue-generation-with-grading"

    def test_step_history_updated(self) -> None:
        clue_json = _build_clue_json()
        eval_json = _build_eval_json(score=85.0)
        llm = SequentialMockLLM([clue_json, eval_json])

        grader = ClueGrader(llm, min_passing_score=70)
        step = ClueWithGradingStep(llm, grader)
        result = step.run(_make_envelope())

        assert "clue-generation-with-grading" in result.step_history
        assert "clue-generation" not in result.step_history

    def test_quality_scores_populated_on_clues(self) -> None:
        clue_json = _build_clue_json()
        eval_json = _build_eval_json(score=90.0)
        llm = SequentialMockLLM([clue_json, eval_json])

        grader = ClueGrader(llm, min_passing_score=70)
        step = ClueWithGradingStep(llm, grader)
        result = step.run(_make_envelope())

        for clue in result.clues:
            assert clue.quality_score is not None
            assert clue.quality_score == 90.0


class TestRegeneration:
    def test_regenerates_on_low_score(self) -> None:
        """First attempt scores low → regenerate → second passes."""
        clue_json_1 = _build_clue_json()
        eval_json_1 = _build_eval_json(score=40.0)  # Fail
        clue_json_2 = _build_clue_json()
        eval_json_2 = _build_eval_json(score=85.0)  # Pass

        llm = SequentialMockLLM([
            clue_json_1, eval_json_1,
            clue_json_2, eval_json_2,
        ])
        grader = ClueGrader(llm, min_passing_score=70)
        step = ClueWithGradingStep(llm, grader, max_retries=3)
        result = step.run(_make_envelope())

        assert result.clue_grade_report is not None
        assert result.clue_grade_report.passing is True
        assert result.clue_grade_report.overall_score == 85.0
        assert result.errors == []


class TestAllRetriesExhausted:
    def test_keeps_best_result_on_all_failures(self) -> None:
        """All attempts score low → keeps best, records error."""
        clue_json = _build_clue_json()
        eval_low = _build_eval_json(score=40.0)
        eval_medium = _build_eval_json(score=55.0)

        llm = SequentialMockLLM([
            clue_json, eval_low,
            clue_json, eval_medium,
        ])
        grader = ClueGrader(llm, min_passing_score=70)
        step = ClueWithGradingStep(llm, grader, max_retries=2)
        result = step.run(_make_envelope())

        assert result.clue_grade_report is not None
        assert result.clue_grade_report.overall_score == 55.0
        assert result.clue_grade_report.passing is False
        assert any("below threshold" in e for e in result.errors)


class TestRegenerationDisabled:
    def test_single_attempt_when_disabled(self) -> None:
        clue_json = _build_clue_json()
        eval_json = _build_eval_json(score=40.0)
        llm = SequentialMockLLM([clue_json, eval_json])

        grader = ClueGrader(llm, min_passing_score=70)
        step = ClueWithGradingStep(
            llm, grader, max_retries=3, regenerate_on_fail=False
        )
        result = step.run(_make_envelope())

        assert result.clue_grade_report is not None
        assert result.clue_grade_report.passing is False
        # Only 2 LLM calls (1 generate + 1 grade)
        assert llm.call_count == 2


class TestValidation:
    def test_rejects_no_fill(self) -> None:
        llm = SequentialMockLLM([])
        grader = ClueGrader(llm)
        step = ClueWithGradingStep(llm, grader)
        envelope = _make_envelope(grid=None)

        with pytest.raises(ValueError, match="no fill result"):
            step.run(envelope)

    def test_rejects_existing_clues(self) -> None:
        llm = SequentialMockLLM([])
        grader = ClueGrader(llm)
        step = ClueWithGradingStep(llm, grader)
        envelope = _make_envelope(
            clues=[ClueEntry(number=1, direction="across", answer="CAT", clue="test")]
        )

        with pytest.raises(ValueError, match="already has clues"):
            step.run(envelope)
