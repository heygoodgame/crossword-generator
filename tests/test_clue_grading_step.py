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
# 1A: CAT, 1D: CAB, 2D: ARE, 3D: TED, 4A: ARE, 5A: BED
EXPECTED_ENTRIES = [
    {"number": 1, "direction": "across", "answer": "CAT"},
    {"number": 1, "direction": "down", "answer": "CAB"},
    {"number": 2, "direction": "down", "answer": "ARE"},
    {"number": 3, "direction": "down", "answer": "TED"},
    {"number": 4, "direction": "across", "answer": "ARE"},
    {"number": 5, "direction": "across", "answer": "BED"},
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


def _build_eval_json(
    *,
    accuracy: float = 20.0,
    freshness: float = 20.0,
    craft: float = 20.0,
    fairness: float = 20.0,
) -> str:
    """Build a valid clue evaluation JSON response with sub-scores."""
    return json.dumps(
        [
            {
                "number": e["number"],
                "direction": e["direction"],
                "accuracy": accuracy,
                "freshness": freshness,
                "craft": craft,
                "fairness": fairness,
                "feedback": "Good",
            }
            for e in EXPECTED_ENTRIES
        ]
    )


def _build_eval_json_mixed(
    low_accuracy_entries: set[tuple[int, str]] | None = None,
    low_accuracy: float = 5.0,
    normal_accuracy: float = 22.0,
) -> str:
    """Build eval JSON where specific entries have low accuracy."""
    items = []
    for e in EXPECTED_ENTRIES:
        key = (e["number"], e["direction"])
        acc = (
            low_accuracy
            if low_accuracy_entries and key in low_accuracy_entries
            else normal_accuracy
        )
        items.append(
            {
                "number": e["number"],
                "direction": e["direction"],
                "accuracy": acc,
                "freshness": 20,
                "craft": 20,
                "fairness": 20,
                "feedback": (
                    "Factually wrong"
                    if low_accuracy_entries and key in low_accuracy_entries
                    else "Good"
                ),
            }
        )
    return json.dumps(items)


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
        """Generate clues -> grade -> pass on first attempt."""
        clue_json = _build_clue_json()
        eval_json = _build_eval_json(
            accuracy=22, freshness=21, craft=22, fairness=20
        )
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
        eval_json = _build_eval_json()
        llm = SequentialMockLLM([clue_json, eval_json])

        grader = ClueGrader(llm, min_passing_score=70)
        step = ClueWithGradingStep(llm, grader)
        result = step.run(_make_envelope())

        assert "clue-generation-with-grading" in result.step_history
        assert "clue-generation" not in result.step_history

    def test_quality_scores_populated_on_clues(self) -> None:
        clue_json = _build_clue_json()
        eval_json = _build_eval_json(
            accuracy=23, freshness=23, craft=22, fairness=22
        )
        llm = SequentialMockLLM([clue_json, eval_json])

        grader = ClueGrader(llm, min_passing_score=70)
        step = ClueWithGradingStep(llm, grader)
        result = step.run(_make_envelope())

        for clue in result.clues:
            assert clue.quality_score is not None
            assert clue.quality_score == 90.0

    def test_sub_scores_in_grade_report(self) -> None:
        clue_json = _build_clue_json()
        eval_json = _build_eval_json(
            accuracy=22, freshness=18, craft=20, fairness=15
        )
        llm = SequentialMockLLM([clue_json, eval_json])

        grader = ClueGrader(llm, min_passing_score=70)
        step = ClueWithGradingStep(llm, grader)
        result = step.run(_make_envelope())

        assert result.clue_grade_report is not None
        for grade in result.clue_grade_report.clue_grades:
            assert grade.accuracy == 22.0
            assert grade.freshness == 18.0
            assert grade.craft == 20.0
            assert grade.fairness == 15.0


class TestRegeneration:
    def test_regenerates_on_low_score(self) -> None:
        """First attempt scores low -> regenerate -> second passes."""
        clue_json_1 = _build_clue_json()
        eval_json_1 = _build_eval_json(
            accuracy=10, freshness=10, craft=10, fairness=10
        )  # 40 → fail
        clue_json_2 = _build_clue_json()
        eval_json_2 = _build_eval_json(
            accuracy=22, freshness=21, craft=22, fairness=20
        )  # 85 → pass

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
        """All attempts score low -> keeps best, records error."""
        clue_json = _build_clue_json()
        eval_low = _build_eval_json(
            accuracy=10, freshness=10, craft=10, fairness=10
        )  # 40
        eval_medium = _build_eval_json(
            accuracy=14, freshness=14, craft=14, fairness=13
        )  # 55

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
        eval_json = _build_eval_json(
            accuracy=10, freshness=10, craft=10, fairness=10
        )  # 40
        llm = SequentialMockLLM([clue_json, eval_json])

        grader = ClueGrader(llm, min_passing_score=70)
        step = ClueWithGradingStep(
            llm, grader, max_retries=3, regenerate_on_fail=False,
            accuracy_repair_threshold=0,
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


class TestAccuracyRepair:
    """Tests for the accuracy-repair pass."""

    def test_repair_triggered_on_low_accuracy(self) -> None:
        """Clues with accuracy below threshold get repaired."""
        clue_json = _build_clue_json()
        # First eval: 1-across has low accuracy (5/25)
        eval_json = _build_eval_json_mixed(
            low_accuracy_entries={(1, "across")},
            low_accuracy=5.0,
        )
        # Repair response: replacement clue for 1-across
        repair_json = json.dumps([
            {"number": 1, "direction": "across", "clue": "Repaired clue for CAT"},
        ])
        # Re-grade after repair: all good now
        re_eval_json = _build_eval_json(
            accuracy=22, freshness=20, craft=20, fairness=20
        )

        llm = SequentialMockLLM([
            clue_json,    # clue generation
            eval_json,    # first grade (1A has low accuracy)
            repair_json,  # repair LLM call
            re_eval_json, # re-grade after repair
        ])
        grader = ClueGrader(llm, min_passing_score=70)
        step = ClueWithGradingStep(
            llm, grader, max_retries=1, accuracy_repair_threshold=12,
        )
        result = step.run(_make_envelope())

        # The repaired clue should be in the result
        clue_1a = next(
            c for c in result.clues
            if c.number == 1 and c.direction == "across"
        )
        assert clue_1a.clue == "Repaired clue for CAT"
        # 4 LLM calls: generate + grade + repair + re-grade
        assert llm.call_count == 4

    def test_no_repair_when_all_accuracy_ok(self) -> None:
        """No repair pass if all accuracy scores are above threshold."""
        clue_json = _build_clue_json()
        eval_json = _build_eval_json(
            accuracy=22, freshness=20, craft=20, fairness=20
        )

        llm = SequentialMockLLM([clue_json, eval_json])
        grader = ClueGrader(llm, min_passing_score=70)
        step = ClueWithGradingStep(
            llm, grader, max_retries=1, accuracy_repair_threshold=12
        )
        result = step.run(_make_envelope())

        assert result.clue_grade_report is not None
        assert result.clue_grade_report.passing is True
        # Only 2 LLM calls (generate + grade, no repair)
        assert llm.call_count == 2

    def test_repair_parse_failure_keeps_original(self) -> None:
        """If repair LLM response is unparseable, keep original clues."""
        clue_json = _build_clue_json()
        eval_json = _build_eval_json_mixed(
            low_accuracy_entries={(1, "across")},
            low_accuracy=5.0,
        )

        llm = SequentialMockLLM([
            clue_json,            # clue generation
            eval_json,            # grade (1A has low accuracy)
            "not valid json!!!",  # repair fails to parse
        ])
        grader = ClueGrader(llm, min_passing_score=70)
        step = ClueWithGradingStep(
            llm, grader, max_retries=1, accuracy_repair_threshold=12,
        )
        result = step.run(_make_envelope())

        # Original clue preserved (repair failed gracefully)
        clue_1a = next(
            c for c in result.clues
            if c.number == 1 and c.direction == "across"
        )
        assert clue_1a.clue == "Clue for CAT"
        # 3 LLM calls: generate + grade + failed repair
        assert llm.call_count == 3

    def test_repair_multiple_clues(self) -> None:
        """Multiple clues with low accuracy get repaired together."""
        clue_json = _build_clue_json()
        eval_json = _build_eval_json_mixed(
            low_accuracy_entries={(1, "across"), (3, "down")},
            low_accuracy=5.0,
        )
        repair_json = json.dumps([
            {"number": 1, "direction": "across", "clue": "Fixed CAT clue"},
            {"number": 3, "direction": "down", "clue": "Fixed TED clue"},
        ])
        re_eval_json = _build_eval_json(
            accuracy=22, freshness=20, craft=20, fairness=20
        )

        llm = SequentialMockLLM([
            clue_json, eval_json, repair_json, re_eval_json,
        ])
        grader = ClueGrader(llm, min_passing_score=70)
        step = ClueWithGradingStep(
            llm, grader, max_retries=1, accuracy_repair_threshold=12,
        )
        result = step.run(_make_envelope())

        clue_1a = next(
            c for c in result.clues
            if c.number == 1 and c.direction == "across"
        )
        clue_3d = next(
            c for c in result.clues
            if c.number == 3 and c.direction == "down"
        )
        assert clue_1a.clue == "Fixed CAT clue"
        assert clue_3d.clue == "Fixed TED clue"
