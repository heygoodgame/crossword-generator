"""Tests for multi-board fill collection and LLM selection."""

from __future__ import annotations

import json

import pytest

from crossword_generator.dictionary import Dictionary
from crossword_generator.fillers.base import FilledGrid, GridFiller, GridSpec
from crossword_generator.graders.fill_grader import FillGrader
from crossword_generator.llm.base import LLMProvider
from crossword_generator.llm.prompts.fill_selection import (
    _extract_words,
    build_fill_selection_prompt,
)
from crossword_generator.models import (
    FillGradeReport,
    FillResult,
    PuzzleEnvelope,
    PuzzleType,
)
from crossword_generator.steps.fill_step import (
    FillWithGradingStep,
    _CandidateCollector,
    _parse_selection_response,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

HIGH_QUALITY_GRID = [
    ["S", "T", "A", "R", "E"],
    ["T", "O", "N", "E", "S"],
    ["A", "R", "E", "N", "A"],
    ["R", "E", "S", "E", "T"],
    ["S", "P", "E", "E", "D"],
]

ALT_GRID = [
    ["C", "R", "A", "N", "E"],
    ["L", "I", "N", "E", "S"],
    ["A", "R", "E", "N", "A"],
    ["R", "E", "S", "E", "T"],
    ["S", "P", "E", "E", "D"],
]

GOOD_WORDS = {
    "STARE": 80, "TONES": 75, "ARENA": 85, "RESET": 70, "SPEED": 90,
    "STARS": 80, "TORED": 60, "ANISE": 65, "RENEE": 60, "EASET": 55,
    "STORE": 70, "PARSE": 75, "ENTER": 80,
    "CRANE": 85, "LINES": 75, "CLASS": 70, "RIRES": 50, "ANNEE": 50,
    "EASED": 60,
}


def _make_dict(words: dict[str, int]) -> Dictionary:
    return Dictionary(words, min_word_score=0, min_2letter_score=0)


class CyclingMockFiller(GridFiller):
    """Returns grids from a list, cycling through them."""

    def __init__(self, grids: list[list[list[str]]]) -> None:
        self._grids = grids
        self._call_count = 0

    @property
    def name(self) -> str:
        return "cycling-mock"

    def fill(self, spec: GridSpec) -> FilledGrid:
        idx = self._call_count % len(self._grids)
        self._call_count += 1
        return FilledGrid(grid=self._grids[idx])


class MockLLMProvider(LLMProvider):
    """Mock LLM provider that returns a fixed response."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.call_count = 0

    @property
    def name(self) -> str:
        return "mock-llm"

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.call_count += 1
        return self._response

    def is_available(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# _CandidateCollector tests
# ---------------------------------------------------------------------------


class TestCandidateCollector:
    def test_add_passing_result(self) -> None:
        collector = _CandidateCollector(target=3)
        result = FillResult(
            grid=HIGH_QUALITY_GRID,
            quality_score=75.0,
            grade_report=FillGradeReport(
                overall_score=75.0, word_count=10, passing=True
            ),
        )
        collector.add(result, ["A", "B"])
        assert len(collector.passing_results) == 1
        assert collector.best_result is result

    def test_rejects_duplicate_grids(self) -> None:
        collector = _CandidateCollector(target=3)
        result1 = FillResult(
            grid=HIGH_QUALITY_GRID,
            quality_score=75.0,
            grade_report=FillGradeReport(
                overall_score=75.0, word_count=10, passing=True
            ),
        )
        result2 = FillResult(
            grid=HIGH_QUALITY_GRID,
            quality_score=80.0,
            grade_report=FillGradeReport(
                overall_score=80.0, word_count=10, passing=True
            ),
        )
        collector.add(result1, ["A"])
        collector.add(result2, ["B"])
        assert len(collector.passing_results) == 1

    def test_accepts_different_grids(self) -> None:
        collector = _CandidateCollector(target=3)
        result1 = FillResult(
            grid=HIGH_QUALITY_GRID,
            quality_score=75.0,
            grade_report=FillGradeReport(
                overall_score=75.0, word_count=10, passing=True
            ),
        )
        result2 = FillResult(
            grid=ALT_GRID,
            quality_score=80.0,
            grade_report=FillGradeReport(
                overall_score=80.0, word_count=10, passing=True
            ),
        )
        collector.add(result1, ["A"])
        collector.add(result2, ["B"])
        assert len(collector.passing_results) == 2

    def test_is_full(self) -> None:
        collector = _CandidateCollector(target=2)
        assert not collector.is_full()
        collector.add(
            FillResult(
                grid=HIGH_QUALITY_GRID,
                quality_score=75.0,
                grade_report=FillGradeReport(
                    overall_score=75.0, word_count=10, passing=True
                ),
            ),
            [],
        )
        assert not collector.is_full()
        collector.add(
            FillResult(
                grid=ALT_GRID,
                quality_score=80.0,
                grade_report=FillGradeReport(
                    overall_score=80.0, word_count=10, passing=True
                ),
            ),
            [],
        )
        assert collector.is_full()

    def test_tracks_best_non_passing(self) -> None:
        collector = _CandidateCollector(target=3)
        result = FillResult(
            grid=HIGH_QUALITY_GRID,
            quality_score=40.0,
            grade_report=FillGradeReport(
                overall_score=40.0, word_count=10, passing=False
            ),
        )
        collector.add(result, ["X"])
        assert collector.best_result is result
        assert len(collector.passing_results) == 0


# ---------------------------------------------------------------------------
# _parse_selection_response tests
# ---------------------------------------------------------------------------


class TestParseSelectionResponse:
    def test_valid_response(self) -> None:
        raw = json.dumps({"selected_board": 2, "rationale": "Board 2 is best"})
        idx, rationale = _parse_selection_response(raw, 3)
        assert idx == 1  # 0-based
        assert rationale == "Board 2 is best"

    def test_response_with_surrounding_text(self) -> None:
        raw = 'Here is my answer: {"selected_board": 1, "rationale": "Good"} done'
        idx, rationale = _parse_selection_response(raw, 2)
        assert idx == 0

    def test_out_of_range_raises(self) -> None:
        raw = json.dumps({"selected_board": 5, "rationale": "x"})
        with pytest.raises(ValueError, match="out of range"):
            _parse_selection_response(raw, 3)

    def test_zero_index_raises(self) -> None:
        raw = json.dumps({"selected_board": 0, "rationale": "x"})
        with pytest.raises(ValueError, match="out of range"):
            _parse_selection_response(raw, 3)

    def test_no_json_raises(self) -> None:
        with pytest.raises(ValueError, match="No JSON"):
            _parse_selection_response("just some text", 3)

    def test_missing_rationale_defaults_empty(self) -> None:
        raw = json.dumps({"selected_board": 1})
        idx, rationale = _parse_selection_response(raw, 2)
        assert idx == 0
        assert rationale == ""


# ---------------------------------------------------------------------------
# build_fill_selection_prompt tests
# ---------------------------------------------------------------------------


class TestBuildFillSelectionPrompt:
    def test_includes_all_boards(self) -> None:
        grids = [HIGH_QUALITY_GRID, ALT_GRID]
        prompt = build_fill_selection_prompt(grids)
        assert "BOARD 1" in prompt
        assert "BOARD 2" in prompt

    def test_no_numeric_scores(self) -> None:
        grids = [HIGH_QUALITY_GRID, ALT_GRID]
        prompt = build_fill_selection_prompt(grids)
        assert "numeric score" not in prompt
        assert "score:" not in prompt.lower().split("scoring")[0]

    def test_includes_criteria(self) -> None:
        grids = [HIGH_QUALITY_GRID]
        prompt = build_fill_selection_prompt(grids)
        assert "VOCABULARY QUALITY" in prompt
        assert "LIVELINESS" in prompt
        assert "CLUE POTENTIAL" in prompt
        assert "AVOIDING JUNK" in prompt

    def test_extract_words(self) -> None:
        words = _extract_words(HIGH_QUALITY_GRID)
        # Should find across and down words
        assert "STARE" in words
        assert "TONES" in words
        assert "ARENA" in words
        assert "SPEED" in words


# ---------------------------------------------------------------------------
# Integration: collect_boards=1 preserves old behavior
# ---------------------------------------------------------------------------


class TestCollectBoardsDefault:
    def test_collect_1_stops_at_first_pass(self) -> None:
        """With collect_boards=1 (default), stops at first passing fill."""
        dictionary = _make_dict(GOOD_WORDS)
        grader = FillGrader(dictionary, min_passing_score=30)
        filler = CyclingMockFiller([HIGH_QUALITY_GRID, ALT_GRID])
        step = FillWithGradingStep(
            filler, grader, max_retries=5, collect_boards=1
        )

        envelope = PuzzleEnvelope(puzzle_type=PuzzleType.MINI, grid_size=5)
        result = step.run(envelope)

        assert result.fill is not None
        assert result.fill.grade_report is not None
        assert result.fill.grade_report.passing is True
        # Should have stopped after 1 attempt since first grid passes
        assert filler._call_count == 1


# ---------------------------------------------------------------------------
# Integration: collect_boards > 1 collects multiple
# ---------------------------------------------------------------------------


class TestCollectMultipleBoards:
    def test_collect_2_gathers_two_passing(self) -> None:
        """With collect_boards=2, collects 2 unique passing boards."""
        dictionary = _make_dict(GOOD_WORDS)
        grader = FillGrader(dictionary, min_passing_score=30)
        filler = CyclingMockFiller([HIGH_QUALITY_GRID, ALT_GRID])
        step = FillWithGradingStep(
            filler, grader, max_retries=10, collect_boards=2
        )

        envelope = PuzzleEnvelope(puzzle_type=PuzzleType.MINI, grid_size=5)
        result = step.run(envelope)

        assert result.fill is not None
        assert result.fill.selection_metadata is not None
        assert result.fill.selection_metadata.candidates_collected == 2
        assert result.fill.selection_metadata.selection_method == "numeric_best"


# ---------------------------------------------------------------------------
# Integration: LLM selection
# ---------------------------------------------------------------------------


class TestLLMSelection:
    def test_llm_selects_board(self) -> None:
        """With llm_select=True, LLM picks the best board."""
        dictionary = _make_dict(GOOD_WORDS)
        grader = FillGrader(dictionary, min_passing_score=30)
        filler = CyclingMockFiller([HIGH_QUALITY_GRID, ALT_GRID])

        llm_response = json.dumps({
            "selected_board": 2,
            "rationale": "Board 2 has better vocab",
        })
        llm = MockLLMProvider(llm_response)

        step = FillWithGradingStep(
            filler, grader, max_retries=10,
            collect_boards=2, llm_select=True, llm_provider=llm,
        )

        envelope = PuzzleEnvelope(puzzle_type=PuzzleType.MINI, grid_size=5)
        result = step.run(envelope)

        assert result.fill is not None
        assert result.fill.selection_metadata is not None
        assert result.fill.selection_metadata.selection_method == "llm"
        assert "Board 2" in result.fill.selection_metadata.llm_rationale
        assert llm.call_count == 1

    def test_llm_parse_failure_falls_back(self) -> None:
        """When LLM returns garbage, falls back to numeric best."""
        dictionary = _make_dict(GOOD_WORDS)
        grader = FillGrader(dictionary, min_passing_score=30)
        filler = CyclingMockFiller([HIGH_QUALITY_GRID, ALT_GRID])

        llm = MockLLMProvider("I don't know how to respond")

        step = FillWithGradingStep(
            filler, grader, max_retries=10,
            collect_boards=2, llm_select=True, llm_provider=llm,
        )

        envelope = PuzzleEnvelope(puzzle_type=PuzzleType.MINI, grid_size=5)
        result = step.run(envelope)

        assert result.fill is not None
        assert result.fill.selection_metadata is not None
        assert result.fill.selection_metadata.selection_method == "numeric_best"
        assert llm.call_count == 3  # 3 retries

    def test_single_board_no_llm_call(self) -> None:
        """With only 1 passing board, no LLM call even if llm_select=True."""
        dictionary = _make_dict(GOOD_WORDS)
        grader = FillGrader(dictionary, min_passing_score=30)
        # Only one grid type, so duplicates will be rejected
        filler = CyclingMockFiller([HIGH_QUALITY_GRID])

        llm = MockLLMProvider("should not be called")

        step = FillWithGradingStep(
            filler, grader, max_retries=5,
            collect_boards=3, llm_select=True, llm_provider=llm,
        )

        envelope = PuzzleEnvelope(puzzle_type=PuzzleType.MINI, grid_size=5)
        result = step.run(envelope)

        assert result.fill is not None
        assert result.fill.selection_metadata is not None
        assert result.fill.selection_metadata.selection_method == "single"
        assert llm.call_count == 0
