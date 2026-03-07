"""Tests for the fill evaluation framework."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from crossword_generator.evaluation import EvalResult, FillerEvaluator
from crossword_generator.fillers.base import FilledGrid, FillError, GridFiller, GridSpec


class FakeSuccessFiller(GridFiller):
    """Filler that always succeeds with a trivial grid."""

    @property
    def name(self) -> str:
        return "fake-ok"

    def fill(self, spec: GridSpec, *, seed: int | None = None) -> FilledGrid:
        grid = [["A"] * spec.cols for _ in range(spec.rows)]
        return FilledGrid(grid=grid)


class FakeFailFiller(GridFiller):
    """Filler that always raises FillError."""

    @property
    def name(self) -> str:
        return "fake-fail"

    def fill(self, spec: GridSpec, *, seed: int | None = None) -> FilledGrid:
        raise FillError("always fails")


class FakeUnavailableFiller(GridFiller):
    """Filler that reports itself as unavailable."""

    @property
    def name(self) -> str:
        return "unavailable"

    def fill(self, spec: GridSpec, *, seed: int | None = None) -> FilledGrid:
        raise FillError("should not be called")

    def is_available(self) -> bool:
        return False


@pytest.fixture
def mock_grader() -> MagicMock:
    grader = MagicMock()
    report = MagicMock()
    report.overall_score = 75.0
    report.word_count = 10
    report.passing = True
    grader.grade.return_value = report
    return grader


class TestFillerEvaluator:
    def test_evaluate_with_success_filler(
        self, mock_grader: MagicMock
    ) -> None:
        evaluator = FillerEvaluator([FakeSuccessFiller()], mock_grader)
        results = evaluator.evaluate([5], [42])
        assert len(results) == 1
        r = results[0]
        assert r.success is True
        assert r.filler_name == "fake-ok"
        assert r.grid_size == 5
        assert r.seed == 42
        assert r.quality_score == 75.0
        assert r.word_count == 10
        assert r.passing is True
        assert r.error is None

    def test_evaluate_handles_filler_error(
        self, mock_grader: MagicMock
    ) -> None:
        evaluator = FillerEvaluator([FakeFailFiller()], mock_grader)
        results = evaluator.evaluate([5], [42])
        assert len(results) == 1
        r = results[0]
        assert r.success is False
        assert r.quality_score is None
        assert r.error == "always fails"
        assert r.word_count == 0
        assert r.passing is False

    def test_evaluate_records_timing(self, mock_grader: MagicMock) -> None:
        evaluator = FillerEvaluator([FakeSuccessFiller()], mock_grader)
        results = evaluator.evaluate([5], [1])
        assert results[0].time_seconds >= 0

    def test_evaluate_skips_unavailable(
        self, mock_grader: MagicMock
    ) -> None:
        evaluator = FillerEvaluator(
            [FakeUnavailableFiller(), FakeSuccessFiller()], mock_grader
        )
        results = evaluator.evaluate([5], [1])
        assert len(results) == 1
        assert results[0].filler_name == "fake-ok"

    def test_evaluate_multiple_sizes_and_seeds(
        self, mock_grader: MagicMock
    ) -> None:
        evaluator = FillerEvaluator([FakeSuccessFiller()], mock_grader)
        results = evaluator.evaluate([5, 7], [1, 2, 3])
        assert len(results) == 6  # 1 filler × 2 sizes × 3 seeds

    def test_evaluate_multiple_fillers(
        self, mock_grader: MagicMock
    ) -> None:
        evaluator = FillerEvaluator(
            [FakeSuccessFiller(), FakeFailFiller()], mock_grader
        )
        results = evaluator.evaluate([5], [1, 2])
        assert len(results) == 4  # 2 fillers × 1 size × 2 seeds
        successes = [r for r in results if r.success]
        failures = [r for r in results if not r.success]
        assert len(successes) == 2
        assert len(failures) == 2


class TestFormatReport:
    def test_empty_results(self) -> None:
        assert FillerEvaluator.format_report([]) == "No evaluation results."

    def test_report_structure(self) -> None:
        results = [
            EvalResult(
                filler_name="test",
                grid_size=5,
                seed=1,
                success=True,
                quality_score=80.0,
                time_seconds=1.5,
                error=None,
                word_count=10,
                passing=True,
            ),
            EvalResult(
                filler_name="test",
                grid_size=5,
                seed=2,
                success=True,
                quality_score=70.0,
                time_seconds=2.0,
                error=None,
                word_count=10,
                passing=True,
            ),
        ]
        report = FillerEvaluator.format_report(results)
        assert "## Fill Evaluation Results" in report
        assert "### 5x5 Grid" in report
        assert "test" in report
        assert "2/2" in report
        assert "75.0" in report  # mean of 80 and 70

    def test_report_with_failures(self) -> None:
        results = [
            EvalResult(
                filler_name="flaky",
                grid_size=5,
                seed=1,
                success=True,
                quality_score=60.0,
                time_seconds=1.0,
                error=None,
                word_count=10,
                passing=False,
            ),
            EvalResult(
                filler_name="flaky",
                grid_size=5,
                seed=2,
                success=False,
                quality_score=None,
                time_seconds=0.5,
                error="timeout",
                word_count=0,
                passing=False,
            ),
        ]
        report = FillerEvaluator.format_report(results)
        assert "1/2" in report

    def test_report_multiple_sizes(self) -> None:
        results = [
            EvalResult(
                filler_name="a",
                grid_size=5,
                seed=1,
                success=True,
                quality_score=80.0,
                time_seconds=1.0,
                error=None,
                word_count=10,
                passing=True,
            ),
            EvalResult(
                filler_name="a",
                grid_size=7,
                seed=1,
                success=True,
                quality_score=70.0,
                time_seconds=2.0,
                error=None,
                word_count=14,
                passing=True,
            ),
        ]
        report = FillerEvaluator.format_report(results)
        assert "### 5x5 Grid" in report
        assert "### 7x7 Grid" in report
