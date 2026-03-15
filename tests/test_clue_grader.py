"""Tests for the ClueGrader (LLM-based clue evaluation)."""

from __future__ import annotations

import json

from crossword_generator.graders.clue_grader import ClueGrader
from crossword_generator.llm.base import LLMProvider
from crossword_generator.models import (
    ClueEntry,
    FillResult,
    PuzzleEnvelope,
    PuzzleType,
)

# Simple 3x3 grid for testing (no black squares)
MOCK_GRID = [
    ["C", "A", "T"],
    ["A", "R", "E"],
    ["B", "E", "D"],
]

# Clues matching the grid entries
MOCK_CLUES = [
    ClueEntry(number=1, direction="across", answer="CAT", clue="Feline pet"),
    ClueEntry(number=2, direction="across", answer="ARE", clue="Exist, for two"),
    ClueEntry(number=3, direction="across", answer="BED", clue="Place to sleep"),
    ClueEntry(number=1, direction="down", answer="CAB", clue="Yellow taxi"),
    ClueEntry(number=2, direction="down", answer="ARE", clue="Second person verb"),
    ClueEntry(number=3, direction="down", answer="TED", clue="Common male name"),
]


def _make_envelope(
    clues: list[ClueEntry] | None = None,
) -> PuzzleEnvelope:
    return PuzzleEnvelope(
        puzzle_type=PuzzleType.MINI,
        grid_size=3,
        fill=FillResult(grid=MOCK_GRID, filler_used="mock"),
        clues=clues or [],
    )


def _build_evaluation_json(
    clues: list[ClueEntry],
    *,
    accuracy: float = 20.0,
    freshness: float = 20.0,
    craft: float = 20.0,
    fairness: float = 20.0,
) -> str:
    """Build a valid evaluation JSON response with sub-scores."""
    return json.dumps(
        [
            {
                "number": c.number,
                "direction": c.direction,
                "accuracy": accuracy,
                "freshness": freshness,
                "craft": craft,
                "fairness": fairness,
                "feedback": f"Solid clue for {c.answer}.",
            }
            for c in clues
        ]
    )


class MockLLM(LLMProvider):
    def __init__(
        self,
        response: str | None = None,
        *,
        responses: list[str] | None = None,
    ) -> None:
        self._response = response
        self._responses = list(responses) if responses else []
        self._call_count = 0

    @property
    def name(self) -> str:
        return "mock-llm"

    def generate(self, prompt: str, **kwargs: object) -> str:
        self._call_count += 1
        if self._responses:
            return self._responses.pop(0)
        return self._response or ""

    def is_available(self) -> bool:
        return True

    @property
    def call_count(self) -> int:
        return self._call_count


class TestHappyPath:
    def test_grades_clues_correctly(self) -> None:
        response = _build_evaluation_json(
            MOCK_CLUES, accuracy=22, freshness=21, craft=22, fairness=20
        )
        grader = ClueGrader(MockLLM(response=response), min_passing_score=70)
        envelope = _make_envelope(clues=MOCK_CLUES)

        report = grader.grade(envelope)

        assert report.passing is True
        assert report.overall_score == 85.0
        assert report.clue_count == len(MOCK_CLUES)
        assert len(report.clue_grades) == len(MOCK_CLUES)

    def test_per_clue_scores_populated(self) -> None:
        response = _build_evaluation_json(MOCK_CLUES)  # 20+20+20+20=80
        grader = ClueGrader(MockLLM(response=response))
        envelope = _make_envelope(clues=MOCK_CLUES)

        report = grader.grade(envelope)

        for grade in report.clue_grades:
            assert grade.score == 80.0
            assert grade.feedback != ""

    def test_sub_scores_populated(self) -> None:
        response = _build_evaluation_json(
            MOCK_CLUES, accuracy=22, freshness=18, craft=20, fairness=15
        )
        grader = ClueGrader(MockLLM(response=response))
        envelope = _make_envelope(clues=MOCK_CLUES)

        report = grader.grade(envelope)

        for grade in report.clue_grades:
            assert grade.accuracy == 22.0
            assert grade.freshness == 18.0
            assert grade.craft == 20.0
            assert grade.fairness == 15.0
            assert grade.score == 75.0

    def test_feedback_text_preserved(self) -> None:
        response = json.dumps(
            [
                {
                    "number": c.number,
                    "direction": c.direction,
                    "accuracy": 20,
                    "freshness": 20,
                    "craft": 20,
                    "fairness": 20,
                    "feedback": f"Good clue for {c.answer}",
                }
                for c in MOCK_CLUES
            ]
        )
        grader = ClueGrader(MockLLM(response=response))
        envelope = _make_envelope(clues=MOCK_CLUES)

        report = grader.grade(envelope)

        assert any("Good clue" in g.feedback for g in report.clue_grades)


class TestFailingScores:
    def test_low_scores_fail(self) -> None:
        response = _build_evaluation_json(
            MOCK_CLUES, accuracy=10, freshness=10, craft=10, fairness=10
        )
        grader = ClueGrader(MockLLM(response=response), min_passing_score=70)
        envelope = _make_envelope(clues=MOCK_CLUES)

        report = grader.grade(envelope)

        assert report.passing is False
        assert report.overall_score == 40.0

    def test_borderline_score(self) -> None:
        # 17.5 * 4 = 70
        response = _build_evaluation_json(
            MOCK_CLUES, accuracy=17.5, freshness=17.5, craft=17.5, fairness=17.5
        )
        grader = ClueGrader(MockLLM(response=response), min_passing_score=70)
        envelope = _make_envelope(clues=MOCK_CLUES)

        report = grader.grade(envelope)

        assert report.passing is True


class TestParseFailureRetry:
    def test_retries_on_bad_json(self) -> None:
        good_response = _build_evaluation_json(MOCK_CLUES)
        llm = MockLLM(responses=["not valid json", good_response])
        grader = ClueGrader(llm, max_parse_retries=3)
        envelope = _make_envelope(clues=MOCK_CLUES)

        report = grader.grade(envelope)

        assert report.passing is True
        assert llm.call_count == 2

    def test_all_retries_exhausted_returns_failing_report(self) -> None:
        llm = MockLLM(response="garbage")
        grader = ClueGrader(llm, max_parse_retries=2)
        envelope = _make_envelope(clues=MOCK_CLUES)

        report = grader.grade(envelope)

        assert report.passing is False
        assert "parse failed" in report.summary.lower()
        assert llm.call_count == 2


class TestAggregateScoring:
    def test_mixed_scores_averaged(self) -> None:
        """Scores are averaged across all clues."""
        items = []
        for i, c in enumerate(MOCK_CLUES):
            # Alternate between total 60 and total 80
            if i % 2 == 0:
                acc, fre, cra, fai = 15, 15, 15, 15  # 60
            else:
                acc, fre, cra, fai = 20, 20, 20, 20  # 80
            items.append(
                {
                    "number": c.number,
                    "direction": c.direction,
                    "accuracy": acc,
                    "freshness": fre,
                    "craft": cra,
                    "fairness": fai,
                    "feedback": "ok",
                }
            )
        response = json.dumps(items)
        grader = ClueGrader(MockLLM(response=response), min_passing_score=70)
        envelope = _make_envelope(clues=MOCK_CLUES)

        report = grader.grade(envelope)

        assert report.overall_score == 70.0
        assert report.passing is True

    def test_sub_scores_clamped_to_0_25(self) -> None:
        items = [
            {
                "number": c.number,
                "direction": c.direction,
                "accuracy": 30,  # Should be clamped to 25
                "freshness": -5,  # Should be clamped to 0
                "craft": 25,
                "fairness": 25,
                "feedback": "extreme",
            }
            for c in MOCK_CLUES
        ]
        response = json.dumps(items)
        grader = ClueGrader(MockLLM(response=response))
        envelope = _make_envelope(clues=MOCK_CLUES)

        report = grader.grade(envelope)

        for grade in report.clue_grades:
            assert grade.accuracy == 25.0
            assert grade.freshness == 0.0
            assert grade.craft == 25.0
            assert grade.fairness == 25.0
            assert grade.score == 75.0  # 25+0+25+25


class TestEmptyClues:
    def test_no_clues_returns_failing(self) -> None:
        grader = ClueGrader(MockLLM())
        envelope = _make_envelope(clues=[])

        report = grader.grade(envelope)

        assert report.passing is False
        assert report.clue_count == 0


class TestMarkdownWrappedResponse:
    def test_handles_markdown_fences(self) -> None:
        raw_json = _build_evaluation_json(MOCK_CLUES)
        wrapped = f"```json\n{raw_json}\n```"
        grader = ClueGrader(MockLLM(response=wrapped))
        envelope = _make_envelope(clues=MOCK_CLUES)

        report = grader.grade(envelope)

        assert report.passing is True
        assert report.clue_count == len(MOCK_CLUES)
