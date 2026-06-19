"""Tests for the ClueWithGradingStep composite pipeline step."""

from __future__ import annotations

import json

import pytest

from crossword_generator.clue_history import ClueHistoryIndex, DuplicateClueHit
from crossword_generator.graders.clue_fact_checker import ClueFactChecker
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


def _build_eval_json_subset(
    keys: set[tuple[int, str]],
    *,
    accuracy: float = 20.0,
    freshness: float = 20.0,
    craft: float = 20.0,
    fairness: float = 20.0,
) -> str:
    """Build an eval response covering ONLY the given (number, direction) keys.

    Repair now re-grades just the repaired clues (ClueGrader.grade_subset), and
    the grader's parser enforces that the response count matches the clues sent.
    So a post-repair mock must return grades for exactly the repaired keys.
    """
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
            if (e["number"], e["direction"]) in keys
        ]
    )


def _build_clue_json() -> str:
    """Build a valid clue generation JSON response."""
    return json.dumps(
        [
            {
                "number": e["number"],
                "direction": e["direction"],
                "clue": f"Generated clue {e['number']} {e['direction']}",
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
        self.prompts: list[str] = []

    @property
    def name(self) -> str:
        return "sequential-mock-llm"

    def generate(self, prompt: str, **kwargs: object) -> str:
        self._call_count += 1
        self.prompts.append(prompt)
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
        eval_json = _build_eval_json(accuracy=22, freshness=21, craft=22, fairness=20)
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
        eval_json = _build_eval_json(accuracy=23, freshness=23, craft=22, fairness=22)
        llm = SequentialMockLLM([clue_json, eval_json])

        grader = ClueGrader(llm, min_passing_score=70)
        step = ClueWithGradingStep(llm, grader)
        result = step.run(_make_envelope())

        for clue in result.clues:
            assert clue.quality_score is not None
            assert clue.quality_score == 90.0

    def test_sub_scores_in_grade_report(self) -> None:
        clue_json = _build_clue_json()
        eval_json = _build_eval_json(accuracy=22, freshness=18, craft=20, fairness=15)
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

        llm = SequentialMockLLM(
            [
                clue_json_1,
                eval_json_1,
                clue_json_2,
                eval_json_2,
            ]
        )
        grader = ClueGrader(llm, min_passing_score=70)
        step = ClueWithGradingStep(llm, grader, max_retries=3)
        result = step.run(_make_envelope())

        assert result.clue_grade_report is not None
        assert result.clue_grade_report.passing is True
        assert result.clue_grade_report.overall_score == 85.0
        assert result.errors == []


class TestMostlyGoodShortcut:
    def test_skips_regeneration_when_mostly_good(self) -> None:
        """A failing puzzle with most clues clean goes straight to surgical
        repair instead of regenerating the whole puzzle again."""
        clue_json = _build_clue_json()
        # Only 1 of 6 clues has low accuracy -> 5/6 clean = 0.83 >= 0.8 ratio.
        eval_json = _build_eval_json_mixed(
            low_accuracy_entries={(1, "across")},
            low_accuracy=5.0,
            normal_accuracy=22.0,
        )
        # Repair fixes the one bad clue; re-grade still has the low entry only
        # if repair fails, so provide a clean re-grade.
        repair_json = json.dumps(
            [{"number": 1, "direction": "across", "clue": "Fixed clue"}]
        )
        re_eval_json = _build_eval_json_subset(
            {(1, "across")}, accuracy=22, freshness=21, craft=22, fairness=20
        )
        llm = SequentialMockLLM([clue_json, eval_json, repair_json, re_eval_json])
        grader = ClueGrader(llm, min_passing_score=70)
        # max_retries=3 would allow 3 regenerations; mostly-good must stop it
        # after the first attempt (1 generate + 1 grade), then surgical repair.
        step = ClueWithGradingStep(
            llm, grader, max_retries=3, surgical_repair_pass_ratio=0.8
        )
        result = step.run(_make_envelope())

        clue_1a = next(
            c for c in result.clues if c.number == 1 and c.direction == "across"
        )
        assert clue_1a.clue == "Fixed clue"
        # No second whole-puzzle regeneration happened.
        assert llm.call_count == 4

    def test_regenerates_when_many_clues_bad(self) -> None:
        """When too many clues are bad, the whole puzzle is regenerated."""
        clue_json = _build_clue_json()
        # 3 of 6 bad -> only 0.5 clean < 0.8 -> should regenerate.
        eval_bad = _build_eval_json_mixed(
            low_accuracy_entries={
                (1, "across"),
                (1, "down"),
                (2, "down"),
            },
            low_accuracy=5.0,
        )
        eval_good = _build_eval_json(accuracy=22, freshness=21, craft=22, fairness=20)
        llm = SequentialMockLLM([clue_json, eval_bad, clue_json, eval_good])
        grader = ClueGrader(llm, min_passing_score=70)
        step = ClueWithGradingStep(
            llm, grader, max_retries=3, surgical_repair_pass_ratio=0.8
        )
        result = step.run(_make_envelope())

        assert result.clue_grade_report is not None
        assert result.clue_grade_report.passing is True


class TestFairnessRepair:
    def test_low_fairness_triggers_repair(self) -> None:
        """A clue with low fairness (e.g. a collocation give-away the grader
        flags) is routed to surgical repair via the fairness threshold."""
        clue_json = _build_clue_json()
        # All accuracy fine, but 1A fairness is very low -> must repair.
        items = []
        for e in EXPECTED_ENTRIES:
            low = (e["number"], e["direction"]) == (1, "across")
            items.append(
                {
                    "number": e["number"],
                    "direction": e["direction"],
                    "accuracy": 22,
                    "freshness": 20,
                    "craft": 20,
                    "fairness": 5 if low else 20,
                    "feedback": "Collocation give-away" if low else "Good",
                }
            )
        eval_json = json.dumps(items)
        repair_json = json.dumps(
            [{"number": 1, "direction": "across", "clue": "Plain definition"}]
        )
        re_eval_json = _build_eval_json(
            accuracy=22, freshness=20, craft=20, fairness=20
        )
        llm = SequentialMockLLM([clue_json, eval_json, repair_json, re_eval_json])
        grader = ClueGrader(llm, min_passing_score=70)
        step = ClueWithGradingStep(
            llm, grader, max_retries=1, fairness_repair_threshold=15
        )
        result = step.run(_make_envelope())

        clue_1a = next(
            c for c in result.clues if c.number == 1 and c.direction == "across"
        )
        assert clue_1a.clue == "Plain definition"

    def test_repair_uses_separate_repair_llm(self) -> None:
        clue_json = _build_clue_json()
        eval_json = _build_eval_json_mixed(
            low_accuracy_entries={(1, "across")},
            low_accuracy=5.0,
            normal_accuracy=22.0,
        )
        repair_json = json.dumps(
            [{"number": 1, "direction": "across", "clue": "Repair model clue"}]
        )
        re_eval_json = _build_eval_json_subset(
            {(1, "across")}, accuracy=22, freshness=20, craft=20, fairness=20
        )
        generation_llm = SequentialMockLLM([clue_json])
        grading_llm = SequentialMockLLM([eval_json, re_eval_json])
        repair_llm = SequentialMockLLM([repair_json])
        grader = ClueGrader(grading_llm, min_passing_score=70)
        step = ClueWithGradingStep(
            generation_llm,
            grader,
            repair_llm=repair_llm,
            max_retries=1,
        )

        result = step.run(_make_envelope())

        clue_1a = next(
            c for c in result.clues if c.number == 1 and c.direction == "across"
        )
        assert clue_1a.clue == "Repair model clue"
        assert generation_llm.call_count == 1
        assert grading_llm.call_count == 2
        assert repair_llm.call_count == 1
        assert "CLUES TO REPAIR" in repair_llm.prompts[0]


class TestFreshnessRepair:
    """Hard configs route evaluator-flagged too-easy clues to repair."""

    def _eval_json_low_freshness(self, feedback: str) -> str:
        items = []
        for e in EXPECTED_ENTRIES:
            low = (e["number"], e["direction"]) == (1, "across")
            items.append(
                {
                    "number": e["number"],
                    "direction": e["direction"],
                    "accuracy": 22,
                    "freshness": 5 if low else 20,
                    "craft": 20,
                    "fairness": 20,
                    "feedback": feedback if low else "Good",
                }
            )
        return json.dumps(items)

    def test_low_freshness_triggers_repair_when_threshold_set(self) -> None:
        """With freshness_repair_threshold set (Hard configs), a too-easy
        clue (freshness 0-9) is surgically repaired."""
        clue_json = _build_clue_json()
        eval_json = self._eval_json_low_freshness("Plain but fine")
        repair_json = json.dumps(
            [{"number": 1, "direction": "across", "clue": "Harder angle"}]
        )
        re_eval_json = _build_eval_json_subset(
            {(1, "across")}, accuracy=22, freshness=20, craft=20, fairness=20
        )
        llm = SequentialMockLLM([clue_json, eval_json, repair_json, re_eval_json])
        grader = ClueGrader(llm, min_passing_score=70)
        step = ClueWithGradingStep(
            llm, grader, max_retries=1, freshness_repair_threshold=10
        )
        result = step.run(_make_envelope())

        clue_1a = next(
            c for c in result.clues if c.number == 1 and c.direction == "across"
        )
        assert clue_1a.clue == "Harder angle"

    def test_low_freshness_ignored_by_default(self) -> None:
        """With the default threshold (0, Easy configs), low freshness alone
        does not trigger repair."""
        clue_json = _build_clue_json()
        eval_json = self._eval_json_low_freshness("Plain but fine")
        llm = SequentialMockLLM([clue_json, eval_json])
        grader = ClueGrader(llm, min_passing_score=70)
        step = ClueWithGradingStep(llm, grader, max_retries=1)
        result = step.run(_make_envelope())

        clue_1a = next(
            c for c in result.clues if c.number == 1 and c.direction == "across"
        )
        assert clue_1a.clue == "Generated clue 1 across"
        assert llm.call_count == 2

    def test_too_easy_feedback_marker_triggers_repair(self) -> None:
        """Evaluator feedback containing "too easy" triggers repair even when
        the numeric sub-scores all pass their thresholds."""
        clue_json = _build_clue_json()
        items = []
        for e in EXPECTED_ENTRIES:
            low = (e["number"], e["direction"]) == (1, "across")
            items.append(
                {
                    "number": e["number"],
                    "direction": e["direction"],
                    "accuracy": 22,
                    "freshness": 12 if low else 20,
                    "craft": 20,
                    "fairness": 20,
                    "feedback": ("Too easy for a Hard puzzle" if low else "Good"),
                }
            )
        eval_json = json.dumps(items)
        repair_json = json.dumps(
            [{"number": 1, "direction": "across", "clue": "Harder angle"}]
        )
        re_eval_json = _build_eval_json_subset(
            {(1, "across")}, accuracy=22, freshness=20, craft=20, fairness=20
        )
        llm = SequentialMockLLM([clue_json, eval_json, repair_json, re_eval_json])
        grader = ClueGrader(llm, min_passing_score=70)
        step = ClueWithGradingStep(llm, grader, max_retries=1)
        result = step.run(_make_envelope())

        clue_1a = next(
            c for c in result.clues if c.number == 1 and c.direction == "across"
        )
        assert clue_1a.clue == "Harder angle"


class TestVerifyAfterRepair:
    def test_repairs_again_when_still_flagged(self) -> None:
        """A clue still flagged after one repair is repaired a second time."""
        clue_json = _build_clue_json()
        eval_json = _build_eval_json_mixed(
            low_accuracy_entries={(1, "across")}, low_accuracy=5.0
        )
        # First repair returns a clue that the re-grade still flags as low;
        # second repair returns one the re-grade accepts.
        repair_1 = json.dumps(
            [{"number": 1, "direction": "across", "clue": "Still bad"}]
        )
        re_eval_still_bad = _build_eval_json_subset({(1, "across")}, accuracy=5.0)
        repair_2 = json.dumps(
            [{"number": 1, "direction": "across", "clue": "Now good"}]
        )
        re_eval_good = _build_eval_json_subset(
            {(1, "across")}, accuracy=22, freshness=21, craft=22, fairness=20
        )
        llm = SequentialMockLLM(
            [
                clue_json,
                eval_json,
                repair_1,
                re_eval_still_bad,
                repair_2,
                re_eval_good,
            ]
        )
        grader = ClueGrader(llm, min_passing_score=70)
        step = ClueWithGradingStep(
            llm,
            grader,
            max_retries=1,
            repair_verify_attempts=2,
            surgical_repair_pass_ratio=None,
        )
        result = step.run(_make_envelope())

        clue_1a = next(
            c for c in result.clues if c.number == 1 and c.direction == "across"
        )
        assert clue_1a.clue == "Now good"


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

        llm = SequentialMockLLM(
            [
                clue_json,
                eval_low,
                clue_json,
                eval_medium,
            ]
        )
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
            llm,
            grader,
            max_retries=3,
            regenerate_on_fail=False,
            accuracy_repair_threshold=0,
            fairness_repair_threshold=0,
            craft_repair_threshold=0,
            individual_repair_score_threshold=0,
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
        repair_json = json.dumps(
            [
                {"number": 1, "direction": "across", "clue": "Repaired feline clue"},
            ]
        )
        # Re-grade after repair: only the repaired clue is re-graded
        re_eval_json = _build_eval_json_subset(
            {(1, "across")}, accuracy=22, freshness=20, craft=20, fairness=20
        )

        llm = SequentialMockLLM(
            [
                clue_json,  # clue generation
                eval_json,  # first grade (1A has low accuracy)
                repair_json,  # repair LLM call
                re_eval_json,  # re-grade after repair
            ]
        )
        grader = ClueGrader(llm, min_passing_score=70)
        step = ClueWithGradingStep(
            llm,
            grader,
            max_retries=1,
            accuracy_repair_threshold=12,
        )
        result = step.run(_make_envelope())

        # The repaired clue should be in the result
        clue_1a = next(
            c for c in result.clues if c.number == 1 and c.direction == "across"
        )
        assert clue_1a.clue == "Repaired feline clue"
        # 4 LLM calls: generate + grade + repair + re-grade
        assert llm.call_count == 4

    def test_repair_triggered_on_low_individual_score(self) -> None:
        """A clue can be repaired even if the LLM overstates accuracy."""
        clue_json = _build_clue_json()
        items = []
        for e in EXPECTED_ENTRIES:
            if e["number"] == 1 and e["direction"] == "across":
                items.append(
                    {
                        "number": e["number"],
                        "direction": e["direction"],
                        "accuracy": 18,
                        "freshness": 15,
                        "craft": 14,
                        "fairness": 12,
                        "feedback": "MAJOR ISSUES: factual error.",
                    }
                )
            else:
                items.append(
                    {
                        "number": e["number"],
                        "direction": e["direction"],
                        "accuracy": 22,
                        "freshness": 20,
                        "craft": 20,
                        "fairness": 20,
                        "feedback": "Good",
                    }
                )
        eval_json = json.dumps(items)
        repair_json = json.dumps(
            [
                {"number": 1, "direction": "across", "clue": "Small house pet"},
            ]
        )
        re_eval_json = _build_eval_json_subset(
            {(1, "across")}, accuracy=22, freshness=20, craft=20, fairness=20
        )

        llm = SequentialMockLLM(
            [
                clue_json,
                eval_json,
                repair_json,
                re_eval_json,
            ]
        )
        grader = ClueGrader(llm, min_passing_score=70)
        step = ClueWithGradingStep(
            llm,
            grader,
            max_retries=1,
            accuracy_repair_threshold=12,
        )
        result = step.run(_make_envelope())

        clue_1a = next(
            c for c in result.clues if c.number == 1 and c.direction == "across"
        )
        assert clue_1a.clue == "Small house pet"
        assert llm.call_count == 4

    def test_no_repair_when_all_accuracy_ok(self) -> None:
        """No repair pass if all accuracy scores are above threshold."""
        clue_json = _build_clue_json()
        eval_json = _build_eval_json(accuracy=22, freshness=20, craft=20, fairness=20)

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

        llm = SequentialMockLLM(
            [
                clue_json,  # clue generation
                eval_json,  # grade (1A has low accuracy)
                "not valid json!!!",  # repair fails to parse
            ]
        )
        grader = ClueGrader(llm, min_passing_score=70)
        step = ClueWithGradingStep(
            llm,
            grader,
            max_retries=1,
            accuracy_repair_threshold=12,
        )
        result = step.run(_make_envelope())

        # Original clue preserved (repair failed gracefully)
        clue_1a = next(
            c for c in result.clues if c.number == 1 and c.direction == "across"
        )
        assert clue_1a.clue == "Generated clue 1 across"
        # 3 LLM calls: generate + grade + failed repair
        assert llm.call_count == 3

    def test_repair_multiple_clues(self) -> None:
        """Multiple clues with low accuracy get repaired together."""
        clue_json = _build_clue_json()
        eval_json = _build_eval_json_mixed(
            low_accuracy_entries={(1, "across"), (3, "down")},
            low_accuracy=5.0,
        )
        repair_json = json.dumps(
            [
                {"number": 1, "direction": "across", "clue": "Fixed feline clue"},
                {"number": 3, "direction": "down", "clue": "Fixed name clue"},
            ]
        )
        re_eval_json = _build_eval_json_subset(
            {(1, "across"), (3, "down")},
            accuracy=22,
            freshness=20,
            craft=20,
            fairness=20,
        )

        llm = SequentialMockLLM(
            [
                clue_json,
                eval_json,
                repair_json,
                re_eval_json,
            ]
        )
        grader = ClueGrader(llm, min_passing_score=70)
        step = ClueWithGradingStep(
            llm,
            grader,
            max_retries=1,
            accuracy_repair_threshold=12,
        )
        result = step.run(_make_envelope())

        clue_1a = next(
            c for c in result.clues if c.number == 1 and c.direction == "across"
        )
        clue_3d = next(
            c for c in result.clues if c.number == 3 and c.direction == "down"
        )
        assert clue_1a.clue == "Fixed feline clue"
        assert clue_3d.clue == "Fixed name clue"


class TestFactCheckRepair:
    """Tests for the fact-check repair pass."""

    def test_fact_check_repairs_high_scoring_risky_clue(self) -> None:
        """Fact checker can repair a clue even when normal grading passes."""
        clue_items = []
        for e in EXPECTED_ENTRIES:
            clue = f"Generated clue {e['number']} {e['direction']}"
            if e["number"] == 1 and e["direction"] == "across":
                clue = "Hit song by Taylor Swift"
            clue_items.append(
                {
                    "number": e["number"],
                    "direction": e["direction"],
                    "clue": clue,
                }
            )
        clue_json = json.dumps(clue_items)
        eval_json = _build_eval_json(accuracy=23, freshness=22, craft=22, fairness=22)
        fact_json = json.dumps(
            [
                {
                    "number": 1,
                    "direction": "across",
                    "status": "uncertain",
                    "reason": "Specific song claim is not reliable enough.",
                }
            ]
        )
        repair_json = json.dumps(
            [
                {
                    "number": 1,
                    "direction": "across",
                    "clue": "Common feline pet",
                },
            ]
        )
        re_eval_json = _build_eval_json_subset(
            {(1, "across")}, accuracy=22, freshness=20, craft=20, fairness=20
        )

        clue_llm = SequentialMockLLM([clue_json, repair_json])
        grade_llm = SequentialMockLLM([eval_json, re_eval_json])
        fact_llm = SequentialMockLLM([fact_json])
        grader = ClueGrader(grade_llm, min_passing_score=70)
        fact_checker = ClueFactChecker(fact_llm)
        step = ClueWithGradingStep(
            clue_llm,
            grader,
            max_retries=1,
            fact_checker=fact_checker,
        )

        result = step.run(_make_envelope())

        clue_1a = next(
            c for c in result.clues if c.number == 1 and c.direction == "across"
        )
        assert clue_1a.clue == "Common feline pet"
        assert result.metadata["clue_fact_check"][0]["status"] == "uncertain"
        assert clue_llm.call_count == 2
        assert grade_llm.call_count == 2
        assert fact_llm.call_count == 1

    def test_fact_check_reverifies_and_fixes_a_bad_repair(self) -> None:
        """A repair that swaps in a fresh factual error gets re-checked.

        Mirrors the ELENA bug: the first fact-check flags a clue, repair
        rewrites it into another still-risky clue, the re-check flags that
        too, and a second repair finally lands a safe clue. The puzzle ships
        clean with no FACT: soft error.
        """
        clue_items = []
        for e in EXPECTED_ENTRIES:
            clue = f"Generated clue {e['number']} {e['direction']}"
            if e["number"] == 1 and e["direction"] == "across":
                clue = "Hit song by a famous singer"
            clue_items.append(
                {"number": e["number"], "direction": e["direction"], "clue": clue}
            )
        clue_json = json.dumps(clue_items)
        eval_json = _build_eval_json(accuracy=23, freshness=22, craft=22, fairness=22)

        # Round 1 flags the original "song" clue; round 2 flags the bad repair
        # (still a risky "Actress ..." trivia clue); round 3 sees a safe clue.
        fact_round1 = json.dumps(
            [
                {
                    "number": 1,
                    "direction": "across",
                    "status": "incorrect",
                    "reason": "Wrong song attribution.",
                }
            ]
        )
        fact_round2 = json.dumps(
            [
                {
                    "number": 1,
                    "direction": "across",
                    "status": "incorrect",
                    "reason": "Wrong actress attribution.",
                }
            ]
        )
        repair_bad = json.dumps(
            [
                {
                    "number": 1,
                    "direction": "across",
                    "clue": "Actress Wrongname of a hit film",
                }
            ]
        )
        repair_good = json.dumps(
            [{"number": 1, "direction": "across", "clue": "Common feline pet"}]
        )
        re_eval = _build_eval_json_subset(
            {(1, "across")}, accuracy=22, freshness=20, craft=20, fairness=20
        )

        clue_llm = SequentialMockLLM([clue_json, repair_bad, repair_good])
        grade_llm = SequentialMockLLM([eval_json, re_eval, re_eval])
        fact_llm = SequentialMockLLM([fact_round1, fact_round2])
        grader = ClueGrader(grade_llm, min_passing_score=70)
        fact_checker = ClueFactChecker(fact_llm)
        step = ClueWithGradingStep(
            clue_llm,
            grader,
            max_retries=1,
            fact_checker=fact_checker,
            fact_check_repair_attempts=2,
        )

        result = step.run(_make_envelope())

        clue_1a = next(
            c for c in result.clues if c.number == 1 and c.direction == "across"
        )
        assert clue_1a.clue == "Common feline pet"
        # Two fact-check rounds ran (original + re-check of the bad repair).
        assert fact_llm.call_count == 2
        # The final clue is clean, so no fact soft-error is surfaced.
        assert not any(err.startswith("FACT:") for err in result.errors)

    def test_fact_check_surfaces_soft_error_when_repair_stays_wrong(self) -> None:
        """If repair keeps producing an incorrect clue, surface a FACT: error.

        The clue is still flagged incorrect after the repair budget is spent,
        so it must be held back from upload via a blocking soft error rather
        than shipped silently.
        """
        clue_items = []
        for e in EXPECTED_ENTRIES:
            clue = f"Generated clue {e['number']} {e['direction']}"
            if e["number"] == 1 and e["direction"] == "across":
                clue = "Hit song by a famous singer"
            clue_items.append(
                {"number": e["number"], "direction": e["direction"], "clue": clue}
            )
        clue_json = json.dumps(clue_items)
        eval_json = _build_eval_json(accuracy=23, freshness=22, craft=22, fairness=22)

        fact_incorrect = json.dumps(
            [
                {
                    "number": 1,
                    "direction": "across",
                    "status": "incorrect",
                    "reason": "Wrong attribution.",
                }
            ]
        )
        repair_still_bad = json.dumps(
            [
                {
                    "number": 1,
                    "direction": "across",
                    "clue": "Actress Stillwrong of a hit film",
                }
            ]
        )
        re_eval = _build_eval_json_subset(
            {(1, "across")}, accuracy=22, freshness=20, craft=20, fairness=20
        )

        # One repair attempt: round 1 flags + repairs, round 2 flags again and
        # the budget is spent, so the bad clue is surfaced as a soft error.
        clue_llm = SequentialMockLLM([clue_json, repair_still_bad])
        grade_llm = SequentialMockLLM([eval_json, re_eval])
        fact_llm = SequentialMockLLM([fact_incorrect, fact_incorrect])
        grader = ClueGrader(grade_llm, min_passing_score=70)
        fact_checker = ClueFactChecker(fact_llm)
        step = ClueWithGradingStep(
            clue_llm,
            grader,
            max_retries=1,
            fact_checker=fact_checker,
            fact_check_repair_attempts=1,
        )

        result = step.run(_make_envelope())

        fact_errors = [err for err in result.errors if err.startswith("FACT:")]
        assert len(fact_errors) == 1
        assert "CAT" in fact_errors[0]
        assert "1-across" in fact_errors[0]


class TestDuplicateHistoryRepair:
    """Tests for exact duplicate clue-history repair."""

    def test_duplicate_history_hit_triggers_repair(self) -> None:
        """Exact clue reuse gets repaired even when regular grading passes."""
        clue_json = _build_clue_json()
        eval_json = _build_eval_json(accuracy=22, freshness=20, craft=20, fairness=20)
        repair_json = json.dumps(
            [
                {"number": 1, "direction": "across", "clue": "Small house pet"},
            ]
        )
        re_eval_json = _build_eval_json_subset(
            {(1, "across")}, accuracy=22, freshness=20, craft=20, fairness=20
        )
        history = ClueHistoryIndex()
        history.add("CAT", "Generated clue 1 across")

        llm = SequentialMockLLM(
            [
                clue_json,
                eval_json,
                repair_json,
                re_eval_json,
            ]
        )
        grader = ClueGrader(llm, min_passing_score=70)
        step = ClueWithGradingStep(
            llm,
            grader,
            max_retries=1,
            clue_history=history,
        )
        result = step.run(_make_envelope())

        clue_1a = next(
            c for c in result.clues if c.number == 1 and c.direction == "across"
        )
        assert clue_1a.clue == "Small house pet"
        assert llm.call_count == 4

    def test_stuck_duplicate_becomes_soft_error_not_raise(self) -> None:
        """A duplicate that survives all repair attempts is a DUPLICATE: soft
        error — the puzzle still completes, no exception."""
        clue_json = _build_clue_json()
        eval_json = _build_eval_json(accuracy=22, freshness=20, craft=20, fairness=20)
        # Repair keeps returning the SAME already-used clue, so it never clears.
        stuck_repair = json.dumps(
            [{"number": 1, "direction": "across", "clue": "Generated clue 1 across"}]
        )
        history = ClueHistoryIndex()
        history.add("CAT", "Generated clue 1 across")

        # gen, grade, then (repair, re-grade) per duplicate attempt.
        responses = [clue_json, eval_json]
        for _ in range(4):
            responses += [stuck_repair, eval_json]
        llm = SequentialMockLLM(responses)
        grader = ClueGrader(llm, min_passing_score=70)
        step = ClueWithGradingStep(
            llm,
            grader,
            max_retries=1,
            clue_history=history,
            duplicate_repair_attempts=4,
        )

        # Must NOT raise.
        result = step.run(_make_envelope())
        dup_errors = [e for e in result.errors if e.startswith("DUPLICATE:")]
        assert len(dup_errors) == 1
        assert "CAT" in dup_errors[0]

    def test_repair_feedback_lists_all_used_clues(self) -> None:
        """The duplicate-repair prompt names every clue already used for the
        answer, not just the one collided with."""
        clue_json = _build_clue_json()
        eval_json = _build_eval_json(accuracy=22, freshness=20, craft=20, fairness=20)
        fixed = json.dumps(
            [{"number": 1, "direction": "across", "clue": "A fresh feline clue"}]
        )
        history = ClueHistoryIndex()
        # The generated clue duplicates one of several known CAT clues.
        history.add("CAT", "Generated clue 1 across")
        history.add("CAT", "Purring pet")
        history.add("CAT", "Whiskered animal")

        llm = SequentialMockLLM([clue_json, eval_json, fixed, eval_json])
        grader = ClueGrader(llm, min_passing_score=70)
        step = ClueWithGradingStep(llm, grader, max_retries=1, clue_history=history)
        step.run(_make_envelope())

        # The repair prompt (3rd call) should mention all three used clues.
        repair_prompt = llm.prompts[2]
        assert "Generated clue 1 across" in repair_prompt
        assert "Purring pet" in repair_prompt
        assert "Whiskered animal" in repair_prompt


class TestExternalDuplicateRepair:
    """Tests for repair_external_duplicates (the post-batch sweep path)."""

    def _completed_envelope(self) -> PuzzleEnvelope:
        clues = [
            ClueEntry(
                number=e["number"],
                direction=e["direction"],
                answer=e["answer"],
                clue=f"Generated clue {e['number']} {e['direction']}",
            )
            for e in EXPECTED_ENTRIES
        ]
        return _make_envelope(clues=clues)

    def test_external_hits_repaired_without_self_matches(self) -> None:
        """Sweep-supplied hits get repaired, and the puzzle's own clues —
        which already live in the shared history after batch completion —
        are not re-flagged as self-matches."""
        envelope = self._completed_envelope()
        history = ClueHistoryIndex()
        # Shared history after a parallel batch contains this puzzle's own
        # clues; the other puzzle's colliding clue is the identical entry.
        history.add_clues(envelope.clues)

        cat_clue = next(c for c in envelope.clues if c.answer == "CAT")
        hits = [DuplicateClueHit(clue=cat_clue, existing_clue=cat_clue.clue)]

        repair_json = json.dumps(
            [{"number": 1, "direction": "across", "clue": "Fresh feline clue"}]
        )
        # One repair pass = repair call + post-repair re-grade (subset: only
        # the repaired clue is re-graded).
        llm = SequentialMockLLM([repair_json, _build_eval_json_subset({(1, "across")})])
        grader = ClueGrader(llm, min_passing_score=70)
        step = ClueWithGradingStep(llm, grader, clue_history=history)

        result = step.repair_external_duplicates(envelope, hits)

        clue_1a = next(c for c in result.clues if c.answer == "CAT")
        assert clue_1a.clue == "Fresh feline clue"
        assert not [e for e in result.errors if e.startswith("DUPLICATE:")]
        # Exactly one repair pass (repair + re-grade): the untouched clues
        # all exist in the shared history, so a full re-scan would have
        # flagged them as self-matches and run pointless extra rounds.
        assert llm.call_count == 2

    def test_stuck_external_duplicate_becomes_soft_error(self) -> None:
        """A sweep duplicate that survives all repair attempts becomes a
        DUPLICATE: soft error, same as the in-run path."""
        envelope = self._completed_envelope()
        history = ClueHistoryIndex()
        history.add_clues(envelope.clues)

        cat_clue = next(c for c in envelope.clues if c.answer == "CAT")
        hits = [DuplicateClueHit(clue=cat_clue, existing_clue=cat_clue.clue)]

        # Repair keeps landing back on the same already-used clue.
        stuck = json.dumps(
            [{"number": 1, "direction": "across", "clue": cat_clue.clue}]
        )
        responses = [stuck, _build_eval_json_subset({(1, "across")})] * 4
        llm = SequentialMockLLM(responses)
        grader = ClueGrader(llm, min_passing_score=70)
        step = ClueWithGradingStep(
            llm, grader, clue_history=history, duplicate_repair_attempts=4
        )

        result = step.repair_external_duplicates(envelope, hits)

        dup_errors = [e for e in result.errors if e.startswith("DUPLICATE:")]
        assert len(dup_errors) == 1
        assert "CAT" in dup_errors[0]
        assert llm.call_count == 8

    def test_no_hits_returns_envelope_unchanged(self) -> None:
        envelope = self._completed_envelope()
        history = ClueHistoryIndex()
        llm = SequentialMockLLM([])
        grader = ClueGrader(llm, min_passing_score=70)
        step = ClueWithGradingStep(llm, grader, clue_history=history)

        assert step.repair_external_duplicates(envelope, []) is envelope
        assert llm.call_count == 0


def _build_clue_json_with(overrides: dict[tuple[int, str], str]) -> str:
    """Clue-gen JSON, overriding specific entries' clue text."""
    items = []
    for e in EXPECTED_ENTRIES:
        key = (e["number"], e["direction"])
        items.append(
            {
                "number": e["number"],
                "direction": e["direction"],
                "clue": overrides.get(
                    key, f"Generated clue {e['number']} {e['direction']}"
                ),
            }
        )
    return json.dumps(items)


def _build_repair_json(repairs: dict[tuple[int, str], str]) -> str:
    items = [
        {"number": n, "direction": d, "clue": clue} for (n, d), clue in repairs.items()
    ]
    return json.dumps(items)


class TestLeakRepair:
    def test_leaked_clue_is_repaired_away(self) -> None:
        """A clue echoing its answer is caught and repaired; no soft error."""
        # 1A=CAT clued with the answer word -> exact leak.
        clue_json = _build_clue_json_with({(1, "across"): "A pet cat"})
        eval_json = _build_eval_json(accuracy=22, freshness=21, craft=22, fairness=20)
        # Leak repair returns a clean clue, then the grader re-grades.
        repair_json = _build_repair_json({(1, "across"): "Common pet"})
        llm = SequentialMockLLM([clue_json, eval_json, repair_json, eval_json])
        grader = ClueGrader(llm, min_passing_score=70)
        step = ClueWithGradingStep(llm, grader, max_retries=1)

        result = step.run(_make_envelope())

        clue_1a = next(
            c for c in result.clues if c.number == 1 and c.direction == "across"
        )
        assert clue_1a.clue == "Common pet"
        assert not any(e.startswith("LEAK:") for e in result.errors)

    def test_surviving_leak_becomes_soft_error(self) -> None:
        """A leak that repair fails to fix is surfaced as a LEAK: soft error."""
        clue_json = _build_clue_json_with({(1, "across"): "A pet cat"})
        eval_json = _build_eval_json(accuracy=22, freshness=21, craft=22, fairness=20)
        # Every repair attempt still leaks "cat"; after leak_repair_attempts
        # the leak survives and must be reported.
        still_leaking = _build_repair_json({(1, "across"): "A wild cat"})
        llm = SequentialMockLLM(
            [
                clue_json,
                eval_json,
                still_leaking,
                eval_json,
                still_leaking,
                eval_json,
                still_leaking,
                eval_json,
            ]
        )
        grader = ClueGrader(llm, min_passing_score=70)
        step = ClueWithGradingStep(llm, grader, max_retries=1, leak_repair_attempts=3)

        result = step.run(_make_envelope())

        leak_errors = [e for e in result.errors if e.startswith("LEAK:")]
        assert len(leak_errors) == 1
        assert "CAT" in leak_errors[0]
