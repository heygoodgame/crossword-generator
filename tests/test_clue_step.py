"""Tests for the ClueGenerationStep pipeline step."""

from __future__ import annotations

import json

import pytest

from crossword_generator.exporters.numbering import (
    NumberedEntry,
    compute_crossing_words,
)
from crossword_generator.llm.base import LLMProvider
from crossword_generator.models import (
    ClueEntry,
    FillResult,
    PuzzleEnvelope,
    PuzzleType,
)
from crossword_generator.steps.clue_step import ClueGenerationStep

# A simple 5x5 grid with no black squares for testing
MOCK_GRID = [
    ["A", "B", "C", "D", "E"],
    ["F", "G", "H", "I", "J"],
    ["K", "L", "M", "N", "O"],
    ["P", "Q", "R", "S", "T"],
    ["U", "V", "W", "X", "Y"],
]

# Expected entries from compute_numbering for the mock grid
# 1-Across: ABCDE, 1-Down: AFKPU
# 2-Down: BGLQV, 3-Down: CHMRW, 4-Down: DINSX, 5-Down: EJOTY
# 6-Across: FGHIJ, 7-Across: KLMNO, 8-Across: PQRST, 9-Across: UVWXY


def _build_mock_clue_json(entries: list[NumberedEntry]) -> str:
    """Build a valid JSON response matching the expected entries."""
    clues = []
    for entry in entries:
        clues.append(
            {
                "number": entry.number,
                "direction": entry.direction,
                "clue": f"Clue for {entry.answer}",
            }
        )
    return json.dumps(clues)


class MockLLM(LLMProvider):
    """Mock LLM that returns a canned response."""

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


def _make_envelope(
    *,
    grid: list[list[str]] | None = None,
    clues: list[ClueEntry] | None = None,
) -> PuzzleEnvelope:
    fill = None
    if grid is not None:
        fill = FillResult(grid=grid, filler_used="mock")
    return PuzzleEnvelope(
        puzzle_type=PuzzleType.MINI,
        grid_size=5,
        fill=fill,
        clues=clues or [],
    )


class TestClueGenerationStep:
    def test_happy_path(self) -> None:
        """Fill → clues populated correctly."""
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(MOCK_GRID)
        mock_response = _build_mock_clue_json(entries)

        step = ClueGenerationStep(MockLLM(response=mock_response))
        envelope = _make_envelope(grid=MOCK_GRID)
        result = step.run(envelope)

        assert len(result.clues) == len(entries)
        for clue_entry in result.clues:
            assert clue_entry.clue.startswith("Clue for ")
            assert clue_entry.answer != ""

    def test_step_name(self) -> None:
        step = ClueGenerationStep(MockLLM())
        assert step.name == "clue-generation"

    def test_step_history_updated(self) -> None:
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(MOCK_GRID)
        mock_response = _build_mock_clue_json(entries)

        step = ClueGenerationStep(MockLLM(response=mock_response))
        envelope = _make_envelope(grid=MOCK_GRID)
        result = step.run(envelope)

        assert "clue-generation" in result.step_history

    def test_validation_rejects_no_fill(self) -> None:
        step = ClueGenerationStep(MockLLM())
        envelope = _make_envelope(grid=None)
        with pytest.raises(ValueError, match="no fill result"):
            step.run(envelope)

    def test_validation_rejects_existing_clues(self) -> None:
        step = ClueGenerationStep(MockLLM())
        envelope = _make_envelope(
            grid=MOCK_GRID,
            clues=[
                ClueEntry(number=1, direction="across", answer="ABCDE", clue="test")
            ],
        )
        with pytest.raises(ValueError, match="already has clues"):
            step.run(envelope)

    def test_retry_on_malformed_response(self) -> None:
        """First response is garbage, second is valid JSON."""
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(MOCK_GRID)
        good_response = _build_mock_clue_json(entries)

        step = ClueGenerationStep(
            MockLLM(responses=["not valid json!!!", good_response]),
            max_retries=3,
        )
        envelope = _make_envelope(grid=MOCK_GRID)
        result = step.run(envelope)

        assert len(result.clues) == len(entries)

    def test_all_retries_exhausted(self) -> None:
        """All retries return garbage → raises ValueError."""
        step = ClueGenerationStep(
            MockLLM(response="not json"),
            max_retries=2,
        )
        envelope = _make_envelope(grid=MOCK_GRID)
        with pytest.raises(ValueError, match="Failed to parse clue response"):
            step.run(envelope)

    def test_handles_markdown_wrapped_json(self) -> None:
        """LLM wraps JSON in ```json ... ``` fences."""
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(MOCK_GRID)
        raw_json = _build_mock_clue_json(entries)
        wrapped = f"```json\n{raw_json}\n```"

        step = ClueGenerationStep(MockLLM(response=wrapped))
        envelope = _make_envelope(grid=MOCK_GRID)
        result = step.run(envelope)

        assert len(result.clues) == len(entries)

    def test_autocorrects_wrong_direction(self) -> None:
        """LLM returns correct number but wrong direction for
        entries that only exist in one direction."""
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(MOCK_GRID)
        # Build response with some directions flipped for
        # entries that only exist in one direction
        clues = []
        for entry in entries:
            d = entry.direction
            # Entries 6,7,8,9 across → flip to "down" (only
            # exist in one direction each, except 9 which has both)
            if entry.number in (6, 7, 8) and d == "across":
                d = "down"  # wrong, should be auto-corrected
            clues.append(
                {
                    "number": entry.number,
                    "direction": d,
                    "clue": f"Clue for {entry.answer}",
                }
            )
        mock_response = json.dumps(clues)

        step = ClueGenerationStep(MockLLM(response=mock_response))
        envelope = _make_envelope(grid=MOCK_GRID)
        result = step.run(envelope)

        assert len(result.clues) == len(entries)
        # Verify the corrected entries have the right direction
        clue_map = {
            (c.number, c.direction): c for c in result.clues
        }
        assert (6, "across") in clue_map
        assert (7, "across") in clue_map
        assert (8, "across") in clue_map


        """LLM adds preamble text before the JSON array."""
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(MOCK_GRID)
        raw_json = _build_mock_clue_json(entries)
        preamble = f"Here are the clues:\n\n{raw_json}"

        step = ClueGenerationStep(MockLLM(response=preamble))
        envelope = _make_envelope(grid=MOCK_GRID)
        result = step.run(envelope)

        assert len(result.clues) == len(entries)

    def test_original_envelope_unchanged(self) -> None:
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(MOCK_GRID)
        mock_response = _build_mock_clue_json(entries)

        step = ClueGenerationStep(MockLLM(response=mock_response))
        envelope = _make_envelope(grid=MOCK_GRID)
        step.run(envelope)

        assert envelope.clues == []  # Original unchanged


class TestComputeCrossingWords:
    def test_simple_grid(self) -> None:
        """Across entries cross all down entries in a full grid."""
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(MOCK_GRID)
        crossings = compute_crossing_words(entries, MOCK_GRID)

        # 1-Across (ABCDE) should cross 1-Down (AFKPU), 2-Down (BGLQV), etc.
        across_1_crossings = crossings[(1, "across")]
        assert len(across_1_crossings) == 5  # Crosses all 5 down entries

        # 1-Down (AFKPU) should cross 1-Across (ABCDE), 6-Across (FGHIJ), etc.
        down_1_crossings = crossings[(1, "down")]
        assert len(down_1_crossings) == 5  # Crosses all 5 across entries

    def test_grid_with_black_squares(self) -> None:
        """Grid with black squares has fewer crossings."""
        grid = [
            ["A", "B", "."],
            ["C", "D", "E"],
            [".", "F", "G"],
        ]
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(grid)
        crossings = compute_crossing_words(entries, grid)

        # All entries should have crossings computed
        for entry in entries:
            key = (entry.number, entry.direction)
            assert key in crossings

    def test_returns_answer_words(self) -> None:
        """Crossing words should be the full answer words, not individual letters."""
        from crossword_generator.exporters.numbering import compute_numbering

        entries = compute_numbering(MOCK_GRID)
        crossings = compute_crossing_words(entries, MOCK_GRID)

        # All crossing words should be strings of length > 1
        for words in crossings.values():
            for word in words:
                assert len(word) > 1
                assert word.isalpha()
