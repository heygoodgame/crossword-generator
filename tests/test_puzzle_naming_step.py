"""Tests for the PuzzleNamingStep pipeline step."""

from __future__ import annotations

import json

import pytest

from crossword_generator.llm.base import LLMProvider
from crossword_generator.llm.prompts.puzzle_naming import (
    build_puzzle_naming_prompt,
)
from crossword_generator.models import (
    ClueEntry,
    FillResult,
    PuzzleEnvelope,
    PuzzleType,
    ThemeConcept,
)
from crossword_generator.steps.puzzle_naming_step import PuzzleNamingStep

MOCK_GRID = [
    ["A", "B", "C", "D", "E"],
    ["F", "G", "H", "I", "J"],
    ["K", "L", "M", "N", "O"],
    ["P", "Q", "R", "S", "T"],
    ["U", "V", "W", "X", "Y"],
]

MOCK_CLUES = [
    ClueEntry(number=1, direction="across", answer="ABCDE", clue="First row"),
    ClueEntry(number=6, direction="across", answer="FGHIJ", clue="Second row"),
    ClueEntry(number=7, direction="across", answer="KLMNO", clue="Third row"),
    ClueEntry(number=8, direction="across", answer="PQRST", clue="Fourth row"),
    ClueEntry(number=9, direction="across", answer="UVWXY", clue="Fifth row"),
    ClueEntry(number=1, direction="down", answer="AFKPU", clue="First col"),
    ClueEntry(number=2, direction="down", answer="BGLQV", clue="Second col"),
    ClueEntry(number=3, direction="down", answer="CHMRW", clue="Third col"),
    ClueEntry(number=4, direction="down", answer="DINSX", clue="Fourth col"),
    ClueEntry(number=5, direction="down", answer="EJOTY", clue="Fifth col"),
]


class MockLLM(LLMProvider):
    """Mock LLM that returns canned responses."""

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
    theme: ThemeConcept | None = None,
) -> PuzzleEnvelope:
    fill = None
    if grid is not None:
        fill = FillResult(grid=grid, filler_used="mock")
    return PuzzleEnvelope(
        puzzle_type=PuzzleType.MINI,
        grid_size=5,
        fill=fill,
        clues=clues or [],
        theme=theme,
    )


class TestPuzzleNamingStep:
    def test_happy_path(self) -> None:
        """LLM returns valid JSON, title is set on envelope."""
        response = json.dumps({"title": "Midas Touch"})
        step = PuzzleNamingStep(MockLLM(response=response))
        envelope = _make_envelope(grid=MOCK_GRID, clues=MOCK_CLUES)
        result = step.run(envelope)

        assert result.title == "Midas Touch"
        assert "puzzle-naming" in result.step_history

    def test_step_name(self) -> None:
        step = PuzzleNamingStep(MockLLM())
        assert step.name == "puzzle-naming"

    def test_parse_retry_then_success(self) -> None:
        """Bad JSON first, good JSON second."""
        good = json.dumps({"title": "Second Try"})
        step = PuzzleNamingStep(
            MockLLM(responses=["not valid json!!!", good]),
            max_retries=3,
        )
        envelope = _make_envelope(grid=MOCK_GRID, clues=MOCK_CLUES)
        result = step.run(envelope)

        assert result.title == "Second Try"

    def test_total_failure_falls_back(self) -> None:
        """All retries fail — falls back to generic title."""
        step = PuzzleNamingStep(
            MockLLM(response="garbage"),
            max_retries=2,
        )
        envelope = _make_envelope(grid=MOCK_GRID, clues=MOCK_CLUES)
        result = step.run(envelope)

        assert result.title == "Mini Crossword"

    def test_fallback_for_midi(self) -> None:
        """Fallback title uses puzzle type."""
        step = PuzzleNamingStep(
            MockLLM(response="garbage"),
            max_retries=1,
        )
        envelope = PuzzleEnvelope(
            puzzle_type=PuzzleType.MIDI,
            grid_size=9,
            fill=FillResult(grid=MOCK_GRID, filler_used="mock"),
            clues=MOCK_CLUES,
        )
        result = step.run(envelope)

        assert result.title == "Midi Crossword"

    def test_validate_input_no_fill(self) -> None:
        step = PuzzleNamingStep(MockLLM())
        envelope = _make_envelope(grid=None, clues=MOCK_CLUES)
        with pytest.raises(ValueError, match="no fill result"):
            step.run(envelope)

    def test_validate_input_no_clues(self) -> None:
        step = PuzzleNamingStep(MockLLM())
        envelope = _make_envelope(grid=MOCK_GRID, clues=[])
        with pytest.raises(ValueError, match="no clues"):
            step.run(envelope)

    def test_original_envelope_unchanged(self) -> None:
        response = json.dumps({"title": "Fresh Title"})
        step = PuzzleNamingStep(MockLLM(response=response))
        envelope = _make_envelope(grid=MOCK_GRID, clues=MOCK_CLUES)
        step.run(envelope)

        assert envelope.title == ""

    def test_handles_markdown_wrapped_json(self) -> None:
        """LLM wraps JSON in ```json fences."""
        raw = json.dumps({"title": "Wrapped Title"})
        wrapped = f"```json\n{raw}\n```"
        step = PuzzleNamingStep(MockLLM(response=wrapped))
        envelope = _make_envelope(grid=MOCK_GRID, clues=MOCK_CLUES)
        result = step.run(envelope)

        assert result.title == "Wrapped Title"

    def test_handles_preamble(self) -> None:
        """LLM adds preamble text before JSON."""
        raw = json.dumps({"title": "After Preamble"})
        preamble = f"Here is the title:\n\n{raw}"
        step = PuzzleNamingStep(MockLLM(response=preamble))
        envelope = _make_envelope(grid=MOCK_GRID, clues=MOCK_CLUES)
        result = step.run(envelope)

        assert result.title == "After Preamble"

    def test_strips_whitespace_from_title(self) -> None:
        response = json.dumps({"title": "  Spacey Title  "})
        step = PuzzleNamingStep(MockLLM(response=response))
        envelope = _make_envelope(grid=MOCK_GRID, clues=MOCK_CLUES)
        result = step.run(envelope)

        assert result.title == "Spacey Title"


class TestPuzzleNamingPrompt:
    def test_themed_prompt_includes_theme_context(self) -> None:
        theme = ThemeConcept(
            topic="Things that are golden",
            wordplay_type="double meaning",
            revealer="GOLDEN",
            seed_entries=["GATE", "RULE", "RATIO"],
        )
        prompt = build_puzzle_naming_prompt(
            PuzzleType.MIDI, 9, MOCK_CLUES, MOCK_GRID, theme
        )

        assert "Things that are golden" in prompt
        assert "GOLDEN" in prompt
        assert "GATE, RULE, RATIO" in prompt
        assert "OBLIQUELY" in prompt

    def test_themeless_prompt(self) -> None:
        prompt = build_puzzle_naming_prompt(
            PuzzleType.MINI, 5, MOCK_CLUES, MOCK_GRID
        )

        assert "THEMELESS" in prompt
        assert "standout fill" in prompt

    def test_prompt_includes_fill_words(self) -> None:
        prompt = build_puzzle_naming_prompt(
            PuzzleType.MINI, 5, MOCK_CLUES, MOCK_GRID
        )

        assert "ABCDE" in prompt
        assert "FGHIJ" in prompt

    def test_prompt_includes_sample_clues(self) -> None:
        prompt = build_puzzle_naming_prompt(
            PuzzleType.MINI, 5, MOCK_CLUES, MOCK_GRID
        )

        assert "First row" in prompt

    def test_prompt_requests_json(self) -> None:
        prompt = build_puzzle_naming_prompt(
            PuzzleType.MINI, 5, MOCK_CLUES, MOCK_GRID
        )

        assert "JSON" in prompt
        assert '"title"' in prompt
