"""Tests for the ThemeGenerationStep pipeline step."""

from __future__ import annotations

import json

import pytest

from crossword_generator.dictionary import Dictionary
from crossword_generator.llm.base import LLMProvider
from crossword_generator.models import PuzzleEnvelope, PuzzleType, ThemeConcept
from crossword_generator.steps.theme_step import (
    ThemeGenerationStep,
    _parse_theme_response,
    _validate_theme_entries,
)


class MockLLM(LLMProvider):
    """A mock LLM that returns a fixed response."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = responses or []
        self._call_count = 0

    @property
    def name(self) -> str:
        return "mock-llm"

    def generate(self, prompt: str, **kwargs: object) -> str:
        if self._call_count < len(self._responses):
            response = self._responses[self._call_count]
        else:
            response = self._responses[-1] if self._responses else ""
        self._call_count += 1
        return response

    def is_available(self) -> bool:
        return True


@pytest.fixture
def theme_dictionary() -> Dictionary:
    """Dictionary with theme-relevant words."""
    words = {
        "EAGLE": 60,
        "KITE": 60,
        "ARROW": 60,
        "AIRBORNE": 60,
        "HAWK": 60,
        "FALCON": 60,
        "PLANE": 60,
        "JET": 60,
        "SOAR": 60,
        "FLY": 60,
        # Some common fill words
        "ACE": 60,
        "ATE": 60,
        "ARE": 60,
        "THE": 60,
        "CAT": 60,
        "DOG": 60,
    }
    return Dictionary(words)


def _make_valid_response(
    topic: str = "Things that fly",
    seed_entries: list[str] | None = None,
    revealer: str = "SOAR",
) -> str:
    if seed_entries is None:
        seed_entries = ["EAGLE", "KITE", "HAWK"]
    return json.dumps({
        "topic": topic,
        "wordplay_type": "literal",
        "seed_entries": seed_entries,
        "revealer": revealer,
        "revealer_clue": "Take flight",
    })


class TestThemeGenerationStep:
    def test_happy_path(self, theme_dictionary: Dictionary) -> None:
        response = _make_valid_response()
        llm = MockLLM([response])
        step = ThemeGenerationStep(
            llm, theme_dictionary, grid_size=9, max_retries=3
        )
        envelope = PuzzleEnvelope(
            puzzle_type=PuzzleType.MIDI, grid_size=9, metadata={"seed": 1}
        )
        result = step.run(envelope)

        assert result.theme is not None
        assert result.theme.topic == "Things that fly"
        assert len(result.theme.seed_entries) == 3
        assert result.theme.revealer == "SOAR"

    def test_step_name(self, theme_dictionary: Dictionary) -> None:
        llm = MockLLM([])
        step = ThemeGenerationStep(llm, theme_dictionary)
        assert step.name == "theme-generation"

    def test_step_history_updated(self, theme_dictionary: Dictionary) -> None:
        response = _make_valid_response()
        llm = MockLLM([response])
        step = ThemeGenerationStep(
            llm, theme_dictionary, grid_size=9
        )
        envelope = PuzzleEnvelope(
            puzzle_type=PuzzleType.MIDI, grid_size=9, metadata={"seed": 1}
        )
        result = step.run(envelope)
        assert "theme-generation" in result.step_history

    def test_rejects_existing_theme(self, theme_dictionary: Dictionary) -> None:
        llm = MockLLM([])
        step = ThemeGenerationStep(llm, theme_dictionary, grid_size=9)
        envelope = PuzzleEnvelope(
            puzzle_type=PuzzleType.MIDI,
            grid_size=9,
            theme=ThemeConcept(topic="existing"),
        )
        with pytest.raises(ValueError, match="already has a theme"):
            step.run(envelope)

    def test_rejects_mini_puzzles(self, theme_dictionary: Dictionary) -> None:
        llm = MockLLM([])
        step = ThemeGenerationStep(llm, theme_dictionary, grid_size=5)
        envelope = PuzzleEnvelope(
            puzzle_type=PuzzleType.MINI, grid_size=5
        )
        with pytest.raises(ValueError, match="only for midi"):
            step.run(envelope)

    def test_retry_on_malformed_json(self, theme_dictionary: Dictionary) -> None:
        valid = _make_valid_response()
        llm = MockLLM(["this is not json", valid])
        step = ThemeGenerationStep(
            llm, theme_dictionary, grid_size=9, max_retries=3
        )
        envelope = PuzzleEnvelope(
            puzzle_type=PuzzleType.MIDI, grid_size=9, metadata={"seed": 1}
        )
        result = step.run(envelope)
        assert result.theme is not None
        assert llm._call_count == 2

    def test_retry_on_entries_not_in_dictionary(
        self, theme_dictionary: Dictionary
    ) -> None:
        bad = _make_valid_response(seed_entries=["XYZZY", "PLUGH", "ZORK"])
        valid = _make_valid_response()
        llm = MockLLM([bad, valid])
        step = ThemeGenerationStep(
            llm, theme_dictionary, grid_size=9, max_retries=3
        )
        envelope = PuzzleEnvelope(
            puzzle_type=PuzzleType.MIDI, grid_size=9, metadata={"seed": 1}
        )
        result = step.run(envelope)
        assert result.theme is not None
        assert llm._call_count == 2

    def test_all_retries_exhausted(self, theme_dictionary: Dictionary) -> None:
        llm = MockLLM(["bad json"] * 3)
        step = ThemeGenerationStep(
            llm, theme_dictionary, grid_size=9, max_retries=3
        )
        envelope = PuzzleEnvelope(
            puzzle_type=PuzzleType.MIDI, grid_size=9, metadata={"seed": 1}
        )
        with pytest.raises(ValueError, match="Failed to generate valid theme"):
            step.run(envelope)

    def test_revealer_validated(self, theme_dictionary: Dictionary) -> None:
        bad = _make_valid_response(revealer="NOTINDICT")
        valid = _make_valid_response()
        llm = MockLLM([bad, valid])
        step = ThemeGenerationStep(
            llm, theme_dictionary, grid_size=9, max_retries=3
        )
        envelope = PuzzleEnvelope(
            puzzle_type=PuzzleType.MIDI, grid_size=9, metadata={"seed": 1}
        )
        result = step.run(envelope)
        assert result.theme is not None
        assert llm._call_count == 2


class TestParseThemeResponse:
    def test_valid_json(self) -> None:
        response = _make_valid_response()
        theme = _parse_theme_response(response)
        assert theme.topic == "Things that fly"
        assert theme.seed_entries == ["EAGLE", "KITE", "HAWK"]
        assert theme.revealer == "SOAR"

    def test_markdown_fenced_json(self) -> None:
        inner = _make_valid_response()
        response = f"```json\n{inner}\n```"
        theme = _parse_theme_response(response)
        assert theme.topic == "Things that fly"

    def test_json_with_preamble(self) -> None:
        inner = _make_valid_response()
        response = f"Here is a theme:\n{inner}\nHope you like it!"
        theme = _parse_theme_response(response)
        assert theme.topic == "Things that fly"

    def test_no_json_raises(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            _parse_theme_response("No JSON here")

    def test_missing_topic_raises(self) -> None:
        response = json.dumps({"seed_entries": ["A"], "revealer": "B"})
        with pytest.raises(KeyError):
            _parse_theme_response(response)

    def test_empty_topic_raises(self) -> None:
        response = json.dumps({
            "topic": "",
            "seed_entries": ["A"],
            "revealer": "B",
        })
        with pytest.raises(ValueError, match="topic is empty"):
            _parse_theme_response(response)

    def test_entries_uppercased(self) -> None:
        response = json.dumps({
            "topic": "test",
            "seed_entries": ["eagle", "kite"],
            "revealer": "soar",
        })
        theme = _parse_theme_response(response)
        assert theme.seed_entries == ["EAGLE", "KITE"]
        assert theme.revealer == "SOAR"


class TestValidateThemeEntries:
    def test_valid_entries(self, theme_dictionary: Dictionary) -> None:
        theme = ThemeConcept(
            topic="test",
            seed_entries=["EAGLE", "KITE", "HAWK"],
            revealer="SOAR",
        )
        errors = _validate_theme_entries(theme, theme_dictionary, 9, [3, 4, 5])
        assert errors == []

    def test_word_not_in_dictionary(self, theme_dictionary: Dictionary) -> None:
        theme = ThemeConcept(
            topic="test",
            seed_entries=["XYZZY"],
            revealer="SOAR",
        )
        errors = _validate_theme_entries(theme, theme_dictionary, 9, [4, 5])
        assert any("not in the dictionary" in e for e in errors)

    def test_word_too_long(self, theme_dictionary: Dictionary) -> None:
        theme = ThemeConcept(
            topic="test",
            seed_entries=["EAGLE"],
            revealer="SOAR",
        )
        errors = _validate_theme_entries(theme, theme_dictionary, 4, [4, 5])
        assert any("outside range" in e for e in errors)

    def test_word_too_short(self, theme_dictionary: Dictionary) -> None:
        # Create a dictionary with a 2-letter word
        d = Dictionary({"AB": 60, "SOAR": 60}, min_2letter_score=30)
        theme = ThemeConcept(
            topic="test",
            seed_entries=["AB"],
            revealer="SOAR",
        )
        errors = _validate_theme_entries(theme, d, 9, [2, 4])
        assert any("outside range" in e for e in errors)

    def test_duplicate_entries(self, theme_dictionary: Dictionary) -> None:
        theme = ThemeConcept(
            topic="test",
            seed_entries=["EAGLE", "EAGLE"],
            revealer="SOAR",
        )
        errors = _validate_theme_entries(theme, theme_dictionary, 9, [4, 5])
        assert any("Duplicate" in e for e in errors)

    def test_length_not_in_available_slots(
        self, theme_dictionary: Dictionary
    ) -> None:
        theme = ThemeConcept(
            topic="test",
            seed_entries=["EAGLE"],
            revealer="SOAR",
        )
        # Only length 3 and 4 available, but EAGLE is 5
        errors = _validate_theme_entries(theme, theme_dictionary, 9, [3, 4])
        assert any("doesn't match any available" in e for e in errors)
