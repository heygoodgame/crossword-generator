"""Tests for the ThemeGenerationStep pipeline step."""

from __future__ import annotations

import json

import pytest

from crossword_generator.dictionary import Dictionary
from crossword_generator.llm.base import LLMProvider
from crossword_generator.llm.prompts.theme_generation import (
    build_theme_generation_prompt,
)
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
        # With default num_candidates=6 > num_seed_entries=3,
        # entries go to candidate_entries and seed_entries is empty
        assert len(result.theme.candidate_entries) == 3
        assert result.theme.seed_entries == []
        assert result.theme.revealer == "SOAR"

    def test_happy_path_no_surplus(self, theme_dictionary: Dictionary) -> None:
        """When num_candidates == num_seed_entries, seed_entries is populated."""
        response = _make_valid_response()
        llm = MockLLM([response])
        step = ThemeGenerationStep(
            llm, theme_dictionary, grid_size=9, max_retries=3,
            num_candidates=3,  # same as num_seed_entries
        )
        envelope = PuzzleEnvelope(
            puzzle_type=PuzzleType.MIDI, grid_size=9, metadata={"seed": 1}
        )
        result = step.run(envelope)

        assert result.theme is not None
        assert len(result.theme.seed_entries) == 3
        assert result.theme.candidate_entries == []
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

    def test_json_with_trailing_text_and_braces(self) -> None:
        """Handles extra text with braces after the JSON object."""
        inner = _make_valid_response()
        response = (
            f"{inner}\n\nNote: you could also try "
            '{"alternative": "theme"} for variety.'
        )
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

    def test_length_not_in_available_slots_strict(
        self, theme_dictionary: Dictionary
    ) -> None:
        """In strict mode, seed entries must match available slot lengths."""
        theme = ThemeConcept(
            topic="test",
            seed_entries=["EAGLE"],
            revealer="SOAR",
        )
        # Only length 3 and 4 available, but EAGLE is 5
        errors = _validate_theme_entries(theme, theme_dictionary, 9, [3, 4])
        assert any("doesn't fit any grid slot" in e for e in errors)

    def test_length_not_in_available_slots_relaxed(
        self, theme_dictionary: Dictionary
    ) -> None:
        """In relaxed mode, seed entry slot-length mismatch is accepted."""
        theme = ThemeConcept(
            topic="test",
            seed_entries=["EAGLE", "KITE", "HAWK"],
            revealer="SOAR",
        )
        # Only length 3 and 4 available — EAGLE is 5, but relaxed mode
        # skips the slot-length check for candidates. All are in dict
        # and within 3-9 range.
        errors = _validate_theme_entries(
            theme, theme_dictionary, 9, [3, 4], min_valid_entries=2
        )
        assert errors == []
        # All 3 should be valid (no slot-length filtering)
        assert theme.seed_entries == ["EAGLE", "KITE", "HAWK"]

    def test_relaxed_still_checks_dictionary(
        self, theme_dictionary: Dictionary
    ) -> None:
        """In relaxed mode, words not in dictionary are still filtered."""
        theme = ThemeConcept(
            topic="test",
            seed_entries=["EAGLE", "XYZZY", "HAWK"],
            revealer="SOAR",
        )
        errors = _validate_theme_entries(
            theme, theme_dictionary, 9, [3, 4], min_valid_entries=2
        )
        assert errors == []
        # XYZZY filtered out, EAGLE and HAWK remain
        assert theme.seed_entries == ["EAGLE", "HAWK"]


    def test_error_message_does_not_expose_slot_lengths(
        self, theme_dictionary: Dictionary
    ) -> None:
        """Sanitized error messages should not leak available_lengths."""
        theme = ThemeConcept(
            topic="test",
            seed_entries=["EAGLE"],
            revealer="AIRBORNE",
        )
        # AIRBORNE is 8 letters, not in available_lengths [3, 4, 5]
        errors = _validate_theme_entries(theme, theme_dictionary, 9, [3, 4, 5])
        for e in errors:
            assert "available_lengths" not in e
            assert "[3, 4, 5]" not in e
        # But should still report the issue
        assert any("doesn't fit any grid slot" in e for e in errors)


class TestBuildThemeGenerationPrompt:
    def test_prompt_has_length_constraints(self) -> None:
        prompt = build_theme_generation_prompt(
            grid_size=9,
            available_slot_lengths=[3, 5, 9],
        )
        # Relaxed prompt should mention 3-9 range, not strict slot lengths
        assert "between 3 and 9" in prompt
        assert "revealer" in prompt.lower()

    def test_prompt_no_strict_slot_constraint(self) -> None:
        prompt = build_theme_generation_prompt(
            grid_size=9,
            available_slot_lengths=[3, 5, 9],
        )
        # Should NOT have the old "ONLY allowed word lengths" wording
        assert "ONLY allowed word lengths" not in prompt

    def test_prompt_has_length_diversity_guidance(self) -> None:
        prompt = build_theme_generation_prompt(grid_size=9)
        assert "3 DIFFERENT lengths" in prompt
        assert "short words (3-4 letters)" in prompt
        assert "7-9 letters" in prompt

    def test_prompt_requests_candidate_count(self) -> None:
        prompt = build_theme_generation_prompt(
            grid_size=9,
            num_candidates=12,
            num_seed_entries=3,
        )
        assert "12" in prompt
        assert "seed entries" in prompt
