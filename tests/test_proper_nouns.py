"""Tests for proper-noun classification helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from crossword_generator.proper_nouns import (
    build_classification_prompt,
    classify_words,
    load_classifications,
    load_proper_noun_set,
    parse_classification_response,
    save_classifications,
)


class TestParseClassificationResponse:
    def test_parses_labeled_lines(self) -> None:
        raw = "AARON P\nAMBER C\nOREO P\n"
        labels = parse_classification_response(raw, ["AARON", "AMBER", "OREO"])
        assert labels == {"AARON": "P", "AMBER": "C", "OREO": "P"}

    def test_ignores_unexpected_words_and_noise(self) -> None:
        raw = "Here you go:\nAARON P\nZEBRA C\nAMBER maybe\n"
        labels = parse_classification_response(raw, ["AARON", "AMBER"])
        assert labels == {"AARON": "P"}

    def test_case_insensitive(self) -> None:
        raw = "aaron p"
        labels = parse_classification_response(raw, ["AARON"])
        assert labels == {"AARON": "P"}


class TestClassifyWords:
    def test_requeues_omitted_words(self) -> None:
        calls: list[str] = []

        class FakeProvider:
            def generate(self, prompt: str, **kwargs: object) -> str:
                calls.append(prompt)
                # First call omits AMBER; the retry round labels it.
                if len(calls) == 1:
                    return "AARON P"
                return "AMBER C"

        labels = classify_words(
            FakeProvider(),  # type: ignore[arg-type]
            ["aaron", "amber"],
            batch_size=10,
            max_workers=1,
        )
        assert labels == {"AARON": "P", "AMBER": "C"}
        assert len(calls) == 2

    def test_unresolvable_words_are_omitted(self) -> None:
        class SilentProvider:
            def generate(self, prompt: str, **kwargs: object) -> str:
                return ""

        labels = classify_words(
            SilentProvider(),  # type: ignore[arg-type]
            ["AARON"],
            batch_size=10,
            max_workers=1,
        )
        assert labels == {}


class TestClassificationFile:
    def test_round_trip_sorted(self, tmp_path: Path) -> None:
        path = tmp_path / "classifications.txt"
        save_classifications(path, {"OREO": "P", "AMBER": "C"})
        assert path.read_text() == "AMBER;C\nOREO;P\n"
        assert load_classifications(path) == {"AMBER": "C", "OREO": "P"}

    def test_load_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert load_classifications(tmp_path / "missing.txt") == {}

    def test_load_proper_noun_set_filters_common(self, tmp_path: Path) -> None:
        path = tmp_path / "classifications.txt"
        path.write_text("AMBER;C\nOREO;P\nNBA;P\n")
        assert load_proper_noun_set(path) == frozenset({"OREO", "NBA"})

    def test_load_proper_noun_set_missing_file_raises(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError):
            load_proper_noun_set(tmp_path / "missing.txt")


def test_prompt_lists_words() -> None:
    prompt = build_classification_prompt(["AARON", "AMBER"])
    assert "AARON" in prompt
    assert "AMBER" in prompt
