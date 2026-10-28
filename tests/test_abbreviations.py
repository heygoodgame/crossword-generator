"""Tests for abbreviation classification helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from crossword_generator.abbreviations import (
    abbreviation_cap_for_size,
    classify_abbreviations,
    load_abbreviation_classifications,
    load_abbreviation_set,
    save_abbreviation_classifications,
)


class TestClassifyAbbreviations:
    def test_uses_abbreviation_prompt_and_labels(self) -> None:
        systems: list[str] = []

        class FakeProvider:
            def generate(self, prompt: str, **kwargs: object) -> str:
                systems.append(str(kwargs["system"]))
                # A proper-noun label is not a valid abbreviation label.
                return "ETA A\nMEMO W\nOREO P"

        labels = classify_abbreviations(
            FakeProvider(),  # type: ignore[arg-type]
            ["eta", "memo"],
            batch_size=10,
            max_workers=1,
        )
        assert labels == {"ETA": "A", "MEMO": "W"}
        assert "abbreviations or ordinary words" in systems[0]


class TestLoadAndSave:
    def test_round_trip_ignores_other_labels(self, tmp_path: Path) -> None:
        path = tmp_path / "abbr.txt"
        path.write_text("ETA;A\nMEMO;W\nOREO;P\n")
        assert load_abbreviation_classifications(path) == {
            "ETA": "A",
            "MEMO": "W",
        }
        save_abbreviation_classifications(path, {"NBA": "A", "AMP": "W"})
        assert path.read_text() == "AMP;W\nNBA;A\n"
        assert load_abbreviation_set(path) == frozenset({"NBA"})

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_abbreviation_set(tmp_path / "missing.txt")

    def test_committed_file_labels_jeffs_examples(
        self, project_root: Path
    ) -> None:
        abbreviations = load_abbreviation_set(
            project_root / "dictionaries" / "HggAbbreviationClassifications.txt"
        )
        for word in ("ETA", "NBA", "OTS", "IDS", "ATM", "SIS"):
            assert word in abbreviations, word
        for word in ("AMP", "MEMO", "INFO", "PRO", "SCUBA"):
            assert word not in abbreviations, word


class TestCapForSize:
    CAPS = {5: 1, 7: 2, 9: 3}

    @pytest.mark.parametrize(
        ("size", "cap"), [(5, 1), (7, 2), (9, 3), (11, 3), (6, 1), (4, None)]
    )
    def test_cap_lookup(self, size: int, cap: int | None) -> None:
        assert abbreviation_cap_for_size(self.CAPS, size) == cap

    def test_empty_map_is_uncapped(self) -> None:
        assert abbreviation_cap_for_size({}, 9) is None
