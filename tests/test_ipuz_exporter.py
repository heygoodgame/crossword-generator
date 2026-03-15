"""Tests for .ipuz exporter."""

import json
from pathlib import Path

import ipuz
import pytest

from crossword_generator.exporters.ipuz_exporter import IpuzExporter
from crossword_generator.models import (
    ClueEntry,
    FillResult,
    PuzzleEnvelope,
    PuzzleType,
    ThemeConcept,
)


@pytest.fixture
def sample_grid() -> list[list[str]]:
    return [
        ["O", "H", ".", ".", "."],
        ["F", ".", "P", ".", "F"],
        [".", "H", "O", "W", "L"],
        ["A", ".", "S", ".", "I"],
        ["S", "L", "E", "E", "P"],
    ]


@pytest.fixture
def envelope_with_fill(sample_grid: list[list[str]]) -> PuzzleEnvelope:
    return PuzzleEnvelope(
        puzzle_type=PuzzleType.MINI,
        grid_size=5,
        fill=FillResult(grid=sample_grid, filler_used="test"),
    )


@pytest.fixture
def envelope_with_clues(envelope_with_fill: PuzzleEnvelope) -> PuzzleEnvelope:
    clues = [
        ClueEntry(
            number=1, direction="across", answer="OH", clue="Expression of surprise"
        ),
        ClueEntry(number=1, direction="down", answer="OF", clue="Belonging to"),
    ]
    return envelope_with_fill.model_copy(update={"clues": clues})


class TestIpuzExporter:
    def test_export_creates_file(
        self, envelope_with_fill: PuzzleEnvelope, tmp_path: Path
    ) -> None:
        exporter = IpuzExporter()
        path = exporter.export(envelope_with_fill, tmp_path)
        assert path.exists()
        assert path.suffix == ".ipuz"

    def test_file_extension(self) -> None:
        assert IpuzExporter().file_extension == ".ipuz"

    def test_round_trip_without_clues(
        self, envelope_with_fill: PuzzleEnvelope, tmp_path: Path
    ) -> None:
        exporter = IpuzExporter()
        path = exporter.export(envelope_with_fill, tmp_path)

        data = ipuz.read(path.read_text())
        assert data["dimensions"]["width"] == 5
        assert data["dimensions"]["height"] == 5

    def test_round_trip_with_clues(
        self, envelope_with_clues: PuzzleEnvelope, tmp_path: Path
    ) -> None:
        exporter = IpuzExporter()
        path = exporter.export(envelope_with_clues, tmp_path)

        data = ipuz.read(path.read_text())
        across_clues = data["clues"]["Across"]
        # Find the clue with number 1
        clue_1 = [c for c in across_clues if c[0] == 1]
        assert len(clue_1) == 1
        assert clue_1[0][1] == "Expression of surprise"

    def test_black_squares_in_puzzle_grid(
        self, envelope_with_fill: PuzzleEnvelope, tmp_path: Path
    ) -> None:
        exporter = IpuzExporter()
        path = exporter.export(envelope_with_fill, tmp_path)

        data = ipuz.read(path.read_text())
        puzzle_grid = data["puzzle"]
        # Row 0, col 2 should be "#" (black square)
        assert puzzle_grid[0][2] == "#"

    def test_solution_grid(
        self, envelope_with_fill: PuzzleEnvelope, tmp_path: Path
    ) -> None:
        exporter = IpuzExporter()
        path = exporter.export(envelope_with_fill, tmp_path)

        data = ipuz.read(path.read_text())
        solution = data["solution"]
        assert solution[0][0] == "O"
        assert solution[0][2] == "#"

    def test_numbered_cells(
        self, envelope_with_fill: PuzzleEnvelope, tmp_path: Path
    ) -> None:
        exporter = IpuzExporter()
        path = exporter.export(envelope_with_fill, tmp_path)

        data = ipuz.read(path.read_text())
        puzzle_grid = data["puzzle"]
        # Cell (0,0) should have a number > 0
        assert puzzle_grid[0][0] > 0

    def test_no_fill_raises(self, tmp_path: Path) -> None:
        envelope = PuzzleEnvelope()
        exporter = IpuzExporter()
        with pytest.raises(ValueError, match="no fill result"):
            exporter.export(envelope, tmp_path)

    def test_filename_format(
        self, envelope_with_fill: PuzzleEnvelope, tmp_path: Path
    ) -> None:
        exporter = IpuzExporter()
        path = exporter.export(envelope_with_fill, tmp_path)
        assert path.name.startswith("mini_5x5_")
        assert path.name.endswith(".ipuz")

    def test_valid_ipuz_json(
        self, envelope_with_fill: PuzzleEnvelope, tmp_path: Path
    ) -> None:
        exporter = IpuzExporter()
        path = exporter.export(envelope_with_fill, tmp_path)
        # Should be valid JSON
        data = json.loads(path.read_text())
        assert "version" in data
        assert "kind" in data


@pytest.fixture
def themed_envelope(sample_grid: list[list[str]]) -> PuzzleEnvelope:
    """Envelope with theme, fill, and clues for reference testing.

    Sample grid words: 1-Across OH, 1-Down OF, 2-Down POSE, 3-Down FLIP,
    4-Across HOWL, 5-Down AS, 6-Across SLEEP.
    """
    clues = [
        ClueEntry(number=1, direction="across", answer="OH", clue="Surprise!"),
        ClueEntry(number=1, direction="down", answer="OF", clue="Belonging to"),
        ClueEntry(number=2, direction="down", answer="POSE", clue="Strike a ___"),
        ClueEntry(number=3, direction="down", answer="FLIP", clue="Turn over"),
        ClueEntry(number=4, direction="across", answer="HOWL", clue="Wolf cry"),
        ClueEntry(number=5, direction="down", answer="AS", clue="Like"),
        ClueEntry(number=6, direction="across", answer="SLEEP", clue="Rest"),
    ]
    theme = ThemeConcept(
        topic="Night time",
        revealer="SLEEP",
        seed_entries=["HOWL", "FLIP"],
    )
    return PuzzleEnvelope(
        puzzle_type=PuzzleType.MIDI,
        grid_size=5,
        fill=FillResult(grid=sample_grid, filler_used="test"),
        clues=clues,
        theme=theme,
    )


class TestClueReferences:
    def test_themed_puzzle_has_references(
        self, themed_envelope: PuzzleEnvelope, tmp_path: Path
    ) -> None:
        exporter = IpuzExporter()
        path = exporter.export(themed_envelope, tmp_path)
        data = json.loads(path.read_text())

        assert "hgg.references" in data
        refs = data["hgg.references"]
        assert len(refs) == 3  # 1 revealer + 2 theme entries

    def test_no_theme_no_references(
        self, envelope_with_fill: PuzzleEnvelope, tmp_path: Path
    ) -> None:
        exporter = IpuzExporter()
        path = exporter.export(envelope_with_fill, tmp_path)
        data = json.loads(path.read_text())

        assert "hgg.references" not in data

    def test_partial_seed_placement(
        self, sample_grid: list[list[str]], tmp_path: Path
    ) -> None:
        """Seeds not in the grid are excluded from references."""
        clues = [
            ClueEntry(number=4, direction="across", answer="HOWL", clue="Wolf cry"),
            ClueEntry(number=6, direction="across", answer="SLEEP", clue="Rest"),
        ]
        theme = ThemeConcept(
            topic="Night time",
            revealer="SLEEP",
            # HOWL is in the grid, MOON and STARS are not
            seed_entries=["HOWL", "MOON", "STARS"],
        )
        envelope = PuzzleEnvelope(
            puzzle_type=PuzzleType.MIDI,
            grid_size=5,
            fill=FillResult(grid=sample_grid, filler_used="test"),
            clues=clues,
            theme=theme,
        )
        exporter = IpuzExporter()
        path = exporter.export(envelope, tmp_path)
        data = json.loads(path.read_text())

        refs = data["hgg.references"]
        revealer = refs[0]
        assert revealer["role"] == "revealer"
        # Only HOWL should appear, not MOON or STARS
        assert len(revealer["references"]) == 1
        assert revealer["references"][0] == [4, "Across"]

    def test_references_are_bidirectional(
        self, themed_envelope: PuzzleEnvelope, tmp_path: Path
    ) -> None:
        exporter = IpuzExporter()
        path = exporter.export(themed_envelope, tmp_path)
        data = json.loads(path.read_text())

        refs = data["hgg.references"]
        revealer = next(r for r in refs if r["role"] == "revealer")
        theme_entries = [r for r in refs if r["role"] == "theme_entry"]

        # Revealer references both theme entries
        assert [4, "Across"] in revealer["references"]
        assert [3, "Down"] in revealer["references"]

        # Each theme entry references the revealer
        for entry in theme_entries:
            assert entry["references"] == [[6, "Across"]]

    def test_revealer_clue_format(
        self, themed_envelope: PuzzleEnvelope, tmp_path: Path
    ) -> None:
        exporter = IpuzExporter()
        path = exporter.export(themed_envelope, tmp_path)
        data = json.loads(path.read_text())

        refs = data["hgg.references"]
        revealer = refs[0]
        assert revealer["clue"] == [6, "Across"]
        assert revealer["role"] == "revealer"
