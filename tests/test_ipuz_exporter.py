"""Tests for .ipuz exporter."""

import json
from pathlib import Path

import ipuz
import pytest

from crossword_generator.exporters.ipuz_exporter import IpuzExporter
from crossword_generator.models import ClueEntry, FillResult, PuzzleEnvelope, PuzzleType


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
