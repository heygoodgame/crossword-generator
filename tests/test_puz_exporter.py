"""Tests for .puz exporter."""

from pathlib import Path

import puz
import pytest

from crossword_generator.exporters.puz_exporter import PuzExporter
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


class TestPuzExporter:
    def test_export_creates_file(
        self, envelope_with_fill: PuzzleEnvelope, tmp_path: Path
    ) -> None:
        exporter = PuzExporter()
        path = exporter.export(envelope_with_fill, tmp_path)
        assert path.exists()
        assert path.suffix == ".puz"

    def test_file_extension(self) -> None:
        assert PuzExporter().file_extension == ".puz"

    def test_round_trip_without_clues(
        self, envelope_with_fill: PuzzleEnvelope, tmp_path: Path
    ) -> None:
        exporter = PuzExporter()
        path = exporter.export(envelope_with_fill, tmp_path)

        p = puz.read(str(path))
        assert p.width == 5
        assert p.height == 5
        # Solution should have letters and "."
        assert p.solution[0] == "O"
        assert p.solution[1] == "H"
        assert p.solution[2] == "."

    def test_round_trip_with_clues(
        self, envelope_with_clues: PuzzleEnvelope, tmp_path: Path
    ) -> None:
        exporter = PuzExporter()
        path = exporter.export(envelope_with_clues, tmp_path)

        p = puz.read(str(path))
        assert "Expression of surprise" in p.clues
        assert "Belonging to" in p.clues

    def test_fill_string(
        self, envelope_with_fill: PuzzleEnvelope, tmp_path: Path
    ) -> None:
        exporter = PuzExporter()
        path = exporter.export(envelope_with_fill, tmp_path)

        p = puz.read(str(path))
        for i, char in enumerate(p.fill):
            if p.solution[i] == ".":
                assert char == "."
            else:
                assert char == "-"

    def test_no_fill_raises(self, tmp_path: Path) -> None:
        envelope = PuzzleEnvelope()
        exporter = PuzExporter()
        with pytest.raises(ValueError, match="no fill result"):
            exporter.export(envelope, tmp_path)

    def test_filename_format(
        self, envelope_with_fill: PuzzleEnvelope, tmp_path: Path
    ) -> None:
        exporter = PuzExporter()
        path = exporter.export(envelope_with_fill, tmp_path)
        assert path.name.startswith("mini_5x5_")
        assert path.name.endswith(".puz")
