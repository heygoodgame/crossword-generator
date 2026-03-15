"""Export puzzles to .puz format via puzpy."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import puz

from crossword_generator.exporters.base import Exporter
from crossword_generator.exporters.numbering import compute_numbering
from crossword_generator.models import PuzzleEnvelope

logger = logging.getLogger(__name__)


class PuzExporter(Exporter):
    """Export a PuzzleEnvelope to .puz format."""

    @property
    def file_extension(self) -> str:
        return ".puz"

    def export(self, envelope: PuzzleEnvelope, output_path: Path) -> Path:
        """Export the puzzle to a .puz file in output_path directory."""
        if envelope.fill is None:
            raise ValueError("Cannot export: envelope has no fill result")
        output_path.mkdir(parents=True, exist_ok=True)
        grid = envelope.fill.grid
        rows = len(grid)
        cols = len(grid[0])
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        filename = f"{envelope.puzzle_type.value}_{rows}x{cols}_{timestamp}.puz"
        return self._write(envelope, output_path / filename)

    def _write(self, envelope: PuzzleEnvelope, filepath: Path) -> Path:
        if envelope.fill is None:
            raise ValueError("Cannot export: envelope has no fill result")

        grid = envelope.fill.grid
        rows = len(grid)
        cols = len(grid[0])

        p = puz.Puzzle()
        p.height = rows
        p.width = cols
        p.title = envelope.title or f"{envelope.puzzle_type.value.title()} Crossword"
        p.author = "Hey Good Game, Inc."

        p.solution = "".join(
            cell if cell != "." else "." for row in grid for cell in row
        )
        p.fill = "".join("." if cell == "." else "-" for row in grid for cell in row)

        numbered = compute_numbering(grid)

        clue_map: dict[tuple[int, str], str] = {}
        for clue_entry in envelope.clues:
            clue_map[(clue_entry.number, clue_entry.direction)] = clue_entry.clue

        clue_list: list[str] = []
        for entry in numbered:
            clue_text = clue_map.get((entry.number, entry.direction), "")
            clue_list.append(clue_text)

        p.clues = clue_list

        p.save(str(filepath))
        logger.info("Exported .puz to %s", filepath)
        return filepath
