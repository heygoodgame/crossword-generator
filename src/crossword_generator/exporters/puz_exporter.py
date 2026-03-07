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
        """Export the puzzle to a .puz file.

        Args:
            envelope: The puzzle envelope with at least a filled grid.
            output_path: Directory to write the file to.

        Returns:
            Path to the written .puz file.
        """
        if envelope.fill is None:
            raise ValueError("Cannot export: envelope has no fill result")

        grid = envelope.fill.grid
        rows = len(grid)
        cols = len(grid[0])

        p = puz.Puzzle()
        p.height = rows
        p.width = cols
        p.title = f"{envelope.puzzle_type.value.title()} Crossword"
        p.author = "Crossword Generator"

        # Solution: flat string, "." for black squares
        p.solution = "".join(
            cell if cell != "." else "." for row in grid for cell in row
        )

        # Fill: "-" for unsolved letter cells, "." for black squares
        p.fill = "".join("." if cell == "." else "-" for row in grid for cell in row)

        # Build clues in standard .puz order:
        # sorted by number, across before down at the same number
        numbered = compute_numbering(grid)

        # Map existing clues from envelope
        clue_map: dict[tuple[int, str], str] = {}
        for clue_entry in envelope.clues:
            clue_map[(clue_entry.number, clue_entry.direction)] = clue_entry.clue

        clue_list: list[str] = []
        for entry in numbered:
            clue_text = clue_map.get((entry.number, entry.direction), "")
            clue_list.append(clue_text)

        p.clues = clue_list

        # Write file
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        filename = f"{envelope.puzzle_type.value}_{rows}x{cols}_{timestamp}.puz"
        filepath = output_path / filename
        p.save(str(filepath))
        logger.info("Exported .puz to %s", filepath)
        return filepath
