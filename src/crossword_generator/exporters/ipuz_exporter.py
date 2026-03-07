"""Export puzzles to .ipuz format."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import ipuz

from crossword_generator.exporters.base import Exporter
from crossword_generator.exporters.numbering import compute_numbering
from crossword_generator.models import PuzzleEnvelope

logger = logging.getLogger(__name__)


class IpuzExporter(Exporter):
    """Export a PuzzleEnvelope to .ipuz format."""

    @property
    def file_extension(self) -> str:
        return ".ipuz"

    def export(self, envelope: PuzzleEnvelope, output_path: Path) -> Path:
        """Export the puzzle to an .ipuz file.

        Args:
            envelope: The puzzle envelope with at least a filled grid.
            output_path: Directory to write the file to.

        Returns:
            Path to the written .ipuz file.
        """
        if envelope.fill is None:
            raise ValueError("Cannot export: envelope has no fill result")

        grid = envelope.fill.grid
        rows = len(grid)
        cols = len(grid[0])

        numbered = compute_numbering(grid)

        # Build a number lookup: (row, col) → cell_number
        number_map: dict[tuple[int, int], int] = {}
        for entry in numbered:
            number_map.setdefault((entry.row, entry.col), entry.number)

        # puzzle grid: cell numbers or "#" for black squares, 0 for unnumbered
        puzzle_grid: list[list[int | str]] = []
        for r in range(rows):
            row_data: list[int | str] = []
            for c in range(cols):
                if grid[r][c] == ".":
                    row_data.append("#")
                else:
                    num = number_map.get((r, c), 0)
                    row_data.append(num)
            puzzle_grid.append(row_data)

        # solution grid: letters or "#" for black squares
        solution_grid: list[list[str]] = []
        for r in range(rows):
            row_data_str: list[str] = []
            for c in range(cols):
                if grid[r][c] == ".":
                    row_data_str.append("#")
                else:
                    row_data_str.append(grid[r][c])
            solution_grid.append(row_data_str)

        # Build clues
        clue_map: dict[tuple[int, str], str] = {}
        for clue_entry in envelope.clues:
            clue_map[(clue_entry.number, clue_entry.direction)] = clue_entry.clue

        across_clues: list[list[int | str]] = []
        down_clues: list[list[int | str]] = []
        for entry in numbered:
            clue_text = clue_map.get((entry.number, entry.direction), "")
            pair: list[int | str] = [entry.number, clue_text]
            if entry.direction == "across":
                across_clues.append(pair)
            else:
                down_clues.append(pair)

        ipuz_dict = {
            "version": "http://ipuz.org/v2",
            "kind": ["http://ipuz.org/crossword#1"],
            "dimensions": {"width": cols, "height": rows},
            "title": f"{envelope.puzzle_type.value.title()} Crossword",
            "author": "Crossword Generator",
            "puzzle": puzzle_grid,
            "solution": solution_grid,
            "clues": {
                "Across": across_clues,
                "Down": down_clues,
            },
        }

        # Validate via ipuz library
        validated = ipuz.read(json.dumps(ipuz_dict))

        # Write file
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        filename = f"{envelope.puzzle_type.value}_{rows}x{cols}_{timestamp}.ipuz"
        filepath = output_path / filename
        filepath.write_text(ipuz.write(validated))
        logger.info("Exported .ipuz to %s", filepath)
        return filepath
