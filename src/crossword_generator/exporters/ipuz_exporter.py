"""Export puzzles to .ipuz format."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
        """Export the puzzle to an .ipuz file in output_path directory."""
        if envelope.fill is None:
            raise ValueError("Cannot export: envelope has no fill result")
        output_path.mkdir(parents=True, exist_ok=True)
        grid = envelope.fill.grid
        rows = len(grid)
        cols = len(grid[0])
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        filename = f"{envelope.puzzle_type.value}_{rows}x{cols}_{timestamp}.ipuz"
        return self._write(envelope, output_path / filename)

    def _write(self, envelope: PuzzleEnvelope, filepath: Path) -> Path:
        if envelope.fill is None:
            raise ValueError("Cannot export: envelope has no fill result")

        grid = envelope.fill.grid
        rows = len(grid)
        cols = len(grid[0])

        numbered = compute_numbering(grid)

        number_map: dict[tuple[int, int], int] = {}
        for entry in numbered:
            number_map.setdefault((entry.row, entry.col), entry.number)

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

        solution_grid: list[list[str]] = []
        for r in range(rows):
            row_data_str: list[str] = []
            for c in range(cols):
                if grid[r][c] == ".":
                    row_data_str.append("#")
                else:
                    row_data_str.append(grid[r][c])
            solution_grid.append(row_data_str)

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
            "title": (
                envelope.title
                or f"{envelope.puzzle_type.value.title()} Crossword"
            ),
            "author": "Hey Good Game, Inc.",
            "puzzle": puzzle_grid,
            "solution": solution_grid,
            "clues": {
                "Across": across_clues,
                "Down": down_clues,
            },
        }

        refs = self._build_clue_references(envelope)
        if refs:
            ipuz_dict["hgg.references"] = refs

        validated = ipuz.read(json.dumps(ipuz_dict))

        filepath.write_text(ipuz.write(validated))
        logger.info("Exported .ipuz to %s", filepath)
        return filepath

    def _build_clue_references(
        self, envelope: PuzzleEnvelope
    ) -> list[dict[str, Any]] | None:
        """Build clue cross-reference data for themed puzzles.

        Returns None if the envelope has no theme, no revealer, or no
        matching seed entries in the clue list.
        """
        if envelope.theme is None or not envelope.theme.revealer:
            return None

        theme = envelope.theme

        # Build answer -> (number, direction) lookup from clue list
        answer_to_clue: dict[str, tuple[int, str]] = {}
        for clue in envelope.clues:
            answer_to_clue[clue.answer.upper()] = (clue.number, clue.direction)

        # Find revealer clue
        revealer_key = theme.revealer.upper()
        if revealer_key not in answer_to_clue:
            return None
        revealer_num, revealer_dir = answer_to_clue[revealer_key]

        # Find theme entry clues (only seeds that appear in the grid)
        theme_clues: list[tuple[int, str]] = []
        for seed in theme.seed_entries:
            key = seed.upper()
            if key in answer_to_clue and key != revealer_key:
                theme_clues.append(answer_to_clue[key])

        if not theme_clues:
            return None

        def _fmt(number: int, direction: str) -> list[int | str]:
            return [number, direction.capitalize()]

        refs: list[dict[str, Any]] = []

        # Revealer → all theme entries
        refs.append({
            "clue": _fmt(revealer_num, revealer_dir),
            "role": "revealer",
            "references": [_fmt(n, d) for n, d in theme_clues],
        })

        # Each theme entry → revealer
        for num, d in theme_clues:
            refs.append({
                "clue": _fmt(num, d),
                "role": "theme_entry",
                "references": [_fmt(revealer_num, revealer_dir)],
            })

        return refs
