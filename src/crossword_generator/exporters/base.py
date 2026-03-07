"""Abstract base class for puzzle exporters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from crossword_generator.models import PuzzleEnvelope


class Exporter(ABC):
    """Abstract interface for exporting puzzles to file formats."""

    @property
    @abstractmethod
    def file_extension(self) -> str:
        """File extension for this format (e.g., '.puz')."""

    @abstractmethod
    def export(self, envelope: PuzzleEnvelope, output_path: Path) -> Path:
        """Export a puzzle to this format.

        Args:
            envelope: The completed puzzle data.
            output_path: Directory to write the output file.

        Returns:
            Path to the written file.
        """
