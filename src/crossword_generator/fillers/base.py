"""Abstract base class for grid fillers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class GridSpec:
    """Specification for a crossword grid."""

    rows: int
    cols: int
    black_cells: list[tuple[int, int]] = field(default_factory=list)
    seed_entries: dict[str, str] = field(default_factory=dict)


@dataclass
class FilledGrid:
    """Result of filling a grid."""

    grid: list[list[str]]
    words_across: list[str] = field(default_factory=list)
    words_down: list[str] = field(default_factory=list)


class GridFiller(ABC):
    """Abstract interface for grid filling backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this filler implementation."""

    @abstractmethod
    def fill(self, spec: GridSpec) -> FilledGrid:
        """Fill a grid according to the given specification.

        Args:
            spec: The grid specification including dimensions and constraints.

        Returns:
            A filled grid with all entries populated.

        Raises:
            FillError: If the grid cannot be filled with valid words.
        """

    def is_available(self) -> bool:
        """Check if this filler's dependencies are available."""
        return True


class FillError(Exception):
    """Raised when a grid cannot be filled."""
