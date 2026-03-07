"""Abstract base class for pipeline steps."""

from __future__ import annotations

from abc import ABC, abstractmethod

from crossword_generator.models import PuzzleEnvelope


class PipelineStep(ABC):
    """A single step in the crossword generation pipeline.

    Each step reads a PuzzleEnvelope, performs its work, and returns
    the updated envelope. Steps are stateless.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this step."""

    @abstractmethod
    def run(self, envelope: PuzzleEnvelope) -> PuzzleEnvelope:
        """Execute this pipeline step.

        Args:
            envelope: The current state of the puzzle.

        Returns:
            The updated puzzle envelope with this step's output added.
        """

    def validate_input(self, envelope: PuzzleEnvelope) -> list[str]:
        """Validate that the envelope has the required inputs for this step.

        Returns:
            A list of validation error messages. Empty list means valid.
        """
        return []
