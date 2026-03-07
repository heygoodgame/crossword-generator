"""Grid fill pipeline step."""

from __future__ import annotations

import logging

from crossword_generator.fillers.base import GridFiller
from crossword_generator.grid_specs import get_grid_spec
from crossword_generator.models import FillResult, PuzzleEnvelope
from crossword_generator.steps.base import PipelineStep

logger = logging.getLogger(__name__)


class FillStep(PipelineStep):
    """Pipeline step that fills an empty grid using a GridFiller backend."""

    def __init__(self, filler: GridFiller) -> None:
        self._filler = filler

    @property
    def name(self) -> str:
        return "grid-fill"

    def run(self, envelope: PuzzleEnvelope) -> PuzzleEnvelope:
        """Fill the grid and return an updated envelope."""
        errors = self.validate_input(envelope)
        if errors:
            raise ValueError(f"FillStep validation failed: {'; '.join(errors)}")

        spec = get_grid_spec(envelope.puzzle_type, envelope.grid_size)

        logger.info(
            "Running grid fill with %s (%dx%d)",
            self._filler.name,
            spec.rows,
            spec.cols,
        )

        filled = self._filler.fill(spec)

        fill_result = FillResult(
            grid=filled.grid,
            filler_used=self._filler.name,
        )

        return envelope.model_copy(
            update={
                "fill": fill_result,
                "step_history": [*envelope.step_history, self.name],
            }
        )

    def validate_input(self, envelope: PuzzleEnvelope) -> list[str]:
        """Validate that the envelope is ready for filling."""
        errors: list[str] = []
        if envelope.fill is not None:
            errors.append("Envelope already has a fill result")
        if not self._filler.is_available():
            errors.append(f"Filler '{self._filler.name}' is not available")
        return errors
