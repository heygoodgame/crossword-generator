"""Evaluation framework for comparing grid filler quality."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from statistics import mean, median

from crossword_generator.fillers.base import FilledGrid, FillError, GridFiller, GridSpec
from crossword_generator.graders.fill_grader import FillGrader

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result of a single filler evaluation run."""

    filler_name: str
    grid_size: int
    seed: int
    success: bool
    quality_score: float | None
    time_seconds: float
    error: str | None
    word_count: int
    passing: bool


class FillerEvaluator:
    """Evaluates grid fillers across multiple sizes and seeds."""

    def __init__(
        self,
        fillers: list[GridFiller],
        grader: FillGrader,
    ) -> None:
        self._fillers = fillers
        self._grader = grader

    def evaluate(
        self,
        grid_sizes: list[int],
        seeds: list[int],
        *,
        max_consecutive_failures: int = 0,
    ) -> list[EvalResult]:
        """Run all fillers across all sizes and seeds.

        Args:
            grid_sizes: Grid dimensions to test (square grids).
            seeds: Random seeds for each trial.
            max_consecutive_failures: If > 0, skip remaining seeds for a
                filler×size combo after this many consecutive failures.
                Set to 0 to disable early abort.

        Returns:
            List of EvalResult for every filler x size x seed combination.
        """
        results: list[EvalResult] = []

        for filler in self._fillers:
            if not filler.is_available():
                logger.warning(
                    "Skipping unavailable filler: %s", filler.name
                )
                continue

            for size in grid_sizes:
                consecutive_failures = 0
                for seed in seeds:
                    if (
                        max_consecutive_failures > 0
                        and consecutive_failures >= max_consecutive_failures
                    ):
                        results.append(EvalResult(
                            filler_name=filler.name,
                            grid_size=size,
                            seed=seed,
                            success=False,
                            quality_score=None,
                            time_seconds=0.0,
                            error="skipped (early abort)",
                            word_count=0,
                            passing=False,
                        ))
                        continue

                    result = self._run_single(filler, size, seed)
                    results.append(result)

                    if result.success:
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1

                    status = (
                        f"score={result.quality_score:.1f}"
                        if result.success
                        else f"FAIL: {result.error}"
                    )
                    logger.info(
                        "%s %dx%d seed=%d: %s (%.2fs)",
                        filler.name,
                        size,
                        size,
                        seed,
                        status,
                        result.time_seconds,
                    )

                if (
                    max_consecutive_failures > 0
                    and consecutive_failures >= max_consecutive_failures
                ):
                    logger.warning(
                        "Early abort: %s at %dx%d after %d consecutive failures",
                        filler.name,
                        size,
                        size,
                        consecutive_failures,
                    )

        return results

    def _run_single(
        self, filler: GridFiller, size: int, seed: int
    ) -> EvalResult:
        """Run a single filler trial and grade the result."""
        spec = GridSpec(rows=size, cols=size)
        start = time.monotonic()

        try:
            filled: FilledGrid = filler.fill(spec, seed=seed)
            elapsed = time.monotonic() - start
        except FillError as e:
            elapsed = time.monotonic() - start
            return EvalResult(
                filler_name=filler.name,
                grid_size=size,
                seed=seed,
                success=False,
                quality_score=None,
                time_seconds=elapsed,
                error=str(e),
                word_count=0,
                passing=False,
            )

        report = self._grader.grade(filled.grid)

        return EvalResult(
            filler_name=filler.name,
            grid_size=size,
            seed=seed,
            success=True,
            quality_score=report.overall_score,
            time_seconds=elapsed,
            error=None,
            word_count=report.word_count,
            passing=report.passing,
        )

    @staticmethod
    def format_report(results: list[EvalResult]) -> str:
        """Format evaluation results as a markdown report.

        Groups results by grid size, then by filler name, showing
        success rate, mean/median quality score, and mean time.
        """
        if not results:
            return "No evaluation results."

        # Collect unique sizes and fillers (preserving order)
        sizes: list[int] = []
        fillers: list[str] = []
        for r in results:
            if r.grid_size not in sizes:
                sizes.append(r.grid_size)
            if r.filler_name not in fillers:
                fillers.append(r.filler_name)

        lines: list[str] = ["## Fill Evaluation Results", ""]

        for size in sorted(sizes):
            lines.append(f"### {size}x{size} Grid")
            lines.append("")
            lines.append(
                "| Filler | Success | Mean Score "
                "| Median Score | Mean Time |"
            )
            lines.append(
                "|--------|---------|------------"
                "|--------------|-----------|"
            )

            for filler_name in fillers:
                group = [
                    r
                    for r in results
                    if r.grid_size == size and r.filler_name == filler_name
                ]
                if not group:
                    continue

                total = len(group)
                successes = [r for r in group if r.success]
                success_count = len(successes)

                scores = [
                    r.quality_score
                    for r in successes
                    if r.quality_score is not None
                ]
                mean_score = f"{mean(scores):.1f}" if scores else "—"
                median_score = f"{median(scores):.1f}" if scores else "—"
                mean_time = f"{mean(r.time_seconds for r in group):.2f}s"

                lines.append(
                    f"| {filler_name:<6} "
                    f"| {success_count}/{total}     "
                    f"| {mean_score:<10} "
                    f"| {median_score:<12} "
                    f"| {mean_time:<9} |"
                )

            lines.append("")

        return "\n".join(lines)
