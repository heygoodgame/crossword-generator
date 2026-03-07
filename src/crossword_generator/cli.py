"""CLI entrypoint for the crossword generator."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from crossword_generator.config import load_config
from crossword_generator.pipeline import create_pipeline


@click.group()
@click.version_option(version="0.1.0")
def main() -> None:
    """Crossword Generator — generate mini and midi crossword puzzles."""


@main.command()
@click.option(
    "--type",
    "puzzle_type",
    type=click.Choice(["mini", "midi"]),
    default="mini",
    help="Puzzle type to generate.",
)
@click.option(
    "--size",
    type=int,
    default=None,
    help="Grid size (5/7 for mini, 9/10/11 for midi).",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducible grid generation.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to config file.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable debug logging.",
)
def generate(
    puzzle_type: str,
    size: int | None,
    seed: int | None,
    config_path: str | None,
    verbose: bool,
) -> None:
    """Generate a crossword puzzle."""
    _setup_logging(verbose)
    logger = logging.getLogger(__name__)

    config = load_config(Path(config_path) if config_path else None)

    # Override config with CLI options
    config.puzzle.type = puzzle_type
    if size is not None:
        config.puzzle.grid_size = size

    logger.info(
        "Generating %s crossword (%dx%d)",
        config.puzzle.type,
        config.puzzle.grid_size,
        config.puzzle.grid_size,
    )

    try:
        pipeline, envelope = create_pipeline(config, seed=seed)
        result = pipeline.run(envelope)
    except Exception as e:
        logger.error("Generation failed: %s", e)
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(
        f"Generated {result.puzzle_type.value} crossword "
        f"({result.grid_size}x{result.grid_size})"
    )
    if result.fill:
        click.echo(f"Filler: {result.fill.filler_used}")
        if result.fill.grade_report:
            report = result.fill.grade_report
            click.echo(
                f"Fill quality: {report.overall_score:.1f}/100 "
                f"({'PASS' if report.passing else 'FAIL'})"
            )
            click.echo(f"Attempt: {result.fill.attempt_number}")
        # Print the grid
        for row in result.fill.grid:
            click.echo(" ".join(c if c != "." else "\u2588" for c in row))


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


if __name__ == "__main__":
    main()
