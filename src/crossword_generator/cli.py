"""CLI entrypoint for the crossword generator."""

import click


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
    "--config",
    "config_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to config file.",
)
def generate(puzzle_type: str, config_path: str | None) -> None:
    """Generate a crossword puzzle."""
    click.echo(f"Generating {puzzle_type} crossword...")
    click.echo("Not yet implemented. See docs/roadmap.md for the development plan.")


if __name__ == "__main__":
    main()
