"""CLI entrypoint for the crossword generator."""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import click

from crossword_generator.config import find_project_root, load_config
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
    "--llm",
    "llm_provider",
    type=click.Choice(["ollama", "claude"]),
    default=None,
    help="LLM provider to use (overrides config).",
)
@click.option(
    "--theme-file",
    type=click.Path(exists=True),
    default=None,
    help="Pre-generated theme file (skips theme generation).",
)
@click.option(
    "--no-theme",
    is_flag=True,
    default=False,
    help="Skip theme generation (themeless midi).",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=None,
    help="Output directory for generated puzzle files (overrides config).",
)
@click.option(
    "--output-file",
    type=click.Path(),
    default=None,
    help="Exact output file path (extension determines format, e.g. .puz or .ipuz).",
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
    llm_provider: str | None,
    theme_file: str | None,
    no_theme: bool,
    output_dir: str | None,
    output_file: str | None,
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
    if llm_provider is not None:
        config.llm.provider = llm_provider

    if no_theme:
        config.theme.enabled = False
    if output_dir is not None:
        config.output.directory = output_dir

    theme_path = Path(theme_file) if theme_file else None
    output_file_path = Path(output_file) if output_file else None

    logger.info(
        "Generating %s crossword (%dx%d)",
        config.puzzle.type,
        config.puzzle.grid_size,
        config.puzzle.grid_size,
    )

    try:
        pipeline, envelope = create_pipeline(
            config, seed=seed, theme_file=theme_path, output_file=output_file_path
        )
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


@main.command(name="generate-themes")
@click.option("--count", type=int, default=5, help="Number of themes to generate.")
@click.option("--size", type=int, default=9, help="Grid size for themes.")
@click.option(
    "--output-dir",
    type=click.Path(),
    default="themes/",
    help="Directory to save theme files.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to config file.",
)
@click.option(
    "--llm",
    "llm_provider",
    type=click.Choice(["ollama", "claude"]),
    default=None,
    help="LLM provider to use (overrides config).",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable debug logging.",
)
def generate_themes(
    count: int,
    size: int,
    output_dir: str,
    config_path: str | None,
    llm_provider: str | None,
    verbose: bool,
) -> None:
    """Generate standalone theme files for later use."""
    _setup_logging(verbose)
    logger = logging.getLogger(__name__)

    from crossword_generator.config import find_project_root
    from crossword_generator.dictionary import Dictionary
    from crossword_generator.llm.claude_provider import ClaudeProvider
    from crossword_generator.llm.ollama_provider import OllamaProvider
    from crossword_generator.steps.theme_step import generate_single_theme
    from crossword_generator.theme_io import load_topic_set, save_theme

    config = load_config(Path(config_path) if config_path else None)
    if llm_provider is not None:
        config.llm.provider = llm_provider

    # Build LLM provider
    if config.llm.provider == "ollama":
        llm = OllamaProvider(config.llm.ollama)
    elif config.llm.provider == "claude":
        llm = ClaudeProvider(config.llm.claude)
    else:
        click.echo(f"Unknown LLM provider: {config.llm.provider}", err=True)
        sys.exit(1)

    # Load dictionary
    project_root = find_project_root()
    dictionary = Dictionary.load(
        project_root / config.dictionary.path,
        min_word_score=config.dictionary.min_word_score,
        min_2letter_score=config.dictionary.min_2letter_score,
    )

    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir

    # Load existing topics for dedup
    avoid_topics = list(load_topic_set(out_dir))
    logger.info("Loaded %d existing topics to avoid", len(avoid_topics))

    generated = 0
    for i in range(count):
        try:
            theme = generate_single_theme(
                llm=llm,
                dictionary=dictionary,
                grid_size=size,
                seed=random.randint(0, 2**31 - 1),
                max_retries=config.theme.max_retries,
                num_seed_entries=config.theme.num_seed_entries,
                num_candidates=config.theme.num_candidates,
                avoid_topics=avoid_topics,
                similarity_threshold=config.theme.similarity_threshold,
                max_avoid_in_prompt=config.theme.max_avoid_in_prompt,
            )
            path = save_theme(theme, size, llm.name, out_dir)
            avoid_topics.append(theme.topic)
            generated += 1
            click.echo(f"  [{i + 1}/{count}] {theme.topic} → {path.name}")
        except Exception as e:
            logger.error("Theme %d/%d failed: %s", i + 1, count, e)
            click.echo(f"  [{i + 1}/{count}] FAILED: {e}", err=True)

    click.echo(f"Generated {generated} themes in {out_dir}")


@main.command(name="dedup-themes")
@click.option(
    "--theme-dir",
    type=click.Path(),
    default="themes/",
    help="Directory containing theme files.",
)
@click.option(
    "--threshold",
    type=float,
    default=0.6,
    help="Jaccard similarity threshold for fuzzy matching (0.0-1.0).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=True,
    help="Preview duplicates without deleting (default).",
)
@click.option(
    "--delete",
    is_flag=True,
    default=False,
    help="Actually delete duplicate theme files.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable debug logging.",
)
def dedup_themes(
    theme_dir: str,
    threshold: float,
    dry_run: bool,
    delete: bool,
    verbose: bool,
) -> None:
    """Find and remove duplicate/similar themes from the theme directory."""
    _setup_logging(verbose)

    from crossword_generator.theme_io import load_theme
    from crossword_generator.topic_dedup import (
        build_normalized_topic_set,
        is_topic_duplicate,
        is_topic_similar,
    )

    project_root = find_project_root()
    theme_path = Path(theme_dir)
    if not theme_path.is_absolute():
        theme_path = project_root / theme_path

    if not theme_path.exists():
        click.echo(f"Theme directory not found: {theme_path}", err=True)
        sys.exit(1)

    # Load all theme files, sorted by name (keeps earliest)
    files = sorted(theme_path.glob("*.json"))
    if not files:
        click.echo("No theme files found.")
        return

    kept_topics: list[str] = []
    kept_normalized: set[str] = set()
    kept_files: list[Path] = []
    duplicates: list[tuple[Path, str, str]] = []  # (path, topic, reason)

    for path in files:
        try:
            tf = load_theme(path)
        except Exception:
            click.echo(f"  SKIP (invalid): {path.name}")
            continue

        topic = tf.theme.topic

        # Check exact duplicate
        if is_topic_duplicate(topic, kept_normalized):
            duplicates.append((path, topic, "exact duplicate"))
            continue

        # Check fuzzy similarity
        similar, closest = is_topic_similar(
            topic, kept_topics, threshold=threshold
        )
        if similar:
            duplicates.append(
                (path, topic, f"similar to {closest!r}")
            )
            continue

        kept_topics.append(topic)
        kept_normalized = build_normalized_topic_set(kept_topics)
        kept_files.append(path)

    if not duplicates:
        click.echo(
            f"No duplicates found among {len(files)} theme files "
            f"(threshold={threshold})."
        )
        return

    click.echo(
        f"Found {len(duplicates)} duplicate(s) among {len(files)} "
        f"theme files (threshold={threshold}):\n"
    )
    for path, topic, reason in duplicates:
        click.echo(f"  {path.name}: {topic!r} ({reason})")

    if delete:
        for path, _, _ in duplicates:
            path.unlink()
        click.echo(
            f"\nDeleted {len(duplicates)} duplicate theme files. "
            f"{len(kept_files)} unique themes remain."
        )
    else:
        click.echo(
            "\nDry run — no files deleted. Use --delete to remove them."
        )


@main.command()
@click.option(
    "--sizes",
    default="5,7",
    help="Comma-separated grid sizes to evaluate.",
)
@click.option(
    "--num-seeds",
    type=int,
    default=5,
    help="Number of random seeds per filler per size.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to config file.",
)
@click.option(
    "--max-consecutive-failures",
    type=int,
    default=5,
    help="Skip remaining seeds after N consecutive failures (0 to disable).",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable debug logging.",
)
def evaluate(
    sizes: str,
    num_seeds: int,
    config_path: str | None,
    max_consecutive_failures: int,
    verbose: bool,
) -> None:
    """Evaluate fill quality across all available fillers."""
    _setup_logging(verbose)
    logger = logging.getLogger(__name__)

    from crossword_generator.dictionary import Dictionary
    from crossword_generator.evaluation import FillerEvaluator
    from crossword_generator.fillers.base import GridFiller
    from crossword_generator.fillers.csp import CSPFiller
    from crossword_generator.graders.fill_grader import FillGrader

    config = load_config(Path(config_path) if config_path else None)

    # Load shared dictionary
    project_root = find_project_root()
    dictionary = Dictionary.load(
        project_root / config.dictionary.path,
        min_word_score=config.dictionary.min_word_score,
        min_2letter_score=config.dictionary.min_2letter_score,
    )
    grader = FillGrader(dictionary, min_passing_score=config.grading.fill.min_score)

    # Build all available fillers
    fillers: list[GridFiller] = []

    # CSP filler (always available)
    csp_filler = CSPFiller(config.fill.csp, dictionary)
    fillers.append(csp_filler)
    logger.info("csp: available")

    if not fillers:
        click.echo("No fillers available.", err=True)
        sys.exit(1)

    # Parse sizes and generate seeds
    grid_sizes = [int(s.strip()) for s in sizes.split(",")]
    seeds = [random.randint(0, 2**31 - 1) for _ in range(num_seeds)]

    click.echo(
        f"Evaluating {len(fillers)} fillers × "
        f"{len(grid_sizes)} sizes × {num_seeds} seeds\n"
    )

    evaluator = FillerEvaluator(fillers, grader)
    results = evaluator.evaluate(
        grid_sizes, seeds, max_consecutive_failures=max_consecutive_failures
    )
    report = FillerEvaluator.format_report(results)
    click.echo(report)


@main.command(name="export-dictionary")
@click.option(
    "--min-score",
    type=int,
    default=50,
    help="Minimum word score to include.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(),
    default="dictionaries/jeff-chen-filtered.txt",
    help="Output file path.",
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
def export_dictionary(
    min_score: int,
    output_path: str,
    config_path: str | None,
    verbose: bool,
) -> None:
    """Export a filtered plain-text dictionary for external tools."""
    _setup_logging(verbose)

    from crossword_generator.dictionary import Dictionary

    config = load_config(Path(config_path) if config_path else None)
    project_root = find_project_root()

    # Load with min_word_score=0 to get all words, then filter via export_plain
    dictionary = Dictionary.load(
        project_root / config.dictionary.path,
        min_word_score=0,
        min_2letter_score=0,
    )

    out = Path(output_path)
    if not out.is_absolute():
        out = project_root / out

    count = dictionary.export_plain(out, min_score=min_score)
    click.echo(f"Exported {count} words (min_score={min_score}) to {out}")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


if __name__ == "__main__":
    main()
