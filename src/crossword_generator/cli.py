"""CLI entrypoint for the crossword generator."""

from __future__ import annotations

import json
import logging
import random
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
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


@main.command(name="generate-pilot-batch")
@click.option(
    "--output-root",
    type=click.Path(),
    default="output/batches/phase-2b-pilot",
    help="Root directory for generated puzzles, logs, and manifest.",
)
@click.option(
    "--batch-id",
    default="phase-2b-pilot",
    help="Batch identifier recorded in the manifest and used downstream "
    "by save-generated-puzzles.",
)
@click.option(
    "--count",
    type=int,
    default=5,
    help="Number of puzzles per difficulty/size bucket.",
)
@click.option(
    "--seed-start",
    type=int,
    default=1,
    help="First deterministic seed for every bucket.",
)
@click.option(
    "--llm",
    "llm_provider",
    type=click.Choice(["ollama", "claude"]),
    default="claude",
    help="LLM provider to use.",
)
@click.option(
    "--per-pattern-attempts",
    type=int,
    default=1,
    help="Batch-mode CSP attempts per grid pattern before trying next variant.",
)
@click.option(
    "--max-grid-variants",
    type=int,
    default=200,
    help="Batch-mode maximum grid variants per puzzle.",
)
@click.option(
    "--timeout-5",
    type=int,
    default=15,
    help="Batch-mode CSP timeout in seconds for 5x5 grids.",
)
@click.option(
    "--timeout-7",
    type=int,
    default=30,
    help="Batch-mode CSP timeout in seconds for 7x7 grids.",
)
@click.option(
    "--timeout-9",
    type=int,
    default=120,
    help="Batch-mode CSP timeout in seconds for 9x9 grids.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Also stream detailed logs to stderr.",
)
def generate_pilot_batch(
    output_root: str,
    batch_id: str,
    count: int,
    seed_start: int,
    llm_provider: str,
    per_pattern_attempts: int,
    max_grid_variants: int,
    timeout_5: int,
    timeout_7: int,
    timeout_9: int,
    verbose: bool,
) -> None:
    """Generate the Phase 2B pilot batch and write a JSON manifest."""
    _setup_logging(verbose)

    project_root = find_project_root()
    root = Path(output_root)
    if not root.is_absolute():
        root = project_root / root
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    buckets = [
        ("easy", 5, "mini", project_root / "config.easy.yaml"),
        ("easy", 7, "mini", project_root / "config.easy.yaml"),
        ("easy", 9, "midi", project_root / "config.easy.yaml"),
        ("hard", 5, "mini", project_root / "config.hard.yaml"),
        ("hard", 7, "mini", project_root / "config.hard.yaml"),
        ("hard", 9, "midi", project_root / "config.hard.yaml"),
    ]

    started_at = _utc_timestamp()
    results: list[dict[str, object]] = []
    for difficulty, size, puzzle_type, config_path in buckets:
        for seed in range(seed_start, seed_start + count):
            results.append(
                _run_batch_item(
                    difficulty=difficulty,
                    size=size,
                    puzzle_type=puzzle_type,
                    seed=seed,
                    config_path=config_path,
                    output_root=root,
                    logs_dir=logs_dir,
                    llm_provider=llm_provider,
                    per_pattern_attempts=per_pattern_attempts,
                    max_grid_variants=max_grid_variants,
                    timeout_by_size={5: timeout_5, 7: timeout_7, 9: timeout_9},
                )
            )
            status = "ok" if results[-1]["success"] else "failed"
            click.echo(
                f"{difficulty} {size}x{size} seed {seed}: {status} "
                f"({results[-1]['runtime_seconds']}s)"
            )

    manifest = {
        "batch": batch_id,
        "started_at": started_at,
        "finished_at": _utc_timestamp(),
        "output_root": str(root),
        "logs_dir": str(logs_dir),
        "count_per_bucket": count,
        "seed_start": seed_start,
        "llm_provider": llm_provider,
        "batch_fill": {
            "per_pattern_attempts": per_pattern_attempts,
            "max_grid_variants": max_grid_variants,
            "timeout_by_size": {"5": timeout_5, "7": timeout_7, "9": timeout_9},
        },
        "results": results,
        "summary": _summarize_batch_results(results),
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    click.echo(f"Manifest: {manifest_path}")


@main.command(name="save-generated-puzzles")
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True),
    required=True,
    help="Batch manifest produced by generate-pilot-batch.",
)
@click.option(
    "--batch-id",
    default=None,
    help="Override the manifest batch id for data-store metadata and keys.",
)
@click.option(
    "--api-base",
    default=None,
    help="Override HEYGG_API_BASE_URL for this upload.",
)
@click.option(
    "--mini-game-key",
    type=click.Choice(["minicrossword", "crosswordle"]),
    default="minicrossword",
    help="Game key for 5x5 and 7x7 generated puzzles.",
)
@click.option(
    "--midi-game-key",
    type=click.Choice(["midicrossword", "crosswordle"]),
    default="midicrossword",
    help="Game key for 9x9 generated puzzles.",
)
@click.option(
    "--generator-version",
    default=None,
    help="Generator version metadata override.",
)
@click.option(
    "--generator-commit",
    default=None,
    help="Generator git commit metadata override.",
)
@click.option(
    "--replace-existing",
    is_flag=True,
    default=False,
    help="PATCH existing duplicate-key records instead of skipping them.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Build and validate records without calling the HeyGG API.",
)
def save_generated_puzzles(
    manifest_path: str,
    batch_id: str | None,
    api_base: str | None,
    mini_game_key: str,
    midi_game_key: str,
    generator_version: str | None,
    generator_commit: str | None,
    replace_existing: bool,
    dry_run: bool,
) -> None:
    """Save generated puzzle candidates to the HeyGG admin data store."""
    from crossword_generator.data_store import (
        bulk_save_generated_puzzles,
        records_from_manifest,
    )

    project_root = find_project_root()
    manifest = Path(manifest_path)
    resolved_version = generator_version or _generator_version()
    resolved_commit = generator_commit or _generator_commit(project_root)
    records = records_from_manifest(
        manifest,
        batch_id=batch_id,
        generator_version=resolved_version,
        generator_commit=resolved_commit,
        mini_game_key=mini_game_key,
        midi_game_key=midi_game_key,
    )

    click.echo(f"Prepared {len(records)} generated puzzle record(s).")
    if not records:
        return

    if dry_run:
        click.echo("Dry run: no HeyGG API calls made.")
        for record in records[:5]:
            click.echo(f"  {record['game_key']} {record['key']}")
        if len(records) > 5:
            click.echo(f"  ... {len(records) - 5} more")
        return

    try:
        results = bulk_save_generated_puzzles(
            records,
            replace_existing=replace_existing,
            api_base=api_base,
        )
    except KeyError as exc:
        missing = exc.args[0]
        click.echo(f"Missing required environment variable: {missing}", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"Save failed: {exc}", err=True)
        sys.exit(1)

    counts: dict[str, int] = {}
    for result in results:
        counts[result.action] = counts.get(result.action, 0) + 1
    click.echo(
        "Saved generated puzzles: "
        + ", ".join(f"{action}={count}" for action, count in sorted(counts.items()))
    )


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


@main.command(name="prepare-dictionaries")
@click.option(
    "--easy-source",
    type=click.Path(exists=True),
    required=True,
    help="Path to Jeff's WordpleteCulledJYC.txt source file.",
)
@click.option(
    "--hard-source",
    type=click.Path(exists=True),
    default="dictionaries/HggCuratedCrosswordList.txt",
    help="Path to the curated hard source dictionary.",
)
@click.option(
    "--easy-output",
    type=click.Path(),
    default="dictionaries/hgg-easy-flat-55.txt",
    help="Output path for the normalized easy dictionary.",
)
@click.option(
    "--hard-output",
    type=click.Path(),
    default="dictionaries/hgg-hard-flat-55.txt",
    help="Output path for the normalized hard dictionary.",
)
@click.option(
    "--score",
    type=int,
    default=55,
    help="Flat score to assign to every output entry.",
)
def prepare_dictionaries(
    easy_source: str,
    hard_source: str,
    easy_output: str,
    hard_output: str,
    score: int,
) -> None:
    """Prepare flat-score easy and hard dictionaries for batch experiments."""
    from crossword_generator.dictionary_prep import (
        format_summary,
        prepare_flat_dictionary,
    )

    project_root = find_project_root()
    jobs = [
        ("Easy dictionary", Path(easy_source), Path(easy_output)),
        ("Hard dictionary", Path(hard_source), Path(hard_output)),
    ]

    for label, source, output in jobs:
        if not source.is_absolute():
            source = project_root / source
        if not output.is_absolute():
            output = project_root / output

        summary = prepare_flat_dictionary(source, output, score=score)
        click.echo(f"{label}:")
        click.echo(format_summary(summary))
        click.echo("")


@main.command(name="validate-mini-patterns")
def validate_mini_patterns() -> None:
    """Validate catalogued 5x5 and 7x7 mini grid patterns."""
    from crossword_generator.grid_pattern_validation import (
        summarize_validations,
        validate_weighted_patterns,
    )
    from crossword_generator.grid_specs import get_grid_patterns
    from crossword_generator.models import PuzzleType

    failures = 0
    for size, expected_count, expected_weight in ((5, 34, 95), (7, 50, 86)):
        patterns = [
            (list(pattern.black_cells), pattern.weight)
            for pattern in get_grid_patterns(PuzzleType.MINI, size)
        ]
        results = validate_weighted_patterns(size, patterns)
        summary = summarize_validations(results)
        asymmetric = [
            str(result.index) for result in results if not result.symmetric
        ]

        click.echo(
            f"{size}x{size}: patterns={summary['patterns']} "
            f"total_weight={summary['total_weight']} "
            f"valid={summary['valid']} invalid={summary['invalid']} "
            f"symmetric={summary['symmetric']} "
            f"asymmetric={summary['asymmetric']}"
        )
        click.echo(
            "  asymmetric pattern indexes: "
            + (", ".join(asymmetric) if asymmetric else "none")
        )

        if (
            summary["patterns"] != expected_count
            or summary["total_weight"] != expected_weight
            or summary["invalid"] != 0
        ):
            failures += 1
            for result in results:
                if result.errors:
                    click.echo(
                        f"  pattern {result.index} errors: "
                        + "; ".join(result.errors),
                        err=True,
                    )

    if failures:
        sys.exit(1)


class _BatchLogHandler(logging.Handler):
    """Captures log records needed for batch manifest metadata."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.skipped_incompatible_variants = 0
        self.fill_attempts = 0
        self.grid_variants_seen: set[int] = set()

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if message.startswith("Fill attempt ") and "grid variant" in message:
            self.fill_attempts += 1
            variant = _extract_grid_variant(message)
            if variant is not None:
                self.grid_variants_seen.add(variant)
        elif message.startswith("Trying grid variant "):
            variant = _extract_grid_variant(message)
            if variant is not None:
                self.grid_variants_seen.add(variant)

        if (
            "skipped: slot lengths" in message
            and "unsupported by dictionary" in message
        ):
            self.skipped_incompatible_variants += 1
            variant = _extract_grid_variant(message)
            if variant is not None:
                self.grid_variants_seen.add(variant)


def _run_batch_item(
    *,
    difficulty: str,
    size: int,
    puzzle_type: str,
    seed: int,
    config_path: Path,
    output_root: Path,
    logs_dir: Path,
    llm_provider: str,
    per_pattern_attempts: int,
    max_grid_variants: int,
    timeout_by_size: dict[int, int],
) -> dict[str, object]:
    bucket_dir = output_root / difficulty / f"{size}x{size}"
    bucket_dir.mkdir(parents=True, exist_ok=True)
    output_path = bucket_dir / f"seed-{seed:03d}.ipuz"
    log_path = logs_dir / f"{difficulty}-{size}x{size}-seed-{seed:03d}.log"
    if output_path.exists():
        output_path.unlink()

    config = load_config(config_path)
    config.puzzle.type = puzzle_type
    config.puzzle.grid_size = size
    config.llm.provider = llm_provider
    config.output.directory = str(bucket_dir / "intermediates" / f"seed-{seed:03d}")
    config.output.formats = ["ipuz"]
    config.fill.max_retries = per_pattern_attempts
    config.fill.max_grid_variants = max_grid_variants
    config.fill.csp.timeout_by_size = timeout_by_size
    if puzzle_type == "midi" and size == 9:
        config.theme.enabled = False

    logger = logging.getLogger()
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    capture_handler = _BatchLogHandler()
    logger.addHandler(file_handler)
    logger.addHandler(capture_handler)

    started = time.monotonic()
    result: dict[str, object] = {
        "difficulty": difficulty,
        "size": size,
        "seed": seed,
        "output_path": str(output_path),
        "log_path": str(log_path),
        "success": False,
        "fill_score": None,
        "clue_score": None,
        "title": None,
        "title_reasoning": None,
        "runtime_seconds": 0.0,
        "fill_seconds": None,
        "clue_seconds": None,
        "total_seconds": 0.0,
        "grid_variants": 0,
        "fill_attempts": 0,
        "skipped_incompatible_variants": 0,
        "failure_category": None,
        "error_message": None,
    }
    try:
        pipeline, envelope = create_pipeline(
            config, seed=seed, output_file=output_path
        )
        completed = pipeline.run(envelope)
        result.update(
            {
                "success": output_path.exists(),
                "fill_score": (
                    completed.fill.quality_score if completed.fill else None
                ),
                "clue_score": (
                    completed.clue_grade_report.overall_score
                    if completed.clue_grade_report
                    else None
                ),
                "title": completed.title or None,
                "title_reasoning": completed.title_reasoning or None,
                "fill_seconds": _metadata_timing(
                    completed, "grid-fill-with-grading"
                ),
                "clue_seconds": _metadata_timing(
                    completed, "clue-generation-with-grading"
                ),
                "error_message": "; ".join(completed.errors) or None,
            }
        )
    except Exception as exc:
        logging.getLogger(__name__).exception(
            "Batch item failed: %s %sx%s seed %s",
            difficulty,
            size,
            size,
            seed,
        )
        result["error_message"] = str(exc)
    finally:
        runtime = time.monotonic() - started
        result["runtime_seconds"] = round(runtime, 3)
        result["total_seconds"] = round(runtime, 3)
        result["grid_variants"] = len(capture_handler.grid_variants_seen)
        result["fill_attempts"] = capture_handler.fill_attempts
        result["skipped_incompatible_variants"] = (
            capture_handler.skipped_incompatible_variants
        )
        result["failure_category"] = _failure_category(result)
        logger.removeHandler(file_handler)
        logger.removeHandler(capture_handler)
        file_handler.close()

    return result


def _summarize_batch_results(
    results: list[dict[str, object]]
) -> dict[str, dict[str, float | int | None]]:
    summaries: dict[str, dict[str, float | int | None]] = {}
    bucket_keys = sorted(
        {f"{r['difficulty']}-{r['size']}x{r['size']}" for r in results}
    )
    for key in bucket_keys:
        bucket = [
            r for r in results
            if f"{r['difficulty']}-{r['size']}x{r['size']}" == key
        ]
        successes = [r for r in bucket if r["success"]]
        clue_scores = [
            float(r["clue_score"])
            for r in successes
            if r["clue_score"] is not None
        ]
        runtimes = [float(r["runtime_seconds"]) for r in bucket]
        summaries[key] = {
            "total": len(bucket),
            "successes": len(successes),
            "failures": len(bucket) - len(successes),
            "success_rate": round(len(successes) / len(bucket), 3) if bucket else None,
            "average_runtime_seconds": (
                round(sum(runtimes) / len(runtimes), 3) if runtimes else None
            ),
            "average_clue_score": (
                round(sum(clue_scores) / len(clue_scores), 3)
                if clue_scores
                else None
            ),
        }
    return summaries


def _metadata_timing(envelope: object, step_name: str) -> float | None:
    metadata = getattr(envelope, "metadata", {})
    timings = metadata.get("step_timings_seconds", {})
    value = timings.get(step_name)
    return float(value) if value is not None else None


def _extract_grid_variant(message: str) -> int | None:
    marker = "grid variant "
    if marker not in message:
        marker = "Grid variant "
    if marker not in message:
        return None
    tail = message.split(marker, 1)[1]
    digits = []
    for char in tail:
        if char.isdigit():
            digits.append(char)
        else:
            break
    return int("".join(digits)) if digits else None


def _failure_category(result: dict[str, object]) -> str | None:
    if result.get("success"):
        return None
    error = str(result.get("error_message") or "").lower()
    if int(result.get("skipped_incompatible_variants") or 0) > 0 and int(
        result.get("fill_attempts") or 0
    ) == 0:
        return "incompatible_grid_patterns"
    if "timed out" in error:
        return "fill_timeout"
    if "could not fill grid" in error or "fill" in error:
        return "fill_failed"
    if "clue" in error or "anthropic" in error or "model" in error:
        return "clue_generation_failed"
    if error:
        return "pipeline_failed"
    return "unknown"


def _utc_timestamp() -> str:
    from datetime import UTC, datetime

    return datetime.now(tz=UTC).isoformat()


def _generator_version() -> str:
    try:
        return version("crossword-generator")
    except PackageNotFoundError:
        return "0.1.0"


def _generator_commit(project_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


if __name__ == "__main__":
    main()
