"""CLI entrypoint for the crossword generator."""

from __future__ import annotations

import json
import logging
import random
import subprocess
import sys
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import click

from crossword_generator.clue_history import (
    ClueHistoryIndex,
    duplicate_error_message,
    extract_ipuz_answers,
)
from crossword_generator.config import Config, find_project_root, load_config
from crossword_generator.exporters.base import Exporter
from crossword_generator.models import PuzzleEnvelope
from crossword_generator.pipeline import create_pipeline
from crossword_generator.steps.clue_grading_step import ClueWithGradingStep


@click.group()
@click.version_option(version="0.1.0")
def main() -> None:
    """Crossword Generator — generate mini and midi crossword puzzles."""


@main.group(name="llm-logs")
def llm_logs() -> None:
    """Browse and replay structured LLM call logs."""


@llm_logs.command(name="browse")
@click.argument("target", required=False, type=click.Path())
@click.option("--step", default=None, help="Filter by pipeline step.")
@click.option("--model", default=None, help="Filter by model ID.")
@click.option("--seed", type=int, default=None, help="Filter by puzzle seed.")
@click.option("--difficulty", default=None, help="Filter by puzzle difficulty.")
@click.option("--size", type=int, default=None, help="Filter by puzzle grid size.")
@click.option(
    "--errors-only",
    is_flag=True,
    default=False,
    help="Show only records with logged errors.",
)
@click.option(
    "--experiment-root",
    type=click.Path(),
    default=None,
    help="Directory for transient replay artifacts.",
)
def browse_llm_logs(
    target: str | None,
    step: str | None,
    model: str | None,
    seed: int | None,
    difficulty: str | None,
    size: int | None,
    errors_only: bool,
    experiment_root: str | None,
) -> None:
    """Open the terminal browser for LLM JSONL logs."""
    from crossword_generator.llm.log_browser import LLMLogFilters, load_llm_logs

    filters = LLMLogFilters(
        step=step,
        model=model,
        seed=seed,
        difficulty=difficulty,
        size=size,
        errors_only=errors_only,
    )
    records, problems = load_llm_logs(Path(target) if target else None)
    if not records:
        for problem in problems:
            click.echo(_format_llm_log_problem(problem), err=True)
        click.echo("No LLM log records found.", err=True)
        sys.exit(1)

    _launch_llm_log_browser(
        records,
        problems,
        initial_filters=filters,
        experiment_root=Path(experiment_root) if experiment_root else None,
    )


@llm_logs.command(name="report")
@click.argument("target", required=False, type=click.Path())
def report_llm_logs(target: str | None) -> None:
    """Print a per-step cost and cache-effectiveness report for a batch.

    TARGET is a JSONL log file or a directory of them (e.g. a batch's logs/).
    """
    from crossword_generator.llm.log_report import format_report, report_for_path

    report = report_for_path(Path(target) if target else None)
    if not report.steps:
        click.echo("No LLM log records found.", err=True)
        sys.exit(1)
    click.echo(format_report(report))


def _launch_llm_log_browser(
    records: list[object],
    problems: list[object],
    *,
    initial_filters: object,
    experiment_root: Path | None,
) -> None:
    from crossword_generator.llm.log_tui import run_llm_log_browser

    run_llm_log_browser(
        records,  # type: ignore[arg-type]
        problems,  # type: ignore[arg-type]
        initial_filters=initial_filters,  # type: ignore[arg-type]
        experiment_root=experiment_root,
    )


def _format_llm_log_problem(problem: object) -> str:
    path = getattr(problem, "path", "")
    line_number = getattr(problem, "line_number", None)
    message = getattr(problem, "message", "")
    line_suffix = f":{line_number}" if line_number is not None else ""
    return f"{path}{line_suffix}: {message}"


@main.command()
@click.option(
    "--type",
    "puzzle_type",
    type=click.Choice(["mini", "midi"]),
    default="mini",
    help="Puzzle type to generate.",
)
@click.option(
    "--difficulty",
    type=click.Choice(["easy", "hard"]),
    default=None,
    help="Clue/fill difficulty to generate (overrides config).",
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
    "--llm-log-path",
    type=click.Path(),
    default=None,
    help="Structured JSONL log path for LLM calls.",
)
@click.option(
    "--no-llm-log",
    is_flag=True,
    default=False,
    help="Disable structured LLM call logging for this run.",
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
    difficulty: str | None,
    size: int | None,
    seed: int | None,
    config_path: str | None,
    llm_provider: str | None,
    theme_file: str | None,
    no_theme: bool,
    output_dir: str | None,
    output_file: str | None,
    llm_log_path: str | None,
    no_llm_log: bool,
    verbose: bool,
) -> None:
    """Generate a crossword puzzle."""
    _setup_logging(verbose)
    logger = logging.getLogger(__name__)

    config = load_config(Path(config_path) if config_path else None)

    # Override config with CLI options
    config.puzzle.type = puzzle_type
    if difficulty is not None:
        config.puzzle.difficulty = difficulty
    if size is not None:
        config.puzzle.grid_size = size
    if llm_provider is not None:
        config.llm.provider = llm_provider

    if no_theme:
        config.theme.enabled = False
    if output_dir is not None:
        config.output.directory = output_dir
    if llm_log_path is not None:
        config.llm.logging.path = llm_log_path
    if no_llm_log:
        config.llm.logging.enabled = False

    theme_path = Path(theme_file) if theme_file else None
    output_file_path = Path(output_file) if output_file else None

    logger.info(
        "Generating %s %s crossword (%dx%d)",
        config.puzzle.difficulty,
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
        f"Generated {result.difficulty.value} {result.puzzle_type.value} crossword "
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
    "--bucket-counts",
    default=None,
    help="Comma-separated count overrides by bucket or size, e.g. "
    "'5=5,7=2,9=7' or 'easy/5=5,hard/9=7'.",
)
@click.option(
    "--seed-start",
    type=int,
    default=1,
    help="First deterministic seed for every bucket.",
)
@click.option(
    "--buckets",
    default=None,
    help="Comma-separated subset of buckets to run, formatted "
    "<difficulty>/<size> (e.g. 'easy/9,hard/9'). Defaults to all buckets.",
)
@click.option(
    "--llm",
    "llm_provider",
    type=click.Choice(["ollama", "claude"]),
    default="claude",
    help="LLM provider to use.",
)
@click.option(
    "--avoid-existing-clues/--no-avoid-existing-clues",
    default=True,
    help=(
        "Load existing generated puzzles from the HeyGG admin API and feed "
        "prior clues into generation, so the model avoids exact repeats and "
        "varies its clue angles per answer (default: on). Requires admin API "
        "credentials; pass --no-avoid-existing-clues to skip."
    ),
)
@click.option(
    "--api-base",
    default=None,
    help="Override HEYGG_API_BASE_URL when loading existing clue history.",
)
@click.option(
    "--exclude-scheduled-sixty/--no-exclude-scheduled-sixty",
    default=True,
    help=(
        "For hard 7x7/9x9 buckets, fetch HGG 60 answers already used in "
        "scheduled dailies (180-day rolling window) from the HeyGG admin "
        "API and exclude them from the hgg-60 fill pool, so candidates "
        "stay schedulable under the 60-point no-repeat rule."
    ),
)
@click.option(
    "--exclude-recent-answers/--no-exclude-recent-answers",
    default=True,
    help=(
        "Fetch answers used in recently scheduled dailies (7 days before "
        "the first unscheduled daily slot, plus all scheduled future days) "
        "from the HeyGG admin API and exclude them from every fill pool, "
        "so candidates stay schedulable under the +/-6-day answer "
        "no-repeat rule."
    ),
)
@click.option(
    "--exclude-answers-file",
    "exclude_answers_files",
    multiple=True,
    type=click.Path(exists=True),
    help=(
        "File of answer words (one per line) to exclude from every fill "
        "pool, in addition to the API-driven exclusions. Repeatable. Used "
        "for targeted replacement runs that must avoid every answer "
        "already in a batch (see check-batch-answers --write-answers-file)."
    ),
)
@click.option(
    "--exclude-answers-min-length",
    type=int,
    default=4,
    help=(
        "Minimum answer length for the intra-batch exclusions "
        "(--exclude-answers-file words and answers used by completed "
        "batch-mates in this run). Short glue (<= 3 letters) saturates "
        "9x9 fills — excluding it makes grids unfillable — and the "
        "scheduler tolerates 3-letter repeats between 9x9s placed 3+ days "
        "apart. API-driven exclusions are unaffected."
    ),
)
@click.option(
    "--intra-batch-dedup/--no-intra-batch-dedup",
    default=True,
    help=(
        "Drop answers used by completed batch-mates from each subsequent "
        "puzzle's fill pool, so no two puzzles in the batch share an answer. "
        "On by default for dated dailies (a batch scheduled across consecutive "
        "days would otherwise trip the no-repeat windows). Disable for "
        "unlimited / non-scheduled pools, where puzzles are never "
        "schedule-adjacent and forcing disjoint answer sets only hurts fill "
        "quality. When disabled, the post-batch check-batch-answers gate is "
        "not meaningful — intra-batch answer overlap is expected."
    ),
)
@click.option(
    "--unlimited-answer-novelty/--no-unlimited-answer-novelty",
    default=True,
    help=(
        "For unlimited/non-scheduled batches (when --no-intra-batch-dedup), "
        "fetch active unlimited-pool answers and prefer candidate fills that "
        "reuse the least frequent answers. Generated answers increment the "
        "same in-run weights for later batch items."
    ),
)
@click.option(
    "--answer-novelty-candidates",
    type=int,
    default=8,
    help=(
        "Number of passing fill candidates to collect per puzzle when "
        "--unlimited-answer-novelty is active."
    ),
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
@click.option(
    "--no-llm-log",
    is_flag=True,
    default=False,
    help="Disable structured per-seed LLM call logging.",
)
@click.option(
    "--max-workers",
    type=int,
    default=1,
    help=(
        "Number of puzzles to generate concurrently. 1 (default) runs "
        "sequentially; higher values run that many puzzles in parallel "
        "(bounded by API rate limits)."
    ),
)
def generate_pilot_batch(
    output_root: str,
    batch_id: str,
    count: int,
    bucket_counts: str | None,
    seed_start: int,
    buckets: str | None,
    llm_provider: str,
    avoid_existing_clues: bool,
    api_base: str | None,
    exclude_scheduled_sixty: bool,
    exclude_recent_answers: bool,
    exclude_answers_files: tuple[str, ...],
    exclude_answers_min_length: int,
    intra_batch_dedup: bool,
    unlimited_answer_novelty: bool,
    answer_novelty_candidates: int,
    per_pattern_attempts: int,
    max_grid_variants: int,
    timeout_5: int,
    timeout_7: int,
    timeout_9: int,
    verbose: bool,
    no_llm_log: bool,
    max_workers: int,
) -> None:
    """Generate the Phase 2B pilot batch and write a JSON manifest."""
    _setup_logging(verbose)
    if max_workers < 1:
        raise click.BadParameter("--max-workers must be >= 1")
    if answer_novelty_candidates < 1:
        raise click.BadParameter("--answer-novelty-candidates must be >= 1")

    project_root = find_project_root()
    root = Path(output_root)
    if not root.is_absolute():
        root = project_root / root
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    all_buckets = _batch_bucket_configs(project_root)
    if buckets:
        wanted = {tag.strip() for tag in buckets.split(",") if tag.strip()}
        unknown = wanted - {f"{d}/{s}" for d, s, _, _ in all_buckets}
        if unknown:
            raise click.BadParameter(
                f"Unknown bucket(s): {', '.join(sorted(unknown))}. "
                f"Valid: {', '.join(f'{d}/{s}' for d, s, _, _ in all_buckets)}"
            )
        selected_buckets = [
            b for b in all_buckets if f"{b[0]}/{b[1]}" in wanted
        ]
    else:
        selected_buckets = all_buckets

    count_by_bucket = _parse_batch_count_overrides(
        bucket_counts,
        selected_buckets,
        count,
    )

    clue_history = ClueHistoryIndex()
    if avoid_existing_clues:
        try:
            loaded_records = _load_existing_clue_history(
                clue_history,
                selected_buckets,
                api_base=api_base,
            )
        except KeyError as exc:
            missing = exc.args[0]
            click.echo(
                f"Missing required environment variable: {missing}. "
                "Prior-clue history is loaded by default so generation can "
                "avoid repeats; pass --no-avoid-existing-clues to skip.",
                err=True,
            )
            sys.exit(1)
        except Exception as exc:
            click.echo(
                f"Failed to load existing clue history: {exc}. "
                "Pass --no-avoid-existing-clues to skip.",
                err=True,
            )
            sys.exit(1)
        click.echo(
            "Loaded existing clue history: "
            f"records={loaded_records}, "
            f"answers={clue_history.answer_count}, "
            f"clues={clue_history.clue_count}"
        )

    scheduled_sixty: list[str] = []
    needs_sixty_exclusion = exclude_scheduled_sixty and any(
        difficulty == "hard"
        and size in (7, 9)
        and count_by_bucket[f"{difficulty}/{size}"] > 0
        for difficulty, size, _, _ in selected_buckets
    )
    if needs_sixty_exclusion:
        from crossword_generator.data_store import fetch_recent_sixty_answers

        try:
            scheduled_sixty = fetch_recent_sixty_answers(api_base=api_base)
        except KeyError as exc:
            click.echo(
                f"Missing required environment variable: {exc.args[0]}. "
                "Hard 7x7/9x9 buckets exclude scheduled HGG 60 answers by "
                "default; pass --no-exclude-scheduled-sixty to skip.",
                err=True,
            )
            sys.exit(1)
        except Exception as exc:
            click.echo(
                f"Failed to fetch scheduled HGG 60 answers: {exc}. "
                "Pass --no-exclude-scheduled-sixty to skip.",
                err=True,
            )
            sys.exit(1)
        if not scheduled_sixty:
            click.echo("No scheduled HGG 60 answers to exclude.")

    recent_answers: list[str] = []
    recent_meta = None
    needs_recent_exclusion = exclude_recent_answers and any(
        count > 0 for count in count_by_bucket.values()
    )
    if needs_recent_exclusion:
        from crossword_generator.data_store import fetch_recent_daily_answers

        try:
            recent_meta = fetch_recent_daily_answers(api_base=api_base)
        except KeyError as exc:
            click.echo(
                f"Missing required environment variable: {exc.args[0]}. "
                "Recently scheduled answers are excluded from fill pools by "
                "default so candidates stay schedulable; pass "
                "--no-exclude-recent-answers to skip.",
                err=True,
            )
            sys.exit(1)
        except Exception as exc:
            click.echo(
                f"Failed to fetch recently scheduled answers: {exc}. "
                "Pass --no-exclude-recent-answers to skip.",
                err=True,
            )
            sys.exit(1)
        recent_answers = recent_meta.answers
        click.echo(
            f"Excluding {len(recent_answers)} recently scheduled answer(s) "
            f"from fill pools ({recent_meta.window_days}-day window before "
            f"first unscheduled slot {recent_meta.first_unscheduled_date})."
        )

    extra_excluded_answers: set[str] = set()
    for answers_file in exclude_answers_files:
        extra_excluded_answers.update(
            word.strip().upper()
            for word in Path(answers_file).read_text().splitlines()
            if len(word.strip()) >= exclude_answers_min_length
        )
    if extra_excluded_answers:
        click.echo(
            f"Excluding {len(extra_excluded_answers)} answer(s) of length "
            f">={exclude_answers_min_length} from "
            f"{len(exclude_answers_files)} --exclude-answers-file file(s)."
        )

    # Write filtered copies of every fill dictionary the active buckets
    # reference. The hgg-60 pool excludes the 180-day sixty list plus the
    # recent answers; base pools exclude only the recent answers. Both also
    # exclude any --exclude-answers-file words.
    dictionary_overrides: dict[str, str] = {}
    removed_by_dictionary: dict[str, int] = {}
    for filename in sorted(
        _referenced_dictionary_filenames(selected_buckets, count_by_bucket)
    ):
        excluded = set(recent_answers) | extra_excluded_answers
        if filename == SIXTY_DICTIONARY_FILENAME:
            excluded.update(scheduled_sixty)
        source = project_root / "dictionaries" / filename
        if not excluded or not source.exists():
            continue
        target_filename = (
            SIXTY_FILTERED_FILENAME
            if filename == SIXTY_DICTIONARY_FILENAME
            else f"{Path(filename).stem}-recent-filtered.txt"
        )
        override_path, removed = _write_filtered_dictionary(
            project_root, root, filename, sorted(excluded), target_filename
        )
        dictionary_overrides[filename] = override_path
        removed_by_dictionary[filename] = removed
    if removed_by_dictionary:
        details = ", ".join(
            f"{name}: {removed}"
            for name, removed in sorted(removed_by_dictionary.items())
        )
        click.echo(f"Filtered dictionary rows removed ({details}).")

    started_at = _utc_timestamp()

    # Flatten all (bucket, seed) items into one work-list so they can run
    # concurrently. Puzzles are independent; the only shared state is the
    # thread-safe clue_history and used_answers.
    work_items: list[tuple[str, int, str, Path, int]] = []
    for difficulty, size, puzzle_type, config_path in selected_buckets:
        bucket_count = count_by_bucket[f"{difficulty}/{size}"]
        for seed in range(seed_start, seed_start + bucket_count):
            work_items.append((difficulty, size, puzzle_type, config_path, seed))

    # Answers used by completed batch items. Each new item starts with a
    # snapshot removed from its fill dictionary, so a batch scheduled across
    # one week cannot repeat answers between its own puzzles. The CSP fill
    # is heavily biased toward the same words on the same grid (different
    # seeds converge on near-identical fills), so this exclusion is what
    # actually guarantees intra-batch answer uniqueness. Parallel workers
    # can't see in-flight batch-mates; check-batch-answers catches the
    # residue after the run.
    used_answers = _UsedAnswerSet()
    novelty_active = unlimited_answer_novelty and not intra_batch_dedup
    answer_usage_by_bucket: dict[tuple[str, int], _AnswerUsageCounter] = {}
    if novelty_active:
        try:
            answer_usage_by_bucket = _load_unlimited_answer_usage_by_bucket(
                selected_buckets,
                count_by_bucket,
                api_base=api_base,
            )
        except KeyError as exc:
            click.echo(
                f"Missing required environment variable: {exc.args[0]}. "
                "Unlimited answer novelty is loaded by default for "
                "non-scheduled batches; pass --no-unlimited-answer-novelty "
                "to skip.",
                err=True,
            )
            sys.exit(1)
        except Exception as exc:
            click.echo(
                f"Failed to load unlimited answer usage: {exc}. "
                "Pass --no-unlimited-answer-novelty to skip.",
                err=True,
            )
            sys.exit(1)
        for (difficulty, size), usage in sorted(answer_usage_by_bucket.items()):
            stats = usage.stats()
            click.echo(
                f"Loaded unlimited answer usage for {difficulty}/{size}x{size}: "
                f"{stats['records']} record(s), {stats['answers']} answer(s), "
                f"{stats['unique_answers']} unique."
            )

    def _run_item(item: tuple[str, int, str, Path, int]) -> dict[str, object]:
        difficulty, size, puzzle_type, config_path, seed = item
        # Minis can safely drop 3-letter answers a midi already used (they
        # carry little glue and refill fine); midis keep their 3-letter glue
        # to stay fillable. Never exclude more aggressively than the operator's
        # global floor.
        item_min_length = (
            min(exclude_answers_min_length, 3)
            if size in (5, 7)
            else exclude_answers_min_length
        )
        result = _run_batch_item(
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
            clue_history=clue_history,
            llm_logging_enabled=not no_llm_log,
            dictionary_overrides=dictionary_overrides,
            keep_sweep_context=max_workers > 1,
            excluded_fill_words=used_answers.snapshot(min_length=item_min_length),
            answer_usage_counts=(
                answer_usage_by_bucket[(difficulty, size)].snapshot()
                if novelty_active and (difficulty, size) in answer_usage_by_bucket
                else None
            ),
            answer_novelty_candidates=answer_novelty_candidates,
        )
        if result["success"] and intra_batch_dedup:
            try:
                puzzle = json.loads(
                    Path(str(result["output_path"])).read_text()
                )
                used_answers.add(extract_ipuz_answers(puzzle))
            except (OSError, json.JSONDecodeError):
                logging.getLogger(__name__).warning(
                    "Could not index answers from %s for in-batch "
                    "exclusion",
                    result["output_path"],
                )
        if result["success"] and novelty_active:
            usage = answer_usage_by_bucket.get((difficulty, size))
            if usage is not None:
                try:
                    puzzle = json.loads(
                        Path(str(result["output_path"])).read_text()
                    )
                    usage.add(extract_ipuz_answers(puzzle), records=1)
                except (OSError, json.JSONDecodeError):
                    logging.getLogger(__name__).warning(
                        "Could not index answers from %s for unlimited "
                        "answer-novelty weighting",
                        result["output_path"],
                    )
        return result

    def _report(result: dict[str, object]) -> None:
        status = "ok" if result["success"] else "failed"
        click.echo(
            f"{result['difficulty']} {result['size']}x{result['size']} "
            f"seed {result['seed']}: {status} "
            f"({result['runtime_seconds']}s)"
        )

    results: list[dict[str, object]] = []
    if max_workers <= 1:
        for item in work_items:
            result = _run_item(item)
            results.append(result)
            _report(result)
    else:
        workers = min(max_workers, len(work_items))
        click.echo(
            f"Generating {len(work_items)} puzzle(s) with {workers} worker(s)..."
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_run_item, item) for item in work_items]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                _report(result)

    # Deterministic manifest order regardless of completion order.
    results.sort(
        key=lambda r: (str(r["difficulty"]), int(r["size"]), int(r["seed"]))
    )

    # Parallel workers can't see each other's in-flight clues, so two
    # concurrent puzzles can produce the same clue for a shared answer
    # without either noticing. Sweep completed puzzles against each other
    # and repair collisions; sequential runs have no such race.
    if max_workers > 1:
        sweep_stats = _run_duplicate_sweep(results, clue_history)
    else:
        sweep_stats = {"checked": 0, "repaired": 0, "unresolved": 0}
        for r in results:
            r.pop("_sweep", None)

    manifest = {
        "batch": batch_id,
        "started_at": started_at,
        "finished_at": _utc_timestamp(),
        "output_root": str(root),
        "logs_dir": str(logs_dir),
        "count_per_bucket": count,
        "bucket_counts": count_by_bucket,
        "seed_start": seed_start,
        "llm_provider": llm_provider,
        "avoid_existing_clues": avoid_existing_clues,
        "exclude_scheduled_sixty": {
            "enabled": exclude_scheduled_sixty,
            "applied": SIXTY_DICTIONARY_FILENAME in dictionary_overrides,
            "excluded_count": removed_by_dictionary.get(
                SIXTY_DICTIONARY_FILENAME, 0
            ),
            "dictionary_path": dictionary_overrides.get(
                SIXTY_DICTIONARY_FILENAME
            ),
        },
        "exclude_recent_answers": {
            "enabled": exclude_recent_answers,
            "applied": bool(recent_answers),
            "answer_count": len(recent_answers),
            "window_days": recent_meta.window_days if recent_meta else None,
            "forward_days": recent_meta.forward_days if recent_meta else None,
            "until_date": recent_meta.until_date if recent_meta else None,
            "first_unscheduled_date": (
                recent_meta.first_unscheduled_date if recent_meta else None
            ),
            "removed_by_dictionary": removed_by_dictionary,
            "dictionary_paths": dictionary_overrides,
        },
        "exclude_answers_files": {
            "paths": list(exclude_answers_files),
            "answer_count": len(extra_excluded_answers),
            "min_length": exclude_answers_min_length,
        },
        "intra_batch_dedup": intra_batch_dedup,
        "unlimited_answer_novelty": {
            "enabled": unlimited_answer_novelty,
            "active": novelty_active,
            "answer_novelty_candidates": answer_novelty_candidates,
            "loaded": {
                f"{difficulty}/{size}": usage.stats()
                for (difficulty, size), usage in sorted(
                    answer_usage_by_bucket.items()
                )
            },
        },
        "llm_logging_enabled": not no_llm_log,
        "max_workers": max_workers,
        "duplicate_sweep": {
            "enabled": max_workers > 1,
            **sweep_stats,
        },
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


def _parse_batch_count_overrides(
    raw_counts: str | None,
    selected_buckets: list[tuple[str, int, str, Path]],
    default_count: int,
) -> dict[str, int]:
    if default_count < 0:
        raise click.BadParameter("--count cannot be negative")

    counts = {
        f"{difficulty}/{size}": default_count
        for difficulty, size, _, _ in selected_buckets
    }
    if not raw_counts:
        return counts

    selected_by_size: dict[int, list[str]] = {}
    for difficulty, size, _, _ in selected_buckets:
        selected_by_size.setdefault(size, []).append(f"{difficulty}/{size}")

    for item in raw_counts.split(","):
        entry = item.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise click.BadParameter(
                "Bucket counts must use KEY=COUNT entries, e.g. 5=5,7=2,9=7"
            )
        raw_key, raw_value = (part.strip().lower() for part in entry.split("=", 1))
        try:
            override_count = int(raw_value)
        except ValueError as exc:
            raise click.BadParameter(
                f"Invalid count for {raw_key!r}: {raw_value!r}"
            ) from exc
        if override_count < 0:
            raise click.BadParameter(f"Count for {raw_key!r} cannot be negative")

        if raw_key in counts:
            counts[raw_key] = override_count
            continue

        size = _parse_bucket_count_size_key(raw_key)
        if size is None or size not in selected_by_size:
            valid = sorted(counts) + [
                str(size_key) for size_key in sorted(selected_by_size)
            ]
            raise click.BadParameter(
                f"Unknown bucket count key {raw_key!r}. "
                f"Valid keys: {', '.join(valid)}"
            )
        for bucket_tag in selected_by_size[size]:
            counts[bucket_tag] = override_count

    return counts


def _parse_bucket_count_size_key(raw_key: str) -> int | None:
    if raw_key.isdigit():
        return int(raw_key)
    left, separator, right = raw_key.partition("x")
    if separator and left == right and left.isdigit():
        return int(left)
    return None


SIXTY_DICTIONARY_FILENAME = "hgg-60.txt"
SIXTY_FILTERED_FILENAME = "hgg-60-scheduled-filtered.txt"


class _UsedAnswerSet:
    """Thread-safe set of answers used by completed batch items.

    All answers (length >= 3) are tracked; the length floor is applied per
    consumer at ``snapshot`` time, not at ``add`` time. This lets a mini
    (5x5/7x7) exclude a 3-letter answer a midi already used while a midi keeps
    its 3-letter glue (excluding it makes 9x9 grids unfillable). The
    scheduler's +/-2 short-answer window only protects 9x9 placements, so a
    3-letter answer shared between a mini and a midi within 6 days is a real
    conflict — driven by the mini side — and the mini must avoid it.
    """

    def __init__(self) -> None:
        self._answers: set[str] = set()
        self._lock = threading.Lock()

    def snapshot(self, *, min_length: int = 4) -> set[str]:
        with self._lock:
            return {a for a in self._answers if len(a) >= min_length}

    def add(self, answers: Iterable[str]) -> None:
        with self._lock:
            self._answers.update(
                answer.strip().upper()
                for answer in answers
                if len(answer.strip()) >= 3
            )


class _AnswerUsageCounter:
    """Thread-safe answer frequency map for novelty-aware unlimited fills."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}
        self._records = 0
        self._answers = 0
        self._lock = threading.Lock()

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

    def add(self, answers: Iterable[str], *, records: int = 0) -> None:
        normalized = [
            answer.strip().upper()
            for answer in answers
            if len(answer.strip()) >= 3
        ]
        with self._lock:
            self._records += records
            self._answers += len(normalized)
            for answer in normalized:
                self._counts[answer] = self._counts.get(answer, 0) + 1

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "records": self._records,
                "answers": self._answers,
                "unique_answers": len(self._counts),
            }


def _load_unlimited_answer_usage_by_bucket(
    selected_buckets: list[tuple[str, int, str, Path]],
    count_by_bucket: dict[str, int],
    *,
    api_base: str | None = None,
) -> dict[tuple[str, int], _AnswerUsageCounter]:
    from crossword_generator.data_store import list_unlimited_puzzle_records

    usage_by_bucket: dict[tuple[str, int], _AnswerUsageCounter] = {}
    seen: set[tuple[str, int]] = set()
    for difficulty, size, _, _ in selected_buckets:
        if count_by_bucket[f"{difficulty}/{size}"] <= 0:
            continue
        key = (difficulty, size)
        if key in seen:
            continue
        seen.add(key)

        usage = _AnswerUsageCounter()
        records = list_unlimited_puzzle_records(
            game_key=_game_key_for_size(size),
            size=size,
            difficulty=difficulty,
            api_base=api_base,
        )
        for record in records:
            answers = _extract_record_answers(record)
            if answers:
                usage.add(answers, records=1)
        usage_by_bucket[key] = usage
    return usage_by_bucket


def _extract_record_answers(record: dict[str, object]) -> list[str]:
    data = record.get("data")
    if not isinstance(data, dict):
        return []
    puzzle = data.get("puzzle")
    if isinstance(puzzle, dict):
        return extract_ipuz_answers(puzzle)
    return extract_ipuz_answers(data)


def _referenced_dictionary_filenames(
    selected_buckets: list[tuple[str, int, str, Path]],
    count_by_bucket: dict[str, int],
) -> set[str]:
    """Fill-dictionary filenames referenced by the active buckets' configs."""
    filenames: set[str] = set()
    for difficulty, size, _, config_path in selected_buckets:
        if count_by_bucket[f"{difficulty}/{size}"] <= 0:
            continue
        config = load_config(config_path)
        filenames.update(
            Path(path).name
            for path in (
                config.dictionary.path,
                config.dictionary.themed_path,
                config.fill.csp.dictionary_path,
                *config.dictionary.additional_paths,
                *config.dictionary.themed_additional_paths,
                *config.fill.csp.additional_dictionary_paths,
            )
        )
    return filenames


def _write_filtered_dictionary(
    project_root: Path,
    output_root: Path,
    source_filename: str,
    excluded_answers: list[str],
    target_filename: str,
) -> tuple[str, int]:
    """Write a copy of a dictionary without the excluded answers.

    Returns the absolute path of the filtered file and the number of
    dictionary rows removed.
    """
    source = project_root / "dictionaries" / source_filename
    excluded = {answer.strip().upper() for answer in excluded_answers}
    kept: list[str] = []
    removed = 0
    for line in source.read_text().splitlines():
        word = line.split(";", 1)[0].strip().upper()
        if word and word in excluded:
            removed += 1
            continue
        kept.append(line)
    target = output_root / target_filename
    target.write_text("\n".join(kept) + "\n")
    return str(target), removed


def _apply_dictionary_overrides(
    config: Config, overrides: dict[str, str]
) -> None:
    """Point dictionary references at filtered copies, matched by filename.

    Configs reference each dictionary in several places (pipeline
    dictionary, themed pipeline dictionary, CSP filler); all must agree or
    the filler could pick a word the grader pool no longer contains.
    """

    def _swap_one(path: str) -> str:
        return overrides.get(Path(path).name, path)

    def _swap(paths: list[str]) -> list[str]:
        return [_swap_one(path) for path in paths]

    config.dictionary.path = _swap_one(config.dictionary.path)
    config.dictionary.themed_path = _swap_one(config.dictionary.themed_path)
    config.dictionary.additional_paths = _swap(
        config.dictionary.additional_paths
    )
    config.dictionary.themed_additional_paths = _swap(
        config.dictionary.themed_additional_paths
    )
    config.fill.csp.dictionary_path = _swap_one(
        config.fill.csp.dictionary_path
    )
    config.fill.csp.additional_dictionary_paths = _swap(
        config.fill.csp.additional_dictionary_paths
    )


@main.command(name="check-batch-answers")
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True),
    required=True,
    help="Batch manifest produced by generate-pilot-batch.",
)
@click.option(
    "--write-answers-file",
    type=click.Path(),
    default=None,
    help=(
        "Also write every answer found in the batch (one per line, "
        "deduplicated) for use with generate-pilot-batch "
        "--exclude-answers-file in replacement runs."
    ),
)
@click.option(
    "--allow-short-window",
    is_flag=True,
    default=False,
    help=(
        "Treat short-window duplicates (3-letter answers shared only "
        "between 9x9 puzzles) as warnings rather than failures. The "
        "scheduler allows those repeats when the puzzles are placed 3+ "
        "days apart."
    ),
)
def check_batch_answers(
    manifest_path: str,
    write_answers_file: str | None,
    allow_short_window: bool,
) -> None:
    """Report duplicate answers shared across puzzles in a generated batch.

    A weekly batch is scheduled across consecutive days, so any answer
    appearing in two puzzles would trip the scheduling-time no-repeat
    windows. Exits non-zero when any cross-puzzle duplicate exists.
    Duplicates of 3-letter answers confined to 9x9 puzzles are labelled
    short-window (the scheduler allows those 3+ days apart), but still
    count as failures: an all-unique batch schedules without manual care.
    """
    from crossword_generator.clue_history import extract_ipuz_answers

    manifest = json.loads(Path(manifest_path).read_text())
    puzzles: list[tuple[str, int, list[str]]] = []
    for result in manifest.get("results", []):
        if not result.get("success"):
            continue
        output_path = Path(str(result.get("output_path", "")))
        if not output_path.exists():
            click.echo(f"Missing output file: {output_path}", err=True)
            sys.exit(2)
        label = (
            f"{result['difficulty']}/{result['size']}x{result['size']}"
            f"/seed-{result['seed']}"
        )
        answers = [
            answer.strip().upper()
            for answer in extract_ipuz_answers(
                json.loads(output_path.read_text())
            )
        ]
        puzzles.append((label, int(result["size"]), answers))

    # Dedupe within each puzzle: the fill grader already guards intra-puzzle
    # repeats; this check targets answers shared across puzzles.
    by_answer: dict[str, list[tuple[str, int]]] = {}
    for label, size, answers in puzzles:
        for answer in sorted(set(answers)):
            by_answer.setdefault(answer, []).append((label, size))

    if write_answers_file:
        Path(write_answers_file).write_text(
            "\n".join(sorted(by_answer)) + "\n"
        )
        click.echo(
            f"Wrote {len(by_answer)} unique answer(s) to {write_answers_file}"
        )

    total_answers = sum(len(answers) for _, _, answers in puzzles)
    duplicates = {
        answer: hits
        for answer, hits in by_answer.items()
        if len(hits) > 1
    }
    click.echo(
        f"{manifest.get('batch', '<batch>')}: {len(puzzles)} puzzle(s), "
        f"{total_answers} answer(s), {len(by_answer)} unique."
    )
    if not duplicates:
        click.echo("No duplicate answers across puzzles.")
        return

    blocking_count = 0
    short_window_count = 0
    for answer in sorted(duplicates):
        hits = duplicates[answer]
        short_window = len(answer) <= 3 and all(
            size == 9 for _, size in hits
        )
        if short_window:
            short_window_count += 1
        else:
            blocking_count += 1
        kind = "short-window (9x9-only, +/-2 days)" if short_window else "blocking"
        labels = ", ".join(label for label, _ in hits)
        click.echo(f"DUPLICATE [{kind}]: {answer} — {labels}")
    click.echo(
        f"{len(duplicates)} duplicate answer(s) across puzzles "
        f"({blocking_count} blocking, {short_window_count} short-window)."
    )
    if blocking_count or not allow_short_window:
        click.echo(
            "Regenerate the affected puzzles with --exclude-answers-file "
            "before uploading."
        )
        sys.exit(1)
    click.echo(
        "Short-window duplicates allowed (--allow-short-window): schedule "
        "the affected 9x9s 3+ days apart."
    )


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
    "--delete-existing-size",
    "delete_existing_sizes",
    type=int,
    multiple=True,
    help=(
        "Delete existing generated-puzzle records for this grid size before "
        "uploading. For 5x5/7x7 this uses --mini-game-key; for 9x9+ it uses "
        "--midi-game-key."
    ),
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Build and validate records without calling the HeyGG API.",
)
@click.option(
    "--allow-leaks",
    is_flag=True,
    default=False,
    help=(
        "Upload puzzles even if a clue leak survived repair (LEAK: error). "
        "Off by default — leaked puzzles are refused."
    ),
)
@click.option(
    "--flag-issues",
    is_flag=True,
    default=False,
    help=(
        "Upload puzzles with surviving LEAK/DUPLICATE clue issues instead of "
        "holding them back, attaching the issues to metadata.clue_issues and "
        "setting review_status=needs_attention so the admin UI flags the "
        "specific clues for the editor. Ignored if --allow-leaks is set."
    ),
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
    delete_existing_sizes: tuple[int, ...],
    dry_run: bool,
    allow_leaks: bool,
    flag_issues: bool,
) -> None:
    """Save generated puzzle candidates to the HeyGG admin data store."""
    from crossword_generator.data_store import (
        bulk_save_generated_puzzles,
        delete_generated_puzzle_records,
        list_generated_puzzle_records,
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
        allow_leaks=allow_leaks,
        flag_issues=flag_issues,
    )

    flagged = sum(
        1 for r in records if r["metadata"].get("clue_issues")
    )
    click.echo(f"Prepared {len(records)} generated puzzle record(s).")
    if flagged:
        click.echo(
            f"  {flagged} flagged for review (review_status=needs_attention)."
        )
    if not records:
        return

    if dry_run:
        click.echo("Dry run: no HeyGG API write calls made.")
        for record in records[:5]:
            click.echo(f"  {record['game_key']} {record['key']}")
        if len(records) > 5:
            click.echo(f"  ... {len(records) - 5} more")
        for size in delete_existing_sizes:
            game_key = mini_game_key if size in (5, 7) else midi_game_key
            existing = list_generated_puzzle_records(
                game_key=game_key,
                size=size,
                api_base=api_base,
            )
            click.echo(
                f"Dry run: would delete {len(existing)} existing "
                f"{game_key} {size}x{size} generated record(s)."
            )
        return

    try:
        for size in delete_existing_sizes:
            game_key = mini_game_key if size in (5, 7) else midi_game_key
            existing = list_generated_puzzle_records(
                game_key=game_key,
                size=size,
                api_base=api_base,
            )
            deleted = delete_generated_puzzle_records(
                existing,
                api_base=api_base,
            )
            click.echo(
                f"Deleted existing generated puzzles: "
                f"{game_key} {size}x{size} deleted={len(deleted)}"
            )

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
    dictionary = Dictionary.load_many(
        [
            project_root / path
            for path in [
                config.dictionary.path,
                *config.dictionary.additional_paths,
            ]
        ],
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
    dictionary = Dictionary.load_many(
        [
            project_root / path
            for path in [
                config.dictionary.path,
                *config.dictionary.additional_paths,
            ]
        ],
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
    dictionary = Dictionary.load_many(
        [
            project_root / path
            for path in [
                config.dictionary.path,
                *config.dictionary.additional_paths,
            ]
        ],
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
    "--easy-extra-source",
    "easy_extra_sources",
    type=click.Path(exists=True),
    multiple=True,
    help="Additional source file to merge into the easy dictionary.",
)
@click.option(
    "--easy-exclude-source",
    "easy_exclude_sources",
    type=click.Path(exists=True),
    multiple=True,
    help="Plain or semicolon-delimited word list to exclude from easy output.",
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
    default="dictionaries/hgg-easy.txt",
    help="Output path for the effective HGG Easy dictionary.",
)
@click.option(
    "--sixty-output",
    type=click.Path(),
    default="dictionaries/hgg-60.txt",
    help="Output path for the HGG 60 dictionary.",
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
    default=50,
    help="Flat score to assign to HGG Easy entries.",
)
@click.option(
    "--long-word-min-source-score",
    type=int,
    default=60,
    show_default=True,
    help=(
        "Minimum source score for scored 6-, 7-, 8-, and 9-letter entries. "
        "Use 0 to disable this length-specific source-score filter."
    ),
)
def prepare_dictionaries(
    easy_source: str,
    easy_extra_sources: tuple[str, ...],
    easy_exclude_sources: tuple[str, ...],
    hard_source: str,
    easy_output: str,
    sixty_output: str,
    hard_output: str,
    score: int,
    long_word_min_source_score: int,
) -> None:
    """Prepare flat-score easy and hard dictionaries for batch experiments.

    All input dictionaries (including exclude sources) come from committed
    .txt files. Use `crossword-generator consolidate-list` to refresh those
    files from hey-you before running this command.
    """
    from crossword_generator.dictionary_prep import (
        format_summary,
        load_excluded_words,
        prepare_hgg_easy_dictionary,
        prepare_length_mixed_flat_dictionary,
        prepare_sixty_dictionary,
    )

    project_root = find_project_root()

    def resolve_path(path: str) -> Path:
        resolved = Path(path)
        return resolved if resolved.is_absolute() else project_root / resolved

    resolved_extra_sources = [
        resolve_path(source) for source in easy_extra_sources
    ]
    resolved_base_exclude_sources = [
        resolve_path(source) for source in easy_exclude_sources
    ]

    # Auto-discover the two thumbs-down lists that `consolidate-list`
    # writes. Operators don't need to remember to pass these as flags;
    # they get unioned in if (and only if) the file is on disk. Easy
    # thumbs-down propagates only to easy; hard thumbs-down lands only
    # on hard — matches the "easy reject ≠ hard reject" semantic.
    thumbs_easy = project_root / "dictionaries" / "HggThumbsDownEasy.txt"
    thumbs_hard = project_root / "dictionaries" / "HggThumbsDownHard.txt"
    base_excluded_words = load_excluded_words(resolved_base_exclude_sources)
    extra_easy_exclude: set[str] = (
        load_excluded_words([thumbs_easy]) if thumbs_easy.exists() else set()
    )
    extra_hard_exclude: set[str] = (
        load_excluded_words([thumbs_hard]) if thumbs_hard.exists() else set()
    )

    excluded_easy_words = base_excluded_words | extra_easy_exclude
    hard_exclude_words: set[str] = base_excluded_words | extra_hard_exclude

    if thumbs_easy.exists() or thumbs_hard.exists():
        easy_count = (
            "+" + str(len(extra_easy_exclude))
            if thumbs_easy.exists()
            else "none"
        )
        hard_count = (
            "+" + str(len(extra_hard_exclude))
            if thumbs_hard.exists()
            else "none"
        )
        click.echo(
            f"Thumbs-down lists auto-discovered: "
            f"easy={easy_count}, hard={hard_count}"
        )

    min_source_score_by_length = (
        {
            6: long_word_min_source_score,
            7: long_word_min_source_score,
            8: long_word_min_source_score,
            9: long_word_min_source_score,
        }
        if long_word_min_source_score > 0
        else {}
    )
    easy_source_path = resolve_path(easy_source)
    easy_output_path = resolve_path(easy_output)
    sixty_output_path = resolve_path(sixty_output)
    hard_source_path = resolve_path(hard_source)
    hard_output_path = resolve_path(hard_output)

    easy_summary = prepare_hgg_easy_dictionary(
        easy_source_path,
        hard_source_path,
        easy_output_path,
        score=score,
        extra_input_paths=resolved_extra_sources,
        exclude_words=excluded_easy_words,
    )
    click.echo("Easy dictionary:")
    click.echo(format_summary(easy_summary))
    click.echo("")

    sixty_summary = prepare_sixty_dictionary(
        hard_source_path,
        sixty_output_path,
        score=60,
        exclude_words=base_excluded_words,
    )
    click.echo("HGG 60 dictionary:")
    click.echo(format_summary(sixty_summary))
    click.echo("")

    hard_summary = prepare_length_mixed_flat_dictionary(
        easy_output_path,
        hard_source_path,
        hard_output_path,
        score=55,
        short_max_length=5,
        long_min_length=6,
        exclude_words=hard_exclude_words,
        min_source_score_by_length=min_source_score_by_length,
        flat_score_input_paths=[easy_output_path],
    )
    click.echo("Hard dictionary:")
    click.echo(format_summary(hard_summary))
    click.echo("")


@main.command(name="publish-effective-dictionaries")
@click.option(
    "--easy-source",
    type=click.Path(exists=True),
    default="dictionaries/HGGXW-Easy.txt",
    show_default=True,
    help="Path to the Easy source list (Jeff's consolidated HGGXW Easy).",
)
@click.option(
    "--easy-extra-source",
    "easy_extra_sources",
    type=click.Path(exists=True),
    multiple=True,
    default=(),
    help=(
        "Additional source file to merge into HGG Easy. HGGXW-Easy is "
        "already consolidated, so this is empty by default."
    ),
)
@click.option(
    "--easy-exclude-source",
    "easy_exclude_sources",
    type=click.Path(exists=True),
    multiple=True,
    default=(
        "dictionaries/XwiJeffChenList-NotFamilyFriendly.txt",
        "dictionaries/HggGeneratedSafetyExclude.txt",
    ),
    help="Plain or semicolon-delimited word list to exclude from HGG Easy.",
)
@click.option(
    "--hard-source",
    type=click.Path(exists=True),
    default="dictionaries/HGGXW-Hard.txt",
    show_default=True,
    help="Path to the Hard fill list (Jeff's consolidated HGGXW Hard).",
)
@click.option(
    "--sixty-source",
    type=click.Path(exists=True),
    default="dictionaries/XwiJeffChenList.txt",
    show_default=True,
    help=(
        "Scored master list providing the source-score 60 entries. "
        "HGGXW-Easy/Hard are plain (unscored), so 60-pointers come from here."
    ),
)
@click.option(
    "--easy-output",
    type=click.Path(),
    default="dictionaries/hgg-easy.txt",
    show_default=True,
    help="Local output path to update after a successful publish.",
)
@click.option(
    "--hard-output",
    type=click.Path(),
    default="dictionaries/hgg-hard.txt",
    show_default=True,
    help="Local output path for the combined Easy+Hard fill dictionary.",
)
@click.option(
    "--sixty-output",
    type=click.Path(),
    default="dictionaries/hgg-60.txt",
    show_default=True,
    help="Local output path to update after a successful publish.",
)
@click.option(
    "--api-base",
    default=None,
    help=(
        "Override the HeyGG admin API base URL. Defaults to the "
        "HEYGG_API_BASE_URL env var or https://play.hey.gg/api."
    ),
)
@click.option(
    "--generator-commit",
    default=None,
    help="Override the generator commit recorded in snapshot metadata.",
)
@click.option(
    "--timeout",
    type=int,
    default=120,
    show_default=True,
    help="HTTP timeout for the publish request.",
)
@click.option(
    "--write-local/--no-write-local",
    default=True,
    show_default=True,
    help="Write hgg-easy.txt and hgg-60.txt locally after publish succeeds.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Build and validate the snapshot without uploading or writing outputs.",
)
def publish_effective_dictionaries(
    easy_source: str,
    easy_extra_sources: tuple[str, ...],
    easy_exclude_sources: tuple[str, ...],
    hard_source: str,
    sixty_source: str,
    easy_output: str,
    hard_output: str,
    sixty_output: str,
    api_base: str | None,
    generator_commit: str | None,
    timeout: int,
    write_local: bool,
    dry_run: bool,
) -> None:
    """Build, validate, and atomically publish HGG Easy + HGG 60."""
    import shutil
    import tempfile

    from crossword_generator.dictionary_prep import format_summary
    from crossword_generator.effective_dictionaries import (
        EffectiveDictionaryError,
        build_effective_dictionaries,
        make_effective_dictionary_payload,
    )
    from crossword_generator.effective_dictionaries import (
        publish_effective_dictionaries as publish_snapshot,
    )

    project_root = find_project_root()

    def resolve_path(path: str) -> Path:
        resolved = Path(path)
        return resolved if resolved.is_absolute() else project_root / resolved

    with tempfile.TemporaryDirectory(prefix="hgg-effective-dicts-") as temp_dir:
        try:
            build = build_effective_dictionaries(
                project_root=project_root,
                output_dir=Path(temp_dir),
                easy_source=resolve_path(easy_source),
                easy_extra_sources=tuple(
                    resolve_path(source) for source in easy_extra_sources
                ),
                easy_exclude_sources=tuple(
                    resolve_path(source) for source in easy_exclude_sources
                ),
                hard_source=resolve_path(hard_source),
                sixty_source=resolve_path(sixty_source),
            )
            payload = make_effective_dictionary_payload(
                build,
                generator_commit=generator_commit,
            )
        except EffectiveDictionaryError as exc:
            click.echo(f"ERROR: {exc}", err=True)
            sys.exit(1)

        data_size = len(json.dumps(payload).encode())
        label = "DRY RUN — " if dry_run else ""
        click.echo(f"{label}Effective dictionary snapshot validated.")
        click.echo("")
        click.echo("HGG Easy dictionary:")
        click.echo(format_summary(build.easy_summary))
        click.echo("")
        click.echo("HGG Hard dictionary (Easy + Hard fill):")
        click.echo(format_summary(build.hard_summary))
        click.echo("")
        click.echo("HGG 60 dictionary:")
        click.echo(format_summary(build.sixty_summary))
        click.echo("")
        click.echo(
            "Snapshot payload: "
            "endpoint=/admin/crossword-effective-dictionaries/publish "
            f"bytes={data_size}"
        )

        if dry_run:
            click.echo("Dry run: no API call made and no local outputs written.")
            return

        try:
            result = publish_snapshot(
                build,
                api_base=api_base,
                timeout=timeout,
                generator_commit=generator_commit,
            )
        except Exception as exc:
            click.echo(f"ERROR: publish failed: {exc}", err=True)
            sys.exit(1)

        click.echo(f"Published effective dictionaries: {result.action}")

        if write_local:
            easy_output_path = resolve_path(easy_output)
            hard_output_path = resolve_path(hard_output)
            sixty_output_path = resolve_path(sixty_output)
            for path in (easy_output_path, hard_output_path, sixty_output_path):
                path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(build.easy.path, easy_output_path)
            shutil.copyfile(build.hard.path, hard_output_path)
            shutil.copyfile(build.sixty.path, sixty_output_path)
            click.echo(f"Wrote local output: {easy_output_path}")
            click.echo(f"Wrote local output: {hard_output_path}")
            click.echo(f"Wrote local output: {sixty_output_path}")


@main.command(name="consolidate-list")
@click.argument("slug", required=False)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Fetch and diff without writing files or marking consolidated on the server.",
)
@click.option(
    "--api-base",
    default=None,
    help=(
        "Override the HeyGG admin API base URL. Defaults to the "
        "HEYGG_API_BASE_URL env var or https://play.hey.gg/api."
    ),
)
def consolidate_list(slug: str | None, dry_run: bool, api_base: str | None) -> None:
    """Pull crossword_lists state from hey-you into the committed .txt files.

    Without an argument, consolidates every registered list. Pass a slug
    (e.g. `hgg-curated`) to operate on just that one. Always prints an
    added/removed diff; only writes when --dry-run is not set.

    Auth: requires HEYGG_ADMIN_TOKEN (or HEYGG_ADMIN_API_TOKEN) with the
    `puzzles.edit` permission.
    """
    from crossword_generator.consolidate_list import (
        ConsolidateError,
        consolidate_one,
        list_registered_lists,
    )

    project_root = find_project_root()

    if slug:
        slugs = [slug]
    else:
        try:
            registered = list_registered_lists(api_base=api_base)
        except ConsolidateError as exc:
            click.echo(f"ERROR: {exc}", err=True)
            sys.exit(1)
        slugs = [str(item["slug"]) for item in registered if item.get("slug")]
        if not slugs:
            click.echo("No registered crossword lists found.", err=True)
            sys.exit(1)

    label = "DRY RUN — " if dry_run else ""
    source_label = api_base or "default API"
    click.echo(
        f"{label}Consolidating {len(slugs)} list(s) from {source_label}"
    )
    click.echo("")

    exit_code = 0
    for current_slug in slugs:
        try:
            summary = consolidate_one(
                current_slug,
                project_root,
                api_base=api_base,
                dry_run=dry_run,
            )
        except ConsolidateError as exc:
            click.echo(f"  {current_slug}: ERROR — {exc}", err=True)
            exit_code = 1
            continue

        status = (
            "wrote" if summary.wrote
            else "no changes" if not dry_run
            else "would write" if (summary.added or summary.removed)
            else "no changes"
        )
        click.echo(
            f"  {summary.slug}: +{summary.added} / -{summary.removed} "
            f"({summary.total_after} active) → {summary.file_path} [{status}]"
        )

    click.echo("")
    if dry_run:
        click.echo("Dry run — no files written, no server state changed.")
    else:
        click.echo("Done. Review `git diff` and commit when ready.")

    if exit_code:
        sys.exit(exit_code)


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
    for size, expected_count, expected_weight in ((5, 12, 58), (7, 18, 50)):
        catalog = get_grid_patterns(PuzzleType.MINI, size)
        patterns = [
            (list(pattern.black_cells), pattern.weight)
            for pattern in catalog
        ]
        results = validate_weighted_patterns(size, patterns)
        summary = summarize_validations(results)
        unsupported = [
            str(index)
            for index, pattern in enumerate(catalog, start=1)
            if not pattern.symmetric
        ]
        regular = sum(
            1 for pattern in catalog if "rotational" in pattern.symmetries
        )
        mirror_only = sum(
            1
            for pattern in catalog
            if "vertical" in pattern.symmetries
            and "rotational" not in pattern.symmetries
        )

        click.echo(
            f"{size}x{size}: patterns={summary['patterns']} "
            f"total_weight={summary['total_weight']} "
            f"valid={summary['valid']} invalid={summary['invalid']} "
            f"regular={regular} mirror_only={mirror_only} "
            f"unsupported_symmetry={len(unsupported)}"
        )
        click.echo(
            "  unsupported symmetry pattern indexes: "
            + (", ".join(unsupported) if unsupported else "none")
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


class _ThreadFilter(logging.Filter):
    """Only pass log records emitted from one specific thread.

    Batch items attach their file + capture handlers to the root logger. When
    items run in parallel, this filter keeps each item's handlers from seeing
    other items' records, so logs and per-item metrics stay isolated.
    """

    def __init__(self, thread_id: int) -> None:
        super().__init__()
        self._thread_id = thread_id

    def filter(self, record: logging.LogRecord) -> bool:
        return record.thread == self._thread_id


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
    clue_history: ClueHistoryIndex | None = None,
    llm_logging_enabled: bool = True,
    dictionary_overrides: dict[str, str] | None = None,
    keep_sweep_context: bool = False,
    excluded_fill_words: set[str] | None = None,
    answer_usage_counts: dict[str, int] | None = None,
    answer_novelty_candidates: int = 1,
) -> dict[str, object]:
    bucket_dir = output_root / difficulty / f"{size}x{size}"
    bucket_dir.mkdir(parents=True, exist_ok=True)
    output_path = bucket_dir / f"seed-{seed:03d}.ipuz"
    log_path = logs_dir / f"{difficulty}-{size}x{size}-seed-{seed:03d}.log"
    llm_log_path = (
        logs_dir / f"{difficulty}-{size}x{size}-seed-{seed:03d}.llm.jsonl"
    )
    if output_path.exists():
        output_path.unlink()
    if llm_log_path.exists():
        llm_log_path.unlink()

    config = load_config(config_path)
    config.puzzle.type = puzzle_type
    config.puzzle.difficulty = difficulty
    config.puzzle.grid_size = size
    config.llm.provider = llm_provider
    config.output.directory = str(bucket_dir / "intermediates" / f"seed-{seed:03d}")
    config.output.formats = ["ipuz"]
    config.fill.max_retries = per_pattern_attempts
    config.fill.max_grid_variants = max_grid_variants
    config.fill.csp.timeout_by_size = timeout_by_size
    if answer_usage_counts:
        config.grading.fill.collect_boards = max(
            config.grading.fill.collect_boards,
            answer_novelty_candidates,
        )
    config.llm.logging.enabled = llm_logging_enabled
    config.llm.logging.path = str(llm_log_path)
    if puzzle_type == "midi" and size == 9:
        config.theme.enabled = False
    if dictionary_overrides:
        _apply_dictionary_overrides(config, dictionary_overrides)

    logger = logging.getLogger()
    # When items run in parallel they share the root logger, so scope each
    # item's handlers to its own thread to keep logs/metrics from interleaving.
    thread_filter = _ThreadFilter(threading.get_ident())
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    file_handler.addFilter(thread_filter)
    capture_handler = _BatchLogHandler()
    capture_handler.addFilter(thread_filter)
    logger.addHandler(file_handler)
    logger.addHandler(capture_handler)

    started = time.monotonic()
    result: dict[str, object] = {
        "difficulty": difficulty,
        "size": size,
        "seed": seed,
        "output_path": str(output_path),
        "log_path": str(log_path),
        "llm_log_path": str(llm_log_path) if llm_logging_enabled else None,
        "success": False,
        "fill_score": None,
        "fill_selection": None,
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
            config,
            seed=seed,
            output_file=output_path,
            clue_history=clue_history,
            excluded_fill_words=excluded_fill_words,
            answer_usage_counts=answer_usage_counts,
        )
        completed = pipeline.run(envelope)
        if output_path.exists() and clue_history is not None:
            clue_history.add_clues(completed.clues)
            if keep_sweep_context:
                # Context for the post-batch duplicate sweep (parallel runs
                # only). Holds the completed envelope and the clue step so
                # cross-puzzle duplicates can be repaired after all workers
                # finish; popped from the result before the manifest is
                # written.
                clue_step = next(
                    (
                        s
                        for s in pipeline.steps
                        if isinstance(s, ClueWithGradingStep)
                    ),
                    None,
                )
                result["_sweep"] = {
                    "envelope": completed,
                    "clue_step": clue_step,
                    "output_path": output_path,
                }
        result.update(
            {
                "success": output_path.exists(),
                "fill_score": (
                    completed.fill.quality_score if completed.fill else None
                ),
                "fill_selection": (
                    completed.fill.selection_metadata.model_dump()
                    if completed.fill and completed.fill.selection_metadata
                    else None
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


def _load_existing_clue_history(
    clue_history: ClueHistoryIndex,
    selected_buckets: list[tuple[str, int, str, Path]],
    *,
    api_base: str | None = None,
) -> int:
    from crossword_generator.data_store import list_official_puzzle_records

    loaded_records = 0
    seen_game_keys: set[str] = set()
    for _difficulty, size, _puzzle_type, _config_path in selected_buckets:
        game_key = _game_key_for_size(size)
        if game_key in seen_game_keys:
            continue
        seen_game_keys.add(game_key)
        # The live/official daily-schedule store is the clue corpus solvers
        # actually see; the draft generated-puzzles store is emptied as
        # candidates are promoted, so it cannot dedup against published clues.
        records = list_official_puzzle_records(
            game_key=game_key,
            api_base=api_base,
        )
        loaded_records += len(records)
        for record in records:
            clue_history.add_record(record)
    return loaded_records


def _game_key_for_size(size: int) -> str:
    return "minicrossword" if size in (5, 7) else "midicrossword"


def _run_duplicate_sweep(
    results: list[dict[str, object]],
    clue_history: ClueHistoryIndex,
    *,
    exporter: Exporter | None = None,
) -> dict[str, int]:
    """Repair cross-puzzle duplicate clues that parallel workers missed.

    A puzzle's clues only enter the shared history once it completes, so two
    puzzles generating concurrently can write the same clue for a shared
    answer without either's in-run duplicate check noticing. This replays
    completed puzzles in manifest order: the first occurrence keeps its clue,
    later ones go through the informed duplicate repair and are re-exported
    in place. Duplicates that survive repair become DUPLICATE: soft errors so
    the upload guard holds those puzzles back.
    """
    if exporter is None:
        from crossword_generator.exporters.ipuz_exporter import IpuzExporter

        exporter = IpuzExporter()

    def _duplicate_error_count(envelope: PuzzleEnvelope) -> int:
        return sum(1 for e in envelope.errors if e.startswith("DUPLICATE:"))

    sweep_index = ClueHistoryIndex()
    checked = 0
    repaired_total = 0
    unresolved_total = 0
    for result in results:
        ctx = result.pop("_sweep", None)
        if not isinstance(ctx, dict):
            continue
        envelope = ctx["envelope"]
        checked += 1
        hits = sweep_index.find_duplicates(envelope.clues)
        if hits:
            label = (
                f"{result['difficulty']} {result['size']}x{result['size']} "
                f"seed {result['seed']}"
            )
            click.echo(
                f"Duplicate sweep: {label} shares {len(hits)} clue(s) with "
                "another puzzle in this batch; regenerating..."
            )
            clue_step = ctx.get("clue_step")
            if clue_step is None:
                # No repair machinery available — record soft errors so the
                # upload guard holds the puzzle back instead of shipping dupes.
                envelope = envelope.model_copy(
                    update={
                        "errors": [
                            *envelope.errors,
                            *(duplicate_error_message(h) for h in hits),
                        ]
                    }
                )
                unresolved_total += len(hits)
            else:
                before = _duplicate_error_count(envelope)
                envelope = clue_step.repair_external_duplicates(envelope, hits)
                unresolved = _duplicate_error_count(envelope) - before
                unresolved_total += unresolved
                repaired_total += len(hits) - unresolved
                # Register replacements in the shared history so later
                # puzzles' repair re-checks can't land on them.
                clue_history.add_clues(envelope.clues)
                exporter.export_to_file(envelope, Path(str(ctx["output_path"])))
            result["error_message"] = "; ".join(envelope.errors) or None
            result["failure_category"] = _failure_category(result)
        sweep_index.add_clues(envelope.clues)
    if repaired_total or unresolved_total:
        click.echo(
            f"Duplicate sweep: repaired {repaired_total} clue(s), "
            f"{unresolved_total} unresolved."
        )
    return {
        "checked": checked,
        "repaired": repaired_total,
        "unresolved": unresolved_total,
    }


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


def _batch_bucket_configs(project_root: Path) -> list[tuple[str, int, str, Path]]:
    # Most-constrained first: 9x9 midis carry the heaviest fill (~13 3-letter
    # entries each) and the least slack, so they claim their answers before the
    # less-constrained 7x7 and 5x5 minis, which then avoid them. Combined with
    # the per-item exclusion floor (minis drop 3-letter answers a midi used;
    # midis keep their glue), this prevents the mini<->midi 3-letter scheduling
    # conflicts that the +/-2 short-answer window does not cover for minis.
    return [
        ("easy", 9, "midi", project_root / "config.easy9.yaml"),
        ("hard", 9, "midi", project_root / "config.hard9.yaml"),
        ("easy", 7, "mini", project_root / "config.easy.yaml"),
        ("hard", 7, "mini", project_root / "config.hard7.yaml"),
        ("easy", 5, "mini", project_root / "config.easy.yaml"),
        ("hard", 5, "mini", project_root / "config.hard5.yaml"),
    ]


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
