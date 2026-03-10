"""Compare clue generation across LLM providers for the same grids.

Usage:
    uv run python scripts/compare_clue_providers.py \
        --providers ollama,claude --seeds 1-10 --size 9

Generates one grid per seed, then runs clue generation with each
provider separately. Saves per-seed comparison JSON and a summary.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from crossword_generator.config import load_config
from crossword_generator.exporters.numbering import compute_numbering
from crossword_generator.llm.base import LLMProvider
from crossword_generator.llm.ollama_provider import OllamaProvider
from crossword_generator.models import PuzzleEnvelope, PuzzleType
from crossword_generator.steps.clue_step import ClueGenerationStep
from crossword_generator.steps.fill_step import FillWithGradingStep
from crossword_generator.config import find_project_root
from crossword_generator.dictionary import Dictionary
from crossword_generator.fillers.csp import CSPFiller
from crossword_generator.graders.fill_grader import FillGrader

logger = logging.getLogger(__name__)


def build_provider(name: str) -> LLMProvider:
    config = load_config()
    if name == "ollama":
        return OllamaProvider(config.llm.ollama)
    elif name == "claude":
        from crossword_generator.llm.claude_provider import ClaudeProvider
        return ClaudeProvider(config.llm.claude)
    else:
        raise ValueError(f"Unknown provider: {name}")


def fill_grid(seed: int, size: int) -> PuzzleEnvelope:
    """Fill a grid using the configured filler."""
    config = load_config()
    project_root = find_project_root()
    dictionary = Dictionary.load(
        project_root / config.dictionary.path,
        min_word_score=config.dictionary.min_word_score,
        min_2letter_score=config.dictionary.min_2letter_score,
    )
    filler = CSPFiller(config.fill.csp, dictionary)
    grader = FillGrader(
        dictionary,
        min_passing_score=config.grading.fill.min_score,
    )
    fill_step = FillWithGradingStep(
        filler, grader,
        max_retries=config.fill.max_retries,
        retry_on_fail=config.grading.fill.retry_on_fail,
    )
    envelope = PuzzleEnvelope(
        puzzle_type=PuzzleType.MIDI,
        grid_size=size,
        metadata={"seed": seed},
    )
    return fill_step.run(envelope)


def parse_seeds(seeds_str: str) -> list[int]:
    """Parse '1-10' or '1,2,5' into a list of ints."""
    if "-" in seeds_str and "," not in seeds_str:
        start, end = seeds_str.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(s.strip()) for s in seeds_str.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare clue generation across LLM providers"
    )
    parser.add_argument(
        "--providers", default="ollama,claude",
        help="Comma-separated provider names (default: ollama,claude)",
    )
    parser.add_argument(
        "--seeds", default="1-10",
        help="Seed range '1-10' or list '1,2,5' (default: 1-10)",
    )
    parser.add_argument(
        "--size", type=int, default=9,
        help="Grid size (default: 9)",
    )
    parser.add_argument(
        "--output-dir", default="output/clue_comparison",
        help="Output directory (default: output/clue_comparison)",
    )
    parser.add_argument(
        "--max-retries", type=int, default=5,
        help="Max LLM retries per puzzle (default: 5)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    provider_names = [p.strip() for p in args.providers.split(",")]
    seeds = parse_seeds(args.seeds)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build providers
    providers: dict[str, LLMProvider] = {}
    for name in provider_names:
        try:
            p = build_provider(name)
            if not p.is_available():
                logger.warning("Provider %s not available, skipping", name)
                continue
            providers[name] = p
        except Exception as exc:
            logger.warning("Failed to init provider %s: %s", name, exc)

    if not providers:
        print("No providers available.", file=sys.stderr)
        sys.exit(1)

    print(
        f"Comparing {list(providers.keys())} across "
        f"{len(seeds)} seeds ({args.size}x{args.size})\n"
    )

    summary: list[dict] = []

    for seed in seeds:
        print(f"=== Seed {seed} ===")

        # Fill grid (once per seed)
        logger.info("Filling grid for seed %d", seed)
        try:
            filled_envelope = fill_grid(seed, args.size)
        except Exception as exc:
            print(f"  Fill failed: {exc}")
            summary.append({"seed": seed, "fill_error": str(exc)})
            continue

        grid = filled_envelope.fill.grid
        entries = compute_numbering(grid)
        score = filled_envelope.fill.quality_score

        print(f"  Fill score: {score:.1f}")
        print(f"  Entries: {len(entries)}")

        seed_result: dict = {
            "seed": seed,
            "grid": grid,
            "fill_score": score,
            "entries": len(entries),
            "clues": {},
        }

        for pname, provider in providers.items():
            # Clear clues from filled envelope to re-run clue step
            clue_envelope = filled_envelope.model_copy(
                update={"clues": [], "step_history": []}
            )
            step = ClueGenerationStep(
                provider, max_retries=args.max_retries
            )

            t0 = time.time()
            try:
                result = step.run(clue_envelope)
                elapsed = time.time() - t0
                clue_map = {
                    f"{c.number}-{c.direction}": c.clue
                    for c in result.clues
                }
                seed_result["clues"][pname] = {
                    "success": True,
                    "elapsed_s": round(elapsed, 1),
                    "clues": clue_map,
                }
                print(f"  {pname}: OK ({elapsed:.1f}s)")
            except Exception as exc:
                elapsed = time.time() - t0
                seed_result["clues"][pname] = {
                    "success": False,
                    "elapsed_s": round(elapsed, 1),
                    "error": str(exc),
                }
                print(f"  {pname}: FAIL ({elapsed:.1f}s) — {exc}")

        summary.append(seed_result)

        # Save per-seed comparison
        seed_file = out_dir / f"seed_{seed}.json"
        seed_file.write_text(json.dumps(seed_result, indent=2))
        print()

    # Save summary
    summary_file = out_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))

    # Print summary table
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for pname in providers:
        successes = sum(
            1 for s in summary
            if s.get("clues", {}).get(pname, {}).get("success")
        )
        times = [
            s["clues"][pname]["elapsed_s"]
            for s in summary
            if s.get("clues", {}).get(pname, {}).get("success")
        ]
        avg_time = sum(times) / len(times) if times else 0
        print(
            f"  {pname}: {successes}/{len(seeds)} success, "
            f"avg {avg_time:.1f}s"
        )

    # Print side-by-side clue sample (first successful seed)
    for s in summary:
        clues = s.get("clues", {})
        if all(
            clues.get(p, {}).get("success") for p in providers
        ):
            print(f"\nSample clues (seed {s['seed']}):")
            all_keys = list(
                next(iter(clues.values()))["clues"].keys()
            )
            for key in all_keys[:8]:
                print(f"\n  {key}:")
                for pname in providers:
                    clue_text = clues[pname]["clues"].get(key, "?")
                    print(f"    {pname:>8}: {clue_text}")
            if len(all_keys) > 8:
                print(f"\n  ... and {len(all_keys) - 8} more")
            break

    print(f"\nDetailed results: {out_dir}/")


if __name__ == "__main__":
    main()
