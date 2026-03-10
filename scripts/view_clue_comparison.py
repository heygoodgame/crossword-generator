"""Pretty-print side-by-side clue comparisons from existing output files.

Usage:
    uv run python scripts/view_clue_comparison.py                   # all seeds
    uv run python scripts/view_clue_comparison.py --seeds 1,3,5     # specific seeds
    uv run python scripts/view_clue_comparison.py --seeds 1-3       # range
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Re-use the project's numbering logic to extract answers from grids
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crossword_generator.exporters.numbering import compute_numbering


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "clue_comparison"


def parse_seeds(spec: str | None) -> list[int] | None:
    """Parse '1,3,5' or '1-10' or None (= all)."""
    if spec is None:
        return None
    seeds: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return sorted(set(seeds))


def render_grid(grid: list[list[str]]) -> str:
    """Render grid as a compact text block with blocked squares as '#'."""
    lines: list[str] = []
    for row in grid:
        lines.append("  " + " ".join(c if c != "." else "#" for c in row))
    return "\n".join(lines)


def render_seed(data: dict) -> str:
    """Render one seed's comparison as formatted text."""
    seed = data["seed"]
    grid = data["grid"]
    fill_score = data["fill_score"]
    providers = list(data["clues"].keys())

    # Compute numbered entries (with answers) from the grid
    entries = compute_numbering(grid)
    answer_map = {f"{e.number}-{e.direction}": e.answer for e in entries}

    # Collect all clue keys, sorted by number then direction
    all_keys: set[str] = set()
    for prov in providers:
        all_keys.update(data["clues"][prov].get("clues", {}).keys())

    def sort_key(k: str) -> tuple[int, int]:
        num, direction = k.split("-", 1)
        return (int(num), 0 if direction == "across" else 1)

    sorted_keys = sorted(all_keys, key=sort_key)

    # Build output
    lines: list[str] = []
    lines.append(f"{'=' * 80}")
    lines.append(f"  SEED {seed}   (fill score: {fill_score:.1f}, entries: {data['entries']})")
    lines.append(f"{'=' * 80}")
    lines.append("")
    lines.append(render_grid(grid))
    lines.append("")

    # Timing summary
    timing = []
    for prov in providers:
        info = data["clues"][prov]
        status = "OK" if info["success"] else "FAIL"
        timing.append(f"  {prov}: {info['elapsed_s']:.1f}s ({status})")
    lines.append("  Timing: " + "  |  ".join(timing))
    lines.append("")

    # Side-by-side clues
    # Determine column widths
    prov_width = max(len(p) for p in providers)
    clue_width = 36

    # Header
    hdr = f"  {'#':<12} {'Answer':<12}"
    for prov in providers:
        hdr += f" {prov:<{clue_width}}"
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))

    # Separate across and down
    for section in ("across", "down"):
        section_keys = [k for k in sorted_keys if k.endswith(f"-{section}")]
        if not section_keys:
            continue
        lines.append(f"  --- {section.upper()} ---")
        for key in section_keys:
            answer = answer_map.get(key, "?")
            row = f"  {key:<12} {answer:<12}"
            for prov in providers:
                clue = data["clues"][prov].get("clues", {}).get(key, "—")
                # Truncate long clues for table alignment, but show full text
                if len(clue) > clue_width - 1:
                    display = clue[: clue_width - 4] + "..."
                else:
                    display = clue
                row += f" {display:<{clue_width}}"
            lines.append(row)
        lines.append("")

    # Full clues (untruncated) for any that were truncated
    long_clues = []
    for key in sorted_keys:
        for prov in providers:
            clue = data["clues"][prov].get("clues", {}).get(key, "")
            if len(clue) > clue_width - 1:
                long_clues.append((key, prov, clue))

    if long_clues:
        lines.append("  Full clues (truncated above):")
        for key, prov, clue in long_clues:
            answer = answer_map.get(key, "?")
            lines.append(f"    {key} [{answer}] ({prov}): {clue}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="View clue comparison results")
    parser.add_argument("--seeds", type=str, default=None, help="Seeds to show (e.g. 1,3,5 or 1-10)")
    parser.add_argument("--dir", type=str, default=None, help="Output directory override")
    args = parser.parse_args()

    output_dir = Path(args.dir) if args.dir else OUTPUT_DIR
    requested = parse_seeds(args.seeds)

    # Find all seed files
    seed_files = sorted(output_dir.glob("seed_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    if not seed_files:
        print(f"No seed files found in {output_dir}", file=sys.stderr)
        sys.exit(1)

    for path in seed_files:
        seed_num = int(path.stem.split("_")[1])
        if requested and seed_num not in requested:
            continue
        data = json.loads(path.read_text())
        print(render_seed(data))


if __name__ == "__main__":
    main()
