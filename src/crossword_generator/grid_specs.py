"""Grid specification catalog for supported puzzle types and sizes."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from crossword_generator.fillers.base import GridSpec
from crossword_generator.grid_pattern_validation import (
    is_rotationally_symmetric,
    validate_pattern,
)
from crossword_generator.models import PuzzleType

# Valid (puzzle_type, grid_size) combinations
_VALID_SPECS: dict[tuple[PuzzleType, int], tuple[int, int]] = {
    (PuzzleType.MINI, 5): (5, 5),
    (PuzzleType.MINI, 7): (7, 7),
    (PuzzleType.MIDI, 9): (9, 9),
    (PuzzleType.MIDI, 10): (10, 10),
    (PuzzleType.MIDI, 11): (11, 11),
}


def _flip_top_to_bottom(
    black_cells: list[tuple[int, int]],
    *,
    size: int,
) -> list[tuple[int, int]]:
    return sorted((size - 1 - r, c) for r, c in black_cells)


def _flip_left_to_right(
    black_cells: list[tuple[int, int]],
    *,
    size: int,
) -> list[tuple[int, int]]:
    return sorted((r, size - 1 - c) for r, c in black_cells)


def _unique_pattern_variants(
    *patterns: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], ...]:
    variants: dict[tuple[tuple[int, int], ...], list[tuple[int, int]]] = {}
    for pattern in patterns:
        normalized = tuple(sorted(pattern))
        variants.setdefault(normalized, list(normalized))
    return tuple(variants.values())


def _pattern_symmetries(
    size: int,
    black_cells: list[tuple[int, int]] | tuple[tuple[int, int], ...],
) -> tuple[str, ...]:
    """Return every symmetry present in a square pattern."""
    black = set(black_cells)
    last = size - 1
    symmetries: list[str] = []

    checks = (
        (
            "rotational",
            lambda r, c: (last - r, last - c),
        ),
        (
            "vertical",
            lambda r, c: (r, last - c),
        ),
        (
            "horizontal",
            lambda r, c: (last - r, c),
        ),
        (
            "main_diagonal",
            lambda r, c: (c, r),
        ),
        (
            "anti_diagonal",
            lambda r, c: (last - c, last - r),
        ),
    )

    for name, transform in checks:
        if all(transform(r, c) in black for r, c in black):
            symmetries.append(name)

    return tuple(symmetries)


def _has_accepted_pattern_symmetry(
    size: int,
    black_cells: list[tuple[int, int]] | tuple[tuple[int, int], ...],
) -> bool:
    """Return whether a pattern has Jeff's accepted professional symmetry."""
    symmetries = _pattern_symmetries(size, black_cells)
    return "rotational" in symmetries or "vertical" in symmetries


def _filter_accepted_symmetry_patterns(
    size: int,
    patterns: list[tuple[list[tuple[int, int]], int]],
) -> list[tuple[list[tuple[int, int]], int]]:
    return [
        (black_cells, weight)
        for black_cells, weight in patterns
        if _has_accepted_pattern_symmetry(size, black_cells)
    ]


def _add_center_black_if_valid(
    size: int,
    black_cells: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    center = (size // 2, size // 2)
    if center in black_cells:
        return black_cells

    candidate = tuple(sorted(set(black_cells) | {center}))
    validation = validate_pattern(size, candidate)
    if validation.valid:
        return candidate

    return black_cells


def _make_mini_7_catalog(
    patterns: list[tuple[list[tuple[int, int]], int]],
    *,
    center_optional_patterns: tuple[list[tuple[int, int]], ...] = (),
) -> list[tuple[list[tuple[int, int]], int]]:
    """Apply Jeff's 7x7 feedback to the raw attachment-derived catalog."""
    transformed: dict[tuple[tuple[int, int], ...], int] = {}
    center_optional = {
        tuple(sorted(pattern)) for pattern in center_optional_patterns
    }

    for black_cells, weight in patterns:
        normalized = tuple(sorted(black_cells))
        if not _has_accepted_pattern_symmetry(7, normalized):
            continue

        updated = _add_center_black_if_valid(7, normalized)
        if _slot_lengths(7, updated).count(7) <= 4:
            transformed[updated] = transformed.get(updated, 0) + weight
        if (
            normalized in center_optional
            and normalized != updated
            and _slot_lengths(7, normalized).count(7) <= 4
        ):
            transformed[normalized] = transformed.get(normalized, 0) + weight

    return [
        (list(black_cells), weight)
        for black_cells, weight in sorted(
            transformed.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]


def _slot_lengths(size: int, black_cells: tuple[tuple[int, int], ...]) -> list[int]:
    black = set(black_cells)
    lengths: list[int] = []

    for r in range(size):
        c = 0
        while c < size:
            if (r, c) in black:
                c += 1
                continue
            start = c
            while c < size and (r, c) not in black:
                c += 1
            lengths.append(c - start)

    for c in range(size):
        r = 0
        while r < size:
            if (r, c) in black:
                r += 1
                continue
            start = r
            while r < size and (r, c) not in black:
                r += 1
            lengths.append(r - start)

    return lengths


def _is_safe_midi_9_pattern(black_cells: tuple[tuple[int, int], ...]) -> bool:
    validation = validate_pattern(9, black_cells)
    if not validation.valid:
        return False

    if _has_corner_perimeter_black_run(9, black_cells, run_length=3):
        return False

    return True


def _has_corner_perimeter_black_run(
    size: int,
    black_cells: tuple[tuple[int, int], ...],
    *,
    run_length: int,
) -> bool:
    black = set(black_cells)
    edge_runs = [
        ((0, c) for c in range(run_length)),
        ((r, 0) for r in range(run_length)),
        ((0, c) for c in range(size - run_length, size)),
        ((r, size - 1) for r in range(run_length)),
        ((size - 1, c) for c in range(run_length)),
        ((r, 0) for r in range(size - run_length, size)),
        ((size - 1, c) for c in range(size - run_length, size)),
        ((r, size - 1) for r in range(size - run_length, size)),
    ]
    for run in edge_runs:
        if all(cell in black for cell in run):
            return True
    return False


_MIDI_9_BASE_PATTERNS: list[list[tuple[int, int]]] = [
    [
        (0, 3), (0, 4), (0, 5), (1, 4), (2, 4),
        (3, 0), (3, 1), (3, 2), (3, 6), (3, 7),
        (3, 8), (4, 0), (4, 1), (4, 7), (4, 8),
        (5, 0), (5, 8), (6, 4), (7, 4), (8, 3),
        (8, 4), (8, 5),
    ],
    [
        (0, 4), (1, 4), (2, 4), (3, 0), (3, 8),
        (4, 0), (4, 1), (4, 2), (4, 6), (4, 7),
        (4, 8), (5, 0), (5, 8), (6, 4), (7, 4),
        (8, 3), (8, 4), (8, 5),
    ],
    [
        (0, 4), (1, 4), (3, 0), (3, 1), (3, 2),
        (3, 6), (3, 7), (3, 8), (4, 0), (4, 8),
        (5, 3), (5, 4), (5, 5), (6, 4), (7, 4),
        (8, 4),
    ],
    [
        (0, 4), (1, 4), (3, 0), (3, 1), (3, 2),
        (3, 6), (3, 7), (3, 8), (5, 3), (5, 4),
        (5, 5), (6, 4), (7, 4), (8, 0), (8, 4),
        (8, 8),
    ],
    [
        (0, 3), (0, 4), (0, 5), (1, 4), (2, 4),
        (4, 0), (4, 1), (4, 7), (4, 8), (5, 0),
        (5, 1), (5, 2), (5, 6), (5, 7), (5, 8),
        (6, 4), (7, 4), (8, 3), (8, 4), (8, 5),
    ],
    [
        (0, 4), (1, 4), (2, 4), (3, 0), (3, 8),
        (4, 0), (4, 1), (4, 7), (4, 8), (5, 0),
        (5, 1), (5, 2), (5, 6), (5, 7), (5, 8),
        (6, 4), (7, 4), (8, 4),
    ],
    [
        (0, 3), (0, 4), (0, 5), (1, 4), (2, 4),
        (3, 0), (3, 4), (3, 8), (4, 0), (4, 1),
        (4, 7), (4, 8), (5, 0), (5, 1), (5, 2),
        (5, 6), (5, 7), (5, 8), (7, 4), (8, 4),
    ],
    [
        (0, 4), (1, 4), (3, 0), (3, 1), (3, 2),
        (3, 6), (3, 7), (3, 8), (5, 3), (5, 4),
        (5, 5), (6, 4), (7, 4), (8, 4),
    ],
    [
        (0, 3), (0, 4), (0, 5), (1, 4), (2, 4),
        (3, 4), (4, 0), (4, 1), (4, 7), (4, 8),
        (5, 0), (5, 1), (5, 2), (5, 6), (5, 7),
        (5, 8), (7, 4), (8, 3), (8, 4), (8, 5),
    ],
]

_MIDI_9_ADDITIVE_MOTIFS: list[list[tuple[int, int]]] = [
    [],
    [(0, 0), (0, 8)],
    [(8, 0), (8, 8)],
    [(0, 0), (1, 0), (2, 0), (0, 8), (1, 8), (2, 8)],
    [(6, 0), (7, 0), (8, 0), (6, 8), (7, 8), (8, 8)],
    [(3, 0), (3, 8)],
    [(5, 0), (5, 8)],
]


_MIDI_9_REGULAR_SYMMETRY_PATTERNS: list[list[tuple[int, int]]] = [
    [
        (0, 3), (0, 4), (1, 3), (3, 0), (3, 1),
        (3, 5), (4, 0), (4, 4), (4, 8), (5, 3),
        (5, 7), (5, 8), (7, 5), (8, 4), (8, 5),
    ],
    [
        (0, 3), (0, 4), (0, 5), (1, 4), (1, 5),
        (2, 5), (3, 0), (4, 0), (4, 1), (4, 2),
        (4, 6), (4, 7), (4, 8), (5, 8), (6, 3),
        (7, 3), (7, 4), (8, 3), (8, 4), (8, 5),
    ],
    [
        (0, 4), (0, 5), (1, 5), (2, 5), (3, 0),
        (3, 8), (4, 0), (4, 1), (4, 2), (4, 6),
        (4, 7), (4, 8), (5, 0), (5, 8), (6, 3),
        (7, 3), (8, 3), (8, 4),
    ],
    [
        (0, 3), (0, 4), (0, 5), (1, 4), (2, 4),
        (3, 0), (3, 8), (4, 0), (4, 1), (4, 2),
        (4, 6), (4, 7), (4, 8), (5, 0), (5, 8),
        (6, 4), (7, 4), (8, 3), (8, 4), (8, 5),
    ],
    [
        (0, 4), (0, 5), (1, 4), (1, 5), (2, 5),
        (3, 0), (3, 1), (3, 2), (4, 0), (4, 1),
        (4, 7), (4, 8), (5, 6), (5, 7), (5, 8),
        (6, 3), (7, 3), (7, 4), (8, 3), (8, 4),
    ],
    [
        (0, 3), (0, 4), (0, 5), (1, 5), (2, 5),
        (3, 0), (3, 1), (3, 2), (3, 8), (4, 0),
        (4, 4), (4, 8), (5, 0), (5, 6), (5, 7),
        (5, 8), (6, 3), (7, 3), (8, 3), (8, 4),
        (8, 5),
    ],
    [
        (0, 3), (1, 3), (3, 0), (3, 5), (3, 6),
        (3, 7), (3, 8), (4, 0), (4, 1), (4, 7),
        (4, 8), (5, 0), (5, 1), (5, 2), (5, 3),
        (5, 8), (7, 5), (8, 5),
    ],
    [
        (0, 3), (0, 4), (1, 4), (3, 0), (3, 5),
        (3, 6), (3, 7), (3, 8), (4, 0), (4, 1),
        (4, 7), (4, 8), (5, 0), (5, 1), (5, 2),
        (5, 3), (5, 8), (7, 4), (8, 4), (8, 5),
    ],
    [
        (0, 3), (0, 4), (1, 3), (3, 0), (3, 5),
        (4, 0), (4, 1), (4, 2), (4, 6), (4, 7),
        (4, 8), (5, 3), (5, 8), (7, 5), (8, 4),
        (8, 5),
    ],
    [
        (0, 3), (0, 4), (1, 4), (3, 0), (3, 1),
        (3, 5), (4, 0), (4, 1), (4, 2), (4, 6),
        (4, 7), (4, 8), (5, 3), (5, 7), (5, 8),
        (7, 4), (8, 4), (8, 5),
    ],
]

_MIDI_9_ROTATIONAL_ADDITIVE_MOTIFS: list[list[tuple[int, int]]] = [
    [],
    [(0, 0), (8, 8)],
    [(0, 8), (8, 0)],
    [(0, 0), (0, 1), (1, 0), (7, 8), (8, 7), (8, 8)],
    [(0, 7), (0, 8), (1, 8), (7, 0), (8, 0), (8, 1)],
]


def _make_midi_9_catalog() -> list[tuple[list[tuple[int, int]], int]]:
    patterns: dict[tuple[tuple[int, int], ...], int] = {}
    for base in _MIDI_9_BASE_PATTERNS:
        for source in (base, _flip_top_to_bottom(base, size=9)):
            for motif in _MIDI_9_ADDITIVE_MOTIFS:
                candidate = tuple(sorted(set(source) | set(motif)))
                if _is_safe_midi_9_pattern(candidate):
                    weight = 2 if not motif else 1
                    patterns[candidate] = max(patterns.get(candidate, 0), weight)

    for base in _MIDI_9_REGULAR_SYMMETRY_PATTERNS:
        for source in _unique_pattern_variants(
            base,
            _flip_left_to_right(base, size=9),
            _flip_top_to_bottom(base, size=9),
        ):
            for motif in _MIDI_9_ROTATIONAL_ADDITIVE_MOTIFS:
                candidate = tuple(sorted(set(source) | set(motif)))
                if _is_safe_midi_9_pattern(candidate):
                    weight = 2 if not motif else 1
                    patterns[candidate] = max(patterns.get(candidate, 0), weight)

    return [(list(black_cells), weight) for black_cells, weight in patterns.items()]


# Black cell patterns for each (puzzle_type, grid_size).
# Each entry is (black_cell_positions, weight) for weighted random selection.
_GRID_PATTERNS: dict[
    tuple[PuzzleType, int],
    list[tuple[list[tuple[int, int]], int]],
] = {
    # 5x5: raw frequency-based catalog filtered to accepted professional
    # symmetry: regular 180-degree rotation or left-right mirror.
    (PuzzleType.MINI, 5): _filter_accepted_symmetry_patterns(5, [
        # --- 9x frequency ---
        ([], 9),
        ([(0, 0), (4, 4)], 9),
        # --- 8x frequency ---
        ([(0, 0), (0, 4), (4, 0), (4, 4)], 8),
        # --- 7x frequency ---
        ([(0, 0), (0, 1), (1, 0), (3, 4), (4, 3), (4, 4)], 7),
        # --- 6x frequency ---
        ([(0, 4), (1, 4), (3, 0), (4, 0)], 6),
        ([(0, 0), (1, 0), (3, 4), (4, 4)], 6),
        # --- 4x frequency ---
        ([(4, 0), (4, 4)], 4),
        ([(0, 4), (4, 0)], 4),
        ([(3, 4), (4, 3), (4, 4)], 4),
        ([(0, 4)], 4),
        # --- 2x frequency ---
        ([(0, 0), (3, 4), (4, 4)], 2),
        ([(0, 3), (0, 4), (1, 4), (3, 0), (4, 0), (4, 1)], 2),
        ([(0, 0), (0, 1), (1, 0), (4, 4)], 2),
        ([(0, 0), (0, 1), (1, 0), (3, 4), (4, 4)], 2),
        ([(0, 0), (0, 1), (1, 0)], 2),
        ([(4, 0), (4, 1)], 2),
        ([(0, 4), (4, 0), (4, 4)], 2),
        ([(0, 0), (4, 0)], 2),
        ([(0, 3), (0, 4), (1, 4)], 2),
        ([(0, 4), (3, 0), (4, 0), (4, 1)], 2),
        # --- 1x frequency ---
        ([(0, 3), (0, 4), (4, 3), (4, 4)], 1),
        ([(0, 0), (3, 4), (4, 3), (4, 4)], 1),
        ([(0, 0), (1, 0)], 1),
        ([(0, 4), (3, 0), (4, 0)], 1),
        ([(0, 0), (0, 4), (1, 0), (1, 4)], 1),
        ([(0, 0)], 1),
        ([(3, 0), (4, 0)], 1),
        ([(0, 3), (0, 4), (4, 0), (4, 1)], 1),
        ([(4, 3), (4, 4)], 1),
        ([(0, 0), (0, 4), (4, 4)], 1),
        ([(0, 0), (4, 3), (4, 4)], 1),
        ([(0, 4), (1, 4), (4, 0), (4, 1)], 1),
        ([(0, 4), (4, 4)], 1),
        ([(3, 0), (3, 4), (4, 0), (4, 4)], 1),
    ]),
    # 7x7: raw frequency-based catalog with Jeff feedback applied:
    # - keep only regular 180-degree rotation or left-right mirror symmetry
    #   (no asymmetric, diagonal-only, or up-down-only patterns)
    # - add the center black square whenever doing so preserves 3+ letter slots
    # - drop patterns with more than four 7-letter slots because recent fill
    #   logs showed a materially lower accepted-fill rate
    (PuzzleType.MINI, 7): _make_mini_7_catalog([
        # --- Jeff attachment examples ---
        # "7x7 1": reduced raw whitespace. The center square stays open
        # because adding it would create one-letter slots.
        ([
            (0, 3), (1, 3), (3, 0), (3, 1), (3, 5),
            (3, 6), (5, 3), (6, 3),
        ], 1),
        # "7x7 2": Utah blocks. Jeff noted this can work with or without
        # the center square, so the exception below keeps both variants.
        ([
            (0, 3), (1, 3), (2, 3), (4, 0), (4, 6),
            (5, 0), (5, 1), (5, 5), (5, 6), (6, 0),
            (6, 1), (6, 5), (6, 6),
        ], 1),
        # --- 13x frequency ---
        ([
            (0, 0), (0, 1), (0, 2), (1, 0), (1, 1),
            (2, 0), (4, 6), (5, 5), (5, 6), (6, 4),
            (6, 5), (6, 6),
        ], 13),
        # --- 9x frequency ---
        ([
            (0, 0), (0, 1), (0, 5), (0, 6), (1, 0),
            (1, 6), (3, 3), (5, 0), (5, 6), (6, 0),
            (6, 1), (6, 5), (6, 6),
        ], 9),
        # --- 6x frequency ---
        ([
            (0, 0), (0, 1), (0, 5), (0, 6), (1, 0),
            (1, 6), (5, 0), (5, 6), (6, 0), (6, 1),
            (6, 5), (6, 6),
        ], 6),
        # --- 3x frequency ---
        ([
            (0, 0), (0, 5), (0, 6), (1, 6), (3, 3),
            (5, 0), (6, 0), (6, 1), (6, 6),
        ], 3),
        ([
            (0, 0), (0, 1), (0, 5), (0, 6), (1, 0),
            (1, 6), (3, 3), (5, 0), (5, 6), (6, 0),
            (6, 6),
        ], 3),
        ([(3, 0), (3, 1), (3, 5), (3, 6)], 3),
        # --- 2x frequency ---
        ([
            (0, 0), (0, 1), (0, 6), (1, 0), (3, 3),
            (5, 6), (6, 0), (6, 5), (6, 6),
        ], 2),
        ([
            (0, 5), (0, 6), (1, 6), (3, 3), (5, 0),
            (6, 0), (6, 1),
        ], 2),
        ([
            (0, 0), (0, 1), (0, 5), (0, 6), (1, 0),
            (1, 6), (3, 3), (6, 0), (6, 6),
        ], 2),
        ([
            (0, 0), (0, 1), (1, 0), (3, 3), (4, 6),
            (5, 5), (5, 6), (6, 4), (6, 5), (6, 6),
        ], 2),
        ([(0, 3), (3, 0), (3, 6), (6, 3)], 2),
        # --- 1x frequency ---
        ([
            (0, 0), (0, 6), (1, 0), (1, 6), (3, 3),
            (6, 0), (6, 1), (6, 5), (6, 6),
        ], 1),
        ([
            (0, 0), (0, 1), (0, 5), (0, 6), (1, 0),
            (1, 6), (4, 3), (5, 3), (6, 3),
        ], 1),
        ([
            (0, 0), (0, 1), (0, 5), (0, 6), (1, 0),
            (3, 3), (5, 6), (6, 0), (6, 1), (6, 5),
            (6, 6),
        ], 1),
        ([
            (0, 0), (0, 1), (0, 5), (0, 6), (1, 0),
            (1, 6), (2, 0), (2, 6), (5, 3), (6, 3),
        ], 1),
        ([
            (0, 0), (3, 3), (5, 0), (5, 6), (6, 0),
            (6, 1), (6, 5), (6, 6),
        ], 1),
        ([
            (0, 0), (0, 1), (0, 5), (0, 6), (1, 0),
            (1, 6), (5, 3), (6, 3),
        ], 1),
        ([
            (0, 0), (0, 1), (0, 5), (0, 6), (1, 0),
            (1, 6), (3, 3),
        ], 1),
        ([
            (0, 4), (0, 5), (0, 6), (1, 5), (1, 6),
            (2, 6), (4, 0), (5, 0), (5, 1), (6, 0),
            (6, 1), (6, 2),
        ], 1),
        ([
            (0, 3), (1, 3), (3, 0), (3, 6), (5, 3),
            (6, 3),
        ], 1),
        ([
            (0, 0), (0, 1), (0, 2), (1, 0), (2, 0),
            (3, 3), (4, 6), (5, 6), (6, 4), (6, 5),
            (6, 6),
        ], 1),
        ([
            (0, 0), (0, 1), (0, 2), (1, 0), (1, 1),
            (2, 0), (3, 0), (5, 6), (6, 5), (6, 6),
        ], 1),
        ([
            (0, 0), (0, 1), (0, 6), (1, 0), (3, 3),
            (5, 0), (5, 6), (6, 0), (6, 1), (6, 5),
            (6, 6),
        ], 1),
        ([
            (0, 0), (0, 1), (1, 0), (3, 3), (4, 6),
            (5, 6), (6, 4), (6, 5), (6, 6),
        ], 1),
        ([(0, 0), (0, 6), (3, 3), (6, 0), (6, 6)], 1),
        ([
            (0, 4), (0, 5), (0, 6), (3, 3), (4, 0),
            (5, 0), (5, 5), (5, 6), (6, 0), (6, 5),
            (6, 6),
        ], 1),
        ([
            (0, 0), (0, 1), (1, 0), (3, 3), (4, 6),
            (5, 6), (6, 5), (6, 6),
        ], 1),
        ([
            (0, 4), (0, 5), (0, 6), (3, 3), (6, 0),
            (6, 1), (6, 5), (6, 6),
        ], 1),
        ([
            (0, 0), (0, 1), (0, 6), (1, 0), (3, 3),
            (5, 0), (6, 0), (6, 1), (6, 6),
        ], 1),
        ([
            (0, 4), (0, 5), (0, 6), (1, 5), (1, 6),
            (3, 0), (5, 3), (6, 3),
        ], 1),
        ([
            (0, 0), (0, 1), (0, 2), (1, 0), (1, 1),
            (2, 0), (6, 6),
        ], 1),
        ([
            (0, 0), (0, 1), (0, 2), (1, 0), (1, 1),
            (5, 5), (5, 6), (6, 4), (6, 5), (6, 6),
        ], 1),
        ([
            (0, 0), (0, 6), (1, 0), (1, 6), (5, 0),
            (5, 6), (6, 0), (6, 6),
        ], 1),
        ([
            (0, 0), (0, 6), (3, 3), (5, 0), (5, 6),
            (6, 0), (6, 1), (6, 5), (6, 6),
        ], 1),
        ([
            (0, 0), (0, 5), (0, 6), (3, 3), (5, 0),
            (5, 6), (6, 0), (6, 5), (6, 6),
        ], 1),
        ([
            (0, 0), (1, 0), (2, 0), (3, 0), (3, 1),
            (3, 5), (3, 6), (4, 6), (5, 6), (6, 6),
        ], 1),
        ([
            (0, 0), (0, 5), (0, 6), (1, 0), (5, 0),
            (6, 0), (6, 5), (6, 6),
        ], 1),
        ([
            (0, 0), (0, 6), (1, 0), (1, 6), (3, 3),
            (5, 0), (5, 6), (6, 0), (6, 6),
        ], 1),
        ([
            (0, 0), (0, 6), (1, 0), (1, 6), (4, 3),
            (5, 3), (6, 3),
        ], 1),
        ([
            (3, 3), (5, 6), (6, 0), (6, 1), (6, 5),
            (6, 6),
        ], 1),
        ([(0, 3), (3, 0), (5, 3), (6, 3)], 1),
        ([(3, 0), (3, 1), (3, 6), (6, 3)], 1),
        ([
            (0, 3), (4, 6), (5, 5), (5, 6), (6, 4),
            (6, 5), (6, 6),
        ], 1),
        ([
            (0, 3), (1, 3), (2, 3), (4, 0), (4, 6),
            (5, 0), (5, 6), (6, 0), (6, 6),
        ], 1),
        ([
            (0, 6), (1, 6), (3, 0), (3, 1), (3, 2),
            (5, 6), (6, 6),
        ], 1),
        ([
            (0, 0), (0, 1), (0, 2), (0, 3), (1, 0),
            (5, 6), (6, 3), (6, 4), (6, 5), (6, 6),
        ], 1),
        ([
            (0, 0), (0, 1), (1, 0), (4, 6), (5, 5),
            (5, 6), (6, 4), (6, 5), (6, 6),
        ], 1),
        ([
            (0, 0), (0, 1), (0, 5), (0, 6), (1, 0),
            (1, 6), (6, 0), (6, 1), (6, 5), (6, 6),
        ], 1),
        ([
            (0, 0), (0, 1), (0, 2), (1, 0), (1, 1),
            (2, 0), (5, 6), (6, 5), (6, 6),
        ], 1),
        ([
            (0, 3), (1, 3), (5, 0), (5, 6), (6, 0),
            (6, 1), (6, 5), (6, 6),
        ], 1),
    ], center_optional_patterns=(
        [
            (0, 3), (1, 3), (2, 3), (4, 0), (4, 6),
            (5, 0), (5, 1), (5, 5), (5, 6), (6, 0),
            (6, 1), (6, 5), (6, 6),
        ],
    )),
    # 9x9 midi: curated mirror-style and regular-symmetry examples from
    # Jeff's feedback, expanded with top-to-bottom flips and conservative
    # cheater-square variants.
    # Exclude patterns with three consecutive black squares pressed into a
    # corner along any perimeter edge.
    # Avoid procedural rotational windmills that can read as swastika-like.
    (PuzzleType.MIDI, 9): _make_midi_9_catalog(),
}


@dataclass(frozen=True)
class WeightedGridPattern:
    """A catalogued black-cell pattern with its selection weight."""

    black_cells: tuple[tuple[int, int], ...]
    weight: int
    symmetric: bool
    rotationally_symmetric: bool
    symmetries: tuple[str, ...]


def get_grid_patterns(
    puzzle_type: PuzzleType | str,
    grid_size: int,
    *,
    symmetric_only: bool = False,
) -> tuple[WeightedGridPattern, ...]:
    """Return structured weighted grid patterns for a supported mini size.

    ``symmetric_only`` keeps patterns with Jeff's accepted professional
    symmetry: regular 180-degree rotation or left-right mirror.
    """
    pt = PuzzleType(puzzle_type)
    key = (pt, grid_size)
    if key not in _VALID_SPECS:
        valid = [f"{t.value}/{s}" for (t, s) in _VALID_SPECS]
        raise ValueError(
            f"Unsupported puzzle_type/grid_size: {pt.value}/{grid_size}. "
            f"Valid combinations: {', '.join(valid)}"
        )

    rows, cols = _VALID_SPECS[key]
    if rows != cols:
        raise ValueError("Only square grid pattern catalogs are supported")

    patterns = tuple(
        WeightedGridPattern(
            black_cells=tuple(sorted(black_cells)),
            weight=weight,
            symmetric=_has_accepted_pattern_symmetry(rows, black_cells),
            rotationally_symmetric=is_rotationally_symmetric(rows, black_cells),
            symmetries=_pattern_symmetries(rows, black_cells),
        )
        for black_cells, weight in _GRID_PATTERNS.get(key, [])
    )
    if symmetric_only:
        return tuple(pattern for pattern in patterns if pattern.symmetric)
    return patterns


def _apply_short_slot_bias(
    size: int,
    patterns: tuple[list[tuple[int, int]], ...],
    weights: tuple[int, ...],
    strength: float,
    glut_strength: float = 0.0,
    glut_threshold: int = 12,
) -> tuple[float, ...]:
    """Down-weight 3-letter-heavy patterns, and 4-letter-glut patterns.

    Each pattern's catalog weight is multiplied by
    ``exp(-strength * (threes - fewest_threes))``, so the leanest grids in
    the catalog keep their full weight and heavier ones decay smoothly.
    Anchoring on the catalog minimum keeps the curve meaningful for sizes
    like 9x9, where every pattern has a large absolute 3-letter count.

    The 3-letter penalty alone is not safe on its own: in the 9x9 catalog,
    fewer 3-letter slots almost always means *more* 4-letter slots (only 6
    of 47 patterns have both few threes and a healthy 5-letter count).
    Grids saturated with 4-letter slots and no mid-length slots are the ones
    that fail Hard fill — the pool cannot supply that many 4-letter answers
    without forcing two Jeff Hard-list entries to cross. So patterns with
    more than ``glut_threshold`` 4-letter slots take a second penalty of
    ``exp(-glut_strength * (fours - glut_threshold))``.
    """
    three_counts: list[int] = []
    four_counts: list[int] = []
    for black_cells in patterns:
        lengths = _slot_lengths(size, tuple(sorted(black_cells)))
        three_counts.append(sum(1 for length in lengths if length == 3))
        four_counts.append(sum(1 for length in lengths if length == 4))

    fewest = min(three_counts)
    return tuple(
        weight
        * math.exp(-strength * (threes - fewest))
        * math.exp(-glut_strength * max(0, fours - glut_threshold))
        for weight, threes, fours in zip(weights, three_counts, four_counts)
    )


def get_grid_spec(
    puzzle_type: PuzzleType | str,
    grid_size: int,
    *,
    seed: int | None = None,
    short_slot_bias: float = 0.0,
    four_glut_bias: float = 0.0,
) -> GridSpec:
    """Return a GridSpec for the given puzzle type and size.

    Args:
        puzzle_type: "mini" or "midi" (or PuzzleType enum).
        grid_size: Grid dimension (e.g., 5, 7, 9, 10, 11).
        seed: Optional seed to randomly select a black cell pattern
              using weighted sampling. When None, uses the most common
              pattern as default.
        short_slot_bias: Strength of the penalty applied to patterns with
              many 3-letter slots. 0.0 (default) leaves the catalog
              weights untouched. Larger values shift selection toward
              grids with fewer 3-letter answers. Only affects seeded
              (weighted) selection.
        four_glut_bias: Strength of the penalty applied to patterns with
              more than 12 four-letter slots. Pair this with
              short_slot_bias: penalizing 3-letter slots alone pushes
              selection toward 4-letter-saturated grids, which fail Hard
              fill by forcing Jeff Hard-list entries to cross.

    Returns:
        GridSpec with the appropriate rows, cols, and black cells.

    Raises:
        ValueError: If the puzzle_type/grid_size combination is not supported.
    """
    pt = PuzzleType(puzzle_type)
    key = (pt, grid_size)

    if key not in _VALID_SPECS:
        valid = [f"{t.value}/{s}" for (t, s) in _VALID_SPECS]
        raise ValueError(
            f"Unsupported puzzle_type/grid_size: {pt.value}/{grid_size}. "
            f"Valid combinations: {', '.join(valid)}"
        )

    rows, cols = _VALID_SPECS[key]

    pattern_data = _GRID_PATTERNS.get(key)
    if pattern_data:
        if seed is not None:
            patterns, weights = zip(*pattern_data)
            if short_slot_bias or four_glut_bias:
                weights = _apply_short_slot_bias(
                    rows,
                    patterns,
                    weights,
                    short_slot_bias,
                    four_glut_bias,
                )
            rng = random.Random(seed)
            black_cells = rng.choices(patterns, weights=weights, k=1)[0]
        else:
            black_cells = pattern_data[0][0]
    elif pt == PuzzleType.MIDI:
        from crossword_generator.grid_pattern_generator import generate_pattern

        effective_seed = seed if seed is not None else 0
        black_cells = generate_pattern(rows, cols, seed=effective_seed)
    else:
        black_cells = []

    return GridSpec(rows=rows, cols=cols, black_cells=list(black_cells))
