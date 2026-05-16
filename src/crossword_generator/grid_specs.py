"""Grid specification catalog for supported puzzle types and sizes."""

from __future__ import annotations

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
        for source in (base, _flip_top_to_bottom(base, size=9)):
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
    # 5x5: 34 patterns with frequency-based weights.
    (PuzzleType.MINI, 5): [
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
    ],
    # 7x7: 50 patterns with frequency-based weights.
    (PuzzleType.MINI, 7): [
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
    ],
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


def get_grid_patterns(
    puzzle_type: PuzzleType | str,
    grid_size: int,
    *,
    symmetric_only: bool = False,
) -> tuple[WeightedGridPattern, ...]:
    """Return structured weighted grid patterns for a supported mini size.

    ``symmetric_only`` makes it possible for future generation experiments to
    filter Jeff's asymmetric mini patterns without changing today's defaults.
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
            symmetric=is_rotationally_symmetric(rows, black_cells),
        )
        for black_cells, weight in _GRID_PATTERNS.get(key, [])
    )
    if symmetric_only:
        return tuple(pattern for pattern in patterns if pattern.symmetric)
    return patterns


def get_grid_spec(
    puzzle_type: PuzzleType | str,
    grid_size: int,
    *,
    seed: int | None = None,
) -> GridSpec:
    """Return a GridSpec for the given puzzle type and size.

    Args:
        puzzle_type: "mini" or "midi" (or PuzzleType enum).
        grid_size: Grid dimension (e.g., 5, 7, 9, 10, 11).
        seed: Optional seed to randomly select a black cell pattern
              using weighted sampling. When None, uses the most common
              pattern as default.

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
