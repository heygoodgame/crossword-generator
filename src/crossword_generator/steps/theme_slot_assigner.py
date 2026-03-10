"""Assign themed seed entries to grid slots by matching length."""

from __future__ import annotations

from dataclasses import dataclass

from crossword_generator.fillers.csp import Slot


@dataclass
class SlotAssignment:
    """A seed entry assigned to a specific grid slot."""

    word: str
    row: int
    col: int
    direction: str
    length: int


def assign_seed_entries_to_slots(
    seed_entries: list[str],
    revealer: str,
    slots: list[Slot],
) -> list[SlotAssignment]:
    """Assign theme words (seed entries + revealer) to matching grid slots.

    Algorithm:
    - Collect all theme words (seed_entries + revealer)
    - Sort by length descending (most constrained first)
    - For each word, find best unassigned slot of matching length
      - Prefer across over down (convention)
      - Prefer slots closer to center (visual prominence)

    Args:
        seed_entries: The themed seed entry words (uppercase).
        revealer: The revealer word (uppercase).
        slots: Extracted grid slots from extract_slots().

    Returns:
        A list of SlotAssignment objects.

    Raises:
        ValueError: If any theme word has no matching available slot.
    """
    all_words = list(seed_entries)
    if revealer:
        all_words.append(revealer)

    # Sort by length descending (most constrained first)
    all_words.sort(key=len, reverse=True)

    # Compute grid center for distance scoring
    if slots:
        max_row = max(
            s.row + s.length - 1 if s.direction == "down" else s.row
            for s in slots
        )
        max_col = max(
            s.col + s.length - 1 if s.direction == "across" else s.col
            for s in slots
        )
        center_row = max_row / 2.0
        center_col = max_col / 2.0
    else:
        center_row = center_col = 0.0

    assigned_indices: set[int] = set()
    assignments: list[SlotAssignment] = []

    for word in all_words:
        best_slot: Slot | None = None
        best_score = float("inf")

        for slot in slots:
            if slot.index in assigned_indices:
                continue
            if slot.length != len(word):
                continue

            # Score: prefer across (0) over down (1), then center
            direction_penalty = (
                0.0 if slot.direction == "across" else 1.0
            )
            if slot.direction == "down":
                mid_row = slot.row + (slot.length - 1) / 2.0
            else:
                mid_row = float(slot.row)
            if slot.direction == "across":
                mid_col = slot.col + (slot.length - 1) / 2.0
            else:
                mid_col = float(slot.col)
            distance = abs(mid_row - center_row) + abs(mid_col - center_col)
            score = direction_penalty * 100 + distance

            if score < best_score:
                best_score = score
                best_slot = slot

        if best_slot is None:
            raise ValueError(
                f"No available slot of length {len(word)} for theme word {word!r}"
            )

        assigned_indices.add(best_slot.index)
        assignments.append(
            SlotAssignment(
                word=word,
                row=best_slot.row,
                col=best_slot.col,
                direction=best_slot.direction,
                length=best_slot.length,
            )
        )

    return assignments
