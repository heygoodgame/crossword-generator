"""Tests for inflectional variant expansion of excluded answers."""

from __future__ import annotations

import pytest

from crossword_generator.answer_variants import answer_variants, expand_answer_variants


def test_expands_regular_suffixes() -> None:
    variants = expand_answer_variants(["WALK"])
    assert {"WALKS", "WALKED", "WALKING"} <= variants
    assert "WALK" not in variants


def test_drops_trailing_e_before_vowel_suffixes() -> None:
    variants = expand_answer_variants(["bake"])
    assert {"BAKES", "BAKED", "BAKING"} <= variants


def test_consonant_y_both_directions() -> None:
    assert {"PARTIES", "PARTIED"} <= expand_answer_variants(["PARTY"])
    assert "PARTY" in expand_answer_variants(["PARTIES"])
    # Vowel + Y is regular: DELAYS, not DELAIES.
    delay = expand_answer_variants(["DELAY"])
    assert "DELAYS" in delay
    assert "DELAIES" not in delay


def test_no_variant_shorter_than_min_base_length() -> None:
    variants = expand_answer_variants(["ARTS", "ERAS", "BOXES", "TIES"])
    assert all(len(v) >= 4 for v in variants)
    assert "TIE" not in variants


def test_strips_terminal_s_and_es() -> None:
    assert "PART" in expand_answer_variants(["PARTS"])
    assert "WATCH" in expand_answer_variants(["WATCHES"])
    # BOX is 3 letters: glue is never excluded, in either direction.
    assert "BOX" not in expand_answer_variants(["BOXES"])
    assert "ART" not in expand_answer_variants(["ARTS"])
    assert "ART" in expand_answer_variants(["ARTS"], min_base_length=3)


def test_short_bases_are_not_expanded_by_default() -> None:
    assert expand_answer_variants(["ART", "RIO", "ETA"]) == set()
    assert "ARTS" in expand_answer_variants(["ART"], min_base_length=3)


def test_bases_never_returned_and_input_normalized() -> None:
    variants = expand_answer_variants([" part ", "PARTS", ""])
    assert "PART" not in variants
    assert "PARTS" not in variants
    assert "PARTED" in variants


def test_non_alpha_ignored() -> None:
    assert expand_answer_variants(["A1B2"]) == set()


# The pair tables below match hey-you's CrosswordAnswerVariantsTest, which pins
# the scheduler side of the same rule. Change both together.
VARIANT_PAIRS = [
    ("USER", "USERS"),  # the reported pair (2026-10-26)
    ("IDEA", "IDEAS"),
    ("WATCH", "WATCHES"),
    ("ECHO", "ECHOED"),
    ("TIDY", "TIDYING"),
    ("RACE", "RACED"),
    ("SNIPE", "SNIPING"),
    ("ENTRY", "ENTRIES"),
    ("PARTY", "PARTIED"),
    ("STRIP", "STRIPPED"),
    ("SLAM", "SLAMMING"),
]

UNRELATED_PAIRS = [
    ("LINE", "LINER"),  # agentive -ER
    ("DORM", "DORMER"),  # noun -ER
    ("STAT", "STATES"),  # -ES after a non-sibilant
    ("ART", "ARTS"),  # three-letter base
    ("USE", "USER"),
]


@pytest.mark.parametrize(("base", "form"), VARIANT_PAIRS)
def test_variants_match_in_both_directions(base: str, form: str) -> None:
    assert form in answer_variants(base)
    assert base in answer_variants(form)


@pytest.mark.parametrize(("left", "right"), UNRELATED_PAIRS)
def test_unrelated_words_are_not_variants(left: str, right: str) -> None:
    assert right not in answer_variants(left)
    assert left not in answer_variants(right)


def test_relation_is_symmetric_for_every_generated_variant() -> None:
    for answer in (
        "USER",
        "USERS",
        "TIDYING",
        "STRIPPED",
        "ENTRIES",
        "BAKING",
        "WATCHES",
        "FREE",
    ):
        for variant in answer_variants(answer):
            assert answer in answer_variants(variant), f"{variant} -> {answer}"
