"""Tests for inflectional variant expansion of excluded answers."""

from __future__ import annotations

from crossword_generator.answer_variants import expand_answer_variants


def test_expands_regular_suffixes() -> None:
    variants = expand_answer_variants(["WALK"])
    assert {"WALKS", "WALKES", "WALKED", "WALKER", "WALKING"} <= variants
    assert "WALK" not in variants


def test_drops_trailing_e_before_vowel_suffixes() -> None:
    variants = expand_answer_variants(["bake"])
    assert {"BAKES", "BAKED", "BAKER", "BAKING"} <= variants


def test_consonant_y_both_directions() -> None:
    assert {"PARTIES", "PARTIED", "PARTIER"} <= expand_answer_variants(["PARTY"])
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
    boxes = expand_answer_variants(["BOXES"])
    assert {"BOXE"} <= boxes
    # BOX is 3 letters: glue is never excluded, in either direction.
    assert "BOX" not in boxes
    assert "ART" not in expand_answer_variants(["ARTS"])
    assert "ART" in expand_answer_variants(["ARTS"], min_base_length=3)
    # Double-S words are not plurals.
    assert "GLAS" not in expand_answer_variants(["GLASS"])


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
