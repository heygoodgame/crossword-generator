"""Tests for Phase 1 dictionary preparation."""

from pathlib import Path

from crossword_generator.dictionary import Dictionary
from crossword_generator.dictionary_prep import (
    load_excluded_words,
    prepare_flat_dictionary,
    prepare_length_mixed_flat_dictionary,
)


def test_prepare_plain_dictionary_normalizes_case_and_shape(tmp_path: Path) -> None:
    source = tmp_path / "plain.txt"
    output = tmp_path / "flat.txt"
    source.write_text("apple\nBanana\n\ncarrot\n")

    summary = prepare_flat_dictionary(source, output, score=55)

    assert summary.total_lines == 4
    assert summary.nonempty_lines == 3
    assert summary.output_count == 3
    assert output.read_text().splitlines() == [
        "APPLE;55",
        "BANANA;55",
        "CARROT;55",
    ]


def test_prepare_scored_dictionary_flattens_scores(tmp_path: Path) -> None:
    source = tmp_path / "scored.txt"
    output = tmp_path / "flat.txt"
    source.write_text("apple;50\nbanana;60\n")

    summary = prepare_flat_dictionary(source, output, score=55)

    assert summary.output_count == 2
    assert output.read_text().splitlines() == ["APPLE;55", "BANANA;55"]


def test_prepare_dictionary_filters_scored_words_by_length(
    tmp_path: Path,
) -> None:
    source = tmp_path / "scored.txt"
    output = tmp_path / "flat.txt"
    source.write_text(
        "sevenss;59\n"
        "eighties;59\n"
        "ninetieth;60\n"
        "sixsix;50\n"
        "unscored\n"
    )

    summary = prepare_flat_dictionary(
        source,
        output,
        score=55,
        min_source_score_by_length={7: 60, 8: 60, 9: 60},
    )

    assert summary.output_count == 3
    assert summary.skipped_below_source_score == 2
    assert output.read_text().splitlines() == [
        "NINETIETH;55",
        "SIXSIX;55",
        "UNSCORED;55",
    ]


def test_prepare_dictionary_ignores_scores_from_flat_inputs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "flat.txt"
    output = tmp_path / "prepared.txt"
    source.write_text("sevenss;55\neighties;55\nninetieth;55\n")

    summary = prepare_flat_dictionary(
        source,
        output,
        score=55,
        min_source_score_by_length={7: 60, 8: 60, 9: 60},
        flat_score_input_paths=[source],
    )

    assert summary.output_count == 3
    assert summary.skipped_below_source_score == 0
    assert output.read_text().splitlines() == [
        "EIGHTIES;55",
        "NINETIETH;55",
        "SEVENSS;55",
    ]


def test_prepare_dictionary_filters_by_word_length(tmp_path: Path) -> None:
    source = tmp_path / "words.txt"
    output = tmp_path / "flat.txt"
    source.write_text("ape\napple\nbanana\ncucumber\n")

    summary = prepare_flat_dictionary(
        source,
        output,
        score=55,
        min_word_length=4,
        max_word_length=6,
    )

    assert summary.output_count == 2
    assert summary.skipped_outside_length == 2
    assert output.read_text().splitlines() == ["APPLE;55", "BANANA;55"]


def test_prepare_length_mixed_dictionary_uses_easy_shorts_and_hard_longs(
    tmp_path: Path,
) -> None:
    easy_source = tmp_path / "easy.txt"
    hard_source = tmp_path / "hard.txt"
    output = tmp_path / "mixed.txt"
    easy_source.write_text("ACE;55\nPITT;55\nAGAR;55\nPUZZLER;55\n")
    hard_source.write_text(
        "NAFF;70\n"
        "ENYA;70\n"
        "BRADPITT;70\n"
        "LONGWORD;59\n"
        "LONGGOOD;60\n"
    )

    summary = prepare_length_mixed_flat_dictionary(
        easy_source,
        hard_source,
        output,
        score=55,
        short_max_length=5,
        long_min_length=6,
        min_source_score_by_length={8: 60},
        flat_score_input_paths=[easy_source],
    )

    assert summary.output_count == 5
    assert summary.skipped_below_source_score == 1
    assert summary.skipped_outside_length == 3
    assert output.read_text().splitlines() == [
        "ACE;55",
        "AGAR;55",
        "BRADPITT;55",
        "LONGGOOD;55",
        "PITT;55",
    ]


def test_prepare_dictionary_skips_duplicates_and_malformed_rows(
    tmp_path: Path,
) -> None:
    source = tmp_path / "messy.txt"
    output = tmp_path / "flat.txt"
    source.write_text(
        "\n"
        "apple\n"
        "APPLE;50\n"
        "bad;not-a-score\n"
        "too;many;parts\n"
        "two words\n"
        "BERRY;60\n"
    )

    summary = prepare_flat_dictionary(source, output, score=55)

    assert summary.total_lines == 7
    assert summary.nonempty_lines == 6
    assert summary.output_count == 2
    assert summary.duplicates == 1
    assert summary.skipped_malformed == 2
    assert summary.skipped_invalid_word == 1
    assert output.read_text().splitlines() == ["APPLE;55", "BERRY;55"]


def test_prepared_dictionary_loads_through_existing_loader(tmp_path: Path) -> None:
    source = tmp_path / "plain.txt"
    output = tmp_path / "flat.txt"
    source.write_text("apple\nbanana\n")

    prepare_flat_dictionary(source, output, score=55)
    dictionary = Dictionary.load(output, min_word_score=55, min_2letter_score=55)

    assert dictionary.contains("apple")
    assert dictionary.contains("BANANA")
    assert dictionary.score("apple") == 55


def test_prepare_dictionary_merges_extra_sources(tmp_path: Path) -> None:
    source = tmp_path / "base.txt"
    extra = tmp_path / "extra.txt"
    output = tmp_path / "flat.txt"
    source.write_text("apple\nbanana\n")
    extra.write_text("carrot\napple\n")

    summary = prepare_flat_dictionary(
        source,
        output,
        score=55,
        extra_input_paths=[extra],
    )

    assert summary.input_paths == (source, extra)
    assert summary.output_count == 3
    assert summary.duplicates == 1
    assert output.read_text().splitlines() == [
        "APPLE;55",
        "BANANA;55",
        "CARROT;55",
    ]


def test_prepare_dictionary_excludes_words(tmp_path: Path) -> None:
    source = tmp_path / "plain.txt"
    output = tmp_path / "flat.txt"
    source.write_text("apple\nbanana\ncarrot\n")

    summary = prepare_flat_dictionary(
        source,
        output,
        score=55,
        exclude_words={"banana"},
    )

    assert summary.output_count == 2
    assert summary.skipped_excluded == 1
    assert output.read_text().splitlines() == ["APPLE;55", "CARROT;55"]


def test_load_excluded_words_accepts_plain_and_reasoned_lists(
    tmp_path: Path,
) -> None:
    plain = tmp_path / "plain.txt"
    reasoned = tmp_path / "reasoned.txt"
    plain.write_text("apple\n")
    reasoned.write_text("banana;55;reason\n")

    assert load_excluded_words([plain, reasoned]) == {"APPLE", "BANANA"}
