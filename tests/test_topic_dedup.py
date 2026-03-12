"""Tests for topic deduplication utilities."""

from __future__ import annotations

from crossword_generator.topic_dedup import (
    build_normalized_topic_set,
    extract_content_words,
    is_topic_duplicate,
    is_topic_similar,
    normalize_topic,
)


class TestNormalizeTopic:
    def test_lowercase(self) -> None:
        assert normalize_topic("Things That FLY") == "things that fly"

    def test_strip_quotes(self) -> None:
        assert normalize_topic("Things that can be 'missed'") == (
            "things that can be missed"
        )

    def test_strip_double_quotes(self) -> None:
        assert normalize_topic('Things that are "sticky"') == (
            "things that are sticky"
        )

    def test_collapse_whitespace(self) -> None:
        assert normalize_topic("things   that   fly") == "things that fly"

    def test_strip_special_chars(self) -> None:
        assert normalize_topic("things! that? fly.") == "things that fly"

    def test_smart_quotes(self) -> None:
        assert normalize_topic("Things that are \u201csticky\u201d") == (
            "things that are sticky"
        )

    def test_empty_string(self) -> None:
        assert normalize_topic("") == ""

    def test_only_special_chars(self) -> None:
        assert normalize_topic("!!!???") == ""


class TestExtractContentWords:
    def test_removes_stopwords(self) -> None:
        words = extract_content_words("Things that are sticky or clingy")
        assert words == {"sticky", "clingy"}

    def test_multi_word_topic(self) -> None:
        words = extract_content_words("Things you can do with a ball")
        assert words == {"ball"}

    def test_no_stopwords_remaining(self) -> None:
        words = extract_content_words("Things that are")
        assert words == set()

    def test_all_content_words(self) -> None:
        words = extract_content_words("pizza toppings")
        assert words == {"pizza", "toppings"}

    def test_preserves_meaningful_words(self) -> None:
        words = extract_content_words("Easy to forget")
        assert words == {"easy", "forget"}


class TestBuildNormalizedTopicSet:
    def test_basic(self) -> None:
        topics = ["Things That Fly", "things that fly", "THINGS THAT FLY"]
        result = build_normalized_topic_set(topics)
        assert result == {"things that fly"}

    def test_empty(self) -> None:
        assert build_normalized_topic_set([]) == set()


class TestIsTopicDuplicate:
    def test_exact_match(self) -> None:
        existing = {"things that can be missed"}
        assert is_topic_duplicate("Things that can be missed", existing)

    def test_case_insensitive(self) -> None:
        existing = {"things that fly"}
        assert is_topic_duplicate("THINGS THAT FLY", existing)

    def test_quote_variants(self) -> None:
        existing = {"things that can be missed"}
        assert is_topic_duplicate(
            "Things that can be 'missed'", existing
        )

    def test_no_match(self) -> None:
        existing = {"things that fly"}
        assert not is_topic_duplicate("Things that swim", existing)

    def test_empty_existing(self) -> None:
        assert not is_topic_duplicate("anything", set())


class TestIsTopicSimilar:
    def test_identical_words(self) -> None:
        existing = ["Things that are twisted"]
        similar, match = is_topic_similar(
            "Things that can be twisted", existing, threshold=0.6
        )
        # Both extract to {"twisted"} → Jaccard 1.0
        assert similar
        assert match == "Things that are twisted"

    def test_partial_overlap_above_threshold(self) -> None:
        # "sticky clingy" vs "sticky gooey" → {"sticky","clingy"} vs
        # {"sticky","gooey"} → intersection=1, union=3 → 0.33
        existing = ["Things that are sticky or gooey"]
        similar, _ = is_topic_similar(
            "Things that are sticky or clingy", existing, threshold=0.3
        )
        assert similar

    def test_partial_overlap_below_threshold(self) -> None:
        existing = ["Things that are easy to forget"]
        similar, _ = is_topic_similar(
            "Things that are easy to overlook", existing, threshold=0.6
        )
        # {"easy","forget"} vs {"easy","overlook"} → 1/3 = 0.33
        assert not similar

    def test_no_overlap(self) -> None:
        existing = ["Things that fly"]
        similar, _ = is_topic_similar(
            "Things that swim", existing, threshold=0.6
        )
        assert not similar

    def test_empty_existing_list(self) -> None:
        similar, match = is_topic_similar("anything", [], threshold=0.6)
        assert not similar
        assert match is None

    def test_all_stopwords_topic(self) -> None:
        """Topic with only stopwords should not match."""
        similar, match = is_topic_similar(
            "Things that are", ["Things that can be"], threshold=0.6
        )
        assert not similar
        assert match is None

    def test_threshold_boundary(self) -> None:
        """Exact threshold value should count as similar."""
        # "sticky" vs "sticky clingy" → {"sticky"} vs {"sticky","clingy"}
        # → 1/2 = 0.5
        existing = ["Things that are sticky or clingy"]
        similar, _ = is_topic_similar(
            "Things that are sticky", existing, threshold=0.5
        )
        assert similar

    def test_below_threshold_boundary(self) -> None:
        existing = ["Things that are sticky or clingy"]
        similar, _ = is_topic_similar(
            "Things that are sticky", existing, threshold=0.51
        )
        assert not similar
