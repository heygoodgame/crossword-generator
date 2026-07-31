"""Tests for required-entry seeding in the fill step.

The exact_score_count rule (e.g. "exactly one 7-letter score-60 entry" on
Hard 7x7) was originally enforced only as a post-fill grading penalty. That
made the filler rediscover whichever qualifying entries were easiest to fill
around, so a handful recurred across a batch while most of the eligible pool
never appeared. These tests pin the sampling behaviour that replaced it.
"""

from __future__ import annotations

import random

from crossword_generator.dictionary import Dictionary
from crossword_generator.steps.fill_step import (
    _select_seed_candidates,
    _weighted_sample_without_replacement,
)


def _dictionary(words: dict[str, int]) -> Dictionary:
    return Dictionary(words, min_word_score=50, min_2letter_score=50)


class TestWeightedSample:
    def test_returns_at_most_k_distinct_words(self) -> None:
        words = [f"WORD{i:03d}" for i in range(50)]
        sample = _weighted_sample_without_replacement(
            words, [1.0] * len(words), 5, random.Random(0)
        )
        assert len(sample) == 5
        assert len(set(sample)) == 5

    def test_zero_weight_words_are_never_drawn(self) -> None:
        words = ["KEEP", "DROP"]
        sample = _weighted_sample_without_replacement(
            words, [1.0, 0.0], 2, random.Random(0)
        )
        assert sample == ["KEEP"]

    def test_higher_weight_is_drawn_more_often(self) -> None:
        words = ["COMMON", "RARE"]
        draws = [
            _weighted_sample_without_replacement(
                words, [10.0, 1.0], 1, random.Random(seed)
            )[0]
            for seed in range(200)
        ]
        assert draws.count("COMMON") > draws.count("RARE")

    def test_k_larger_than_pool_returns_whole_pool(self) -> None:
        words = ["ONE", "TWO"]
        sample = _weighted_sample_without_replacement(
            words, [1.0, 1.0], 10, random.Random(0)
        )
        assert sorted(sample) == ["ONE", "TWO"]


class TestSelectSeedCandidates:
    def test_only_returns_words_of_requested_length_and_score(self) -> None:
        d = _dictionary(
            {
                "SEVENLT": 60,  # 7 letters, qualifying
                "ALSOSEV": 60,  # 7 letters, qualifying
                "LOWSCOR": 50,  # 7 letters, below min_score
                "SIXLTRS": 60,
                "SHORTER": 60,
                "TOOSHRT": 60,
            }
        )
        picks = _select_seed_candidates(
            d, length=7, min_score=60, count=10, rng=random.Random(1)
        )
        assert picks
        for word in picks:
            assert len(word) == 7
            assert d.score(word) is not None
            assert d.score(word) >= 60
        assert "LOWSCOR" not in picks

    def test_empty_pool_returns_empty_list(self) -> None:
        d = _dictionary({"AAAAAAA": 50})
        picks = _select_seed_candidates(
            d, length=7, min_score=60, count=3, rng=random.Random(0)
        )
        assert picks == []

    def test_draws_spread_across_the_whole_pool(self) -> None:
        """The core regression: sampling must not concentrate on a few words."""
        pool = {f"WORD{i:03d}": 60 for i in range(100)}
        d = _dictionary(pool)
        seen = {
            _select_seed_candidates(
                d, length=7, min_score=60, count=1, rng=random.Random(seed)
            )[0]
            for seed in range(300)
        }
        # A biased sampler collapses onto a handful of entries; a uniform one
        # covers most of a 100-word pool within 300 draws.
        assert len(seen) > 60

    def test_usage_counts_downweight_already_used_entries(self) -> None:
        pool = {f"WORD{i:03d}": 60 for i in range(20)}
        d = _dictionary(pool)
        heavily_used = {"WORD000": 50}
        draws = [
            _select_seed_candidates(
                d,
                length=7,
                min_score=60,
                count=1,
                rng=random.Random(seed),
                usage_counts=heavily_used,
            )[0]
            for seed in range(100)
        ]
        # Down-weighted, not hard-excluded: rare but not necessarily absent.
        assert draws.count("WORD000") <= 2

    def test_used_entries_still_reachable_when_pool_exhausted(self) -> None:
        """Weighting must never make a required entry unplaceable."""
        d = _dictionary({"ONLYONE": 60})
        picks = _select_seed_candidates(
            d,
            length=7,
            min_score=60,
            count=1,
            rng=random.Random(0),
            usage_counts={"ONLYONE": 99},
        )
        assert picks == ["ONLYONE"]
