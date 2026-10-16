"""Tests for LLM cost estimation."""

from __future__ import annotations

from crossword_generator.llm.costs import estimate_llm_cost


def test_opus_4_8_uses_current_pricing() -> None:
    result = estimate_llm_cost(
        "claude",
        "claude-opus-4-8",
        {
            "input_tokens": 5_000,
            "output_tokens": 2_000,
            "cache_creation_input_tokens": 1_000,
            "cache_read_input_tokens": 5_000,
        },
    )

    assert result["pricing"] == "anthropic_opus_4_5_plus_usd_per_mtok"
    assert result["rates_usd_per_mtok"] == {
        "input_tokens": 5.0,
        "output_tokens": 25.0,
        "cache_creation_input_tokens": 6.25,
        "cache_read_input_tokens": 0.5,
    }
    assert result["estimated_cost_usd"] == 0.08375


def test_cache_tokens_are_additive_to_input_tokens() -> None:
    """Anthropic's input_tokens excludes cache tokens; never subtract them.

    A well-cached call reports a small uncached ``input_tokens`` next to a
    large ``cache_read_input_tokens``. The uncached part must still be billed
    at the full input rate.
    """
    result = estimate_llm_cost(
        "claude",
        "claude-sonnet-5",
        {
            "input_tokens": 400,
            "output_tokens": 1_000,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 4_000,
        },
    )

    # 400 * $2 + 1,000 * $10 + 4,000 * $0.20 (per MTok)
    assert result["estimated_cost_usd"] == 0.0116


def test_opus_5_5_uses_its_own_tier() -> None:
    result = estimate_llm_cost(
        "claude",
        "claude-opus-5-5",
        {
            "input_tokens": 1_000_000,
            "output_tokens": 1_000_000,
            "cache_creation_input_tokens": 1_000_000,
            "cache_read_input_tokens": 1_000_000,
        },
    )

    assert result["pricing"] == "anthropic_opus_5_5_usd_per_mtok"
    assert result["estimated_cost_usd"] == 29.2


def test_sonnet_5_5_has_reduced_cache_reads() -> None:
    result = estimate_llm_cost(
        "claude",
        "claude-sonnet-5-5",
        {"input_tokens": 1_000_000, "cache_read_input_tokens": 1_000_000},
    )

    assert result["pricing"] == "anthropic_sonnet_5_5_usd_per_mtok"
    assert result["rates_usd_per_mtok"]["cache_read_input_tokens"] == 0.10
    assert result["estimated_cost_usd"] == 2.1


def test_haiku_5_5_short_prompt_rate_card() -> None:
    result = estimate_llm_cost(
        "claude",
        "claude-haiku-5-5",
        {"input_tokens": 1_000, "output_tokens": 1_000},
    )

    assert result["pricing"] == "anthropic_haiku_5_5_usd_per_mtok"
    assert result["estimated_cost_usd"] == 0.0006


def test_haiku_5_5_long_prompt_rate_card_counts_cache_reads() -> None:
    result = estimate_llm_cost(
        "claude",
        "claude-haiku-5-5",
        {
            "input_tokens": 1_000,
            "output_tokens": 1_000,
            "cache_read_input_tokens": 100_000,
        },
    )

    assert result["pricing"] == "anthropic_haiku_5_5_long_usd_per_mtok"
    # 1,000 * $0.50 + 1,000 * $2.50 + 100,000 * $0.05 (per MTok)
    assert result["estimated_cost_usd"] == 0.008


def test_haiku_4_5_keeps_haiku_pricing() -> None:
    result = estimate_llm_cost(
        "claude",
        "claude-haiku-4-5-20251001",
        {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
    )

    assert result["pricing"] == "anthropic_haiku_usd_per_mtok"
    assert result["estimated_cost_usd"] == 6.0


def test_opus_5_shares_opus_4_5_plus_tier() -> None:
    result = estimate_llm_cost(
        "claude",
        "claude-opus-5",
        {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
    )

    assert result["pricing"] == "anthropic_opus_4_5_plus_usd_per_mtok"
    assert result["estimated_cost_usd"] == 30.0


def test_sonnet_5_uses_reduced_pricing() -> None:
    result = estimate_llm_cost(
        "claude",
        "claude-sonnet-5",
        {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
    )

    assert result["pricing"] == "anthropic_sonnet_5_usd_per_mtok"
    assert result["rates_usd_per_mtok"]["cache_read_input_tokens"] == 0.20
    assert result["estimated_cost_usd"] == 12.0


def test_sonnet_4_6_keeps_sonnet_4_pricing() -> None:
    result = estimate_llm_cost(
        "claude",
        "claude-sonnet-4-6",
        {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
    )

    assert result["pricing"] == "anthropic_sonnet_usd_per_mtok"
    assert result["estimated_cost_usd"] == 18.0


def test_legacy_opus_family_keeps_legacy_pricing() -> None:
    result = estimate_llm_cost(
        "claude",
        "claude-opus-4-20250514",
        {"input_tokens": 1_000, "output_tokens": 1_000},
    )

    assert result["pricing"] == "anthropic_opus_usd_per_mtok"
    assert result["rates_usd_per_mtok"]["input_tokens"] == 15.0
    assert result["rates_usd_per_mtok"]["output_tokens"] == 75.0
    assert result["estimated_cost_usd"] == 0.09


def test_openai_gpt5_pricing() -> None:
    result = estimate_llm_cost(
        "openai",
        "gpt-5",
        {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
    )

    assert result["pricing"] == "openai_gpt-5_usd_per_mtok"
    assert result["estimated_cost_usd"] == 11.25


def test_openai_gpt5_mini_wins_over_gpt5_prefix() -> None:
    result = estimate_llm_cost(
        "openai",
        "gpt-5-mini",
        {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
    )

    assert result["pricing"] == "openai_gpt-5-mini_usd_per_mtok"
    assert result["estimated_cost_usd"] == 2.25


def test_openai_unknown_model_returns_none() -> None:
    result = estimate_llm_cost("openai", "gpt-foo", {"input_tokens": 10})

    assert result["estimated_cost_usd"] is None
    assert result["pricing"] == "unknown_openai_model"
