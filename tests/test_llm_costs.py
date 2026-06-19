"""Tests for LLM cost estimation."""

from __future__ import annotations

from crossword_generator.llm.costs import estimate_llm_cost


def test_opus_4_8_uses_current_pricing() -> None:
    result = estimate_llm_cost(
        "claude",
        "claude-opus-4-8",
        {
            "input_tokens": 11_000,
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
