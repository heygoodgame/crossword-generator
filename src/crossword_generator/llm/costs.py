"""Best-effort LLM cost estimation."""


_ANTHROPIC_PRICING_USD_PER_MTOK = {
    "haiku": {
        "input_tokens": 1.00,
        "output_tokens": 5.00,
        "cache_creation_input_tokens": 1.25,
        "cache_read_input_tokens": 0.10,
    },
    "sonnet": {
        "input_tokens": 3.00,
        "output_tokens": 15.00,
        "cache_creation_input_tokens": 3.75,
        "cache_read_input_tokens": 0.30,
    },
    "opus_4_5_plus": {
        "input_tokens": 5.00,
        "output_tokens": 25.00,
        "cache_creation_input_tokens": 6.25,
        "cache_read_input_tokens": 0.50,
    },
    "opus": {
        "input_tokens": 15.00,
        "output_tokens": 75.00,
        "cache_creation_input_tokens": 18.75,
        "cache_read_input_tokens": 1.50,
    },
}


def estimate_llm_cost(
    provider: str,
    model: str | None,
    usage: dict[str, int | float],
) -> dict[str, object]:
    """Estimate request cost from provider usage metadata.

    Prices are intentionally approximate and recorded alongside the log record
    so downstream analysis can recalculate if provider pricing changes.
    """
    if provider != "claude" or not model:
        return {
            "estimated_cost_usd": 0.0 if provider == "ollama" else None,
            "currency": "USD",
            "pricing": "local_ollama_no_api_metering"
            if provider == "ollama"
            else "unknown",
        }

    pricing_key = _anthropic_pricing_key(model)
    if pricing_key is None:
        return {
            "estimated_cost_usd": None,
            "currency": "USD",
            "pricing": "unknown_anthropic_model",
        }

    prices = _ANTHROPIC_PRICING_USD_PER_MTOK[pricing_key]
    input_tokens = _usage_value(usage, "input_tokens")
    output_tokens = _usage_value(usage, "output_tokens")
    cache_creation_tokens = _usage_value(usage, "cache_creation_input_tokens")
    cache_read_tokens = _usage_value(usage, "cache_read_input_tokens")

    # Anthropic usage reports cache tokens separately. Subtract them from base
    # input to avoid double-counting when present.
    base_input_tokens = max(
        0.0,
        input_tokens - cache_creation_tokens - cache_read_tokens,
    )
    total = (
        base_input_tokens * prices["input_tokens"]
        + output_tokens * prices["output_tokens"]
        + cache_creation_tokens * prices["cache_creation_input_tokens"]
        + cache_read_tokens * prices["cache_read_input_tokens"]
    ) / 1_000_000

    return {
        "estimated_cost_usd": round(total, 8),
        "currency": "USD",
        "pricing": f"anthropic_{pricing_key}_usd_per_mtok",
        "rates_usd_per_mtok": dict(prices),
    }


def _anthropic_pricing_key(model: str) -> str | None:
    normalized = model.lower()
    if "opus" in normalized and any(
        version in normalized
        for version in ("opus-4-5", "opus-4-6", "opus-4-7", "opus-4-8")
    ):
        return "opus_4_5_plus"
    for family in _ANTHROPIC_PRICING_USD_PER_MTOK:
        if family in normalized:
            return family
    return None


def _usage_value(usage: dict[str, int | float], key: str) -> float:
    value = usage.get(key, 0)
    return float(value) if isinstance(value, int | float) else 0.0
