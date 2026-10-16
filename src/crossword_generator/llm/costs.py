"""Best-effort LLM cost estimation."""

import re

# Cache writes are priced at the 5-minute TTL (the only TTL the provider
# requests). Cache-read multipliers differ by model: 0.05x base input on
# Opus 5.5 / Sonnet 5.5, 0.1x elsewhere.
_ANTHROPIC_PRICING_USD_PER_MTOK = {
    # Haiku 5.5 has two rate cards keyed on total prompt length (see
    # _HAIKU_5_5_LONG_PROMPT_TOKENS); this is the <=100K-token card.
    "haiku_5_5": {
        "input_tokens": 0.10,
        "output_tokens": 0.50,
        "cache_creation_input_tokens": 0.125,
        "cache_read_input_tokens": 0.01,
    },
    "haiku_5_5_long": {
        "input_tokens": 0.50,
        "output_tokens": 2.50,
        "cache_creation_input_tokens": 0.625,
        "cache_read_input_tokens": 0.05,
    },
    "haiku": {
        "input_tokens": 1.00,
        "output_tokens": 5.00,
        "cache_creation_input_tokens": 1.25,
        "cache_read_input_tokens": 0.10,
    },
    # Sonnet 5.5 keeps Sonnet 5's $2/$10 but halves the cache-read rate.
    "sonnet_5_5": {
        "input_tokens": 2.00,
        "output_tokens": 10.00,
        "cache_creation_input_tokens": 2.50,
        "cache_read_input_tokens": 0.10,
    },
    # Sonnet 5 dropped to $2/$10 (Sonnet 4.x stays at $3/$15).
    "sonnet_5": {
        "input_tokens": 2.00,
        "output_tokens": 10.00,
        "cache_creation_input_tokens": 2.50,
        "cache_read_input_tokens": 0.20,
    },
    "sonnet": {
        "input_tokens": 3.00,
        "output_tokens": 15.00,
        "cache_creation_input_tokens": 3.75,
        "cache_read_input_tokens": 0.30,
    },
    "opus_5_5": {
        "input_tokens": 4.00,
        "output_tokens": 20.00,
        "cache_creation_input_tokens": 5.00,
        "cache_read_input_tokens": 0.20,
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


# Haiku 5.5 bills a request on its long-prompt rate card when the prompt
# (uncached input + cache writes + cache reads) exceeds this many tokens.
_HAIKU_5_5_LONG_PROMPT_TOKENS = 100_000


# OpenAI text pricing (USD per million tokens). Approximate; recorded with
# the log record so downstream analysis can recompute if pricing changes.
_OPENAI_PRICING_USD_PER_MTOK = {
    "gpt-5-mini": {"input_tokens": 0.25, "output_tokens": 2.00},
    "gpt-5-nano": {"input_tokens": 0.05, "output_tokens": 0.40},
    "gpt-5": {"input_tokens": 1.25, "output_tokens": 10.00},
    "gpt-4.1-mini": {"input_tokens": 0.40, "output_tokens": 1.60},
    "gpt-4.1": {"input_tokens": 2.00, "output_tokens": 8.00},
    "o4-mini": {"input_tokens": 1.10, "output_tokens": 4.40},
    "o3": {"input_tokens": 2.00, "output_tokens": 8.00},
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
    if provider == "openai" and model:
        return _estimate_openai_cost(model, usage)

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

    input_tokens = _usage_value(usage, "input_tokens")
    output_tokens = _usage_value(usage, "output_tokens")
    cache_creation_tokens = _usage_value(usage, "cache_creation_input_tokens")
    cache_read_tokens = _usage_value(usage, "cache_read_input_tokens")

    # Anthropic's ``input_tokens`` is already the uncached remainder: the
    # cache fields are additive, so the full prompt is the sum of all three.
    # (Subtracting the cache fields from ``input_tokens`` zeroed out uncached
    # input on well-cached steps and under-reported cost by ~17%.)
    prompt_tokens = input_tokens + cache_creation_tokens + cache_read_tokens
    if pricing_key == "haiku_5_5" and prompt_tokens > _HAIKU_5_5_LONG_PROMPT_TOKENS:
        pricing_key = "haiku_5_5_long"
    prices = _ANTHROPIC_PRICING_USD_PER_MTOK[pricing_key]
    total = (
        input_tokens * prices["input_tokens"]
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


def _estimate_openai_cost(
    model: str,
    usage: dict[str, int | float],
) -> dict[str, object]:
    pricing_key = _openai_pricing_key(model)
    if pricing_key is None:
        return {
            "estimated_cost_usd": None,
            "currency": "USD",
            "pricing": "unknown_openai_model",
        }
    prices = _OPENAI_PRICING_USD_PER_MTOK[pricing_key]
    input_tokens = _usage_value(usage, "input_tokens")
    output_tokens = _usage_value(usage, "output_tokens")
    total = (
        input_tokens * prices["input_tokens"] + output_tokens * prices["output_tokens"]
    ) / 1_000_000
    return {
        "estimated_cost_usd": round(total, 8),
        "currency": "USD",
        "pricing": f"openai_{pricing_key}_usd_per_mtok",
        "rates_usd_per_mtok": dict(prices),
    }


def _openai_pricing_key(model: str) -> str | None:
    normalized = model.lower()
    # Longest keys first so "gpt-5-mini" wins over "gpt-5".
    for family in sorted(_OPENAI_PRICING_USD_PER_MTOK, key=len, reverse=True):
        if normalized.startswith(family):
            return family
    return None


def _anthropic_pricing_key(model: str) -> str | None:
    normalized = model.lower()
    # 5.5 generation first: "opus-5-5" would otherwise match the "opus-5"
    # check below, and "sonnet-5-5" the Sonnet 5 regex.
    if re.search(r"opus-5-5(?:\b|-)", normalized):
        return "opus_5_5"
    if re.search(r"sonnet-5-5(?:\b|-)", normalized):
        return "sonnet_5_5"
    if re.search(r"haiku-5-5(?:\b|-)", normalized):
        return "haiku_5_5"
    # Opus 4.5 through Opus 5 share the $5/$25 tier; Opus 4.1 and earlier
    # were $15/$75.
    if "opus" in normalized and any(
        version in normalized
        for version in ("opus-4-5", "opus-4-6", "opus-4-7", "opus-4-8", "opus-5")
    ):
        return "opus_4_5_plus"
    if re.search(r"sonnet-5(?:\b|-)", normalized):
        return "sonnet_5"
    for family in ("haiku", "sonnet", "opus"):
        if family in normalized:
            return family
    return None


def _usage_value(usage: dict[str, int | float], key: str) -> float:
    value = usage.get(key, 0)
    return float(value) if isinstance(value, int | float) else 0.0
