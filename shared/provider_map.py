# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Single source of truth for provider-to-model mapping.

Consumers:
  - shared/llm.py
  - shared/llm_router.py
  - shared/telemetry/gen_ai_conventions.py
  - shared/telemetry/otel_exporter.py

V0.7 Pre-Phase additions:
  - Perplexity and Groq as first-class providers (10-provider catalog)
  - ``data_residency_multiplier`` support via ModelEntry (see catalog_v07)
  - ``long_context_breakpoint`` for tiered-pricing models (Gemini 3.1 Pro 200k)
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


# Model-name prefix → provider name.  Ordered longest-prefix-first within
# each provider so that "open-mistral-" matches before "mistral-".
PROVIDER_ROUTES: Dict[str, str] = {
    "gpt-": "openai",
    "text-embedding-": "openai",
    "o1-": "openai",
    "o3-": "openai",
    "o4-": "openai",
    "claude-": "anthropic",
    "gemini-": "google",
    "deepseek-": "deepseek",
    "open-mistral-": "mistral",
    "mistral-": "mistral",
    "ministral-": "mistral",
    "codestral": "mistral",
    "grok-": "xai",
    "meta-llama/": "openrouter",
    "qwen/": "openrouter",
    # V0.7 — route native Qwen3.5 to DashScope before falling back
    # to OpenRouter for legacy qwen3-* entries.
    "qwen3.5": "qwen",
    "qwen3": "openrouter",
    # 3-layer LLM plan: vllm/ = HPC vLLM (Layer 1, main); ollama/ = local laptop Ollama
    # (Layer 2); commercial providers above = cloud (Layer 3). vllm/ and ollama/ are
    # distinct providers though both default to an OpenAI-compatible endpoint on :11434.
    "vllm/": "vllm",
    "ollama/": "ollama",
    # V0.7: new providers
    "perplexity/": "perplexity",
    "perplexity-": "perplexity",
    "sonar": "perplexity",
    "groq/": "groq",
}

# OpenAI-compatible providers: base URL + API key env var
OPENAI_COMPATIBLE_PROVIDERS: Dict[str, Dict[str, str]] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1/chat/completions",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1/chat/completions",
        "api_key_env": "MISTRAL_API_KEY",
    },
    "xai": {
        "base_url": "https://api.x.ai/v1/chat/completions",
        "api_key_env": "XAI_API_KEY",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "vllm": {
        # Layer 1 — HPC vLLM. Override the endpoint with VLLM_BASE_URL (handled in llm.py);
        # default matches the colocated server. No API key needed for a local server.
        "base_url": "http://localhost:11434/v1/chat/completions",
        "api_key_env": "VLLM_API_KEY",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1/chat/completions",
        "api_key_env": "OLLAMA_API_KEY",
    },
    # V0.7: new first-class providers
    "perplexity": {
        "base_url": "https://api.perplexity.ai/chat/completions",
        "api_key_env": "PERPLEXITY_API_KEY",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1/chat/completions",
        "api_key_env": "GROQ_API_KEY",
    },
}

# Provider → API key env var (for key resolution)
PROVIDER_API_KEY_ENV: Dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    **{p: cfg["api_key_env"] for p, cfg in OPENAI_COMPATIBLE_PROVIDERS.items()},
}


# ---------------------------------------------------------------------------
# V0.7: extended model entry (catalog_v07)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelEntry:
    """Extended model entry with V0.7 pricing fields.

    Legacy MODEL_CATALOG in llm_router.py remains authoritative for routing;
    this structure carries the additional fields that the April 2026 catalog
    needs (tiered long-context pricing, data residency surcharges, automatic
    cache discounts).
    """
    model: str
    provider: str
    input_per_1m: float
    output_per_1m: float
    max_context: int
    long_context_breakpoint: Optional[int] = None
    long_context_input_per_1m: Optional[float] = None
    long_context_output_per_1m: Optional[float] = None
    data_residency_multiplier: float = 1.0
    cached_input_multiplier: float = 1.0
    notes: str = ""

    def effective_input_cost(
        self,
        tokens: int,
        *,
        cached: bool = False,
        data_residency: bool = False,
    ) -> float:
        """Return $ for `tokens` of input under the requested pricing flags."""
        n = tokens / 1_000_000
        if (
            self.long_context_breakpoint
            and tokens > self.long_context_breakpoint
            and self.long_context_input_per_1m is not None
        ):
            rate = self.long_context_input_per_1m
        else:
            rate = self.input_per_1m
        if cached:
            rate = rate * self.cached_input_multiplier
        if data_residency:
            rate = rate * self.data_residency_multiplier
        return n * rate


# April 2026 snapshot — minimal entries for pricing/residency verification.
# The full routing catalog lives in shared/llm_router.py::MODEL_CATALOG.
# Only entries with non-default residency/long-context/caching fields appear here.
CATALOG_V07: Dict[str, ModelEntry] = {
    "gpt-5.4-nano":       ModelEntry("gpt-5.4-nano", "openai", 0.20, 1.25, 270_000,
                                     cached_input_multiplier=0.10,
                                     data_residency_multiplier=1.10),
    "gpt-5.4-mini":       ModelEntry("gpt-5.4-mini", "openai", 0.75, 4.50, 270_000,
                                     cached_input_multiplier=0.10,
                                     data_residency_multiplier=1.10),
    "gpt-5.4":            ModelEntry("gpt-5.4", "openai", 2.50, 15.00, 270_000,
                                     cached_input_multiplier=0.10,
                                     data_residency_multiplier=1.10),
    "claude-opus-4-6":    ModelEntry("claude-opus-4-6", "anthropic", 5.00, 25.00, 1_000_000,
                                     data_residency_multiplier=1.10),
    "claude-sonnet-4-6":  ModelEntry("claude-sonnet-4-6", "anthropic", 3.00, 15.00, 1_000_000,
                                     data_residency_multiplier=1.10),
    "gemini-3.1-pro":     ModelEntry("gemini-3.1-pro", "google", 2.00, 12.00, 1_048_576,
                                     long_context_breakpoint=200_000,
                                     long_context_input_per_1m=4.00,
                                     long_context_output_per_1m=18.00),
    "gemini-3.1-flash-lite": ModelEntry("gemini-3.1-flash-lite", "google",
                                        0.25, 1.50, 1_048_576),
    "deepseek-v4":        ModelEntry("deepseek-v4", "deepseek", 0.30, 0.50, 1_000_000,
                                     cached_input_multiplier=0.10,
                                     notes="V4 1M ctx, 81% SWE-bench, hybrid reasoning (Mar 2026)"),
    "grok-4-20-reasoning": ModelEntry("grok-4-20-reasoning", "xai", 2.00, 6.00, 2_000_000,
                                      cached_input_multiplier=0.10),
    "grok-4-20-multi-agent": ModelEntry("grok-4-20-multi-agent", "xai", 2.00, 6.00, 2_000_000,
                                        cached_input_multiplier=0.10,
                                        notes="Native multi-agent variant (Mar 2026)"),
    "qwen3.5-plus":       ModelEntry("qwen3.5-plus", "qwen", 0.26, 1.56, 128_000),
    "qwen3.5-397b-a17b":  ModelEntry("qwen3.5-397b-a17b", "qwen", 0.39, 0.90, 128_000,
                                     notes="MoE flagship (Mar 2026)"),
    "qwen3.5-flash":      ModelEntry("qwen3.5-flash", "qwen", 0.07, 0.26, 128_000),
    "mistral-medium-3.1": ModelEntry("mistral-medium-3.1", "mistral", 0.40, 2.00, 128_000),
    "mistral-small-3.2":  ModelEntry("mistral-small-3.2", "mistral", 0.075, 0.20, 128_000),
    "ministral-3-8b":     ModelEntry("ministral-3-8b", "mistral", 0.05, 0.10, 128_000),
    "perplexity-sonar":   ModelEntry("perplexity-sonar", "perplexity", 1.00, 1.00, 128_000,
                                     notes="Citation tokens unbilled (2026 reform)"),
    "perplexity-sonar-pro": ModelEntry("perplexity-sonar-pro", "perplexity", 3.00, 15.00, 200_000),
    "perplexity-reasoning-pro": ModelEntry("perplexity-reasoning-pro", "perplexity",
                                           2.00, 8.00, 128_000),
    "perplexity-deep-research": ModelEntry("perplexity-deep-research", "perplexity",
                                           2.00, 8.00, 128_000,
                                           notes="Plus search+reasoning fees ($5-$14/1k)"),
    "groq/llama-4-scout": ModelEntry("groq/llama-4-scout", "groq", 0.11, 0.34, 10_000_000,
                                     cached_input_multiplier=0.50,
                                     notes="LPU 300-1000 tok/s; free dev tier"),
    "groq/llama-3.3-70b": ModelEntry("groq/llama-3.3-70b", "groq", 0.59, 0.79, 128_000,
                                     cached_input_multiplier=0.50),
    "groq/llama-3.1-8b":  ModelEntry("groq/llama-3.1-8b", "groq", 0.05, 0.08, 128_000,
                                     cached_input_multiplier=0.50),
}


def get_catalog_entry(model: str) -> Optional[ModelEntry]:
    """Return the V0.7 ModelEntry for a given model name, or None if absent.

    Unknown/legacy models return None — callers should fall back to the
    tuple-based MODEL_CATALOG in shared.llm_router.
    """
    return CATALOG_V07.get(model)

# ---------------------------------------------------------------------------
# V0.7 post-release (2026-04-20): Web-crawl provider registry.
#
# Introduced by the Firecrawl → WebCrawl refactor. See
# ``shared/tools/webcrawl_tool.py`` for the facade and
# ``shared/webcrawl/providers/`` for the concrete adapters.
#
# Mirrors the shape of ``OPENAI_COMPATIBLE_PROVIDERS``: one entry per backend
# with the env var(s) that configure it and the capability set it exposes.
# ---------------------------------------------------------------------------

WEBCRAWL_PROVIDERS: Dict[str, Dict] = {
    "firecrawl": {
        "api_key_env": "FIRECRAWL_API_KEY",
        "base_url": "https://api.firecrawl.dev",
        "capabilities": {"scrape", "crawl", "extract", "map"},
    },
    "tavily": {
        "api_key_env": "TAVILY_API_KEY",
        "base_url": "https://api.tavily.com",
        "capabilities": {"scrape", "crawl", "extract"},
    },
    "jina": {
        "api_key_env": "JINA_API_KEY",  # optional — rate-limited without
        "base_url": "https://r.jina.ai",
        "capabilities": {"scrape", "extract"},
    },
    "crawl4ai": {
        "api_key_env": "CRAWL4AI_API_KEY",  # optional
        "base_url_env": "CRAWL4AI_BASE_URL",  # required (self-hosted)
        "capabilities": {"scrape", "crawl", "extract"},
    },
}


# All known providers (useful for validation)
ALL_PROVIDERS = set(PROVIDER_ROUTES.values())


def detect_provider(model: str, default: str = "openai") -> str:
    """Detect the provider for a given model name.

    Args:
        model: Model name (e.g. "gpt-4o-mini", "claude-sonnet-4-6").
        default: Provider to return if no prefix matches.

    Returns:
        Provider name string (e.g. "openai", "anthropic").
    """
    for prefix, provider in PROVIDER_ROUTES.items():
        if model.startswith(prefix):
            return provider
    return default
