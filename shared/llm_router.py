# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Model Router — Provider-agnostic model selection for AEL workflows.

Routes LLM requests to the optimal model+provider based on task type,
with automatic fallback chains when a provider's API key is unavailable.

Supports 7+ providers and 20+ models. All providers auto-detected from
environment variables; each hosted provider needs its own key, local models none.

Usage:
    from shared.llm_router import ModelRouter

    router = ModelRouter()

    # Select model by task type
    model = router.select_model("extraction")       # → gpt-4.1-nano
    model = router.select_model("reasoning")         # → claude-sonnet-4-6
    model = router.select_model("structured_output")  # → gpt-4.1

    # Check available providers
    providers = router.available_providers  # {"openai", "anthropic", ...}

    # Custom routing config
    router = ModelRouter(config={"extraction": "mistral-nemo"})
"""

import os
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Provider → Model Catalog
# ---------------------------------------------------------------------------

# Maps model name → (provider, input_$/1M, output_$/1M, context_window)
MODEL_CATALOG: Dict[str, Tuple[str, float, float, int]] = {
    # OpenAI (direct API)
    "gpt-4.1-nano":     ("openai",      0.02,   0.15,   1_000_000),
    "gpt-4o-mini":      ("openai",      0.15,   0.60,   128_000),
    "gpt-5-mini":       ("openai",      0.25,   2.00,   400_000),
    "gpt-4.1-mini":     ("openai",      0.40,   1.60,   1_000_000),
    "o4-mini":          ("openai",      1.10,   4.40,   200_000),
    "gpt-4.1":          ("openai",      2.00,   8.00,   1_000_000),
    "gpt-5.4":          ("openai",      2.50,   20.00,  1_000_000),
    # Anthropic (direct API)
    "claude-haiku-4-5-20251001": ("anthropic", 1.00, 5.00, 200_000),
    "claude-sonnet-4-6":         ("anthropic", 3.00, 15.00, 200_000),
    "claude-opus-4-6":           ("anthropic", 5.00, 25.00, 1_000_000),
    # Google (direct API)
    "gemini-flash-lite":  ("google",   0.10,   0.40,   1_000_000),
    "gemini-2.5-flash":   ("google",   0.15,   0.60,   1_000_000),
    "gemini-2.5-pro":     ("google",   1.25,   10.00,  1_000_000),
    # DeepSeek (OpenAI-compatible)
    "deepseek-chat":     ("deepseek",   0.14,   0.28,   128_000),
    "deepseek-reasoner": ("deepseek",   0.50,   2.18,   128_000),
    # Mistral (OpenAI-compatible)
    "open-mistral-nemo":    ("mistral",  0.02,   0.04,   128_000),
    "codestral-latest":     ("mistral",  0.30,   0.90,   256_000),
    "mistral-large-latest": ("mistral",  0.50,   1.50,   256_000),
    # xAI (OpenAI-compatible)
    "grok-3-mini-fast":     ("xai",      0.20,   0.50,   2_000_000),
    # OpenRouter (for Llama, Qwen, any model)
    "meta-llama/llama-4-scout":    ("openrouter", 0.08, 0.30, 10_000_000),
    "meta-llama/llama-4-maverick": ("openrouter", 0.15, 0.60, 1_000_000),
    "qwen/qwen3-32b":             ("openrouter", 0.15, 0.75, 128_000),
    # Ollama (local — zero cost)
    "ollama/llama-4-scout": ("ollama", 0.0, 0.0, 128_000),
    "ollama/qwen3-32b":    ("ollama", 0.0, 0.0, 128_000),
    # V0.6 additions — March 2026 models
    "gemini-3.1-pro":      ("google",     2.00,  12.00, 1_000_000),
    "gemini-3-flash":      ("google",     0.50,   3.00, 1_000_000),
    "deepseek-v3.2":       ("deepseek",   0.28,   0.42, 128_000),
    "qwen3.5-plus":        ("openrouter", 0.26,   1.56, 128_000),
    "mistral-medium-3":    ("mistral",    0.40,   2.00, 256_000),
    "mistral-small-3.2":   ("mistral",    0.06,   0.18, 256_000),
    "qwen3-8b":            ("openrouter", 0.05,   0.40, 128_000),
    # V0.7 additions — April 2026 catalog refresh
    "gpt-5.4-nano":        ("openai",     0.20,   1.25, 270_000),
    "gpt-5.4-mini":        ("openai",     0.75,   4.50, 270_000),
    "gemini-3.1-flash-lite": ("google",   0.25,   1.50, 1_048_576),
    "deepseek-v4":         ("deepseek",   0.30,   0.50, 1_000_000),
    "grok-4-20-reasoning": ("xai",        2.00,   6.00, 2_000_000),
    "grok-4-20-multi-agent": ("xai",      2.00,   6.00, 2_000_000),
    "qwen3.5-397b-a17b":   ("qwen",       0.39,   0.90, 128_000),
    "qwen3.5-flash":       ("qwen",       0.07,   0.26, 128_000),
    "mistral-medium-3.1":  ("mistral",    0.40,   2.00, 128_000),
    "ministral-3-8b":      ("mistral",    0.05,   0.10, 128_000),
    # Perplexity (new provider in V0.7)
    "perplexity-sonar":    ("perplexity",       1.00,   1.00, 128_000),
    "perplexity-sonar-pro": ("perplexity",      3.00,  15.00, 200_000),
    "perplexity-reasoning-pro": ("perplexity",  2.00,   8.00, 128_000),
    "perplexity-deep-research": ("perplexity",  2.00,   8.00, 128_000),
    # Groq (new provider in V0.7)
    "groq/llama-4-scout":  ("groq",       0.11,   0.34, 10_000_000),
    "groq/llama-3.3-70b":  ("groq",       0.59,   0.79, 128_000),
    "groq/llama-3.1-8b":   ("groq",       0.05,   0.08, 128_000),
}

# Friendly aliases → canonical model names (for convenience)
MODEL_ALIASES: Dict[str, str] = {
    "deepseek-v3": "deepseek-chat",
    "deepseek-r1": "deepseek-reasoner",
    "mistral-nemo": "open-mistral-nemo",
    "codestral": "codestral-latest",
    "mistral-large-3": "mistral-large-latest",
    "grok-4.1-fast": "grok-3-mini-fast",
    "llama-4-scout": "meta-llama/llama-4-scout",
    "llama-4-maverick": "meta-llama/llama-4-maverick",
    "qwen3-32b": "qwen/qwen3-32b",
}

# Provider → env var for API key
PROVIDER_ENV_KEYS: Dict[str, str] = {
    "openai":     "OPENAI_API_KEY",
    "anthropic":  "ANTHROPIC_API_KEY",
    "google":     "GOOGLE_API_KEY",
    "deepseek":   "DEEPSEEK_API_KEY",
    "mistral":    "MISTRAL_API_KEY",
    "xai":        "XAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "ollama":     "OLLAMA_HOST",  # Always available if Ollama is running
    # V0.7 additions
    "qwen":       "DASHSCOPE_API_KEY",
    "perplexity": "PERPLEXITY_API_KEY",
    "groq":       "GROQ_API_KEY",
}

# Provider → base URL for API calls
PROVIDER_BASE_URLS: Dict[str, str] = {
    "openai":     "https://api.openai.com/v1",
    "anthropic":  "https://api.anthropic.com/v1",
    "google":     "https://generativelanguage.googleapis.com/v1beta/models",
    "deepseek":   "https://api.deepseek.com/v1",
    "mistral":    "https://api.mistral.ai/v1",
    "xai":        "https://api.x.ai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama":     "http://localhost:11434/v1",
    # V0.7 additions
    "qwen":       "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "perplexity": "https://api.perplexity.ai",
    "groq":       "https://api.groq.com/openai/v1",
}


# ---------------------------------------------------------------------------
# Default Task → Model Routes
# ---------------------------------------------------------------------------

DEFAULT_ROUTES: Dict[str, List[str]] = {
    # Task type → ordered preference list (first available wins)
    "extraction":        ["gpt-4.1-nano", "open-mistral-nemo", "deepseek-chat", "gpt-4o-mini"],
    "formatting":        ["gpt-4.1-nano", "open-mistral-nemo", "deepseek-chat", "gpt-4o-mini"],
    "structured_output": ["gpt-4.1", "gpt-4.1-mini", "gpt-4o-mini"],
    "reasoning":         ["claude-sonnet-4-6", "gpt-5.4", "gemini-2.5-pro", "gpt-4o-mini"],
    "summarization":     ["gemini-2.5-flash", "meta-llama/llama-4-scout", "deepseek-chat", "gpt-4o-mini"],
    "calibration":       ["gpt-4.1", "deepseek-reasoner", "gpt-4o-mini"],
    "code_generation":   ["codestral-latest", "gpt-4.1", "gpt-4o-mini"],
    "reasoning_cot":     ["o4-mini", "deepseek-reasoner", "gpt-4o-mini"],
    "evaluation":        ["gpt-4o-mini", "qwen/qwen3-32b", "claude-haiku-4-5-20251001"],
    "batch":             ["deepseek-chat", "meta-llama/llama-4-maverick", "gpt-4o-mini"],
    "offline":           ["ollama/llama-4-scout", "ollama/qwen3-32b", "gpt-4o-mini"],
    # V0.6 additions — specialized routes
    "tool_heavy":        ["gpt-5.4", "gpt-4.1", "gemini-2.5-flash"],
    "reasoning_heavy":   ["gemini-3.1-pro", "claude-opus-4-6", "deepseek-reasoner"],
    "budget_extraction": ["qwen3-8b", "mistral-small-3.2", "deepseek-v3.2", "gpt-4.1-nano"],
    # Default fallback for unrecognized task types
    "default":           ["gpt-4o-mini", "gpt-4.1-nano", "deepseek-chat"],
}


# ---------------------------------------------------------------------------
# ModelRouter
# ---------------------------------------------------------------------------

class ModelRouter:
    """
    Provider-agnostic model selection based on task characteristics.

    Auto-detects available providers from environment variables and selects
    the optimal model for each task type with automatic fallback chains.

    Attributes:
        available_providers: Set of providers with configured API keys.
        routes: Task type → ordered model preference list.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: Optional dict overriding default routes.
                    Keys are task types, values are model names or lists of
                    model names (fallback chain).
        """
        self.routes: Dict[str, List[str]] = dict(DEFAULT_ROUTES)
        if config:
            for task_type, models in config.items():
                if isinstance(models, str):
                    models = [models]
                self.routes[task_type] = models
        self.available_providers: Set[str] = self._detect_available_providers()

    def select_model(
        self,
        task_type: str = "default",
        required_context: int = 0,
    ) -> str:
        """
        Select optimal model based on task type with provider fallback.

        Args:
            task_type: Type of task (extraction, reasoning, etc.).
            required_context: Minimum context window size needed.

        Returns:
            Model name string suitable for LLMClient.
        """
        chain = self.routes.get(task_type, self.routes.get("default", ["gpt-4o-mini"]))
        for model in chain:
            canonical = self.resolve_alias(model)
            if self._model_available(canonical, required_context):
                return canonical
        # Ultimate fallback: return gpt-4o-mini (always assumed available)
        return "gpt-4o-mini"

    def get_provider(self, model: str) -> str:
        """Get the provider name for a model."""
        canonical = self.resolve_alias(model)
        entry = MODEL_CATALOG.get(canonical)
        if entry:
            return entry[0]
        # Fall back to prefix detection
        return _detect_provider_from_prefix(canonical)

    def get_pricing(self, model: str) -> Tuple[float, float]:
        """
        Get pricing for a model as (input_$/1M_tokens, output_$/1M_tokens).

        Returns (0.0, 0.0) for unknown models.
        """
        canonical = self.resolve_alias(model)
        entry = MODEL_CATALOG.get(canonical)
        if entry:
            return (entry[1], entry[2])
        return (0.0, 0.0)

    def get_context_window(self, model: str) -> int:
        """Get context window size for a model. Returns 128000 for unknown."""
        canonical = self.resolve_alias(model)
        entry = MODEL_CATALOG.get(canonical)
        if entry:
            return entry[3]
        return 128_000

    def list_available_models(self) -> List[str]:
        """List all models whose provider is available."""
        result = []
        for model, (provider, _, _, _) in MODEL_CATALOG.items():
            if provider in self.available_providers:
                result.append(model)
        return sorted(result)

    def list_models_for_provider(self, provider: str) -> List[str]:
        """List all models for a specific provider."""
        return sorted(
            model for model, (p, _, _, _) in MODEL_CATALOG.items()
            if p == provider
        )

    @staticmethod
    def resolve_alias(model: str) -> str:
        """Resolve a friendly alias to the canonical model name."""
        return MODEL_ALIASES.get(model, model)

    def _model_available(self, model: str, required_context: int = 0) -> bool:
        """Check if a model's provider is available and context is sufficient."""
        entry = MODEL_CATALOG.get(model)
        if entry is None:
            # Unknown model — assume available if it looks like a known provider
            provider = _detect_provider_from_prefix(model)
            return provider in self.available_providers
        provider, _, _, context = entry
        if provider not in self.available_providers:
            return False
        if required_context > 0 and context < required_context:
            return False
        return True

    @staticmethod
    def _detect_available_providers() -> Set[str]:
        """Check which API keys are configured in environment variables."""
        available = set()
        for provider, env_key in PROVIDER_ENV_KEYS.items():
            if provider == "ollama":
                # Ollama is available if OLLAMA_HOST is set or localhost is assumed
                if os.getenv(env_key) or os.getenv("OLLAMA_AVAILABLE"):
                    available.add("ollama")
            else:
                if os.getenv(env_key):
                    available.add(provider)
        return available


def _detect_provider_from_prefix(model: str) -> str:
    """Detect provider from model name prefix.

    Delegates to the single source of truth in shared.provider_map (V0.6).
    """
    from shared.provider_map import detect_provider
    return detect_provider(model, default="openai")
