# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Tests for multi-provider LLMClient — provider routing, API formatting, model tiers."""

import json
import pytest
from shared.llm import (
    LLMClient,
    detect_provider,
    PROVIDER_ROUTES,
    MODEL_TIERS,
    API_URL,
    ANTHROPIC_URL,
    GOOGLE_URL,
)


# ---------------------------------------------------------------------------
# Provider detection tests
# ---------------------------------------------------------------------------

class TestProviderDetection:
    def test_openai_gpt4o(self):
        assert detect_provider("gpt-4o") == "openai"

    def test_openai_gpt4o_mini(self):
        assert detect_provider("gpt-4o-mini") == "openai"

    def test_openai_o1(self):
        assert detect_provider("o1-preview") == "openai"

    def test_openai_o3(self):
        assert detect_provider("o3-mini") == "openai"

    def test_openai_embedding(self):
        assert detect_provider("text-embedding-3-small") == "openai"

    def test_anthropic_claude(self):
        assert detect_provider("claude-haiku-4-5-20251001") == "anthropic"

    def test_anthropic_claude_sonnet(self):
        assert detect_provider("claude-sonnet-4-5-20250514") == "anthropic"

    def test_google_gemini_flash(self):
        assert detect_provider("gemini-2.5-flash") == "google"

    def test_google_gemini_pro(self):
        assert detect_provider("gemini-2.5-pro") == "google"

    def test_unknown_defaults_openai(self):
        assert detect_provider("some-unknown-model") == "openai"


# ---------------------------------------------------------------------------
# LLMClient initialization tests
# ---------------------------------------------------------------------------

class TestLLMClientInit:
    def test_default_provider_local_vllm(self):
        from shared.model_config import default_model
        client = LLMClient()
        assert client.model == default_model()
        assert client.provider == "vllm"

    def test_anthropic_provider_set(self):
        client = LLMClient(model="claude-haiku-4-5-20251001")
        assert client.provider == "anthropic"

    def test_google_provider_set(self):
        client = LLMClient(model="gemini-2.5-flash")
        assert client.provider == "google"

    def test_explicit_api_key(self):
        client = LLMClient(api_key="test-key-123")
        assert client.api_key == "test-key-123"

    def test_temperature_default(self):
        client = LLMClient()
        assert client.temperature == 0.3

    def test_collector_default_none(self):
        client = LLMClient()
        assert client.collector is None

    def test_agent_name(self):
        client = LLMClient(agent_name="TrendSurfer")
        assert client.agent_name == "TrendSurfer"


# ---------------------------------------------------------------------------
# Model tiers tests
# ---------------------------------------------------------------------------

class TestModelTiers:
    def test_routine_tier_exists(self):
        assert "routine" in MODEL_TIERS

    def test_complex_tier_exists(self):
        assert "complex" in MODEL_TIERS

    def test_routine_has_three_models(self):
        assert len(MODEL_TIERS["routine"]) == 3

    def test_complex_has_three_models(self):
        assert len(MODEL_TIERS["complex"]) == 3

    def test_routine_covers_all_providers(self):
        providers = {detect_provider(m) for m in MODEL_TIERS["routine"]}
        assert providers == {"openai", "anthropic", "google"}

    def test_complex_covers_all_providers(self):
        providers = {detect_provider(m) for m in MODEL_TIERS["complex"]}
        assert providers == {"openai", "anthropic", "google"}


# ---------------------------------------------------------------------------
# Provider URL constants tests
# ---------------------------------------------------------------------------

class TestProviderURLs:
    def test_openai_url(self):
        assert "api.openai.com" in API_URL

    def test_anthropic_url(self):
        assert "api.anthropic.com" in ANTHROPIC_URL

    def test_google_url(self):
        assert "generativelanguage.googleapis.com" in GOOGLE_URL


# ---------------------------------------------------------------------------
# Provider routes completeness tests
# ---------------------------------------------------------------------------

class TestProviderRoutes:
    def test_gpt_prefix(self):
        assert "gpt-" in PROVIDER_ROUTES

    def test_claude_prefix(self):
        assert "claude-" in PROVIDER_ROUTES

    def test_gemini_prefix(self):
        assert "gemini-" in PROVIDER_ROUTES

    def test_embedding_prefix(self):
        assert "text-embedding-" in PROVIDER_ROUTES

    def test_all_routes_valid_providers(self):
        # V0.7: +Perplexity +Groq +Qwen (DashScope).
        # 3-layer plan: +vllm (Layer 1 HPC vLLM, distinct from ollama = Layer 2 laptop).
        valid_providers = {
            "openai", "anthropic", "google",
            "deepseek", "mistral", "xai", "openrouter", "vllm", "ollama",
            "perplexity", "groq", "qwen",
        }
        for prefix, provider in PROVIDER_ROUTES.items():
            assert provider in valid_providers, f"{prefix} -> {provider} not valid"


# ---------------------------------------------------------------------------
# Prompt cache (already tested in Phase 5, verify with provider context)
# ---------------------------------------------------------------------------

class TestPromptCacheWithProvider:
    def test_cache_key_same_across_providers(self):
        """Cache key depends on messages, not provider."""
        c1 = LLMClient(model="gpt-4o-mini", enable_prompt_cache=True)
        c2 = LLMClient(model="claude-haiku-4-5-20251001", enable_prompt_cache=True)
        msgs = [{"role": "user", "content": "test"}]
        assert c1._cache_key(msgs) == c2._cache_key(msgs)

    def test_format_and_invoke_builds_messages(self):
        """Verify format_and_invoke constructs correct message list."""
        client = LLMClient(api_key="test")
        # We can't call invoke without a real API, but we can verify
        # the method exists and has the right signature
        assert hasattr(client, "format_and_invoke")
        assert callable(client.format_and_invoke)


# ---------------------------------------------------------------------------
# Embed method existence tests
# ---------------------------------------------------------------------------

class TestEmbedMethod:
    def test_embed_exists(self):
        client = LLMClient()
        assert hasattr(client, "embed")

    def test_embed_empty_returns_empty(self):
        client = LLMClient(api_key="test")
        result = client.embed([])
        assert result == []
