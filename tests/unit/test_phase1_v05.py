# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Phase 1 (V0.5) Integration Tests — Multi-LLM Routing & Cost Optimization.

Tests ModelRouter, provider backends, task-type routing, pricing,
CostDashboard per-provider breakdown, and backward compatibility.
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch, MagicMock

import pytest


# ============================================================================
# Test 1: ModelRouter — Core Functionality
# ============================================================================

class TestModelRouterCore:
    """ModelRouter initialization, provider detection, and model selection."""

    def test_import(self):
        from shared.llm_router import ModelRouter
        assert ModelRouter is not None

    def test_default_init(self):
        from shared.llm_router import ModelRouter
        router = ModelRouter()
        assert isinstance(router.routes, dict)
        assert isinstance(router.available_providers, set)

    def test_custom_config(self):
        from shared.llm_router import ModelRouter
        router = ModelRouter(config={"my_task": "gpt-4o-mini"})
        assert "my_task" in router.routes
        assert router.routes["my_task"] == ["gpt-4o-mini"]

    def test_custom_config_list(self):
        from shared.llm_router import ModelRouter
        router = ModelRouter(config={"my_task": ["gpt-4.1", "gpt-4o-mini"]})
        assert router.routes["my_task"] == ["gpt-4.1", "gpt-4o-mini"]

    def test_select_model_default(self):
        """select_model with no available providers falls back to gpt-4o-mini."""
        from shared.llm_router import ModelRouter
        with patch.dict(os.environ, {}, clear=True):
            router = ModelRouter()
            model = router.select_model("extraction")
            # With no API keys, falls back to gpt-4o-mini
            assert isinstance(model, str)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_select_model_with_openai(self):
        from shared.llm_router import ModelRouter
        router = ModelRouter()
        assert "openai" in router.available_providers
        model = router.select_model("extraction")
        # Should pick gpt-4.1-nano (first in extraction chain, OpenAI available)
        assert model == "gpt-4.1-nano"

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_select_model_with_anthropic(self):
        from shared.llm_router import ModelRouter
        router = ModelRouter()
        assert "anthropic" in router.available_providers
        model = router.select_model("reasoning")
        assert model == "claude-sonnet-4-6"

    @patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"})
    def test_select_model_with_deepseek(self):
        from shared.llm_router import ModelRouter
        router = ModelRouter()
        assert "deepseek" in router.available_providers

    @patch.dict(os.environ, {"MISTRAL_API_KEY": "test-key"})
    def test_select_model_with_mistral(self):
        from shared.llm_router import ModelRouter
        router = ModelRouter()
        assert "mistral" in router.available_providers

    @patch.dict(os.environ, {"XAI_API_KEY": "test-key"})
    def test_select_model_with_xai(self):
        from shared.llm_router import ModelRouter
        router = ModelRouter()
        assert "xai" in router.available_providers

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    def test_select_model_with_openrouter(self):
        from shared.llm_router import ModelRouter
        router = ModelRouter()
        assert "openrouter" in router.available_providers

    @patch.dict(os.environ, {"OLLAMA_HOST": "http://localhost:11434"})
    def test_select_model_with_ollama(self):
        from shared.llm_router import ModelRouter
        router = ModelRouter()
        assert "ollama" in router.available_providers

    def test_select_model_unknown_task_type(self):
        from shared.llm_router import ModelRouter
        router = ModelRouter()
        model = router.select_model("nonexistent_task")
        assert isinstance(model, str)


# ============================================================================
# Test 2: ModelRouter — Model Catalog
# ============================================================================

class TestModelCatalog:
    """MODEL_CATALOG completeness and correctness."""

    def test_catalog_has_20_plus_models(self):
        from shared.llm_router import MODEL_CATALOG
        assert len(MODEL_CATALOG) >= 20

    def test_catalog_entries_have_correct_shape(self):
        from shared.llm_router import MODEL_CATALOG
        for model, entry in MODEL_CATALOG.items():
            assert len(entry) == 4, f"{model} entry has {len(entry)} fields"
            provider, input_price, output_price, context = entry
            assert isinstance(provider, str)
            assert isinstance(input_price, (int, float))
            assert isinstance(output_price, (int, float))
            assert isinstance(context, int)
            assert input_price >= 0
            assert output_price >= 0
            assert context > 0

    def test_all_seven_providers_represented(self):
        from shared.llm_router import MODEL_CATALOG
        providers = {entry[0] for entry in MODEL_CATALOG.values()}
        expected = {"openai", "anthropic", "google", "deepseek", "mistral", "xai", "openrouter", "ollama"}
        assert expected.issubset(providers), f"Missing: {expected - providers}"

    def test_openai_models_present(self):
        from shared.llm_router import MODEL_CATALOG
        openai_models = [m for m, e in MODEL_CATALOG.items() if e[0] == "openai"]
        assert len(openai_models) >= 5  # nano, mini, 4.1-mini, 4.1, 5.4, o4-mini, gpt-4o-mini

    def test_anthropic_models_present(self):
        from shared.llm_router import MODEL_CATALOG
        anthropic_models = [m for m, e in MODEL_CATALOG.items() if e[0] == "anthropic"]
        assert len(anthropic_models) >= 3

    def test_google_models_present(self):
        from shared.llm_router import MODEL_CATALOG
        google_models = [m for m, e in MODEL_CATALOG.items() if e[0] == "google"]
        assert len(google_models) >= 3


# ============================================================================
# Test 3: ModelRouter — Aliases
# ============================================================================

class TestModelAliases:
    """Friendly alias resolution."""

    def test_resolve_alias_known(self):
        from shared.llm_router import ModelRouter
        assert ModelRouter.resolve_alias("deepseek-v3") == "deepseek-chat"
        assert ModelRouter.resolve_alias("deepseek-r1") == "deepseek-reasoner"
        assert ModelRouter.resolve_alias("mistral-nemo") == "open-mistral-nemo"
        assert ModelRouter.resolve_alias("codestral") == "codestral-latest"
        assert ModelRouter.resolve_alias("llama-4-scout") == "meta-llama/llama-4-scout"

    def test_resolve_alias_passthrough(self):
        from shared.llm_router import ModelRouter
        assert ModelRouter.resolve_alias("gpt-4o-mini") == "gpt-4o-mini"
        assert ModelRouter.resolve_alias("unknown-model") == "unknown-model"


# ============================================================================
# Test 4: ModelRouter — Pricing
# ============================================================================

class TestModelPricing:
    """Pricing lookups from MODEL_CATALOG."""

    def test_pricing_gpt4_1_nano(self):
        from shared.llm_router import ModelRouter
        router = ModelRouter()
        input_p, output_p = router.get_pricing("gpt-4.1-nano")
        assert input_p == 0.02
        assert output_p == 0.15

    def test_pricing_gpt4o_mini(self):
        from shared.llm_router import ModelRouter
        router = ModelRouter()
        input_p, output_p = router.get_pricing("gpt-4o-mini")
        assert input_p == 0.15
        assert output_p == 0.60

    def test_pricing_claude_sonnet(self):
        from shared.llm_router import ModelRouter
        router = ModelRouter()
        input_p, output_p = router.get_pricing("claude-sonnet-4-6")
        assert input_p == 3.00

    def test_pricing_deepseek_via_alias(self):
        from shared.llm_router import ModelRouter
        router = ModelRouter()
        input_p, output_p = router.get_pricing("deepseek-v3")
        assert input_p == 0.14

    def test_pricing_unknown_model(self):
        from shared.llm_router import ModelRouter
        router = ModelRouter()
        input_p, output_p = router.get_pricing("nonexistent-model")
        assert input_p == 0.0
        assert output_p == 0.0

    def test_pricing_ollama_is_free(self):
        from shared.llm_router import ModelRouter
        router = ModelRouter()
        input_p, output_p = router.get_pricing("ollama/llama-4-scout")
        assert input_p == 0.0
        assert output_p == 0.0


# ============================================================================
# Test 5: ModelRouter — Context Window
# ============================================================================

class TestContextWindow:
    """Context window lookups."""

    def test_gpt41_nano_1m_context(self):
        from shared.llm_router import ModelRouter
        router = ModelRouter()
        assert router.get_context_window("gpt-4.1-nano") == 1_000_000

    def test_llama4_scout_10m_context(self):
        from shared.llm_router import ModelRouter
        router = ModelRouter()
        assert router.get_context_window("meta-llama/llama-4-scout") == 10_000_000

    def test_unknown_model_default_128k(self):
        from shared.llm_router import ModelRouter
        router = ModelRouter()
        assert router.get_context_window("unknown-model") == 128_000

    def test_context_filter_in_select(self):
        """select_model with required_context skips models with small windows."""
        from shared.llm_router import ModelRouter
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            router = ModelRouter()
            # Request model with >= 500K context
            model = router.select_model("extraction", required_context=500_000)
            ctx = router.get_context_window(model)
            assert ctx >= 500_000


# ============================================================================
# Test 6: ModelRouter — Provider Detection
# ============================================================================

class TestProviderDetection:
    """get_provider and detect_provider_from_prefix."""

    def test_get_provider_openai(self):
        from shared.llm_router import ModelRouter
        router = ModelRouter()
        assert router.get_provider("gpt-4o-mini") == "openai"
        assert router.get_provider("gpt-4.1-nano") == "openai"
        assert router.get_provider("o4-mini") == "openai"

    def test_get_provider_anthropic(self):
        from shared.llm_router import ModelRouter
        router = ModelRouter()
        assert router.get_provider("claude-sonnet-4-6") == "anthropic"
        assert router.get_provider("claude-haiku-4-5-20251001") == "anthropic"

    def test_get_provider_google(self):
        from shared.llm_router import ModelRouter
        router = ModelRouter()
        assert router.get_provider("gemini-2.5-flash") == "google"

    def test_get_provider_deepseek(self):
        from shared.llm_router import ModelRouter
        router = ModelRouter()
        assert router.get_provider("deepseek-chat") == "deepseek"

    def test_get_provider_mistral(self):
        from shared.llm_router import ModelRouter
        router = ModelRouter()
        assert router.get_provider("open-mistral-nemo") == "mistral"
        assert router.get_provider("codestral-latest") == "mistral"

    def test_get_provider_xai(self):
        from shared.llm_router import ModelRouter
        router = ModelRouter()
        assert router.get_provider("grok-3-mini-fast") == "xai"

    def test_get_provider_openrouter(self):
        from shared.llm_router import ModelRouter
        router = ModelRouter()
        assert router.get_provider("meta-llama/llama-4-scout") == "openrouter"
        assert router.get_provider("qwen/qwen3-32b") == "openrouter"

    def test_get_provider_ollama(self):
        from shared.llm_router import ModelRouter
        router = ModelRouter()
        assert router.get_provider("ollama/llama-4-scout") == "ollama"


# ============================================================================
# Test 7: ModelRouter — List Operations
# ============================================================================

class TestModelRouterListing:
    """list_available_models and list_models_for_provider."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test", "ANTHROPIC_API_KEY": "test"})
    def test_list_available_models(self):
        from shared.llm_router import ModelRouter
        router = ModelRouter()
        models = router.list_available_models()
        assert isinstance(models, list)
        assert len(models) > 0
        # All returned models should be from available providers
        for m in models:
            provider = router.get_provider(m)
            assert provider in router.available_providers

    def test_list_models_for_openai(self):
        from shared.llm_router import ModelRouter
        router = ModelRouter()
        models = router.list_models_for_provider("openai")
        assert "gpt-4o-mini" in models
        assert "gpt-4.1-nano" in models

    def test_list_models_for_anthropic(self):
        from shared.llm_router import ModelRouter
        router = ModelRouter()
        models = router.list_models_for_provider("anthropic")
        assert any("claude" in m for m in models)


# ============================================================================
# Test 8: LLMClient — Provider Routing (detect_provider extended)
# ============================================================================

class TestLLMClientProviderRouting:
    """LLMClient detect_provider routes to correct provider."""

    def test_detect_deepseek(self):
        from shared.llm import detect_provider
        assert detect_provider("deepseek-chat") == "deepseek"
        assert detect_provider("deepseek-reasoner") == "deepseek"

    def test_detect_mistral(self):
        from shared.llm import detect_provider
        assert detect_provider("open-mistral-nemo") == "mistral"
        assert detect_provider("mistral-large-latest") == "mistral"
        assert detect_provider("codestral-latest") == "mistral"

    def test_detect_xai(self):
        from shared.llm import detect_provider
        assert detect_provider("grok-3-mini-fast") == "xai"

    def test_detect_openrouter(self):
        from shared.llm import detect_provider
        assert detect_provider("meta-llama/llama-4-scout") == "openrouter"
        assert detect_provider("qwen/qwen3-32b") == "openrouter"

    def test_detect_ollama(self):
        from shared.llm import detect_provider
        assert detect_provider("ollama/llama-4-scout") == "ollama"

    def test_detect_o4_mini(self):
        from shared.llm import detect_provider
        assert detect_provider("o4-mini") == "openai"


# ============================================================================
# Test 9: LLMClient — Task-Type Constructor
# ============================================================================

class TestLLMClientTaskType:
    """LLMClient with task_type parameter uses ModelRouter."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test"})
    def test_task_type_extraction(self):
        from shared.llm import LLMClient
        client = LLMClient(task_type="extraction")
        assert client.model == "gpt-4.1-nano"
        assert client.provider == "openai"

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test"})
    def test_task_type_reasoning_with_anthropic(self):
        from shared.llm import LLMClient
        client = LLMClient(task_type="reasoning")
        assert client.model == "claude-sonnet-4-6"
        assert client.provider == "anthropic"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test"})
    def test_task_type_with_custom_router(self):
        from shared.llm import LLMClient
        from shared.llm_router import ModelRouter
        router = ModelRouter(config={"my_custom": "gpt-4o-mini"})
        client = LLMClient(task_type="my_custom", router=router)
        assert client.model == "gpt-4o-mini"

    def test_no_task_type_uses_model_directly(self):
        """Without task_type, model param is used as-is (backward compat)."""
        from shared.llm import LLMClient
        client = LLMClient(model="gpt-4o-mini", api_key="test")
        assert client.model == "gpt-4o-mini"


# ============================================================================
# Test 10: LLMClient — API Key Resolution
# ============================================================================

class TestAPIKeyResolution:
    """LLMClient resolves API keys from correct env vars per provider."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "openai-key-123"}, clear=True)
    def test_openai_key(self):
        from shared.llm import LLMClient
        client = LLMClient(model="gpt-4o-mini")
        assert client.api_key == "openai-key-123"

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "anthropic-key-123"}, clear=True)
    def test_anthropic_key(self):
        from shared.llm import LLMClient
        client = LLMClient(model="claude-haiku-4-5-20251001")
        assert client.api_key == "anthropic-key-123"

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "google-key-123"}, clear=True)
    def test_google_key(self):
        from shared.llm import LLMClient
        client = LLMClient(model="gemini-2.5-flash")
        assert client.api_key == "google-key-123"

    @patch.dict(os.environ, {"DEEPSEEK_API_KEY": "deepseek-key-123"}, clear=True)
    def test_deepseek_key(self):
        from shared.llm import LLMClient
        client = LLMClient(model="deepseek-chat")
        assert client.api_key == "deepseek-key-123"

    @patch.dict(os.environ, {"MISTRAL_API_KEY": "mistral-key-123"}, clear=True)
    def test_mistral_key(self):
        from shared.llm import LLMClient
        client = LLMClient(model="open-mistral-nemo")
        assert client.api_key == "mistral-key-123"

    @patch.dict(os.environ, {"XAI_API_KEY": "xai-key-123"}, clear=True)
    def test_xai_key(self):
        from shared.llm import LLMClient
        client = LLMClient(model="grok-3-mini-fast")
        assert client.api_key == "xai-key-123"

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "openrouter-key-123"}, clear=True)
    def test_openrouter_key(self):
        from shared.llm import LLMClient
        client = LLMClient(model="meta-llama/llama-4-scout")
        assert client.api_key == "openrouter-key-123"

    def test_explicit_key_overrides_env(self):
        from shared.llm import LLMClient
        client = LLMClient(model="gpt-4o-mini", api_key="explicit-key")
        assert client.api_key == "explicit-key"


# ============================================================================
# Test 11: LLMClient — OpenAI Compatible Backend Config
# ============================================================================

class TestOpenAICompatibleConfig:
    """OPENAI_COMPATIBLE_PROVIDERS has correct config."""

    def test_all_five_providers_configured(self):
        # V0.7 Pre-Phase: Perplexity + Groq added (still a superset of the V0.5 five).
        from shared.llm import OPENAI_COMPATIBLE_PROVIDERS
        v05_five = {"deepseek", "mistral", "xai", "openrouter", "ollama"}
        assert v05_five.issubset(set(OPENAI_COMPATIBLE_PROVIDERS.keys()))

    def test_configs_have_required_keys(self):
        from shared.llm import OPENAI_COMPATIBLE_PROVIDERS
        for provider, config in OPENAI_COMPATIBLE_PROVIDERS.items():
            assert "base_url" in config, f"{provider} missing base_url"
            assert "api_key_env" in config, f"{provider} missing api_key_env"

    def test_base_urls_are_valid(self):
        from shared.llm import OPENAI_COMPATIBLE_PROVIDERS
        for provider, config in OPENAI_COMPATIBLE_PROVIDERS.items():
            url = config["base_url"]
            assert url.startswith("http"), f"{provider} URL invalid: {url}"

    def test_ollama_is_localhost(self):
        from shared.llm import OPENAI_COMPATIBLE_PROVIDERS
        assert "localhost" in OPENAI_COMPATIBLE_PROVIDERS["ollama"]["base_url"]


# ============================================================================
# Test 12: MODEL_PRICING — Updated Pricing Table
# ============================================================================

class TestModelPricingTable:
    """MODEL_PRICING in observability.py includes all new models."""

    def test_new_openai_models_priced(self):
        from shared.observability import MODEL_PRICING
        new_models = ["gpt-4.1-nano", "gpt-5-mini", "gpt-4.1-mini", "o4-mini", "gpt-4.1", "gpt-5.4"]
        for model in new_models:
            assert model in MODEL_PRICING, f"{model} missing from MODEL_PRICING"

    def test_new_anthropic_models_priced(self):
        from shared.observability import MODEL_PRICING
        assert "claude-sonnet-4-6" in MODEL_PRICING
        assert "claude-opus-4-6" in MODEL_PRICING

    def test_google_models_priced(self):
        from shared.observability import MODEL_PRICING
        assert "gemini-flash-lite" in MODEL_PRICING
        assert "gemini-2.5-flash" in MODEL_PRICING
        assert "gemini-2.5-pro" in MODEL_PRICING

    def test_deepseek_models_priced(self):
        from shared.observability import MODEL_PRICING
        assert "deepseek-chat" in MODEL_PRICING
        assert "deepseek-reasoner" in MODEL_PRICING

    def test_mistral_models_priced(self):
        from shared.observability import MODEL_PRICING
        assert "open-mistral-nemo" in MODEL_PRICING
        assert "codestral-latest" in MODEL_PRICING
        assert "mistral-large-latest" in MODEL_PRICING

    def test_xai_model_priced(self):
        from shared.observability import MODEL_PRICING
        assert "grok-3-mini-fast" in MODEL_PRICING

    def test_openrouter_models_priced(self):
        from shared.observability import MODEL_PRICING
        assert "meta-llama/llama-4-scout" in MODEL_PRICING
        assert "meta-llama/llama-4-maverick" in MODEL_PRICING
        assert "qwen/qwen3-32b" in MODEL_PRICING

    def test_ollama_models_free(self):
        from shared.observability import MODEL_PRICING
        for model in ["ollama/llama-4-scout", "ollama/qwen3-32b"]:
            assert model in MODEL_PRICING
            assert MODEL_PRICING[model] == (0.0, 0.0)

    def test_estimate_cost_new_model(self):
        from shared.observability import estimate_cost
        # gpt-4.1-nano: $0.02/1M input, $0.15/1M output
        # = $0.00002/1K input, $0.00015/1K output
        cost = estimate_cost("gpt-4.1-nano", 1000, 1000)
        assert abs(cost - 0.00017) < 0.0001

    def test_estimate_cost_deepseek(self):
        from shared.observability import estimate_cost
        cost = estimate_cost("deepseek-chat", 1000, 1000)
        assert cost > 0

    def test_estimate_cost_ollama_free(self):
        from shared.observability import estimate_cost
        cost = estimate_cost("ollama/llama-4-scout", 10000, 10000)
        assert cost == 0.0


# ============================================================================
# Test 13: CostDashboard — Per-Provider Breakdown
# ============================================================================

class TestCostDashboardProvider:
    """CostDashboard by_provider grouping."""

    def test_cost_report_has_by_provider(self):
        from shared.telemetry.cost_dashboard import CostReport
        report = CostReport(team="Test", mode="Test")
        assert hasattr(report, "by_provider")

    def test_by_provider_in_to_dict(self):
        from shared.telemetry.cost_dashboard import CostReport
        report = CostReport(team="Test", mode="Test")
        d = report.to_dict()
        assert "by_provider" in d

    def test_generate_groups_by_provider(self):
        from shared.telemetry.cost_dashboard import CostDashboard
        from shared.observability import MetricsCollector, LLMCallRecord

        collector = MetricsCollector()
        # Record calls from different providers
        collector.record_llm_call(LLMCallRecord(
            agent="Agent1", model="gpt-4o-mini",
            prompt_tokens=100, completion_tokens=50, cost_usd=0.001,
        ))
        collector.record_llm_call(LLMCallRecord(
            agent="Agent2", model="deepseek-chat",
            prompt_tokens=200, completion_tokens=100, cost_usd=0.0005,
        ))

        dashboard = CostDashboard()
        report = dashboard.generate(collector, team="Test", mode="Test")

        assert len(report.by_provider) >= 2
        provider_names = {b.name for b in report.by_provider}
        assert "openai" in provider_names
        assert "deepseek" in provider_names

    def test_by_provider_cost_attribution(self):
        from shared.telemetry.cost_dashboard import CostDashboard
        from shared.observability import MetricsCollector, LLMCallRecord

        collector = MetricsCollector()
        collector.record_llm_call(LLMCallRecord(
            model="gpt-4o-mini", cost_usd=0.01,
        ))
        collector.record_llm_call(LLMCallRecord(
            model="gpt-4.1-nano", cost_usd=0.001,
        ))
        collector.record_llm_call(LLMCallRecord(
            model="deepseek-chat", cost_usd=0.005,
        ))

        dashboard = CostDashboard()
        report = dashboard.generate(collector, team="T", mode="M")

        openai_breakdown = next(
            (b for b in report.by_provider if b.name == "openai"), None
        )
        assert openai_breakdown is not None
        assert openai_breakdown.llm_calls == 2
        assert abs(openai_breakdown.llm_cost_usd - 0.011) < 0.0001


# ============================================================================
# Test 14: Default Routes — All Task Types
# ============================================================================

class TestDefaultRoutes:
    """DEFAULT_ROUTES covers expected task types."""

    def test_expected_task_types_present(self):
        from shared.llm_router import DEFAULT_ROUTES
        expected_types = [
            "extraction", "formatting", "structured_output", "reasoning",
            "summarization", "calibration", "code_generation", "reasoning_cot",
            "evaluation", "batch", "offline", "default",
        ]
        for task_type in expected_types:
            assert task_type in DEFAULT_ROUTES, f"Missing: {task_type}"

    def test_all_routes_are_lists(self):
        from shared.llm_router import DEFAULT_ROUTES
        for task_type, chain in DEFAULT_ROUTES.items():
            assert isinstance(chain, list), f"{task_type} should be a list"
            assert len(chain) >= 1, f"{task_type} chain is empty"

    def test_all_route_models_in_catalog_or_alias(self):
        from shared.llm_router import DEFAULT_ROUTES, MODEL_CATALOG, MODEL_ALIASES
        for task_type, chain in DEFAULT_ROUTES.items():
            for model in chain:
                canonical = MODEL_ALIASES.get(model, model)
                assert canonical in MODEL_CATALOG, \
                    f"{task_type} references unknown model: {model}"

    def test_each_route_has_gpt4o_mini_in_chain(self):
        """Core route chains should include gpt-4o-mini somewhere as a fallback."""
        from shared.llm_router import DEFAULT_ROUTES
        # V0.6 added specialized routes with different fallback strategies
        v06_routes = {"tool_heavy", "reasoning_heavy", "budget_extraction"}
        for task_type, chain in DEFAULT_ROUTES.items():
            if task_type in v06_routes:
                continue
            assert "gpt-4o-mini" in chain, \
                f"{task_type} chain is missing gpt-4o-mini fallback: {chain}"


# ============================================================================
# Test 15: Provider Env Keys
# ============================================================================

class TestProviderEnvKeys:
    """PROVIDER_ENV_KEYS and PROVIDER_BASE_URLS completeness."""

    def test_seven_plus_providers_in_env_keys(self):
        from shared.llm_router import PROVIDER_ENV_KEYS
        assert len(PROVIDER_ENV_KEYS) >= 7

    def test_seven_plus_providers_in_base_urls(self):
        from shared.llm_router import PROVIDER_BASE_URLS
        assert len(PROVIDER_BASE_URLS) >= 7

    def test_env_keys_match_base_urls(self):
        from shared.llm_router import PROVIDER_ENV_KEYS, PROVIDER_BASE_URLS
        assert set(PROVIDER_ENV_KEYS.keys()) == set(PROVIDER_BASE_URLS.keys())


# ============================================================================
# Test 16: Backward Compatibility
# ============================================================================

class TestBackwardCompatibility:
    """Existing code continues to work unchanged."""

    def test_llmclient_default_model(self):
        from shared.llm import LLMClient
        from shared.model_config import default_model
        client = LLMClient()
        assert client.model == default_model()

    def test_llmclient_model_tiers_still_exist(self):
        from shared.llm import MODEL_TIERS
        assert "routine" in MODEL_TIERS
        assert "complex" in MODEL_TIERS

    def test_detect_provider_backward_compat(self):
        from shared.llm import detect_provider
        assert detect_provider("gpt-4o-mini") == "openai"
        assert detect_provider("claude-haiku-4-5-20251001") == "anthropic"
        assert detect_provider("gemini-2.5-flash") == "google"

    def test_format_and_invoke_exists(self):
        from shared.llm import LLMClient
        client = LLMClient(api_key="test")
        assert hasattr(client, "format_and_invoke")

    def test_embed_exists(self):
        from shared.llm import LLMClient
        client = LLMClient(api_key="test")
        assert hasattr(client, "embed")

    def test_cache_stats_exists(self):
        from shared.llm import LLMClient
        client = LLMClient(api_key="test")
        stats = client.get_cache_stats()
        assert "enabled" in stats
        assert "hits" in stats

    def test_cost_dashboard_backward_compat(self):
        from shared.telemetry.cost_dashboard import CostDashboard
        from shared.observability import MetricsCollector

        dashboard = CostDashboard()
        collector = MetricsCollector()
        report = dashboard.generate(collector, team="Test", mode="Test")
        d = report.to_dict()
        # Old fields still present
        assert "by_agent" in d
        assert "by_stage" in d
        assert "by_model" in d
        assert "by_tool" in d
        # New field also present
        assert "by_provider" in d


# ============================================================================
# Test 17: Multi-Provider Fallback Chain
# ============================================================================

class TestFallbackChain:
    """Model selection falls back correctly when providers unavailable."""

    @patch.dict(os.environ, {}, clear=True)
    def test_no_providers_falls_to_gpt4o_mini(self):
        from shared.llm_router import ModelRouter
        router = ModelRouter()
        model = router.select_model("extraction")
        # With no API keys, falls back to gpt-4o-mini (ultimate fallback)
        assert model == "gpt-4o-mini"

    @patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test"}, clear=True)
    def test_only_deepseek_available(self):
        from shared.llm_router import ModelRouter
        router = ModelRouter()
        model = router.select_model("extraction")
        # Extraction chain: gpt-4.1-nano(no), mistral-nemo(no), deepseek-chat(yes!)
        assert model == "deepseek-chat"

    @patch.dict(os.environ, {"MISTRAL_API_KEY": "test"}, clear=True)
    def test_only_mistral_available(self):
        from shared.llm_router import ModelRouter
        router = ModelRouter()
        model = router.select_model("extraction")
        # Extraction chain: gpt-4.1-nano(no), open-mistral-nemo(yes!)
        assert model == "open-mistral-nemo"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "k1", "ANTHROPIC_API_KEY": "k2"}, clear=True)
    def test_multi_provider_reasoning_prefers_anthropic(self):
        from shared.llm_router import ModelRouter
        router = ModelRouter()
        model = router.select_model("reasoning")
        # Reasoning chain: claude-sonnet-4-6 (anthropic available) → selected
        assert model == "claude-sonnet-4-6"


# ============================================================================
# Test 18: LLMClient — _invoke_openai_compatible method
# ============================================================================

class TestInvokeOpenAICompatible:
    """The _invoke_openai_compatible method exists and is correctly wired."""

    def test_method_exists(self):
        from shared.llm import LLMClient
        client = LLMClient(model="deepseek-chat", api_key="test")
        assert hasattr(client, "_invoke_openai_compatible")

    def test_deepseek_client_uses_correct_provider(self):
        from shared.llm import LLMClient
        client = LLMClient(model="deepseek-chat", api_key="test")
        assert client.provider == "deepseek"

    def test_mistral_client_uses_correct_provider(self):
        from shared.llm import LLMClient
        client = LLMClient(model="open-mistral-nemo", api_key="test")
        assert client.provider == "mistral"

    def test_xai_client_uses_correct_provider(self):
        from shared.llm import LLMClient
        client = LLMClient(model="grok-3-mini-fast", api_key="test")
        assert client.provider == "xai"

    def test_openrouter_client_uses_correct_provider(self):
        from shared.llm import LLMClient
        client = LLMClient(model="meta-llama/llama-4-scout", api_key="test")
        assert client.provider == "openrouter"

    def test_ollama_client_uses_correct_provider(self):
        from shared.llm import LLMClient
        client = LLMClient(model="ollama/llama-4-scout", api_key="")
        assert client.provider == "ollama"

    def test_openai_compat_providers_in_routing(self):
        """All OpenAI-compatible providers are in OPENAI_COMPATIBLE_PROVIDERS."""
        from shared.llm import OPENAI_COMPATIBLE_PROVIDERS
        for provider in ["deepseek", "mistral", "xai", "openrouter", "ollama"]:
            assert provider in OPENAI_COMPATIBLE_PROVIDERS
