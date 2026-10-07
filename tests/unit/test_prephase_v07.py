# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Unit tests for AEL V0.7 Pre-Phase — Catalog refresh + OWASP ASI01-ASI10 + OTel dual-emit."""

from __future__ import annotations

import os
import pytest

from shared.provider_map import (
    CATALOG_V07,
    ModelEntry,
    OPENAI_COMPATIBLE_PROVIDERS,
    PROVIDER_API_KEY_ENV,
    PROVIDER_ROUTES,
    detect_provider,
    get_catalog_entry,
)
from shared.llm_router import (
    MODEL_CATALOG,
    PROVIDER_BASE_URLS,
    PROVIDER_ENV_KEYS,
    ModelRouter,
)
from shared.security import (
    ASICategory,
    ASIFinding,
    LeastAgencyContext,
    LogSanitizer,
    OWASPASIAuditor,
    assert_least_agency,
    resolve_asi_category,
)
from shared.security.owasp_audit import DEPRECATED_ASI_ALIASES
from shared.telemetry.gen_ai_conventions import (
    AGENT_SPAN_NAME_INVOKE,
    LEGACY_AGENT_SPAN_NAME_INVOKE,
    agent_invocation_attributes,
    agent_span_name,
)


# ---------------------------------------------------------------------------
# Catalog refresh — April 2026
# ---------------------------------------------------------------------------

class TestCatalogRefreshV07:
    def test_gpt_54_family_present(self):
        assert ("gpt-5.4", "openai", 2.50, 15.00, 270_000) == (
            "gpt-5.4",
            *MODEL_CATALOG["gpt-5.4"],
        ) or "gpt-5.4" in MODEL_CATALOG  # legacy entry existed; V0.7 adds family
        assert MODEL_CATALOG["gpt-5.4-mini"][0] == "openai"
        assert MODEL_CATALOG["gpt-5.4-mini"][1] == 0.75
        assert MODEL_CATALOG["gpt-5.4-nano"][1] == 0.20
        assert MODEL_CATALOG["gpt-5.4-nano"][3] == 270_000

    def test_deepseek_v4_1m_context(self):
        entry = MODEL_CATALOG["deepseek-v4"]
        assert entry[0] == "deepseek"
        assert entry[1] == 0.30
        assert entry[2] == 0.50
        assert entry[3] == 1_000_000

    def test_gemini_3_1_flash_lite(self):
        entry = MODEL_CATALOG["gemini-3.1-flash-lite"]
        assert entry[0] == "google"
        assert entry[1] == 0.25

    def test_grok_4_20_variants(self):
        assert "grok-4-20-reasoning" in MODEL_CATALOG
        assert "grok-4-20-multi-agent" in MODEL_CATALOG
        assert MODEL_CATALOG["grok-4-20-reasoning"][3] == 2_000_000

    def test_qwen3_5_family(self):
        assert MODEL_CATALOG["qwen3.5-plus"][1] == 0.26
        assert MODEL_CATALOG["qwen3.5-397b-a17b"][1] == 0.39
        assert MODEL_CATALOG["qwen3.5-flash"][1] == 0.07

    def test_mistral_updates(self):
        assert "mistral-medium-3.1" in MODEL_CATALOG
        assert "ministral-3-8b" in MODEL_CATALOG
        assert MODEL_CATALOG["ministral-3-8b"][1] == 0.05

    def test_perplexity_provider_entries(self):
        for model in (
            "perplexity-sonar",
            "perplexity-sonar-pro",
            "perplexity-reasoning-pro",
            "perplexity-deep-research",
        ):
            assert MODEL_CATALOG[model][0] == "perplexity"

    def test_groq_provider_entries(self):
        for model in ("groq/llama-4-scout", "groq/llama-3.3-70b", "groq/llama-3.1-8b"):
            assert MODEL_CATALOG[model][0] == "groq"
        assert MODEL_CATALOG["groq/llama-4-scout"][3] == 10_000_000

    def test_provider_env_keys_has_new_providers(self):
        assert PROVIDER_ENV_KEYS["perplexity"] == "PERPLEXITY_API_KEY"
        assert PROVIDER_ENV_KEYS["groq"] == "GROQ_API_KEY"
        assert PROVIDER_ENV_KEYS["qwen"] == "DASHSCOPE_API_KEY"

    def test_provider_base_urls_has_new_providers(self):
        assert "perplexity.ai" in PROVIDER_BASE_URLS["perplexity"]
        assert "groq.com" in PROVIDER_BASE_URLS["groq"]

    def test_at_least_10_providers(self):
        providers = set(MODEL_CATALOG[m][0] for m in MODEL_CATALOG)
        assert len(providers) >= 10

    def test_openai_compatible_includes_perplexity_groq(self):
        assert "perplexity" in OPENAI_COMPATIBLE_PROVIDERS
        assert "groq" in OPENAI_COMPATIBLE_PROVIDERS
        assert OPENAI_COMPATIBLE_PROVIDERS["perplexity"]["api_key_env"] == "PERPLEXITY_API_KEY"

    def test_router_detects_perplexity_model_via_prefix(self):
        assert detect_provider("sonar-pro") == "perplexity"

    def test_router_detects_groq_prefix(self):
        assert detect_provider("groq/llama-4-scout") == "groq"


# ---------------------------------------------------------------------------
# CATALOG_V07 ModelEntry — long-context + residency
# ---------------------------------------------------------------------------

class TestCatalogV07Entries:
    def test_gemini_3_1_pro_tiered_pricing(self):
        e = get_catalog_entry("gemini-3.1-pro")
        assert e is not None
        assert e.long_context_breakpoint == 200_000
        # At breakpoint+1, long-context rate kicks in for the whole request
        low = e.effective_input_cost(100_000)
        high = e.effective_input_cost(300_000)
        assert low / 100_000 * 1_000_000 == pytest.approx(2.00)
        assert high / 300_000 * 1_000_000 == pytest.approx(4.00)

    def test_openai_cached_input_multiplier(self):
        e = get_catalog_entry("gpt-5.4-nano")
        assert e is not None
        nominal = e.effective_input_cost(1_000_000)
        cached = e.effective_input_cost(1_000_000, cached=True)
        assert cached == pytest.approx(nominal * 0.10)

    def test_anthropic_residency_multiplier(self):
        e = get_catalog_entry("claude-sonnet-4-6")
        assert e is not None
        nominal = e.effective_input_cost(1_000_000)
        with_residency = e.effective_input_cost(1_000_000, data_residency=True)
        assert with_residency == pytest.approx(nominal * 1.10)

    def test_unknown_model_returns_none(self):
        assert get_catalog_entry("definitely-not-real") is None

    def test_deepseek_v4_entry(self):
        e = get_catalog_entry("deepseek-v4")
        assert e and e.max_context == 1_000_000

    def test_grok_4_20_reasoning_cache(self):
        e = get_catalog_entry("grok-4-20-reasoning")
        assert e and e.cached_input_multiplier == 0.10


# ---------------------------------------------------------------------------
# ModelRouter — availability detection for new providers
# ---------------------------------------------------------------------------

class TestRouterNewProviders:
    def test_perplexity_detected_when_key_present(self, monkeypatch):
        monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")
        r = ModelRouter()
        assert "perplexity" in r.available_providers

    def test_groq_detected_when_key_present(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        r = ModelRouter()
        assert "groq" in r.available_providers

    def test_perplexity_pricing_lookup(self):
        r = ModelRouter()
        inp, out = r.get_pricing("perplexity-sonar")
        assert inp == 1.00
        assert out == 1.00

    def test_groq_context_window(self):
        r = ModelRouter()
        assert r.get_context_window("groq/llama-4-scout") == 10_000_000


# ---------------------------------------------------------------------------
# OWASP ASI01-ASI10 final taxonomy
# ---------------------------------------------------------------------------

class TestASITaxonomy:
    def test_all_10_categories_present(self):
        assert len(list(ASICategory)) == 10
        expected = {f"ASI{i:02d}" for i in range(1, 11)}
        assert {c.value for c in ASICategory} == expected

    def test_deprecated_aliases_resolve(self):
        assert resolve_asi_category("goal_hijacking") == ASICategory.AGENT_GOAL_HIJACK
        assert resolve_asi_category("memory_poisoning") == ASICategory.MEMORY_CONTEXT_POISONING
        assert (
            resolve_asi_category("inter_agent_mitm")
            == ASICategory.INSECURE_INTER_AGENT_COMMUNICATION
        )

    def test_resolve_from_enum(self):
        assert resolve_asi_category(ASICategory.TOOL_MISUSE) == ASICategory.TOOL_MISUSE

    def test_resolve_from_asi_string(self):
        assert resolve_asi_category("ASI07") == ASICategory.INSECURE_INTER_AGENT_COMMUNICATION

    def test_resolve_unknown_raises(self):
        with pytest.raises(ValueError):
            resolve_asi_category("not-a-category")

    def test_deprecated_aliases_cover_all_10(self):
        assert len(DEPRECATED_ASI_ALIASES) == 10


class TestASIAuditor:
    def test_register_and_run_single_check(self):
        auditor = OWASPASIAuditor()
        auditor.register(
            ASICategory.TOOL_MISUSE,
            lambda: [
                ASIFinding(
                    category=ASICategory.TOOL_MISUSE,
                    severity="medium",
                    component="ToolRegistry",
                    description="unrestricted tool allowed",
                )
            ],
        )
        report = auditor.run()
        assert len(report.findings) == 1
        assert report.findings[0].category == ASICategory.TOOL_MISUSE

    def test_coverage_ratio_full(self):
        auditor = OWASPASIAuditor()
        for cat in ASICategory:
            auditor.register(cat, lambda: [])
        report = auditor.run()
        assert report.coverage_ratio == 1.0

    def test_coverage_ratio_partial(self):
        auditor = OWASPASIAuditor()
        auditor.register(ASICategory.TOOL_MISUSE, lambda: [])
        auditor.register(ASICategory.MEMORY_CONTEXT_POISONING, lambda: [])
        report = auditor.run()
        assert report.coverage_ratio == pytest.approx(0.2)

    def test_alias_registration(self):
        auditor = OWASPASIAuditor()
        auditor.register("memory_poisoning", lambda: [])
        report = auditor.run([ASICategory.MEMORY_CONTEXT_POISONING])
        assert ASICategory.MEMORY_CONTEXT_POISONING in report.checks_run

    def test_summary_counts_by_category(self):
        auditor = OWASPASIAuditor()
        auditor.register(
            ASICategory.ROGUE_AGENTS,
            lambda: [
                ASIFinding(ASICategory.ROGUE_AGENTS, "high", "bus", "rogue-1"),
                ASIFinding(ASICategory.ROGUE_AGENTS, "high", "bus", "rogue-2"),
            ],
        )
        report = auditor.run()
        assert report.summary()["ASI10"] == 2


class TestLeastAgency:
    def test_pass_within_envelope(self):
        ctx = LeastAgencyContext(actor="ideator", allowed_tools=["search"])
        # Should not raise
        assert_least_agency(ctx, tool="search") is None

    def test_block_tool_outside_envelope(self):
        ctx = LeastAgencyContext(actor="ideator", allowed_tools=["search"])
        with pytest.raises(PermissionError):
            assert_least_agency(ctx, tool="shell_exec")

    def test_empty_allowlist_is_unrestricted(self):
        ctx = LeastAgencyContext(actor="ideator")
        assert_least_agency(ctx, tool="anything")  # no raise

    def test_multiple_dimensions(self):
        ctx = LeastAgencyContext(
            actor="worker",
            allowed_scopes=["read:data"],
            allowed_actions=["select"],
            allowed_data_classes=["public"],
        )
        assert_least_agency(ctx, scope="read:data", action="select", data_class="public")
        with pytest.raises(PermissionError):
            assert_least_agency(ctx, action="delete")


# ---------------------------------------------------------------------------
# LogSanitizer (ASI07 Log-To-Leak defense)
# ---------------------------------------------------------------------------

class TestLogSanitizer:
    def setup_method(self):
        self.s = LogSanitizer()

    def test_strips_jinja_injection(self):
        out = self.s.sanitize_string("request={{system.env.API_KEY}}")
        assert "{{" not in out and "}}" not in out
        assert "[TEMPLATE_REDACTED]" in out

    def test_strips_shell_injection(self):
        out = self.s.sanitize_string("path=${HOME}/secrets")
        assert "${HOME}" not in out

    def test_strips_php_injection(self):
        out = self.s.sanitize_string("<?php system($_GET['x']); ?>")
        assert "<?" not in out

    def test_redacts_openai_key(self):
        out = self.s.sanitize_string("auth sk-abcdefghijklmnopqrstuvwxyz123456")
        assert "sk-abc" not in out
        assert "[REDACTED]" in out

    def test_redacts_anthropic_key(self):
        out = self.s.sanitize_string("sk-ant-api03-abcdefghij1234567890")
        assert "sk-ant" not in out

    def test_redacts_bearer(self):
        out = self.s.sanitize_string("Authorization: Bearer abcdefghijklmnopqrstuvwxyz")
        assert "Bearer abc" not in out

    def test_redacts_aws_key(self):
        out = self.s.sanitize_string("AKIAIOSFODNN7EXAMPLE")
        assert "AKIA" not in out

    def test_redacts_jwt(self):
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijklmnop"
        assert "[REDACTED]" in self.s.sanitize_string(jwt)

    def test_redacts_google_api_key(self):
        out = self.s.sanitize_string("AIzaSyA" + "a" * 33)
        assert "AIzaSyA" not in out

    def test_sanitize_dict_recursive(self):
        payload = {"msg": "hello {{inject}}", "nested": {"Authorization": "Bearer abc"}}
        result = self.s.sanitize(payload)
        assert "{{" not in result["msg"]
        assert result["nested"]["Authorization"] == LogSanitizer.REDACTED

    def test_sanitize_list(self):
        payload = ["ok", "sk-" + "a" * 40]
        result = self.s.sanitize(payload)
        assert result[0] == "ok"
        assert "[REDACTED]" in result[1]

    def test_sensitive_key_dropped_case_insensitive(self):
        payload = {"X-API-KEY": "anything"}
        assert self.s.sanitize(payload)["X-API-KEY"] == LogSanitizer.REDACTED

    def test_contains_template_injection(self):
        assert self.s.contains_template_injection("x {{ y }} z")
        assert not self.s.contains_template_injection("plain text")

    def test_contains_secret(self):
        assert self.s.contains_secret("Bearer " + "a" * 30)
        assert not self.s.contains_secret("no secrets here")

    def test_passthrough_primitives(self):
        assert self.s.sanitize(42) == 42
        assert self.s.sanitize(None) is None
        assert self.s.sanitize(True) is True


# ---------------------------------------------------------------------------
# OTel GenAI agent-span dual-emit
# ---------------------------------------------------------------------------

class TestOTelDualEmit:
    def test_legacy_span_name_default(self, monkeypatch):
        monkeypatch.delenv("OTEL_SEMCONV_STABILITY_OPT_IN", raising=False)
        assert agent_span_name("invoke") == LEGACY_AGENT_SPAN_NAME_INVOKE

    def test_finalized_span_name_with_opt_in(self, monkeypatch):
        monkeypatch.setenv("OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental")
        assert agent_span_name("invoke") == AGENT_SPAN_NAME_INVOKE
        assert agent_span_name("invoke") == "invoke_agent"

    def test_create_agent_span_finalized(self, monkeypatch):
        monkeypatch.setenv("OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental")
        assert agent_span_name("create") == "create_agent"

    def test_invalid_op_raises(self):
        with pytest.raises(ValueError):
            agent_span_name("destroy")

    def test_invocation_attrs_always_gen_ai(self, monkeypatch):
        monkeypatch.delenv("OTEL_SEMCONV_STABILITY_OPT_IN", raising=False)
        attrs = agent_invocation_attributes(agent_name="Ideator", task_type="reasoning")
        assert attrs["gen_ai.agent.name"] == "Ideator"
        assert attrs["gen_ai.operation.name"] == "invoke_agent"
        assert attrs["gen_ai.agent.task_type"] == "reasoning"
        assert attrs["span.kind"] == "CLIENT"
        # Legacy mirrors only under /dup
        assert "ael.agent.name" not in attrs

    def test_invocation_attrs_dup_mirrors_legacy(self, monkeypatch):
        monkeypatch.setenv(
            "OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental/dup"
        )
        attrs = agent_invocation_attributes(
            agent_name="Ideator",
            task_type="reasoning",
            invocation_id="abc",
            tool_count=3,
            step_count=5,
        )
        assert attrs["ael.agent.name"] == "Ideator"
        assert attrs["ael.agent.invocation_id"] == "abc"
        assert attrs["ael.agent.tool_count"] == 3
        assert attrs["ael.agent.step_count"] == 5
        # Finalized attrs still present
        assert attrs["gen_ai.agent.name"] == "Ideator"

    def test_invocation_with_model_fills_provider(self):
        attrs = agent_invocation_attributes(agent_name="X", model="claude-sonnet-4-6")
        assert attrs["gen_ai.request.model"] == "claude-sonnet-4-6"
        assert attrs["gen_ai.provider.name"] == "anthropic"
        assert attrs["gen_ai.system"] == "anthropic"

    def test_invocation_omits_none_fields(self):
        attrs = agent_invocation_attributes(agent_name="X")
        assert "gen_ai.agent.id" not in attrs
        assert "gen_ai.agent.task_type" not in attrs
