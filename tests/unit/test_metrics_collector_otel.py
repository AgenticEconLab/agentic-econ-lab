# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Tests for MetricsCollector OTel span generation and prompt caching."""

import json
import pytest
from shared.observability import MetricsCollector, LLMCallRecord, ToolCallRecord, EmbeddingCallRecord
from shared.telemetry.gen_ai_conventions import GenAIAttributes
from shared.llm import LLMClient


# ---------------------------------------------------------------------------
# MetricsCollector.to_otel_spans tests
# ---------------------------------------------------------------------------

class TestMetricsCollectorOTelSpans:
    def test_empty_collector_returns_empty(self):
        c = MetricsCollector()
        spans = c.to_otel_spans()
        assert spans == []

    def test_llm_call_span(self):
        c = MetricsCollector()
        c.record_llm_call(LLMCallRecord(
            agent="TrendSurfer", stage="SourcingStage", model="gpt-4o-mini",
            prompt_tokens=100, completion_tokens=50, total_tokens=150,
            cost_usd=0.001, latency_seconds=1.5,
        ))
        spans = c.to_otel_spans(trace_id="test-trace")
        assert len(spans) == 1
        span = spans[0]
        assert span["trace_id"] == "test-trace"
        assert "gen_ai.chat" in span["name"]
        assert span["kind"] == "CLIENT"
        assert span["status"] == "OK"
        assert span["attributes"][GenAIAttributes.REQUEST_MODEL] == "gpt-4o-mini"
        assert span["attributes"][GenAIAttributes.AGENT_NAME] == "TrendSurfer"

    def test_tool_call_span(self):
        c = MetricsCollector()
        c.record_tool_call(ToolCallRecord(
            agent="CorpusScout", tool_name="arXiv",
            url="https://arxiv.org/api", http_method="GET",
            status_code=200, latency_seconds=0.5, success=True,
        ))
        spans = c.to_otel_spans()
        assert len(spans) == 1
        assert "tool.arXiv" in spans[0]["name"]
        assert spans[0]["attributes"][GenAIAttributes.TOOL_NAME] == "arXiv"

    def test_embedding_call_span(self):
        c = MetricsCollector()
        c.record_embedding_call(EmbeddingCallRecord(
            agent="Embedder", model="text-embedding-3-small",
            input_tokens=500, cost_usd=0.0001, latency_seconds=0.3,
        ))
        spans = c.to_otel_spans()
        assert len(spans) == 1
        assert "gen_ai.embeddings" in spans[0]["name"]

    def test_team_and_mode_attributes(self):
        c = MetricsCollector()
        c.record_llm_call(LLMCallRecord(agent="A", model="gpt-4o-mini"))
        spans = c.to_otel_spans(team="IdeationTeam", mode="ModeNoWcNoHITL")
        assert spans[0]["attributes"][GenAIAttributes.AEL_TEAM] == "IdeationTeam"
        assert spans[0]["attributes"][GenAIAttributes.AEL_MODE] == "ModeNoWcNoHITL"

    def test_error_span_status(self):
        c = MetricsCollector()
        c.record_llm_call(LLMCallRecord(
            agent="A", model="gpt-4o-mini", error="rate_limit",
        ))
        spans = c.to_otel_spans()
        assert spans[0]["status"] == "ERROR"

    def test_multiple_record_types(self):
        c = MetricsCollector()
        c.record_llm_call(LLMCallRecord(agent="A", model="gpt-4o-mini"))
        c.record_tool_call(ToolCallRecord(agent="A", tool_name="FRED"))
        c.record_embedding_call(EmbeddingCallRecord(agent="A", model="text-embedding-3-small"))
        spans = c.to_otel_spans()
        assert len(spans) == 3

    def test_duration_ms_conversion(self):
        c = MetricsCollector()
        c.record_llm_call(LLMCallRecord(
            agent="A", model="gpt-4o-mini", latency_seconds=2.5,
        ))
        spans = c.to_otel_spans()
        assert spans[0]["duration_ms"] == 2500.0


# ---------------------------------------------------------------------------
# MetricsCollector property accessors tests
# ---------------------------------------------------------------------------

class TestMetricsCollectorProperties:
    def test_llm_calls_property(self):
        c = MetricsCollector()
        c.record_llm_call(LLMCallRecord(agent="A", model="m"))
        calls = c.llm_calls
        assert len(calls) == 1
        # Should be a copy
        c.record_llm_call(LLMCallRecord(agent="B", model="m"))
        assert len(calls) == 1  # Original list unchanged

    def test_tool_calls_property(self):
        c = MetricsCollector()
        c.record_tool_call(ToolCallRecord(agent="A", tool_name="T"))
        assert len(c.tool_calls) == 1

    def test_embedding_calls_property(self):
        c = MetricsCollector()
        c.record_embedding_call(EmbeddingCallRecord(agent="A", model="m"))
        assert len(c.embedding_calls) == 1


# ---------------------------------------------------------------------------
# LLMClient prompt caching tests
# ---------------------------------------------------------------------------

class TestLLMClientPromptCache:
    def test_cache_disabled_by_default(self):
        client = LLMClient()
        assert client.enable_prompt_cache is False
        stats = client.get_cache_stats()
        assert stats["enabled"] is False
        assert stats["entries"] == 0

    def test_cache_key_deterministic(self):
        client = LLMClient(enable_prompt_cache=True)
        msgs = [{"role": "user", "content": "hello"}]
        k1 = client._cache_key(msgs)
        k2 = client._cache_key(msgs)
        assert k1 == k2
        assert len(k1) == 24

    def test_different_messages_different_keys(self):
        client = LLMClient(enable_prompt_cache=True)
        k1 = client._cache_key([{"role": "user", "content": "hello"}])
        k2 = client._cache_key([{"role": "user", "content": "world"}])
        assert k1 != k2

    def test_cache_stats_initial(self):
        client = LLMClient(enable_prompt_cache=True, temperature=0)
        stats = client.get_cache_stats()
        assert stats["enabled"] is True
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0
