# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Tests for OTel GenAI semantic conventions."""

import pytest
from shared.telemetry.gen_ai_conventions import (
    GenAIAttributes,
    SpanKind,
    llm_call_attributes,
    tool_call_attributes,
    embedding_call_attributes,
    OP_CHAT,
    OP_EMBED,
    OP_TOOL,
    _detect_provider,
)


# ---------------------------------------------------------------------------
# Attribute constants tests
# ---------------------------------------------------------------------------

class TestGenAIAttributes:
    def test_standard_attributes_exist(self):
        assert GenAIAttributes.AGENT_NAME == "gen_ai.agent.name"
        assert GenAIAttributes.REQUEST_MODEL == "gen_ai.request.model"
        assert GenAIAttributes.USAGE_INPUT_TOKENS == "gen_ai.usage.input_tokens"
        assert GenAIAttributes.USAGE_OUTPUT_TOKENS == "gen_ai.usage.output_tokens"
        assert GenAIAttributes.PROVIDER_NAME == "gen_ai.provider.name"

    def test_ael_extensions(self):
        assert GenAIAttributes.AEL_STAGE == "ael.stage"
        assert GenAIAttributes.AEL_TEAM == "ael.team"
        assert GenAIAttributes.AEL_COST_USD == "ael.cost_usd"
        assert GenAIAttributes.AEL_CACHE_HIT == "ael.cache_hit"

    def test_tool_attributes(self):
        assert GenAIAttributes.TOOL_NAME == "tool.name"
        assert GenAIAttributes.HTTP_METHOD == "http.request.method"
        assert GenAIAttributes.HTTP_STATUS_CODE == "http.response.status_code"


class TestSpanKind:
    def test_kinds(self):
        assert SpanKind.CLIENT == "CLIENT"
        assert SpanKind.INTERNAL == "INTERNAL"


# ---------------------------------------------------------------------------
# Provider detection tests
# ---------------------------------------------------------------------------

class TestProviderDetection:
    def test_openai_models(self):
        assert _detect_provider("gpt-4o-mini") == "openai"
        assert _detect_provider("gpt-4o") == "openai"
        assert _detect_provider("text-embedding-3-small") == "openai"

    def test_anthropic_models(self):
        assert _detect_provider("claude-haiku-4-5-20251001") == "anthropic"
        assert _detect_provider("claude-3-5-sonnet-20241022") == "anthropic"

    def test_unknown_model(self):
        assert _detect_provider("llama-3") == "unknown"


# ---------------------------------------------------------------------------
# Attribute builder tests
# ---------------------------------------------------------------------------

class _MockLLMRecord:
    def __init__(self, **kwargs):
        self.agent = kwargs.get("agent", "")
        self.stage = kwargs.get("stage", "")
        self.model = kwargs.get("model", "gpt-4o-mini")
        self.prompt_tokens = kwargs.get("prompt_tokens", 100)
        self.completion_tokens = kwargs.get("completion_tokens", 50)
        self.cost_usd = kwargs.get("cost_usd", 0.001)
        self.error = kwargs.get("error", None)


class _MockToolRecord:
    def __init__(self, **kwargs):
        self.agent = kwargs.get("agent", "")
        self.stage = kwargs.get("stage", "")
        self.tool_name = kwargs.get("tool_name", "arXiv")
        self.url = kwargs.get("url", "https://arxiv.org/api")
        self.http_method = kwargs.get("http_method", "GET")
        self.status_code = kwargs.get("status_code", 200)
        self.error = kwargs.get("error", None)


class _MockEmbedRecord:
    def __init__(self, **kwargs):
        self.agent = kwargs.get("agent", "")
        self.stage = kwargs.get("stage", "")
        self.model = kwargs.get("model", "text-embedding-3-small")
        self.input_tokens = kwargs.get("input_tokens", 500)
        self.cost_usd = kwargs.get("cost_usd", 0.0001)
        self.error = kwargs.get("error", None)


class TestLLMCallAttributes:
    def test_basic_attributes(self):
        rec = _MockLLMRecord(agent="TrendSurfer", model="gpt-4o-mini")
        attrs = llm_call_attributes(rec)
        assert attrs[GenAIAttributes.OPERATION_NAME] == OP_CHAT
        assert attrs[GenAIAttributes.REQUEST_MODEL] == "gpt-4o-mini"
        assert attrs[GenAIAttributes.PROVIDER_NAME] == "openai"
        assert attrs[GenAIAttributes.USAGE_INPUT_TOKENS] == 100
        assert attrs[GenAIAttributes.USAGE_OUTPUT_TOKENS] == 50
        assert attrs[GenAIAttributes.AGENT_NAME] == "TrendSurfer"

    def test_anthropic_model(self):
        rec = _MockLLMRecord(model="claude-haiku-4-5-20251001")
        attrs = llm_call_attributes(rec)
        assert attrs[GenAIAttributes.PROVIDER_NAME] == "anthropic"

    def test_error_attribute(self):
        rec = _MockLLMRecord(error="timeout")
        attrs = llm_call_attributes(rec)
        assert attrs["error.message"] == "timeout"

    def test_no_error_no_error_key(self):
        rec = _MockLLMRecord()
        attrs = llm_call_attributes(rec)
        assert "error.message" not in attrs

    def test_stage_attribute(self):
        rec = _MockLLMRecord(stage="SourcingStage")
        attrs = llm_call_attributes(rec)
        assert attrs[GenAIAttributes.AEL_STAGE] == "SourcingStage"


class TestToolCallAttributes:
    def test_basic_attributes(self):
        rec = _MockToolRecord(agent="CorpusScout", tool_name="arXiv")
        attrs = tool_call_attributes(rec)
        assert attrs[GenAIAttributes.OPERATION_NAME] == OP_TOOL
        assert attrs[GenAIAttributes.TOOL_NAME] == "arXiv"
        assert attrs[GenAIAttributes.HTTP_METHOD] == "GET"
        assert attrs[GenAIAttributes.TOOL_URL] == "https://arxiv.org/api"
        assert attrs[GenAIAttributes.HTTP_STATUS_CODE] == 200
        assert attrs[GenAIAttributes.AGENT_NAME] == "CorpusScout"

    def test_no_url(self):
        rec = _MockToolRecord(url="")
        attrs = tool_call_attributes(rec)
        assert GenAIAttributes.TOOL_URL not in attrs


class TestEmbeddingCallAttributes:
    def test_basic_attributes(self):
        rec = _MockEmbedRecord()
        attrs = embedding_call_attributes(rec)
        assert attrs[GenAIAttributes.OPERATION_NAME] == OP_EMBED
        assert attrs[GenAIAttributes.REQUEST_MODEL] == "text-embedding-3-small"
        assert attrs[GenAIAttributes.PROVIDER_NAME] == "openai"
        assert attrs[GenAIAttributes.USAGE_INPUT_TOKENS] == 500
