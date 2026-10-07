# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
OpenTelemetry GenAI Semantic Conventions for AEL workflows.

Implements attribute naming from the OTel GenAI spec (v1.37+) so that
MetricsCollector records can be exported to any OTel-compatible backend
(Jaeger, Phoenix, Datadog, etc.) with standard attribute semantics.

Reference:
    https://opentelemetry.io/docs/specs/semconv/gen-ai/

Usage:
    from shared.telemetry.gen_ai_conventions import llm_call_attributes

    attrs = llm_call_attributes(record)
    # -> {"gen_ai.request.model": "gpt-4o-mini", "gen_ai.usage.input_tokens": 150, ...}
"""

from typing import Any, Dict


# ---------------------------------------------------------------------------
# Standard OTel GenAI attribute names (v1.37+)
# ---------------------------------------------------------------------------

class GenAIAttributes:
    """OTel GenAI semantic convention attribute keys."""

    # Agent identification
    AGENT_NAME = "gen_ai.agent.name"
    AGENT_ID = "gen_ai.agent.id"

    # Operation
    OPERATION_NAME = "gen_ai.operation.name"
    SYSTEM = "gen_ai.system"

    # Request
    REQUEST_MODEL = "gen_ai.request.model"
    REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"

    # Response
    RESPONSE_MODEL = "gen_ai.response.model"
    RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"

    # Usage
    USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

    # Provider
    PROVIDER_NAME = "gen_ai.provider.name"

    # Conversation
    CONVERSATION_ID = "gen_ai.conversation.id"

    # AEL extensions (namespaced under ael.*)
    AEL_STAGE = "ael.stage"
    AEL_TEAM = "ael.team"
    AEL_MODE = "ael.mode"
    AEL_COST_USD = "ael.cost_usd"
    AEL_CACHE_HIT = "ael.cache_hit"
    AEL_PIPELINE_RUN_ID = "ael.pipeline_run_id"
    AEL_PROVIDER = "ael.provider"

    # Tool attributes
    TOOL_NAME = "tool.name"
    TOOL_URL = "url.full"
    HTTP_METHOD = "http.request.method"
    HTTP_STATUS_CODE = "http.response.status_code"

    # Agent invocation span attributes (V0.5 — OTel GenAI agent conventions)
    AGENT_INVOCATION_ID = "gen_ai.agent.invocation_id"
    AGENT_TASK_TYPE = "gen_ai.agent.task_type"
    AGENT_TOOL_COUNT = "gen_ai.agent.tool_count"
    AGENT_STEP_COUNT = "gen_ai.agent.step_count"


class SpanKind:
    """OTel span kind constants (string-based for JSON compatibility)."""

    CLIENT = "CLIENT"
    INTERNAL = "INTERNAL"
    SERVER = "SERVER"


# Operation name constants
OP_CHAT = "chat"
OP_EMBED = "embeddings"
OP_TOOL = "execute_tool"
OP_CREATE_AGENT = "create_agent"
OP_INVOKE_AGENT = "invoke_agent"


# Provider detection — delegates to single source of truth (V0.6)
from shared.provider_map import PROVIDER_ROUTES as _MODEL_PROVIDERS  # noqa: E402
from shared.provider_map import detect_provider as _detect_provider_base  # noqa: E402


def _detect_provider(model: str) -> str:
    """Detect provider from model name.

    Delegates to shared.provider_map.detect_provider with "unknown" default
    (telemetry convention: unknown provider rather than assuming openai).
    """
    return _detect_provider_base(model, default="unknown")


# ---------------------------------------------------------------------------
# Attribute builders — convert existing records to OTel attributes
# ---------------------------------------------------------------------------


def llm_call_attributes(record: Any) -> Dict[str, Any]:
    """
    Convert an LLMCallRecord to OTel GenAI attributes dict.

    Args:
        record: An LLMCallRecord (from shared.observability).

    Returns:
        Dict of OTel-compliant attribute key-value pairs.
    """
    attrs: Dict[str, Any] = {
        GenAIAttributes.OPERATION_NAME: OP_CHAT,
        GenAIAttributes.SYSTEM: _detect_provider(record.model),
        GenAIAttributes.REQUEST_MODEL: record.model,
        GenAIAttributes.USAGE_INPUT_TOKENS: record.prompt_tokens,
        GenAIAttributes.USAGE_OUTPUT_TOKENS: record.completion_tokens,
        GenAIAttributes.PROVIDER_NAME: _detect_provider(record.model),
    }

    if record.agent:
        attrs[GenAIAttributes.AGENT_NAME] = record.agent
    if record.stage:
        attrs[GenAIAttributes.AEL_STAGE] = record.stage
    if record.cost_usd:
        attrs[GenAIAttributes.AEL_COST_USD] = round(record.cost_usd, 8)
    if record.error:
        attrs["error.message"] = record.error

    return attrs


def tool_call_attributes(record: Any) -> Dict[str, Any]:
    """
    Convert a ToolCallRecord to OTel attributes dict.

    Args:
        record: A ToolCallRecord (from shared.observability).

    Returns:
        Dict of OTel-compliant attribute key-value pairs.
    """
    attrs: Dict[str, Any] = {
        GenAIAttributes.OPERATION_NAME: OP_TOOL,
        GenAIAttributes.TOOL_NAME: record.tool_name,
        GenAIAttributes.HTTP_METHOD: record.http_method,
    }

    if record.url:
        attrs[GenAIAttributes.TOOL_URL] = record.url
    if record.status_code is not None:
        attrs[GenAIAttributes.HTTP_STATUS_CODE] = record.status_code
    if record.agent:
        attrs[GenAIAttributes.AGENT_NAME] = record.agent
    if record.stage:
        attrs[GenAIAttributes.AEL_STAGE] = record.stage
    if record.error:
        attrs["error.message"] = record.error

    return attrs


# ---------------------------------------------------------------------------
# V0.7 Pre-Phase — finalized OTel GenAI agent-span conventions
# Reference: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/
# ---------------------------------------------------------------------------

import os as _os

# Legacy (V0.5) AEL agent-span attribute prefix used before OTel finalization.
LEGACY_AGENT_SPAN_NAME_INVOKE = "ael.agent.invoke"
LEGACY_AGENT_SPAN_NAME_CREATE = "ael.agent.create"

# Finalized OTel GenAI operation names
AGENT_SPAN_NAME_INVOKE = OP_INVOKE_AGENT  # "invoke_agent"
AGENT_SPAN_NAME_CREATE = OP_CREATE_AGENT  # "create_agent"


def _dual_emit_enabled() -> bool:
    """Honour OTEL_SEMCONV_STABILITY_OPT_IN (comma list, case-insensitive).

    Supported values:
      "gen_ai_latest_experimental" -> new attrs only
      "gen_ai_latest_experimental/dup" -> legacy + new (dual emit)
      "" / unset -> legacy only (back-compat default)
    """
    raw = _os.getenv("OTEL_SEMCONV_STABILITY_OPT_IN", "").lower()
    return "gen_ai_latest_experimental" in raw


def _dup_emit_enabled() -> bool:
    raw = _os.getenv("OTEL_SEMCONV_STABILITY_OPT_IN", "").lower()
    return "/dup" in raw or "dup" in [t.strip() for t in raw.split(",")]


def agent_span_name(op: str) -> str:
    """Return the appropriate span name based on the semconv opt-in flag.

    ``op`` is one of ``"invoke"`` or ``"create"``.
    """
    if op == "invoke":
        return AGENT_SPAN_NAME_INVOKE if _dual_emit_enabled() else LEGACY_AGENT_SPAN_NAME_INVOKE
    if op == "create":
        return AGENT_SPAN_NAME_CREATE if _dual_emit_enabled() else LEGACY_AGENT_SPAN_NAME_CREATE
    raise ValueError(f"Unknown agent span op: {op!r}")


def agent_invocation_attributes(
    *,
    agent_name: str,
    agent_id: str | None = None,
    task_type: str | None = None,
    tool_count: int | None = None,
    step_count: int | None = None,
    invocation_id: str | None = None,
    span_kind: str = SpanKind.CLIENT,
    model: str | None = None,
) -> Dict[str, Any]:
    """Build attributes for an agent invocation span (dual-emit aware).

    Always emits finalized ``gen_ai.*`` attributes. When the stability opt-in
    env var has ``/dup`` set, also emits the legacy ``ael.agent.*`` mirror
    attributes so dashboards built against V0.5-era span data keep working.
    """
    attrs: Dict[str, Any] = {
        GenAIAttributes.OPERATION_NAME: OP_INVOKE_AGENT,
        GenAIAttributes.AGENT_NAME: agent_name,
        "span.kind": span_kind,
    }
    if agent_id is not None:
        attrs[GenAIAttributes.AGENT_ID] = agent_id
    if task_type is not None:
        attrs[GenAIAttributes.AGENT_TASK_TYPE] = task_type
    if tool_count is not None:
        attrs[GenAIAttributes.AGENT_TOOL_COUNT] = tool_count
    if step_count is not None:
        attrs[GenAIAttributes.AGENT_STEP_COUNT] = step_count
    if invocation_id is not None:
        attrs[GenAIAttributes.AGENT_INVOCATION_ID] = invocation_id
    if model is not None:
        attrs[GenAIAttributes.REQUEST_MODEL] = model
        attrs[GenAIAttributes.SYSTEM] = _detect_provider(model)
        attrs[GenAIAttributes.PROVIDER_NAME] = _detect_provider(model)

    if _dup_emit_enabled():
        # Dual-emit legacy mirrors (V0.5 attribute names)
        if agent_name:
            attrs["ael.agent.name"] = agent_name
        if invocation_id is not None:
            attrs["ael.agent.invocation_id"] = invocation_id
        if task_type is not None:
            attrs["ael.agent.task_type"] = task_type
        if tool_count is not None:
            attrs["ael.agent.tool_count"] = tool_count
        if step_count is not None:
            attrs["ael.agent.step_count"] = step_count

    return attrs


def embedding_call_attributes(record: Any) -> Dict[str, Any]:
    """
    Convert an EmbeddingCallRecord to OTel attributes dict.

    Args:
        record: An EmbeddingCallRecord (from shared.observability).

    Returns:
        Dict of OTel-compliant attribute key-value pairs.
    """
    attrs: Dict[str, Any] = {
        GenAIAttributes.OPERATION_NAME: OP_EMBED,
        GenAIAttributes.SYSTEM: _detect_provider(record.model),
        GenAIAttributes.REQUEST_MODEL: record.model,
        GenAIAttributes.USAGE_INPUT_TOKENS: record.input_tokens,
        GenAIAttributes.PROVIDER_NAME: _detect_provider(record.model),
    }

    if record.agent:
        attrs[GenAIAttributes.AGENT_NAME] = record.agent
    if record.stage:
        attrs[GenAIAttributes.AEL_STAGE] = record.stage
    if record.cost_usd:
        attrs[GenAIAttributes.AEL_COST_USD] = round(record.cost_usd, 8)
    if record.error:
        attrs["error.message"] = record.error

    return attrs
