# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
OTel Exporter — Converts MetricsCollector data to OpenTelemetry-compatible spans.

Supports multiple export backends:
- "json": JSON file export (backward compatible, no external deps)
- "console": Print spans to stdout (development)
- "otlp": OTLP/gRPC export (requires opentelemetry-sdk, optional)

The exporter uses a lightweight SpanRecord dataclass so the core export
logic works without the OTel SDK installed.  When OTLP export is requested,
the SDK is imported lazily.

Usage:
    from shared.telemetry.otel_exporter import OTelExporter
    from shared.observability import MetricsCollector

    collector = MetricsCollector()
    # ... workflow runs ...

    exporter = OTelExporter(backend="json")
    exporter.export_from_collector(collector, output_path="traces.json")
"""

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.telemetry.gen_ai_conventions import (
    GenAIAttributes,
    SpanKind,
    llm_call_attributes,
    tool_call_attributes,
    embedding_call_attributes,
)


@dataclass
class SpanRecord:
    """Lightweight OTel-compatible span representation."""

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_span_id: Optional[str] = None
    name: str = ""
    kind: str = SpanKind.CLIENT
    start_time: str = ""
    end_time: str = ""
    duration_ms: float = 0.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    status: str = "OK"  # OK, ERROR

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "name": self.name,
            "kind": self.kind,
            "start_time": self.start_time,
            "duration_ms": round(self.duration_ms, 2),
            "attributes": self.attributes,
            "status": self.status,
        }
        if self.parent_span_id:
            d["parent_span_id"] = self.parent_span_id
        if self.end_time:
            d["end_time"] = self.end_time
        return d


class OTelExporter:
    """
    Exports MetricsCollector data as OpenTelemetry-compatible spans.

    Converts LLMCallRecord, ToolCallRecord, and EmbeddingCallRecord
    instances into SpanRecord objects with OTel GenAI semantic attributes,
    then writes them to the selected backend.
    """

    BACKENDS = ("json", "console", "otlp")

    def __init__(
        self,
        backend: str = "json",
        service_name: str = "ael-workflow",
        otlp_endpoint: Optional[str] = None,
    ):
        """
        Args:
            backend: Export backend — "json", "console", or "otlp".
            service_name: Service name for OTel resource.
            otlp_endpoint: OTLP collector endpoint (for "otlp" backend).
        """
        if backend not in self.BACKENDS:
            raise ValueError(f"Unknown backend {backend!r}; choose from {self.BACKENDS}")
        self.backend = backend
        self.service_name = service_name
        self.otlp_endpoint = otlp_endpoint or "http://localhost:4317"

    def export_from_collector(
        self,
        collector: Any,
        output_path: Optional[str] = None,
        trace_id: Optional[str] = None,
        team: str = "",
        mode: str = "",
    ) -> List[SpanRecord]:
        """
        Convert all MetricsCollector records to spans and export.

        Args:
            collector: MetricsCollector instance.
            output_path: File path for JSON backend (ignored for others).
            trace_id: Optional shared trace ID for all spans.
            team: Team name to attach as attribute.
            mode: Mode name to attach as attribute.

        Returns:
            List of SpanRecord objects that were exported.
        """
        records = collector.get_detailed_records()
        shared_trace_id = trace_id or uuid.uuid4().hex

        spans: List[SpanRecord] = []

        # Root span for the workflow
        root_span_id = uuid.uuid4().hex[:16]
        root_attrs: Dict[str, Any] = {
            "service.name": self.service_name,
        }
        if team:
            root_attrs[GenAIAttributes.AEL_TEAM] = team
        if mode:
            root_attrs[GenAIAttributes.AEL_MODE] = mode

        # Determine overall duration from child spans
        all_timestamps = []
        for r in records.get("llm_calls", []):
            all_timestamps.append(r.get("timestamp", ""))
        for r in records.get("tool_calls", []):
            all_timestamps.append(r.get("timestamp", ""))
        for r in records.get("embedding_calls", []):
            all_timestamps.append(r.get("timestamp", ""))

        root_start = min(all_timestamps) if all_timestamps else ""

        root_span = SpanRecord(
            trace_id=shared_trace_id,
            span_id=root_span_id,
            name=f"workflow.{team}.{mode}" if team else "workflow",
            kind=SpanKind.INTERNAL,
            start_time=root_start,
            attributes=root_attrs,
        )
        spans.append(root_span)

        # LLM call spans
        for rec in records.get("llm_calls", []):
            attrs = _dict_to_otel_attrs(rec)
            if team:
                attrs[GenAIAttributes.AEL_TEAM] = team
            span = SpanRecord(
                trace_id=shared_trace_id,
                parent_span_id=root_span_id,
                name=f"gen_ai.chat {rec.get('model', '')}",
                kind=SpanKind.CLIENT,
                start_time=rec.get("timestamp", ""),
                duration_ms=rec.get("latency_seconds", 0) * 1000,
                attributes=attrs,
                status="ERROR" if rec.get("error") else "OK",
            )
            spans.append(span)

        # Tool call spans
        for rec in records.get("tool_calls", []):
            attrs = _dict_to_tool_attrs(rec)
            if team:
                attrs[GenAIAttributes.AEL_TEAM] = team
            span = SpanRecord(
                trace_id=shared_trace_id,
                parent_span_id=root_span_id,
                name=f"tool.{rec.get('tool_name', 'unknown')}",
                kind=SpanKind.CLIENT,
                start_time=rec.get("timestamp", ""),
                duration_ms=rec.get("latency_seconds", 0) * 1000,
                attributes=attrs,
                status="ERROR" if rec.get("error") else "OK",
            )
            spans.append(span)

        # Embedding call spans
        for rec in records.get("embedding_calls", []):
            attrs = _dict_to_embed_attrs(rec)
            if team:
                attrs[GenAIAttributes.AEL_TEAM] = team
            span = SpanRecord(
                trace_id=shared_trace_id,
                parent_span_id=root_span_id,
                name=f"gen_ai.embeddings {rec.get('model', '')}",
                kind=SpanKind.CLIENT,
                start_time=rec.get("timestamp", ""),
                duration_ms=rec.get("latency_seconds", 0) * 1000,
                attributes=attrs,
                status="ERROR" if rec.get("error") else "OK",
            )
            spans.append(span)

        # Export
        self._export(spans, output_path)
        return spans

    def _export(self, spans: List[SpanRecord], output_path: Optional[str] = None):
        """Dispatch to the configured backend."""
        if self.backend == "json":
            self._export_json(spans, output_path)
        elif self.backend == "console":
            self._export_console(spans)
        elif self.backend == "otlp":
            self._export_otlp(spans)

    def _export_json(self, spans: List[SpanRecord], output_path: Optional[str] = None):
        """Write spans to a JSON file. Requires an explicit output_path."""
        if not output_path:
            return
        path = Path(output_path)
        data = {
            "service_name": self.service_name,
            "span_count": len(spans),
            "spans": [s.to_dict() for s in spans],
        }
        path.write_text(json.dumps(data, indent=2, default=str))

    def _export_console(self, spans: List[SpanRecord]):
        """Print spans to stdout."""
        for span in spans:
            print(json.dumps(span.to_dict(), indent=2, default=str))

    def _export_otlp(self, spans: List[SpanRecord]):
        """
        Export spans via OTLP/gRPC.

        Requires opentelemetry-sdk and opentelemetry-exporter-otlp-proto-grpc.
        Falls back to JSON export if SDK is not available.
        """
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.resources import Resource

            resource = Resource.create({"service.name": self.service_name})
            provider = TracerProvider(resource=resource)
            exporter = OTLPSpanExporter(endpoint=self.otlp_endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            trace.set_tracer_provider(provider)
            tracer = trace.get_tracer(self.service_name)

            for span_rec in spans:
                with tracer.start_as_current_span(
                    span_rec.name,
                    attributes=span_rec.attributes,
                ) as otel_span:
                    if span_rec.status == "ERROR":
                        otel_span.set_status(trace.StatusCode.ERROR)

            provider.shutdown()

        except ImportError:
            # OTel SDK not installed — fall back to console
            self._export_console(spans)


# ---------------------------------------------------------------------------
# Helpers: dict-based attribute builders (for serialized records)
# ---------------------------------------------------------------------------


def _dict_to_otel_attrs(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a serialized LLM call dict to OTel attributes."""
    attrs: Dict[str, Any] = {
        GenAIAttributes.OPERATION_NAME: "chat",
        GenAIAttributes.REQUEST_MODEL: rec.get("model", ""),
        GenAIAttributes.USAGE_INPUT_TOKENS: rec.get("prompt_tokens", 0),
        GenAIAttributes.USAGE_OUTPUT_TOKENS: rec.get("completion_tokens", 0),
    }
    model = rec.get("model", "")
    from shared.provider_map import detect_provider as _detect
    _provider = _detect(model, default="")
    if _provider:
        attrs[GenAIAttributes.PROVIDER_NAME] = _provider
    if rec.get("agent"):
        attrs[GenAIAttributes.AGENT_NAME] = rec["agent"]
    if rec.get("stage"):
        attrs[GenAIAttributes.AEL_STAGE] = rec["stage"]
    if rec.get("cost_usd"):
        attrs[GenAIAttributes.AEL_COST_USD] = rec["cost_usd"]
    if rec.get("error"):
        attrs["error.message"] = rec["error"]
    return attrs


def _dict_to_tool_attrs(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a serialized tool call dict to OTel attributes."""
    attrs: Dict[str, Any] = {
        GenAIAttributes.OPERATION_NAME: "execute_tool",
        GenAIAttributes.TOOL_NAME: rec.get("tool_name", ""),
        GenAIAttributes.HTTP_METHOD: rec.get("http_method", "GET"),
    }
    if rec.get("url"):
        attrs[GenAIAttributes.TOOL_URL] = rec["url"]
    if rec.get("status_code") is not None:
        attrs[GenAIAttributes.HTTP_STATUS_CODE] = rec["status_code"]
    if rec.get("agent"):
        attrs[GenAIAttributes.AGENT_NAME] = rec["agent"]
    if rec.get("stage"):
        attrs[GenAIAttributes.AEL_STAGE] = rec["stage"]
    if rec.get("error"):
        attrs["error.message"] = rec["error"]
    return attrs


def _dict_to_embed_attrs(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a serialized embedding call dict to OTel attributes."""
    attrs: Dict[str, Any] = {
        GenAIAttributes.OPERATION_NAME: "embeddings",
        GenAIAttributes.REQUEST_MODEL: rec.get("model", ""),
        GenAIAttributes.USAGE_INPUT_TOKENS: rec.get("input_tokens", 0),
    }
    model = rec.get("model", "")
    from shared.provider_map import detect_provider as _detect
    _embed_provider = _detect(model, default="")
    if _embed_provider:
        attrs[GenAIAttributes.PROVIDER_NAME] = _embed_provider
    if rec.get("agent"):
        attrs[GenAIAttributes.AGENT_NAME] = rec["agent"]
    if rec.get("stage"):
        attrs[GenAIAttributes.AEL_STAGE] = rec["stage"]
    if rec.get("cost_usd"):
        attrs[GenAIAttributes.AEL_COST_USD] = rec["cost_usd"]
    if rec.get("error"):
        attrs["error.message"] = rec["error"]
    return attrs
