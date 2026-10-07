# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Tests for OTel exporter."""

import json
import pytest
from pathlib import Path
from shared.observability import MetricsCollector, LLMCallRecord, ToolCallRecord, EmbeddingCallRecord
from shared.telemetry.otel_exporter import OTelExporter, SpanRecord
from shared.telemetry.gen_ai_conventions import GenAIAttributes


# ---------------------------------------------------------------------------
# SpanRecord tests
# ---------------------------------------------------------------------------

class TestSpanRecord:
    def test_default_values(self):
        span = SpanRecord(name="test")
        assert span.name == "test"
        assert span.kind == "CLIENT"
        assert span.status == "OK"
        assert len(span.trace_id) == 32
        assert len(span.span_id) == 16

    def test_to_dict(self):
        span = SpanRecord(
            name="gen_ai.chat gpt-4o-mini",
            duration_ms=150.5,
            attributes={"gen_ai.request.model": "gpt-4o-mini"},
            status="OK",
        )
        d = span.to_dict()
        assert d["name"] == "gen_ai.chat gpt-4o-mini"
        assert d["duration_ms"] == 150.5
        assert d["attributes"]["gen_ai.request.model"] == "gpt-4o-mini"

    def test_parent_span_id(self):
        span = SpanRecord(name="child", parent_span_id="abc123")
        d = span.to_dict()
        assert d["parent_span_id"] == "abc123"

    def test_no_parent_span_id(self):
        span = SpanRecord(name="root")
        d = span.to_dict()
        assert "parent_span_id" not in d


# ---------------------------------------------------------------------------
# OTelExporter init tests
# ---------------------------------------------------------------------------

class TestOTelExporterInit:
    def test_default_backend(self):
        exporter = OTelExporter()
        assert exporter.backend == "json"

    def test_console_backend(self):
        exporter = OTelExporter(backend="console")
        assert exporter.backend == "console"

    def test_invalid_backend(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            OTelExporter(backend="invalid")

    def test_service_name(self):
        exporter = OTelExporter(service_name="my-service")
        assert exporter.service_name == "my-service"


# ---------------------------------------------------------------------------
# Export from collector tests
# ---------------------------------------------------------------------------

class TestExportFromCollector:
    def _make_collector(self):
        c = MetricsCollector()
        c.record_llm_call(LLMCallRecord(
            agent="TrendSurfer",
            stage="SourcingStage",
            model="gpt-4o-mini",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.001,
            latency_seconds=1.5,
        ))
        c.record_tool_call(ToolCallRecord(
            agent="CorpusScout",
            stage="SourcingStage",
            tool_name="arXiv",
            url="https://arxiv.org/api",
            http_method="GET",
            status_code=200,
            latency_seconds=0.5,
            success=True,
        ))
        c.record_embedding_call(EmbeddingCallRecord(
            agent="Embedder",
            model="text-embedding-3-small",
            input_tokens=200,
            num_texts=5,
            cost_usd=0.00004,
            latency_seconds=0.3,
        ))
        return c

    def test_export_json(self, tmp_path):
        collector = self._make_collector()
        exporter = OTelExporter(backend="json")
        output = tmp_path / "traces.json"
        spans = exporter.export_from_collector(
            collector, output_path=str(output), team="IdeationTeam", mode="ModeNoWcNoHITL"
        )
        # Root + 1 LLM + 1 tool + 1 embedding = 4 spans
        assert len(spans) == 4
        assert output.exists()
        data = json.loads(output.read_text())
        assert data["span_count"] == 4
        assert len(data["spans"]) == 4

    def test_spans_have_team_attribute(self):
        collector = self._make_collector()
        exporter = OTelExporter(backend="json")
        spans = exporter.export_from_collector(
            collector, team="IdeationTeam"
        )
        # All non-root spans should have team attribute
        for span in spans[1:]:
            assert span.attributes.get(GenAIAttributes.AEL_TEAM) == "IdeationTeam"

    def test_shared_trace_id(self):
        collector = self._make_collector()
        exporter = OTelExporter(backend="json")
        spans = exporter.export_from_collector(collector, trace_id="abc123")
        for span in spans:
            assert span.trace_id == "abc123"

    def test_child_spans_have_parent(self):
        collector = self._make_collector()
        exporter = OTelExporter(backend="json")
        spans = exporter.export_from_collector(collector)
        root = spans[0]
        for span in spans[1:]:
            assert span.parent_span_id == root.span_id

    def test_llm_span_name(self):
        collector = self._make_collector()
        exporter = OTelExporter(backend="json")
        spans = exporter.export_from_collector(collector)
        llm_span = spans[1]
        assert "gen_ai.chat" in llm_span.name
        assert "gpt-4o-mini" in llm_span.name

    def test_error_span_status(self):
        c = MetricsCollector()
        c.record_llm_call(LLMCallRecord(
            agent="Agent", model="gpt-4o-mini", error="timeout"
        ))
        exporter = OTelExporter(backend="json")
        spans = exporter.export_from_collector(c)
        llm_span = spans[1]
        assert llm_span.status == "ERROR"

    def test_empty_collector(self):
        c = MetricsCollector()
        exporter = OTelExporter(backend="json")
        spans = exporter.export_from_collector(c)
        # Only root span
        assert len(spans) == 1

    def test_console_export(self, capsys):
        c = MetricsCollector()
        c.record_llm_call(LLMCallRecord(agent="A", model="gpt-4o-mini"))
        exporter = OTelExporter(backend="console")
        spans = exporter.export_from_collector(c)
        captured = capsys.readouterr()
        assert "gen_ai.chat" in captured.out
