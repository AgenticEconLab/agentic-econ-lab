# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Telemetry module for OTel-aligned observability.

- gen_ai_conventions: OTel GenAI semantic attribute names
- otel_exporter: Export MetricsCollector data as OTel spans
- cost_dashboard: Per-team cost reports
"""

from shared.telemetry.gen_ai_conventions import (
    GenAIAttributes,
    SpanKind,
    llm_call_attributes,
    tool_call_attributes,
    embedding_call_attributes,
)
from shared.telemetry.otel_exporter import OTelExporter, SpanRecord
from shared.telemetry.cost_dashboard import CostDashboard

__all__ = [
    "GenAIAttributes",
    "SpanKind",
    "llm_call_attributes",
    "tool_call_attributes",
    "embedding_call_attributes",
    "OTelExporter",
    "SpanRecord",
    "CostDashboard",
    "export_telemetry",
]


def export_telemetry(
    collector,
    output_dir: str,
    team: str = "",
    mode: str = "",
):
    """
    Export OTel spans and CostDashboard report at workflow completion.

    Call this after ui.observability_summary() in MasterOrchestrators.
    Safe to call even if collector has no data — produces empty files.
    """
    import os

    try:
        exporter = OTelExporter(backend="json")
        trace_path = os.path.join(output_dir, "traces.json")
        exporter.export_from_collector(
            collector, output_path=trace_path, team=team, mode=mode,
        )
    except Exception:
        pass  # Non-critical; don't crash workflow

    try:
        dashboard = CostDashboard()
        report = dashboard.generate(collector, team=team, mode=mode)
        cost_path = os.path.join(output_dir, "cost_report.json")
        dashboard.save(report, cost_path)
    except Exception:
        pass  # Non-critical; don't crash workflow
