# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Tests for per-team cost dashboard."""

import json
import pytest
from shared.observability import MetricsCollector, LLMCallRecord, ToolCallRecord, EmbeddingCallRecord
from shared.telemetry.cost_dashboard import CostDashboard, CostReport, CostBreakdown


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_collector():
    """Create a collector with sample records across agents/stages."""
    c = MetricsCollector()
    # Agent 1, Stage 1
    c.record_llm_call(LLMCallRecord(
        agent="TrendSurfer", stage="SourcingStage", model="gpt-4o-mini",
        prompt_tokens=200, completion_tokens=100, total_tokens=300,
        cost_usd=0.0009, latency_seconds=1.0,
    ))
    c.record_llm_call(LLMCallRecord(
        agent="TrendSurfer", stage="SourcingStage", model="gpt-4o-mini",
        prompt_tokens=150, completion_tokens=80, total_tokens=230,
        cost_usd=0.0007, latency_seconds=0.8,
    ))
    # Agent 2, Stage 1
    c.record_llm_call(LLMCallRecord(
        agent="CorpusScout", stage="SourcingStage", model="gpt-4o",
        prompt_tokens=500, completion_tokens=200, total_tokens=700,
        cost_usd=0.005, latency_seconds=2.0,
    ))
    # Agent 3, Stage 2
    c.record_llm_call(LLMCallRecord(
        agent="Refiner", stage="RefinementStage", model="gpt-4o-mini",
        prompt_tokens=300, completion_tokens=150, total_tokens=450,
        cost_usd=0.001, latency_seconds=1.2,
    ))
    # Tool calls
    c.record_tool_call(ToolCallRecord(
        agent="CorpusScout", stage="SourcingStage", tool_name="arXiv",
        url="https://arxiv.org", success=True, latency_seconds=0.5,
    ))
    c.record_tool_call(ToolCallRecord(
        agent="CorpusScout", stage="SourcingStage", tool_name="FRED",
        url="https://api.stlouisfed.org", success=False, latency_seconds=5.0,
        error="timeout",
    ))
    # Embedding call
    c.record_embedding_call(EmbeddingCallRecord(
        agent="Refiner", stage="RefinementStage",
        model="text-embedding-3-small",
        input_tokens=1000, num_texts=10, cost_usd=0.00002,
        latency_seconds=0.4,
    ))
    return c


# ---------------------------------------------------------------------------
# CostBreakdown tests
# ---------------------------------------------------------------------------

class TestCostBreakdown:
    def test_total_cost(self):
        b = CostBreakdown(name="agent", llm_cost_usd=0.01, embedding_cost_usd=0.001)
        assert b.total_cost_usd == pytest.approx(0.011)

    def test_to_dict(self):
        b = CostBreakdown(name="test", llm_calls=5, llm_cost_usd=0.1)
        d = b.to_dict()
        assert d["name"] == "test"
        assert d["llm_calls"] == 5
        assert d["total_cost_usd"] == 0.1


# ---------------------------------------------------------------------------
# CostReport tests
# ---------------------------------------------------------------------------

class TestCostReport:
    def test_to_dict(self):
        report = CostReport(
            team="IdeationTeam",
            mode="ModeNoWcNoHITL",
            total_cost_usd=0.05,
            total_llm_calls=10,
        )
        d = report.to_dict()
        assert d["team"] == "IdeationTeam"
        assert d["summary"]["total_cost_usd"] == 0.05
        assert d["summary"]["total_llm_calls"] == 10

    def test_budget_utilization_included(self):
        report = CostReport(
            team="T", mode="M",
            budget_utilization={"utilization_pct": 50.0},
        )
        d = report.to_dict()
        assert d["budget_utilization"]["utilization_pct"] == 50.0


# ---------------------------------------------------------------------------
# CostDashboard tests
# ---------------------------------------------------------------------------

class TestCostDashboard:
    def test_generate_basic(self):
        collector = _make_collector()
        dashboard = CostDashboard()
        report = dashboard.generate(collector, team="IdeationTeam", mode="ModeNoWcNoHITL")
        assert isinstance(report, CostReport)
        assert report.team == "IdeationTeam"
        assert report.total_llm_calls == 4
        assert report.total_tool_calls == 2
        assert report.total_embedding_calls == 1
        assert report.total_cost_usd > 0

    def test_by_agent_breakdown(self):
        collector = _make_collector()
        dashboard = CostDashboard()
        report = dashboard.generate(collector)
        agents = {b.name for b in report.by_agent}
        assert "TrendSurfer" in agents
        assert "CorpusScout" in agents
        assert "Refiner" in agents

    def test_by_stage_breakdown(self):
        collector = _make_collector()
        dashboard = CostDashboard()
        report = dashboard.generate(collector)
        stages = {b.name for b in report.by_stage}
        assert "SourcingStage" in stages
        assert "RefinementStage" in stages

    def test_by_model_breakdown(self):
        collector = _make_collector()
        dashboard = CostDashboard()
        report = dashboard.generate(collector)
        models = {b.name for b in report.by_model}
        assert "gpt-4o-mini" in models
        assert "gpt-4o" in models

    def test_by_tool_breakdown(self):
        collector = _make_collector()
        dashboard = CostDashboard()
        report = dashboard.generate(collector)
        tools = {b.name for b in report.by_tool}
        assert "arXiv" in tools
        assert "FRED" in tools
        # FRED call had an error
        fred_breakdown = next(b for b in report.by_tool if b.name == "FRED")
        assert fred_breakdown.tool_errors == 1

    def test_budget_utilization(self):
        collector = _make_collector()
        dashboard = CostDashboard()
        report = dashboard.generate(
            collector,
            budget_limits={"max_cost_usd": 1.0},
        )
        assert report.budget_utilization is not None
        assert report.budget_utilization["max_cost_usd"] == 1.0
        assert report.budget_utilization["utilization_pct"] < 100
        assert report.budget_utilization["remaining_usd"] > 0

    def test_cache_savings(self):
        collector = _make_collector()
        dashboard = CostDashboard()
        report = dashboard.generate(
            collector,
            cache_stats={"hit_rate": 0.35, "total_cost_saved": 0.02},
        )
        assert report.cache_savings is not None
        assert report.cache_savings["hit_rate"] == 0.35

    def test_save(self, tmp_path):
        collector = _make_collector()
        dashboard = CostDashboard()
        report = dashboard.generate(collector, team="T", mode="M")
        output = tmp_path / "report.json"
        dashboard.save(report, str(output))
        assert output.exists()
        data = json.loads(output.read_text())
        assert data["team"] == "T"

    def test_empty_collector(self):
        collector = MetricsCollector()
        dashboard = CostDashboard()
        report = dashboard.generate(collector)
        assert report.total_cost_usd == 0
        assert report.total_llm_calls == 0
        assert len(report.by_agent) == 0

    def test_sorted_by_cost(self):
        collector = _make_collector()
        dashboard = CostDashboard()
        report = dashboard.generate(collector)
        # by_agent should be sorted by total_cost descending
        costs = [b.total_cost_usd for b in report.by_agent]
        assert costs == sorted(costs, reverse=True)
