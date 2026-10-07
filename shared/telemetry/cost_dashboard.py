# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Cost Dashboard — Per-team cost attribution reports from MetricsCollector data.

Generates JSON reports with cost breakdowns by team, agent, stage, model,
and tool.  Reports include budget utilization if BudgetController limits
are provided.

Usage:
    from shared.telemetry.cost_dashboard import CostDashboard

    dashboard = CostDashboard()
    report = dashboard.generate(collector, team="IdeationTeam", mode="ModeNoWcNoHITL")
    dashboard.save(report, "cost_report.json")
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CostBreakdown:
    """Cost breakdown for a single grouping (agent, stage, model, tool)."""

    name: str
    llm_calls: int = 0
    llm_cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    embedding_calls: int = 0
    embedding_cost_usd: float = 0.0

    @property
    def total_cost_usd(self) -> float:
        return self.llm_cost_usd + self.embedding_cost_usd

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "llm_calls": self.llm_calls,
            "llm_cost_usd": round(self.llm_cost_usd, 6),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
            "embedding_calls": self.embedding_calls,
            "embedding_cost_usd": round(self.embedding_cost_usd, 6),
            "total_cost_usd": round(self.total_cost_usd, 6),
        }


@dataclass
class CostReport:
    """Complete cost report for a workflow run."""

    team: str
    mode: str
    total_cost_usd: float = 0.0
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    total_embedding_calls: int = 0
    total_tokens: int = 0
    by_agent: List[CostBreakdown] = field(default_factory=list)
    by_stage: List[CostBreakdown] = field(default_factory=list)
    by_model: List[CostBreakdown] = field(default_factory=list)
    by_provider: List[CostBreakdown] = field(default_factory=list)
    by_tool: List[CostBreakdown] = field(default_factory=list)
    budget_utilization: Optional[Dict[str, Any]] = None
    cache_savings: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "team": self.team,
            "mode": self.mode,
            "summary": {
                "total_cost_usd": round(self.total_cost_usd, 6),
                "total_llm_calls": self.total_llm_calls,
                "total_tool_calls": self.total_tool_calls,
                "total_embedding_calls": self.total_embedding_calls,
                "total_tokens": self.total_tokens,
            },
            "by_agent": [b.to_dict() for b in self.by_agent],
            "by_stage": [b.to_dict() for b in self.by_stage],
            "by_model": [b.to_dict() for b in self.by_model],
            "by_provider": [b.to_dict() for b in self.by_provider],
            "by_tool": [b.to_dict() for b in self.by_tool],
        }
        if self.budget_utilization:
            d["budget_utilization"] = self.budget_utilization
        if self.cache_savings:
            d["cache_savings"] = self.cache_savings
        return d


class CostDashboard:
    """
    Generates per-team cost reports from MetricsCollector data.

    Reads the detailed records from a MetricsCollector and produces
    breakdowns by agent, stage, model, and tool.
    """

    def generate(
        self,
        collector: Any,
        team: str = "",
        mode: str = "",
        budget_limits: Optional[Dict[str, float]] = None,
        cache_stats: Optional[Dict[str, Any]] = None,
    ) -> CostReport:
        """
        Generate a cost report from collector data.

        Args:
            collector: MetricsCollector instance.
            team: Team name.
            mode: Mode name.
            budget_limits: Optional budget limits for utilization calc.
            cache_stats: Optional cache stats from SemanticCache.

        Returns:
            CostReport with full breakdowns.
        """
        records = collector.get_detailed_records()
        llm_calls = records.get("llm_calls", [])
        tool_calls = records.get("tool_calls", [])
        embedding_calls = records.get("embedding_calls", [])

        # Totals
        total_llm_cost = sum(r.get("cost_usd", 0) for r in llm_calls)
        total_embed_cost = sum(r.get("cost_usd", 0) for r in embedding_calls)
        total_tokens = sum(r.get("total_tokens", 0) for r in llm_calls)

        report = CostReport(
            team=team,
            mode=mode,
            total_cost_usd=total_llm_cost + total_embed_cost,
            total_llm_calls=len(llm_calls),
            total_tool_calls=len(tool_calls),
            total_embedding_calls=len(embedding_calls),
            total_tokens=total_tokens,
        )

        # Group by agent
        report.by_agent = self._group_breakdowns(
            llm_calls, tool_calls, embedding_calls, key="agent"
        )

        # Group by stage
        report.by_stage = self._group_breakdowns(
            llm_calls, tool_calls, embedding_calls, key="stage"
        )

        # Group by model
        report.by_model = self._group_llm_by_key(llm_calls, key="model")

        # Group by provider (derived from model name)
        report.by_provider = self._group_llm_by_provider(llm_calls)

        # Group by tool
        report.by_tool = self._group_tools_by_name(tool_calls)

        # Budget utilization
        if budget_limits:
            max_cost = budget_limits.get("max_cost_usd", 0)
            report.budget_utilization = {
                "max_cost_usd": max_cost,
                "current_cost_usd": round(report.total_cost_usd, 6),
                "utilization_pct": round(
                    (report.total_cost_usd / max_cost * 100) if max_cost > 0 else 0, 2
                ),
                "remaining_usd": round(max(0, max_cost - report.total_cost_usd), 6),
            }

        # Cache savings
        if cache_stats:
            report.cache_savings = cache_stats

        return report

    def save(self, report: CostReport, output_path: str):
        """Save a cost report to JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report.to_dict(), indent=2))

    @staticmethod
    def _group_breakdowns(
        llm_calls: List[Dict],
        tool_calls: List[Dict],
        embedding_calls: List[Dict],
        key: str,
    ) -> List[CostBreakdown]:
        """Group all call types by the specified key."""
        groups: Dict[str, CostBreakdown] = {}

        for r in llm_calls:
            name = r.get(key, "unknown") or "unknown"
            if name not in groups:
                groups[name] = CostBreakdown(name=name)
            b = groups[name]
            b.llm_calls += 1
            b.llm_cost_usd += r.get("cost_usd", 0)
            b.prompt_tokens += r.get("prompt_tokens", 0)
            b.completion_tokens += r.get("completion_tokens", 0)

        for r in tool_calls:
            name = r.get(key, "unknown") or "unknown"
            if name not in groups:
                groups[name] = CostBreakdown(name=name)
            b = groups[name]
            b.tool_calls += 1
            if not r.get("success", True):
                b.tool_errors += 1

        for r in embedding_calls:
            name = r.get(key, "unknown") or "unknown"
            if name not in groups:
                groups[name] = CostBreakdown(name=name)
            b = groups[name]
            b.embedding_calls += 1
            b.embedding_cost_usd += r.get("cost_usd", 0)

        return sorted(groups.values(), key=lambda b: b.total_cost_usd, reverse=True)

    @staticmethod
    def _group_llm_by_key(
        llm_calls: List[Dict], key: str
    ) -> List[CostBreakdown]:
        """Group LLM calls by a specific key (e.g., model)."""
        groups: Dict[str, CostBreakdown] = {}
        for r in llm_calls:
            name = r.get(key, "unknown") or "unknown"
            if name not in groups:
                groups[name] = CostBreakdown(name=name)
            b = groups[name]
            b.llm_calls += 1
            b.llm_cost_usd += r.get("cost_usd", 0)
            b.prompt_tokens += r.get("prompt_tokens", 0)
            b.completion_tokens += r.get("completion_tokens", 0)
        return sorted(groups.values(), key=lambda b: b.llm_cost_usd, reverse=True)

    @staticmethod
    def _group_llm_by_provider(llm_calls: List[Dict]) -> List[CostBreakdown]:
        """Group LLM calls by provider (derived from model name)."""
        from shared.llm import detect_provider

        groups: Dict[str, CostBreakdown] = {}
        for r in llm_calls:
            model = r.get("model", "unknown") or "unknown"
            provider = detect_provider(model)
            if provider not in groups:
                groups[provider] = CostBreakdown(name=provider)
            b = groups[provider]
            b.llm_calls += 1
            b.llm_cost_usd += r.get("cost_usd", 0)
            b.prompt_tokens += r.get("prompt_tokens", 0)
            b.completion_tokens += r.get("completion_tokens", 0)
        return sorted(groups.values(), key=lambda b: b.llm_cost_usd, reverse=True)

    @staticmethod
    def _group_tools_by_name(tool_calls: List[Dict]) -> List[CostBreakdown]:
        """Group tool calls by tool name."""
        groups: Dict[str, CostBreakdown] = {}
        for r in tool_calls:
            name = r.get("tool_name", "unknown") or "unknown"
            if name not in groups:
                groups[name] = CostBreakdown(name=name)
            b = groups[name]
            b.tool_calls += 1
            if not r.get("success", True):
                b.tool_errors += 1
        return sorted(groups.values(), key=lambda b: b.tool_calls, reverse=True)
