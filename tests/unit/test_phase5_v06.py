# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
AEL V0.6 Phase 5 — Smart Orchestration & Efficiency Tests

Tests for ToolRegistry search/lazy_load, LearnedRouter, AdaptiveOrchestrator,
model catalog updates, per-team budget profiles, and routing strategies.
"""

import os
from unittest.mock import patch

import pytest


# ── ToolRegistry Extensions ──────────────────────────────────────────────


class TestToolRegistrySearch:
    def setup_method(self):
        from shared.tools.tool_registry import ToolRegistry
        from pydantic import BaseModel

        self._saved_tools = dict(ToolRegistry._tools)
        ToolRegistry.clear()

        class DummyInput(BaseModel):
            query: str = ""

        ToolRegistry.register(
            "arxiv_search", "Search academic papers on arxiv",
            DummyInput, lambda **kw: [], category="search"
        )
        ToolRegistry.register(
            "fred_data", "Fetch FRED economic data series",
            DummyInput, lambda **kw: [], category="data"
        )
        ToolRegistry.register(
            "web_scrape", "Scrape web pages for content",
            DummyInput, lambda **kw: [], category="web"
        )
        ToolRegistry.register(
            "compute_stats", "Compute statistical summaries",
            DummyInput, lambda **kw: [], category="compute"
        )

    def teardown_method(self):
        from shared.tools.tool_registry import ToolRegistry
        ToolRegistry._tools = self._saved_tools

    def test_search_finds_relevant_tools(self):
        from shared.tools.tool_registry import ToolRegistry
        results = ToolRegistry.search("search academic papers")
        assert len(results) >= 1
        assert results[0].name == "arxiv_search"

    def test_search_data_tools(self):
        from shared.tools.tool_registry import ToolRegistry
        results = ToolRegistry.search("economic data FRED")
        assert any(t.name == "fred_data" for t in results)

    def test_search_empty_query(self):
        from shared.tools.tool_registry import ToolRegistry
        assert ToolRegistry.search("") == []

    def test_search_no_matches(self):
        from shared.tools.tool_registry import ToolRegistry
        results = ToolRegistry.search("quantum physics entanglement")
        # May return some tools but not highly relevant
        assert isinstance(results, list)

    def test_search_max_tools(self):
        from shared.tools.tool_registry import ToolRegistry
        results = ToolRegistry.search("search data", max_tools=2)
        assert len(results) <= 2

    def test_get_stage_tools_ideation(self):
        from shared.tools.tool_registry import ToolRegistry
        tools = ToolRegistry.get_stage_tools("IdeationTeam", "SourcingStage")
        # IdeationTeam maps to search + web categories
        names = [t.name for t in tools]
        assert "arxiv_search" in names or "web_scrape" in names

    def test_get_stage_tools_data_team(self):
        from shared.tools.tool_registry import ToolRegistry
        tools = ToolRegistry.get_stage_tools("DataTeam", "DataSourceStage")
        names = [t.name for t in tools]
        assert "fred_data" in names

    def test_get_stage_tools_with_task(self):
        from shared.tools.tool_registry import ToolRegistry
        tools = ToolRegistry.get_stage_tools(
            "LiteratureTeam", "GatheringStage",
            task_description="search for papers"
        )
        assert len(tools) >= 1

    def test_lazy_load_existing(self):
        from shared.tools.tool_registry import ToolRegistry
        tool = ToolRegistry.lazy_load("arxiv_search")
        assert tool is not None
        assert tool.name == "arxiv_search"

    def test_lazy_load_nonexistent(self):
        from shared.tools.tool_registry import ToolRegistry
        assert ToolRegistry.lazy_load("nonexistent_tool") is None


# ── Model Catalog Updates ────────────────────────────────────────────────


class TestModelCatalogV06:
    def test_new_models_in_catalog(self):
        from shared.llm_router import MODEL_CATALOG
        new_models = [
            "gemini-3.1-pro", "gemini-3-flash", "deepseek-v3.2",
            "qwen3.5-plus", "mistral-medium-3", "mistral-small-3.2", "qwen3-8b",
        ]
        for model in new_models:
            assert model in MODEL_CATALOG, f"{model} not in catalog"

    def test_gpt54_already_existed(self):
        from shared.llm_router import MODEL_CATALOG
        assert "gpt-5.4" in MODEL_CATALOG
        assert MODEL_CATALOG["gpt-5.4"][0] == "openai"

    def test_gemini31_pro_pricing(self):
        from shared.llm_router import MODEL_CATALOG
        entry = MODEL_CATALOG["gemini-3.1-pro"]
        assert entry[0] == "google"
        assert entry[1] == 2.00  # input price
        assert entry[2] == 12.00  # output price

    def test_budget_models_cheap(self):
        from shared.llm_router import MODEL_CATALOG
        for model in ["qwen3-8b", "mistral-small-3.2", "deepseek-v3.2"]:
            assert MODEL_CATALOG[model][1] < 0.50  # input < $0.50/M

    def test_new_routing_strategies(self):
        from shared.llm_router import DEFAULT_ROUTES
        assert "tool_heavy" in DEFAULT_ROUTES
        assert "reasoning_heavy" in DEFAULT_ROUTES
        assert "budget_extraction" in DEFAULT_ROUTES

    def test_tool_heavy_route(self):
        from shared.llm_router import DEFAULT_ROUTES
        assert DEFAULT_ROUTES["tool_heavy"][0] == "gpt-5.4"

    def test_reasoning_heavy_route(self):
        from shared.llm_router import DEFAULT_ROUTES
        assert DEFAULT_ROUTES["reasoning_heavy"][0] == "gemini-3.1-pro"

    def test_total_catalog_size(self):
        from shared.llm_router import MODEL_CATALOG
        assert len(MODEL_CATALOG) >= 28  # 21 original + 7 new


# ── Learned Router ───────────────────────────────────────────────────────


class TestLearnedRouter:
    def test_module_imports(self):
        from shared.llm_router_learned import (
            LearnedRouter,
            TEAM_BUDGET_PROFILES,
            get_team_budget,
        )

    def test_complexity_low(self):
        from shared.llm_router_learned import LearnedRouter
        router = LearnedRouter()
        assert router.get_complexity_estimate("extract the data", "extraction") == "low"

    def test_complexity_medium(self):
        from shared.llm_router_learned import LearnedRouter
        router = LearnedRouter()
        c = router.get_complexity_estimate("summarize the key findings", "summarization")
        assert c == "medium"

    def test_complexity_high(self):
        from shared.llm_router_learned import LearnedRouter
        router = LearnedRouter()
        c = router.get_complexity_estimate("analyze and evaluate the complex multi-step reasoning", "reasoning")
        assert c == "high"

    def test_route_returns_model(self):
        from shared.llm_router_learned import LearnedRouter
        router = LearnedRouter()
        model = router.route("Extract entity names from text", task_type="extraction")
        assert isinstance(model, str)
        assert len(model) > 0

    def test_route_budget_mode(self):
        from shared.llm_router_learned import LearnedRouter
        from shared.llm_router import MODEL_CATALOG
        router = LearnedRouter()
        model = router.route("Do something", budget_mode=True)
        # Budget mode should select cheap models
        if model in MODEL_CATALOG:
            assert MODEL_CATALOG[model][1] <= 1.0  # input < $1/M

    def test_long_prompt_upgrades_complexity(self):
        from shared.llm_router_learned import LearnedRouter
        router = LearnedRouter()
        long_prompt = "word " * 120  # > 500 chars
        c = router.get_complexity_estimate(long_prompt, "extraction")
        assert c in ("medium", "high")  # Upgraded from "low"

    def test_empty_prompt(self):
        from shared.llm_router_learned import LearnedRouter
        router = LearnedRouter()
        c = router.get_complexity_estimate("", "reasoning")
        assert c == "high"  # Falls back to task type default

    def test_no_eval_data(self):
        from shared.llm_router_learned import LearnedRouter
        router = LearnedRouter(eval_data_path="/nonexistent/path.json")
        # Should not crash, falls back to static routing
        model = router.route("test", task_type="default")
        assert isinstance(model, str)


# ── Per-Team Budget Profiles ─────────────────────────────────────────────


class TestTeamBudgetProfiles:
    def test_all_teams_have_profiles(self):
        from shared.llm_router_learned import TEAM_BUDGET_PROFILES
        for team in ["IdeationTeam", "LiteratureTeam", "DataTeam", "ModelTeam"]:
            assert team in TEAM_BUDGET_PROFILES

    def test_profile_structure(self):
        from shared.llm_router_learned import TEAM_BUDGET_PROFILES
        for team, profile in TEAM_BUDGET_PROFILES.items():
            assert "max_cost" in profile
            assert "max_tokens" in profile
            assert "stage_timeouts" in profile
            assert len(profile["stage_timeouts"]) == 3

    def test_literature_team_highest_budget(self):
        from shared.llm_router_learned import TEAM_BUDGET_PROFILES
        lit = TEAM_BUDGET_PROFILES["LiteratureTeam"]
        idea = TEAM_BUDGET_PROFILES["IdeationTeam"]
        assert lit["max_cost"] > idea["max_cost"]
        assert lit["max_tokens"] > idea["max_tokens"]

    def test_get_team_budget_known(self):
        from shared.llm_router_learned import get_team_budget
        budget = get_team_budget("IdeationTeam")
        assert budget["max_cost"] == 1.50
        assert budget["max_tokens"] == 300_000

    def test_get_team_budget_unknown(self):
        from shared.llm_router_learned import get_team_budget
        budget = get_team_budget("UnknownTeam")
        assert "max_cost" in budget
        assert "max_tokens" in budget


# ── Adaptive Orchestrator ────────────────────────────────────────────────


class TestAdaptiveOrchestrator:
    def test_module_imports(self):
        from shared.orchestration.adaptive import (
            AdaptiveOrchestrator,
            ExecutionPlan,
            ExecutionStep,
            TEAM_DEPENDENCIES,
        )

    def test_topology_single_team(self):
        from shared.orchestration.adaptive import AdaptiveOrchestrator
        orch = AdaptiveOrchestrator()
        assert orch.select_topology(["IdeationTeam"]) == "sequential"

    def test_topology_independent_teams(self):
        from shared.orchestration.adaptive import AdaptiveOrchestrator
        orch = AdaptiveOrchestrator()
        topo = orch.select_topology(["IdeationTeam", "DataTeam"])
        assert topo == "parallel"

    def test_topology_dependent_chain(self):
        from shared.orchestration.adaptive import AdaptiveOrchestrator
        orch = AdaptiveOrchestrator()
        topo = orch.select_topology(["LiteratureTeam", "ModelTeam"])
        # LiteratureTeam depends on IdeationTeam (not in set)
        # ModelTeam depends on LiteratureTeam (in set)
        assert topo in ("sequential", "hierarchical")

    def test_topology_hybrid(self):
        from shared.orchestration.adaptive import AdaptiveOrchestrator
        orch = AdaptiveOrchestrator()
        topo = orch.select_topology(["IdeationTeam", "DataTeam", "ModelTeam"])
        # IdeationTeam and DataTeam independent, ModelTeam depends on both
        assert topo == "hybrid"

    def test_independent_teams(self):
        from shared.orchestration.adaptive import AdaptiveOrchestrator
        orch = AdaptiveOrchestrator()
        independent = orch.get_independent_teams(
            ["IdeationTeam", "LiteratureTeam", "DataTeam"]
        )
        assert "IdeationTeam" in independent
        assert "DataTeam" in independent

    def test_execution_order(self):
        from shared.orchestration.adaptive import AdaptiveOrchestrator
        orch = AdaptiveOrchestrator()
        order = orch.get_execution_order(
            ["IdeationTeam", "LiteratureTeam", "DataTeam", "ModelTeam"]
        )
        assert len(order) >= 2
        # First group should contain independent teams
        first_group = order[0]
        assert "IdeationTeam" in first_group or "DataTeam" in first_group
        # ModelTeam should be last (depends on all others)
        last_group = order[-1]
        assert "ModelTeam" in last_group

    def test_create_execution_plan(self):
        from shared.orchestration.adaptive import AdaptiveOrchestrator
        orch = AdaptiveOrchestrator()
        plan = orch.create_execution_plan(
            ["IdeationTeam", "DataTeam", "LiteratureTeam", "ModelTeam"]
        )
        assert plan.topology in AdaptiveOrchestrator.TOPOLOGIES
        assert plan.total_steps >= 2
        assert len(plan.steps) == plan.total_steps

    def test_plan_parallel_group_count(self):
        from shared.orchestration.adaptive import AdaptiveOrchestrator
        orch = AdaptiveOrchestrator()
        plan = orch.create_execution_plan(["IdeationTeam", "DataTeam"])
        # Both independent → one parallel step
        assert plan.parallel_groups >= 1

    def test_custom_dependencies(self):
        from shared.orchestration.adaptive import AdaptiveOrchestrator
        custom = {
            "A": [],
            "B": [],
            "C": ["A", "B"],
        }
        orch = AdaptiveOrchestrator(dependencies=custom)
        topo = orch.select_topology(["A", "B", "C"])
        assert topo == "hybrid"

    def test_all_topologies_valid(self):
        from shared.orchestration.adaptive import AdaptiveOrchestrator
        assert len(AdaptiveOrchestrator.TOPOLOGIES) == 4
        assert "sequential" in AdaptiveOrchestrator.TOPOLOGIES
        assert "parallel" in AdaptiveOrchestrator.TOPOLOGIES


# ── Backward Compatibility ───────────────────────────────────────────────


class TestPhase5BackwardCompat:
    def test_tool_registry_original_api_intact(self):
        from shared.tools.tool_registry import ToolRegistry
        assert hasattr(ToolRegistry, "register")
        assert hasattr(ToolRegistry, "invoke")
        assert hasattr(ToolRegistry, "list_tools")
        assert hasattr(ToolRegistry, "get_tool")

    def test_model_router_original_api_intact(self):
        from shared.llm_router import ModelRouter
        router = ModelRouter()
        assert hasattr(router, "select_model")
        assert hasattr(router, "get_provider")
        assert hasattr(router, "get_pricing")

    def test_existing_routes_preserved(self):
        from shared.llm_router import DEFAULT_ROUTES
        for task in ["extraction", "reasoning", "summarization", "default"]:
            assert task in DEFAULT_ROUTES
