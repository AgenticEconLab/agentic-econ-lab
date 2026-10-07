# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Integration tests for cross-component verification.

Tests that V0.4 components work together correctly:
1. Schema → Guardrails → MasterOrchestrator flow
2. Memory → RAG → LiteratureTeam flow
3. ToolRegistry → Stage file → MetricsCollector flow
4. Pipeline → Team runners → Artifact flow
5. Evaluation → Consensus → Trajectory flow
6. Telemetry → OTel → CostDashboard flow
7. End-to-end fixture loading and evaluation
8. Backward compatibility (all 15 configs)
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest

# repository root for file path lookups (sys.path handled by conftest.py)
_agents_dir = Path(__file__).resolve().parent.parent.parent
_fixtures_dir = Path(__file__).resolve().parent.parent / "fixtures"


def _load_fixture(name: str) -> Dict[str, Any]:
    """Load a JSON fixture file."""
    path = _fixtures_dir / name
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# Test 1: Schema → Guardrails flow
# ============================================================================

class TestSchemaGuardrailsFlow:
    """Schema validation through guardrails pipeline."""

    def test_schema_validator_accepts_valid_output(self):
        """SchemaValidator accepts a correctly-shaped stage output."""
        from shared.guardrails.schema_validator import SchemaValidator

        result = SchemaValidator.validate(
            "IdeationTeam", "SourcingStage",
            {"literature_items": [{
                "title": "AI in Economics", "abstract": "Summary",
                "url": "http://example.com", "year": 2025,
                "source": "arxiv", "authors": ["Smith"], "agent": "CorpusScout",
            }]},
        )
        assert result.valid

    def test_budget_controller_tracks_cost(self):
        """BudgetController initializes with limits."""
        from shared.guardrails.budget_controller import BudgetController

        bc = BudgetController(max_cost_usd=1.0)
        from shared.observability import MetricsCollector
        status = bc.check(MetricsCollector())
        assert status.ok

    def test_pii_detector_scans_text(self):
        """PIIDetector detects PII in text."""
        from shared.guardrails.pii_detector import PIIDetector

        detector = PIIDetector()
        result = detector.scan("Contact john@example.com for details")
        assert result.has_pii

    def test_pii_detector_clean_text(self):
        """PIIDetector passes clean text."""
        from shared.guardrails.pii_detector import PIIDetector

        detector = PIIDetector()
        result = detector.scan("GDP growth was 3.2% in Q4 2025")
        assert not result.has_pii


# ============================================================================
# Test 2: Memory → RAG flow
# ============================================================================

class TestMemoryRAGFlow:
    """Memory and RAG components work together."""

    def test_memory_manager_lifecycle(self):
        """MemoryManager set_context → remember → recall → close."""
        from shared.memory.memory_manager import MemoryManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mm = MemoryManager(memory_dir=tmpdir)
            mm.set_context("topic", "AI economics research")

            mm.remember_fact(
                "Found 25 papers on AI labor markets",
                category="literature",
                source="test",
            )

            facts = mm.recall_facts("AI papers", category="literature")
            assert len(facts) >= 1
            mm.close()

    def test_document_store_lifecycle(self):
        """DocumentStore can be created and closed."""
        from shared.rag.document_store import DocumentStore, Document

        with tempfile.TemporaryDirectory() as tmpdir:
            ds = DocumentStore(store_dir=tmpdir)
            doc = Document(
                doc_id="paper1",
                content="AI affects labor market polarization",
                metadata={"source": "test"},
            )
            count = ds.add_documents([doc], compute_embeddings=False)
            assert count >= 1
            ds.close()

    def test_grounding_checker(self):
        """GroundingChecker can be instantiated and check_citations called."""
        from shared.guardrails.grounding_checker import GroundingChecker

        checker = GroundingChecker()
        report = checker.check_citations([])
        assert hasattr(report, "total")
        assert report.total == 0


# ============================================================================
# Test 3: ToolRegistry → MetricsCollector flow
# ============================================================================

class TestToolRegistryMetricsFlow:
    """ToolRegistry invocations flow through to MetricsCollector."""

    def test_tool_invoke_records_metrics(self):
        """ToolRegistry.invoke() records tool call in MetricsCollector."""
        from shared.tools.tool_registry import ToolRegistry
        from shared.tools.register_all import register_all_tools
        from shared.observability import MetricsCollector
        import shared.tools.register_all as reg_mod

        ToolRegistry.clear()
        reg_mod._registered = False
        register_all_tools()

        collector = MetricsCollector()
        result = ToolRegistry.invoke(
            "arxiv_search",
            {"query": "test", "max_results": 1},
            collector=collector,
            agent="IntegrationTest",
        )

        summary = collector.get_summary()
        tool_calls = summary.get("tools", {}).get("total_calls", 0)
        assert tool_calls >= 1 or not result.success

    def test_tool_registry_lists_by_category(self):
        """ToolRegistry.list_tools() filters by category."""
        from shared.tools.tool_registry import ToolRegistry

        search_tools = ToolRegistry.list_tools(category="search")
        data_tools = ToolRegistry.list_tools(category="data")
        web_tools = ToolRegistry.list_tools(category="web")

        assert any(t.name == "arxiv_search" for t in search_tools)
        assert any(t.name == "fred_get_series" for t in data_tools)
        assert any(t.name == "web_fetch" for t in web_tools)


# ============================================================================
# Test 4: Pipeline config loading
# ============================================================================

class TestPipelineConfigFlow:
    """Pipeline configuration loading and team runner setup."""

    def test_load_yaml_config(self):
        """Pipeline config loads from YAML fixture."""
        import yaml

        config_path = _fixtures_dir / "sample_pipeline_config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        assert config["name"] == "test_pipeline"
        assert len(config["stages"]) == 4
        assert config["budget"]["max_total_cost_usd"] == 5.0

    def test_team_runners_registered(self):
        """All 4 team runners can be imported."""
        from pipeline.team_runners import (
            run_ideation_team,
            run_literature_team,
            run_model_team,
            run_data_team,
        )
        assert callable(run_ideation_team)
        assert callable(run_literature_team)
        assert callable(run_model_team)
        assert callable(run_data_team)

    def test_pipeline_orchestrator_importable(self):
        """ResearchPipelineOrchestrator can be imported."""
        from pipeline.research_pipeline import ResearchPipelineOrchestrator
        orch = ResearchPipelineOrchestrator.__new__(ResearchPipelineOrchestrator)
        assert orch is not None


# ============================================================================
# Test 5: Evaluation → Consensus → Trajectory flow
# ============================================================================

class TestEvaluationFlow:
    """Evaluation components work together."""

    def test_tier1_metrics_from_fixture(self):
        """Tier 1 structural metrics compute from fixture log."""
        from evaluation.run_ael_evaluation import calculate_tier1_metrics

        log = _load_fixture("sample_execution_log.json")
        metrics = calculate_tier1_metrics(log)

        assert metrics["reliability"] > 0
        assert metrics["stage_completion_rate"] == 1.0
        assert metrics["soundness"] == 1.0
        assert metrics["efficiency"] > 0

    def test_trajectory_from_fixture(self):
        """TrajectoryEvaluator scores fixture log."""
        from evaluation.trajectory.trajectory_evaluator import TrajectoryEvaluator

        log = _load_fixture("sample_execution_log.json")
        evaluator = TrajectoryEvaluator()
        score = evaluator.evaluate_trajectory(log)

        assert score.team == "IdeationTeam"
        assert score.overall_score > 0
        assert len(score.decision_points) == 3

    def test_benchmark_with_fixture_output(self):
        """BenchmarkComparator evaluates fixture output."""
        from evaluation.benchmarks.econ_benchmarks import (
            BenchmarkComparator, get_team_benchmark,
        )

        outputs = _load_fixture("sample_ideation_output.json")
        benchmark = get_team_benchmark("IdeationTeam")
        comparator = BenchmarkComparator()
        result = comparator.compare(
            {"finalized_research_questions_automated.json": outputs},
            benchmark,
        )
        assert 0 <= result.relative_quality <= 1

    def test_combine_scores(self):
        """combine_scores produces 12 dimension scores."""
        from evaluation.run_ael_evaluation import calculate_tier1_metrics, combine_scores

        log = _load_fixture("sample_execution_log.json")
        tier1 = calculate_tier1_metrics(log)
        combined = combine_scores(tier1, {})

        dimensions = [
            "reliability", "correctness", "soundness",
            "efficiency", "scalability", "robustness",
            "transparency", "traceability", "reproducibility",
            "innovation_potential", "decision_quality", "economic_rigor",
        ]
        for dim in dimensions:
            assert dim in combined, f"Missing dimension: {dim}"


# ============================================================================
# Test 6: Telemetry flow
# ============================================================================

class TestTelemetryFlow:
    """OTel and CostDashboard produce valid output."""

    def test_full_telemetry_export(self):
        """export_telemetry produces traces.json and cost_report.json."""
        from shared.telemetry import export_telemetry
        from shared.observability import MetricsCollector

        collector = MetricsCollector()

        with tempfile.TemporaryDirectory() as tmpdir:
            export_telemetry(
                collector, tmpdir,
                team="IdeationTeam", mode="ModeNoWcNoHITL",
            )

            traces_path = os.path.join(tmpdir, "traces.json")
            cost_path = os.path.join(tmpdir, "cost_report.json")

            assert os.path.exists(traces_path)
            assert os.path.exists(cost_path)

            with open(traces_path) as f:
                traces = json.load(f)
            assert isinstance(traces, dict)
            assert "spans" in traces

            with open(cost_path) as f:
                cost = json.load(f)
            assert cost["team"] == "IdeationTeam"

    def test_otel_span_attributes(self):
        """OTel spans contain GenAI semantic attributes."""
        from shared.telemetry.gen_ai_conventions import GenAIAttributes

        assert GenAIAttributes.AGENT_NAME == "gen_ai.agent.name"
        assert GenAIAttributes.REQUEST_MODEL == "gen_ai.request.model"
        assert GenAIAttributes.USAGE_INPUT_TOKENS == "gen_ai.usage.input_tokens"


# ============================================================================
# Test 7: Fixture loading
# ============================================================================

class TestFixtures:
    """Verify all fixture files load correctly."""

    def test_execution_log_fixture(self):
        log = _load_fixture("sample_execution_log.json")
        assert log["team"] == "IdeationTeam"
        assert len(log["stages"]) == 3

    def test_ideation_output_fixture(self):
        output = _load_fixture("sample_ideation_output.json")
        assert len(output["research_questions"]) == 3
        assert output["research_questions"][0]["priority_score"] > 0

    def test_llm_response_fixture(self):
        response = _load_fixture("sample_llm_response.json")
        assert response["model"] == "gpt-4o-mini"
        assert response["total_tokens"] == 2000

    def test_pipeline_config_fixture(self):
        import yaml
        config_path = _fixtures_dir / "sample_pipeline_config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert config["name"] == "test_pipeline"


# ============================================================================
# Test 8: Backward compatibility (all 15 configs)
# ============================================================================

class TestBackwardCompatibility:
    """All 15 configurations remain importable and structurally correct."""

    def _get_all_configs(self):
        configs = []
        for team in ["IdeationTeam", "LiteratureTeam", "ModelTeam"]:
            for mode in ["ModeNoWcNoHITL", "ModeNoWcWithHITL",
                          "ModeWithWcNoHITL", "ModeWithWcWithHITL"]:
                configs.append((team, mode))
        for mode in ["ModeOpenSourceAPI", "ModePremiumSubscribed", "ModeUserUploaded"]:
            configs.append(("DataTeam", mode))
        return configs

    def test_15_orchestrator_files_exist(self):
        configs = self._get_all_configs()
        for team, mode in configs:
            path = _agents_dir / team / "ael" / mode / "0-MasterOrchestrator.py"
            assert path.exists(), f"Missing: {team}/{mode}"

    def test_15_stage_directories_exist(self):
        configs = self._get_all_configs()
        for team, mode in configs:
            path = _agents_dir / team / "ael" / mode
            assert path.is_dir(), f"Missing dir: {team}/{mode}"
            # Should have at least 4 Python files (orchestrator + 3 stages)
            py_files = list(path.glob("*.py"))
            assert len(py_files) >= 4, f"{team}/{mode} has {len(py_files)} .py files"

    def test_schema_modules_importable(self):
        for team in ["IdeationTeam", "LiteratureTeam", "ModelTeam", "DataTeam"]:
            mod_path = f"{team}.ael.schemas.stage_outputs"
            __import__(mod_path)

    def test_shared_modules_importable(self):
        """All shared modules can be imported."""
        import shared.llm
        import shared.observability
        import shared.instrumentation
        import shared.guardrails.schema_validator
        import shared.guardrails.budget_controller
        import shared.guardrails.pii_detector
        import shared.guardrails.grounding_checker
        import shared.tools.tool_registry
        import shared.memory.memory_manager
        import shared.rag.document_store
        import shared.rag.hybrid_retriever
        import shared.telemetry


# ============================================================================
# Test 9: NF requirement baselines
# ============================================================================

class TestNFRequirements:
    """Non-functional requirement measurement baselines."""

    def test_nf7_test_count_above_600(self):
        """NF7: Unit + integration tests combined > 600."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q",
             str(_agents_dir / "tests")],
            capture_output=True, text=True, cwd=str(_agents_dir),
        )
        # Parse "N tests collected" from output
        lines = result.stdout.strip().split("\n")
        last_line = lines[-1] if lines else ""
        # Extract number from "N tests collected" or "N test"
        import re
        match = re.search(r"(\d+)\s+tests?\s", last_line)
        if match:
            count = int(match.group(1))
        else:
            # Fallback: count lines with "::" in them
            count = sum(1 for line in lines if "::" in line)
        assert count >= 600, f"Only {count} tests collected; need >= 600"

    def test_nf6_backward_compat_all_configs(self):
        """NF6: All 15 configs have orchestrator + 3 stages."""
        configs = []
        for team in ["IdeationTeam", "LiteratureTeam", "ModelTeam"]:
            for mode in ["ModeNoWcNoHITL", "ModeNoWcWithHITL",
                          "ModeWithWcNoHITL", "ModeWithWcWithHITL"]:
                configs.append((team, mode))
        for mode in ["ModeOpenSourceAPI", "ModePremiumSubscribed", "ModeUserUploaded"]:
            configs.append(("DataTeam", mode))

        for team, mode in configs:
            base = _agents_dir / team / "ael" / mode
            assert (base / "0-MasterOrchestrator.py").exists()
            py_files = [f.name for f in base.glob("[1-3]-*.py")]
            assert len(py_files) >= 3, f"{team}/{mode} only has stages: {py_files}"
