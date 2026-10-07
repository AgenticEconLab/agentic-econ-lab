# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Tests for AEL V0.4 Phase 5: Evaluation Pipeline + Telemetry.

Verifies:
1. ConsensusEngine wired into run_ael_evaluation.py (--consensus flag)
2. TrajectoryEvaluator wired into run_ael_evaluation.py (--trajectory flag)
3. AdversarialReviewer wired into run_ael_evaluation.py (--adversarial flag)
4. BenchmarkComparator wired into run_ael_evaluation.py (--benchmark flag)
5. OTel span export at MasterOrchestrator completion
6. CostDashboard report at end of each workflow run
7. Functional tests for each component
"""

import ast
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# repository root for file path lookups (sys.path handled by conftest.py)
_agents_dir = Path(__file__).resolve().parent.parent.parent


# ============================================================================
# Helpers
# ============================================================================

def _get_orchestrator_files():
    """Return all 15 MasterOrchestrator files."""
    base = _agents_dir
    orchestrators = []
    for team in ["IdeationTeam", "LiteratureTeam", "ModelTeam"]:
        for mode in ["ModeNoWcNoHITL", "ModeNoWcWithHITL",
                      "ModeWithWcNoHITL", "ModeWithWcWithHITL"]:
            f = base / team / "ael" / mode / "0-MasterOrchestrator.py"
            if f.exists():
                orchestrators.append(f)
    for mode in ["ModeOpenSourceAPI", "ModePremiumSubscribed", "ModeUserUploaded"]:
        f = base / "DataTeam" / "ael" / mode / "0-MasterOrchestrator.py"
        if f.exists():
            orchestrators.append(f)
    return orchestrators


def _file_contains_text(filepath, text):
    source = filepath.read_text(encoding="utf-8")
    return text in source


def _file_contains_import(filepath, module_path, name):
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(filepath))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module == module_path:
                for alias in node.names:
                    if alias.name == name:
                        return True
    return False


# Sample execution log for testing
_SAMPLE_LOG = {
    "execution_id": "test-001",
    "team": "IdeationTeam",
    "mode": "ModeNoWcNoHITL",
    "started_at": "2026-03-10T10:00:00",
    "completed_at": "2026-03-10T10:05:00",
    "total_duration_seconds": 300,
    "metadata": {"research_topic": "test topic"},
    "stages": [
        {
            "name": "SourcingStage",
            "number": 1,
            "status": "success",
            "item_count": 25,
            "output_files": ["sourcing_output.json"],
            "duration_seconds": 100,
        },
        {
            "name": "RefinementStage",
            "number": 2,
            "status": "success",
            "item_count": 10,
            "output_files": ["refinement_output.json"],
            "duration_seconds": 100,
        },
        {
            "name": "IntegrationStage",
            "number": 3,
            "status": "success",
            "item_count": 5,
            "output_files": ["integration_output.json"],
            "duration_seconds": 100,
        },
    ],
    "errors": [],
    "summary": {
        "total_stages": 3,
        "successful_stages": 3,
        "total_items": 40,
        "total_errors": 0,
    },
}


# ============================================================================
# Test 1: CLI flags exist in run_ael_evaluation.py
# ============================================================================

class TestCLIFlags:
    """Verify CLI flags are registered in run_ael_evaluation.py."""

    def _get_eval_script(self):
        return _agents_dir / "evaluation" / "run_ael_evaluation.py"

    def test_consensus_flag(self):
        assert _file_contains_text(self._get_eval_script(), "--consensus")

    def test_trajectory_flag(self):
        assert _file_contains_text(self._get_eval_script(), "--trajectory")

    def test_adversarial_flag(self):
        assert _file_contains_text(self._get_eval_script(), "--adversarial")

    def test_benchmark_flag(self):
        assert _file_contains_text(self._get_eval_script(), "--benchmark")


# ============================================================================
# Test 2: Runner functions exist
# ============================================================================

class TestRunnerFunctions:
    """Verify evaluation runner functions exist."""

    def _get_eval_script(self):
        return _agents_dir / "evaluation" / "run_ael_evaluation.py"

    def test_run_consensus_evaluation(self):
        assert _file_contains_text(
            self._get_eval_script(), "def run_consensus_evaluation("
        )

    def test_run_trajectory_evaluation(self):
        assert _file_contains_text(
            self._get_eval_script(), "def run_trajectory_evaluation("
        )

    def test_run_adversarial_evaluation(self):
        assert _file_contains_text(
            self._get_eval_script(), "def run_adversarial_evaluation("
        )

    def test_run_benchmark_evaluation(self):
        assert _file_contains_text(
            self._get_eval_script(), "def run_benchmark_evaluation("
        )


# ============================================================================
# Test 3: ConsensusEngine functional
# ============================================================================

class TestConsensusEngine:
    """Verify ConsensusEngine works."""

    def test_importable(self):
        from evaluation.consensus.consensus_engine import ConsensusEngine
        engine = ConsensusEngine()
        assert engine is not None

    def test_default_models(self):
        from evaluation.consensus.consensus_engine import ConsensusEngine
        engine = ConsensusEngine()
        assert len(engine.model_names) == 2
        # default judges are open-weight vLLM (commercial is fallback-only)
        assert all(m.startswith(("vllm/", "ollama/")) for m in engine.model_names)

    def test_calibration_method(self):
        from evaluation.consensus.consensus_engine import (
            ConsensusEngine, CalibrationMethod,
        )
        engine = ConsensusEngine(calibration=CalibrationMethod.Z_SCORE)
        assert engine.calibration == CalibrationMethod.Z_SCORE

    def test_custom_models(self):
        from evaluation.consensus.consensus_engine import ConsensusEngine
        engine = ConsensusEngine(model_names=["model-a", "model-b", "model-c"])
        assert len(engine.model_names) == 3

    def test_consensus_result_to_dict(self):
        from evaluation.consensus.consensus_engine import ConsensusResult
        cr = ConsensusResult(
            team="TestTeam",
            mode="TestMode",
            models=["m1"],
            calibration_method="mean_shift",
            raw_scores={},
            calibrated_scores={},
            consensus_scores={"correctness": 0.75},
            fleiss_kappa=0.5,
        )
        d = cr.to_dict()
        assert d["team"] == "TestTeam"
        assert d["consensus_scores"]["correctness"] == 0.75


# ============================================================================
# Test 4: TrajectoryEvaluator functional
# ============================================================================

class TestTrajectoryEvaluator:
    """Verify TrajectoryEvaluator works."""

    def test_importable(self):
        from evaluation.trajectory.trajectory_evaluator import TrajectoryEvaluator
        evaluator = TrajectoryEvaluator()
        assert evaluator is not None

    def test_structural_evaluation(self):
        """Structural evaluation (no LLM) should work."""
        from evaluation.trajectory.trajectory_evaluator import TrajectoryEvaluator
        evaluator = TrajectoryEvaluator()
        score = evaluator.evaluate_trajectory(_SAMPLE_LOG)
        assert 0 <= score.overall_score <= 1
        assert score.team == "IdeationTeam"
        assert score.mode == "ModeNoWcNoHITL"
        assert not score.llm_evaluated

    def test_decision_points_extracted(self):
        from evaluation.trajectory.trajectory_evaluator import TrajectoryEvaluator
        evaluator = TrajectoryEvaluator()
        score = evaluator.evaluate_trajectory(_SAMPLE_LOG)
        assert len(score.decision_points) == 3

    def test_score_dimensions(self):
        from evaluation.trajectory.trajectory_evaluator import TrajectoryEvaluator
        evaluator = TrajectoryEvaluator()
        score = evaluator.evaluate_trajectory(_SAMPLE_LOG)
        assert 0 <= score.decision_coherence <= 1
        assert 0 <= score.information_utilization <= 1
        assert 0 <= score.scope_management <= 1
        assert 0 <= score.stage_transition_quality <= 1

    def test_failed_stage_reduces_score(self):
        """A failed stage should reduce trajectory quality."""
        from evaluation.trajectory.trajectory_evaluator import TrajectoryEvaluator
        evaluator = TrajectoryEvaluator()

        failed_log = dict(_SAMPLE_LOG)
        failed_log["stages"] = list(_SAMPLE_LOG["stages"])
        failed_log["stages"][1] = dict(failed_log["stages"][1])
        failed_log["stages"][1]["status"] = "failed"
        failed_log["stages"][1]["output_files"] = []

        good_score = evaluator.evaluate_trajectory(_SAMPLE_LOG)
        bad_score = evaluator.evaluate_trajectory(failed_log)
        assert bad_score.overall_score <= good_score.overall_score


# ============================================================================
# Test 5: AdversarialReviewer functional
# ============================================================================

class TestAdversarialReviewer:
    """Verify AdversarialReviewer works."""

    def test_importable(self):
        from evaluation.consensus.adversarial_reviewer import AdversarialReviewer
        reviewer = AdversarialReviewer()
        assert reviewer is not None

    def test_no_llm_returns_empty_report(self):
        """Without LLM, reviewer returns empty report."""
        from evaluation.consensus.adversarial_reviewer import AdversarialReviewer
        reviewer = AdversarialReviewer()
        report = reviewer.review("TestTeam", "TestMode", {"output.json": {}})
        assert report.total_issues == 0
        assert report.penalty == 0.0

    def test_report_to_dict(self):
        from evaluation.consensus.adversarial_reviewer import (
            AdversarialReviewer, AdversarialReport,
        )
        report = AdversarialReport(
            team="TestTeam",
            mode="TestMode",
            total_issues=2,
            critical_count=1,
            major_count=1,
            penalty=0.23,
        )
        d = report.to_dict()
        assert d["team"] == "TestTeam"
        assert d["total_issues"] == 2
        assert d["penalty"] == 0.23

    def test_issue_model(self):
        from evaluation.consensus.adversarial_reviewer import AdversarialIssue
        issue = AdversarialIssue(
            category="hallucination",
            severity="critical",
            description="Fabricated citation",
            evidence="Smith et al. (2025)",
            affected_stage="LiteratureGatheringStage",
        )
        assert issue.category == "hallucination"
        assert issue.severity == "critical"


# ============================================================================
# Test 6: BenchmarkComparator functional
# ============================================================================

class TestBenchmarkComparator:
    """Verify BenchmarkComparator works."""

    def test_importable(self):
        from evaluation.benchmarks.econ_benchmarks import BenchmarkComparator
        comparator = BenchmarkComparator()
        assert comparator is not None

    def test_get_team_benchmark(self):
        from evaluation.benchmarks.econ_benchmarks import get_team_benchmark
        for team in ["IdeationTeam", "LiteratureTeam", "ModelTeam", "DataTeam"]:
            benchmark = get_team_benchmark(team)
            assert benchmark is not None, f"No benchmark for {team}"
            assert benchmark.team == team

    def test_get_unknown_team(self):
        from evaluation.benchmarks.econ_benchmarks import get_team_benchmark
        assert get_team_benchmark("UnknownTeam") is None

    def test_compare_empty_outputs(self):
        from evaluation.benchmarks.econ_benchmarks import (
            BenchmarkComparator, get_team_benchmark,
        )
        comparator = BenchmarkComparator()
        benchmark = get_team_benchmark("IdeationTeam")
        result = comparator.compare({}, benchmark)
        assert 0 <= result.relative_quality <= 1

    def test_compare_returns_feature_matches(self):
        from evaluation.benchmarks.econ_benchmarks import (
            BenchmarkComparator, get_team_benchmark,
        )
        comparator = BenchmarkComparator()
        benchmark = get_team_benchmark("IdeationTeam")
        result = comparator.compare({}, benchmark)
        assert isinstance(result.feature_matches, dict)

    def test_result_to_dict(self):
        from evaluation.benchmarks.econ_benchmarks import BenchmarkResult
        result = BenchmarkResult(
            team="TestTeam",
            relative_quality=0.65,
            dimension_assessments={"correctness": 0.8},
            feature_matches={"feat1": True},
        )
        d = result.to_dict()
        assert d["team"] == "TestTeam"
        assert d["relative_quality"] == 0.65


# ============================================================================
# Test 7: OTel export in MasterOrchestrators
# ============================================================================

class TestOTelExportWiring:
    """Verify OTel export is wired into MasterOrchestrators."""

    @pytest.mark.parametrize("filepath", _get_orchestrator_files(),
                             ids=lambda p: str(p.relative_to(_agents_dir)))
    def test_import_export_telemetry(self, filepath):
        assert _file_contains_import(
            filepath, "shared.telemetry", "export_telemetry"
        ), f"{filepath.name}: missing export_telemetry import"

    @pytest.mark.parametrize("filepath", _get_orchestrator_files(),
                             ids=lambda p: str(p.relative_to(_agents_dir)))
    def test_calls_export_telemetry(self, filepath):
        assert _file_contains_text(filepath, "export_telemetry("), (
            f"{filepath.name}: export_telemetry() not called"
        )


# ============================================================================
# Test 8: OTelExporter functional
# ============================================================================

class TestOTelExporter:
    """Verify OTelExporter works."""

    def test_importable(self):
        from shared.telemetry.otel_exporter import OTelExporter
        exporter = OTelExporter(backend="json")
        assert exporter is not None

    def test_export_empty_collector(self):
        from shared.telemetry.otel_exporter import OTelExporter
        from shared.observability import MetricsCollector
        collector = MetricsCollector()
        exporter = OTelExporter(backend="json")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            spans = exporter.export_from_collector(
                collector, output_path=path, team="Test", mode="TestMode",
            )
            assert isinstance(spans, list)
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_backends(self):
        from shared.telemetry.otel_exporter import OTelExporter
        for backend in ["json", "console"]:
            exporter = OTelExporter(backend=backend)
            assert exporter.backend == backend

    def test_invalid_backend(self):
        from shared.telemetry.otel_exporter import OTelExporter
        with pytest.raises(ValueError):
            OTelExporter(backend="invalid")


# ============================================================================
# Test 9: CostDashboard functional
# ============================================================================

class TestCostDashboard:
    """Verify CostDashboard works."""

    def test_importable(self):
        from shared.telemetry.cost_dashboard import CostDashboard
        dashboard = CostDashboard()
        assert dashboard is not None

    def test_generate_empty_collector(self):
        from shared.telemetry.cost_dashboard import CostDashboard
        from shared.observability import MetricsCollector
        collector = MetricsCollector()
        dashboard = CostDashboard()
        report = dashboard.generate(collector, team="Test", mode="TestMode")
        assert report.team == "Test"
        assert report.total_cost_usd == 0.0

    def test_save_report(self):
        from shared.telemetry.cost_dashboard import CostDashboard, CostReport
        dashboard = CostDashboard()
        report = CostReport(team="Test", mode="TestMode")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            dashboard.save(report, path)
            with open(path, "r") as f:
                data = json.load(f)
            assert data["team"] == "Test"
        finally:
            os.unlink(path)

    def test_report_to_dict(self):
        from shared.telemetry.cost_dashboard import CostReport
        report = CostReport(
            team="Test", mode="TestMode",
            total_cost_usd=1.23, total_llm_calls=10,
        )
        d = report.to_dict()
        assert d["summary"]["total_cost_usd"] == 1.23
        assert d["summary"]["total_llm_calls"] == 10


# ============================================================================
# Test 10: export_telemetry convenience function
# ============================================================================

class TestExportTelemetry:
    """Verify the export_telemetry convenience function."""

    def test_importable(self):
        from shared.telemetry import export_telemetry
        assert callable(export_telemetry)

    def test_runs_without_error(self):
        from shared.telemetry import export_telemetry
        from shared.observability import MetricsCollector
        collector = MetricsCollector()
        with tempfile.TemporaryDirectory() as tmpdir:
            export_telemetry(
                collector, tmpdir, team="TestTeam", mode="TestMode"
            )
            assert os.path.exists(os.path.join(tmpdir, "traces.json"))
            assert os.path.exists(os.path.join(tmpdir, "cost_report.json"))


# ============================================================================
# Test 11: GenAI conventions
# ============================================================================

class TestGenAIConventions:
    """Verify GenAI semantic convention attributes."""

    def test_importable(self):
        from shared.telemetry.gen_ai_conventions import GenAIAttributes
        assert GenAIAttributes.AGENT_NAME == "gen_ai.agent.name"
        assert GenAIAttributes.REQUEST_MODEL == "gen_ai.request.model"

    def test_ael_extensions(self):
        from shared.telemetry.gen_ai_conventions import GenAIAttributes
        assert GenAIAttributes.AEL_TEAM == "ael.team"
        assert GenAIAttributes.AEL_MODE == "ael.mode"
        assert GenAIAttributes.AEL_COST_USD == "ael.cost_usd"

    def test_attribute_helpers(self):
        from shared.telemetry.gen_ai_conventions import (
            llm_call_attributes, tool_call_attributes, embedding_call_attributes,
        )
        assert callable(llm_call_attributes)
        assert callable(tool_call_attributes)
        assert callable(embedding_call_attributes)


# ============================================================================
# Test 12: Extended results in save_data
# ============================================================================

class TestExtendedResultsOutput:
    """Verify extended evaluation results are included in output."""

    def _get_eval_script(self):
        return _agents_dir / "evaluation" / "run_ael_evaluation.py"

    def test_consensus_results_saved(self):
        assert _file_contains_text(
            self._get_eval_script(), '"consensus"'
        )

    def test_trajectory_results_saved(self):
        assert _file_contains_text(
            self._get_eval_script(), '"trajectory"'
        )

    def test_adversarial_results_saved(self):
        assert _file_contains_text(
            self._get_eval_script(), '"adversarial"'
        )

    def test_benchmark_results_saved(self):
        assert _file_contains_text(
            self._get_eval_script(), '"benchmark"'
        )

    def test_extended_evaluations_key(self):
        assert _file_contains_text(
            self._get_eval_script(), '"extended_evaluations"'
        )


# ============================================================================
# Test 13: Count verification
# ============================================================================

class TestPhase5Counts:
    """Verify count expectations."""

    def test_15_orchestrators_have_export_telemetry(self):
        orchestrators = _get_orchestrator_files()
        count = sum(
            1 for f in orchestrators
            if _file_contains_import(f, "shared.telemetry", "export_telemetry")
        )
        assert count == 15, f"Expected 15, got {count}"

    def test_4_cli_flags(self):
        script = _agents_dir / "evaluation" / "run_ael_evaluation.py"
        source = script.read_text(encoding="utf-8")
        flags = ["--consensus", "--trajectory", "--adversarial", "--benchmark"]
        present = sum(1 for f in flags if f in source)
        assert present == 4, f"Expected 4 CLI flags, got {present}"

    def test_4_runner_functions(self):
        script = _agents_dir / "evaluation" / "run_ael_evaluation.py"
        source = script.read_text(encoding="utf-8")
        fns = [
            "def run_consensus_evaluation(",
            "def run_trajectory_evaluation(",
            "def run_adversarial_evaluation(",
            "def run_benchmark_evaluation(",
        ]
        present = sum(1 for f in fns if f in source)
        assert present == 4, f"Expected 4 runner functions, got {present}"

    def test_benchmark_for_all_4_teams(self):
        from evaluation.benchmarks.econ_benchmarks import get_all_benchmarks
        benchmarks = get_all_benchmarks()
        assert len(benchmarks) >= 4
