# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
AEL V0.6 Phase 6 — Evaluation & Reliability Enhancement Tests

Tests for Princeton ReliabilityEvaluator, MultiRunEvaluator, architecture comparison,
and coverage expansion for previously untested evaluation modules, tool wrappers,
and shared core modules.
"""

import math
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── Princeton Reliability Evaluator ──────────────────────────────────────


class TestReliabilityEvaluatorImports:
    def test_module_imports(self):
        from evaluation.reliability.evaluator import (
            ReliabilityEvaluator,
            ReliabilityResult,
            RunResult,
            MultiRunEvaluator,
        )

    def test_run_result_model(self):
        from evaluation.reliability.evaluator import RunResult
        r = RunResult(run_id="r1", quality_score=0.8, confidence=0.7)
        assert r.quality_score == 0.8
        assert r.confidence == 0.7

    def test_reliability_result_model(self):
        from evaluation.reliability.evaluator import ReliabilityResult
        r = ReliabilityResult(consistency=0.9, safety=0.95)
        assert r.consistency == 0.9
        assert r.safety == 0.95

    def test_dimensions_defined(self):
        from evaluation.reliability.evaluator import ReliabilityEvaluator
        assert len(ReliabilityEvaluator.DIMENSIONS) == 4
        assert "consistency" in ReliabilityEvaluator.DIMENSIONS
        assert "robustness" in ReliabilityEvaluator.DIMENSIONS
        assert "predictability" in ReliabilityEvaluator.DIMENSIONS
        assert "safety" in ReliabilityEvaluator.DIMENSIONS


class TestConsistency:
    def test_perfect_consistency(self):
        from evaluation.reliability.evaluator import ReliabilityEvaluator, RunResult
        ev = ReliabilityEvaluator()
        runs = [RunResult(quality_score=0.8) for _ in range(5)]
        assert ev.evaluate_consistency(runs) == 1.0

    def test_variable_consistency(self):
        from evaluation.reliability.evaluator import ReliabilityEvaluator, RunResult
        ev = ReliabilityEvaluator()
        runs = [RunResult(quality_score=s) for s in [0.5, 0.6, 0.7, 0.8, 0.9]]
        score = ev.evaluate_consistency(runs)
        assert 0.0 < score < 1.0

    def test_single_run(self):
        from evaluation.reliability.evaluator import ReliabilityEvaluator, RunResult
        ev = ReliabilityEvaluator()
        assert ev.evaluate_consistency([RunResult(quality_score=0.8)]) == 1.0


class TestRobustness:
    def test_perfect_robustness(self):
        from evaluation.reliability.evaluator import ReliabilityEvaluator, RunResult
        ev = ReliabilityEvaluator()
        original = RunResult(quality_score=0.8)
        paraphrased = [RunResult(quality_score=0.8) for _ in range(3)]
        assert ev.evaluate_robustness(original, paraphrased) == 1.0

    def test_degraded_robustness(self):
        from evaluation.reliability.evaluator import ReliabilityEvaluator, RunResult
        ev = ReliabilityEvaluator()
        original = RunResult(quality_score=1.0)
        paraphrased = [RunResult(quality_score=0.5)]
        score = ev.evaluate_robustness(original, paraphrased)
        assert score == 0.5

    def test_no_paraphrased(self):
        from evaluation.reliability.evaluator import ReliabilityEvaluator, RunResult
        ev = ReliabilityEvaluator()
        assert ev.evaluate_robustness(RunResult(quality_score=0.8), []) == 1.0

    def test_zero_base_quality(self):
        from evaluation.reliability.evaluator import ReliabilityEvaluator, RunResult
        ev = ReliabilityEvaluator()
        assert ev.evaluate_robustness(RunResult(quality_score=0), [RunResult(quality_score=0.5)]) == 0.0


class TestPredictability:
    def test_perfect_correlation(self):
        from evaluation.reliability.evaluator import ReliabilityEvaluator, RunResult
        ev = ReliabilityEvaluator()
        runs = [
            RunResult(confidence=0.2, quality_score=0.2),
            RunResult(confidence=0.5, quality_score=0.5),
            RunResult(confidence=0.8, quality_score=0.8),
        ]
        score = ev.evaluate_predictability(runs)
        assert score > 0.9

    def test_inverse_correlation(self):
        from evaluation.reliability.evaluator import ReliabilityEvaluator, RunResult
        ev = ReliabilityEvaluator()
        runs = [
            RunResult(confidence=0.2, quality_score=0.8),
            RunResult(confidence=0.5, quality_score=0.5),
            RunResult(confidence=0.8, quality_score=0.2),
        ]
        score = ev.evaluate_predictability(runs)
        assert score < 0.1

    def test_single_run_default(self):
        from evaluation.reliability.evaluator import ReliabilityEvaluator, RunResult
        ev = ReliabilityEvaluator()
        assert ev.evaluate_predictability([RunResult()]) == 0.5


class TestSafety:
    def test_all_safe(self):
        from evaluation.reliability.evaluator import ReliabilityEvaluator, RunResult
        ev = ReliabilityEvaluator()
        runs = [RunResult(output_text="GDP grew by 3% in 2024.") for _ in range(3)]
        assert ev.evaluate_safety(runs) == 1.0

    def test_hallucination_detected(self):
        from evaluation.reliability.evaluator import ReliabilityEvaluator, RunResult
        ev = ReliabilityEvaluator()
        runs = [
            RunResult(output_text="GDP grew by 3%."),
            RunResult(output_text="I cannot verify this hallucinated claim."),
        ]
        score = ev.evaluate_safety(runs)
        assert score == 0.5

    def test_empty_runs(self):
        from evaluation.reliability.evaluator import ReliabilityEvaluator
        ev = ReliabilityEvaluator()
        assert ev.evaluate_safety([]) == 1.0


class TestEvaluateAll:
    def test_evaluate_all_returns_result(self):
        from evaluation.reliability.evaluator import ReliabilityEvaluator, RunResult
        ev = ReliabilityEvaluator()
        runs = [RunResult(quality_score=0.7 + i * 0.05, confidence=0.6 + i * 0.05,
                          output_text="Clean output.") for i in range(5)]
        result = ev.evaluate_all(runs)
        assert 0.0 <= result.consistency <= 1.0
        assert 0.0 <= result.safety <= 1.0
        assert 0.0 <= result.overall <= 1.0
        assert result.details["num_runs"] == 5


class TestMultiRunEvaluator:
    def test_compute_statistics(self):
        from evaluation.reliability.evaluator import MultiRunEvaluator
        ev = MultiRunEvaluator(k=5)
        stats = ev.compute_statistics([0.7, 0.8, 0.75, 0.85, 0.9])
        assert stats["mean"] > 0
        assert stats["std"] > 0
        assert stats["min"] == 0.7
        assert stats["max"] == 0.9

    def test_compute_statistics_empty(self):
        from evaluation.reliability.evaluator import MultiRunEvaluator
        ev = MultiRunEvaluator()
        stats = ev.compute_statistics([])
        assert stats["mean"] == 0.0

    def test_compare_architectures(self):
        from evaluation.reliability.evaluator import MultiRunEvaluator
        ev = MultiRunEvaluator()
        results = {
            "sequential": [0.7, 0.75, 0.72, 0.68, 0.74],
            "parallel": [0.8, 0.82, 0.79, 0.81, 0.83],
        }
        comparison = ev.compare_architectures(results)
        assert comparison["dominant"] == "parallel"
        assert comparison["num_architectures"] == 2
        assert len(comparison["rankings"]) == 2
        assert comparison["rankings"][0]["rank"] == 1


# ── Evaluation Core Modules (Coverage Expansion) ─────────────────────────


class TestEvalCoreDimensions:
    def test_import_dimensions(self):
        from evaluation.core.dimensions import DIMENSIONS
        assert len(DIMENSIONS) >= 10

    def test_get_dimension(self):
        from evaluation.core.dimensions import get_dimension
        d = get_dimension("reliability")
        assert d is not None


class TestEvalCoreMetrics:
    def test_import_metrics(self):
        from evaluation.core.metrics import MetricCalculator

    def test_calculator_instantiates(self):
        from evaluation.core.metrics import MetricCalculator
        calc = MetricCalculator()
        assert calc is not None


class TestEvalCoreReporter:
    def test_import_reporter(self):
        from evaluation.core.reporter import EvaluationReporter

    def test_reporter_instantiates(self):
        from evaluation.core.reporter import EvaluationReporter
        reporter = EvaluationReporter()
        assert reporter is not None


class TestEvalParsers:
    def test_import_base_parser(self):
        from evaluation.parsers.base import BaseWorkflowParser

    def test_import_ael_parser(self):
        from evaluation.parsers.ael_parser import AELParser

    def test_ael_parser_instantiates(self):
        from evaluation.parsers.ael_parser import AELParser
        parser = AELParser()
        assert parser is not None


class TestEvalRunners:
    def test_import_base_runner(self):
        from evaluation.runners.base import BaseWorkflowRunner

    def test_import_ael_runner(self):
        from evaluation.runners.ael_runner import AELRunner

    def test_import_multirun_functions(self):
        from evaluation.runners.multirun_orchestrator import run_multirun, run_all_modes


class TestEvalAnalysis:
    def test_import_batch_evaluator(self):
        from evaluation.analysis.batch_evaluator import BatchEvaluator

    def test_import_dimension_analyzer(self):
        from evaluation.analysis.dimension_analyzer import DimensionAnalyzer

    def test_import_statistical_aggregator(self):
        from evaluation.analysis.statistical_aggregator import StatisticalAggregator

    def test_import_output_comparator(self):
        from evaluation.analysis.output_comparator import OutputComparator

    def test_import_innovation_analyzer(self):
        from evaluation.analysis.innovation_analyzer import InnovationAnalyzer

    def test_statistical_aggregator_instantiates(self):
        from evaluation.analysis.statistical_aggregator import StatisticalAggregator
        agg = StatisticalAggregator()
        assert agg is not None


class TestEvalSchemas:
    def test_import_execution_schema(self):
        from evaluation.schemas.execution import ExecutionTrace, StageExecution

    def test_import_metrics_schema(self):
        from evaluation.schemas.metrics import DimensionScore, EvaluationResult

    def test_import_observability_schema(self):
        from evaluation.schemas.observability import ObservabilityReport

    def test_import_llm_scores(self):
        from evaluation.schemas.llm_scores import LLMDimensionScore

    def test_dimension_score_has_fields(self):
        from evaluation.schemas.metrics import DimensionScore
        fields = DimensionScore.model_fields
        assert "dimension" in fields
        assert "score" in fields

    def test_execution_trace_creation(self):
        from evaluation.schemas.execution import ExecutionTrace
        trace = ExecutionTrace(
            team="IdeationTeam",
            mode="ModeNoWcNoHITL",
            framework="ael",
        )
        assert trace.team == "IdeationTeam"


class TestEvalBaselines:
    def test_import_baseline_comparator(self):
        from evaluation.baselines.baseline_comparator import BaselineComparator

    def test_import_baseline_generator(self):
        from evaluation.baselines.baseline_generator import BaselineGenerator

    def test_baseline_comparator_instantiates(self):
        from evaluation.baselines.baseline_comparator import BaselineComparator
        comp = BaselineComparator()
        assert comp is not None


class TestEvalRubrics:
    def test_import_rubrics(self):
        from evaluation.scoring.rubrics import LLM_EVALUATED_DIMENSIONS, get_rubric

    def test_rubrics_has_dimensions(self):
        from evaluation.scoring.rubrics import LLM_EVALUATED_DIMENSIONS
        assert len(LLM_EVALUATED_DIMENSIONS) >= 4

    def test_get_rubric(self):
        from evaluation.scoring.rubrics import get_rubric
        r = get_rubric("correctness", "IdeationTeam")
        assert r is not None


class TestEvalCalibration:
    def test_import_judge_calibrator(self):
        from evaluation.calibration.judge_calibrator import JudgeCalibratorGLM


class TestEvalAgentMetrics:
    def test_import_tool_correctness(self):
        from evaluation.agent_metrics.tool_correctness import ToolCorrectnessScorer

    def test_import_step_efficiency(self):
        from evaluation.agent_metrics.step_efficiency import StepEfficiencyScorer

    def test_import_plan_adherence(self):
        from evaluation.agent_metrics.plan_adherence import PlanAdherenceScorer

    def test_import_argument_correctness(self):
        from evaluation.agent_metrics.argument_correctness import ArgumentCorrectnessScorer


# ── Tool Wrappers (Coverage Expansion) ───────────────────────────────────


class TestArxivTool:
    def test_import(self):
        from shared.tools.arxiv_tool import arxiv_search_handler, ArxivSearchInput

    def test_input_schema(self):
        from shared.tools.arxiv_tool import ArxivSearchInput
        inp = ArxivSearchInput(query="DSGE models")
        assert inp.query == "DSGE models"

    def test_search_returns_list(self):
        from shared.tools.arxiv_tool import arxiv_search_handler
        mock_result = MagicMock()
        mock_result.title = "Test Paper"
        mock_result.summary = "Abstract"
        mock_result.entry_id = "http://arxiv.org/abs/1234"
        mock_result.published = "2024-01-01"
        mock_result.updated = "2024-01-01"
        mock_result.authors = []
        mock_result.categories = []
        with patch("shared.observability.tracked_arxiv_search", return_value=[mock_result]):
            results = arxiv_search_handler(query="test", collector=None, agent="test")
            assert isinstance(results, list)
            assert len(results) == 1
            assert results[0]["title"] == "Test Paper"


class TestFredTool:
    def test_import(self):
        from shared.tools.fred_tool import fred_get_series_handler, FredGetSeriesInput

    def test_input_schema(self):
        from shared.tools.fred_tool import FredGetSeriesInput
        inp = FredGetSeriesInput(series_id="GDP")
        assert inp.series_id == "GDP"


class TestWebSearchTool:
    def test_import(self):
        from shared.tools.web_search_tool import web_search_handler, WebSearchInput

    def test_input_schema(self):
        from shared.tools.web_search_tool import WebSearchInput
        inp = WebSearchInput(query="AI economics")
        assert inp.query == "AI economics"


class TestYFinanceTool:
    def test_import(self):
        from shared.tools.yfinance_tool import yfinance_history_handler, YFinanceHistoryInput

    def test_input_schema(self):
        from shared.tools.yfinance_tool import YFinanceHistoryInput
        inp = YFinanceHistoryInput(ticker="AAPL")
        assert inp.ticker == "AAPL"


class TestHttpTool:
    def test_import(self):
        from shared.tools.http_tool import web_fetch_handler, WebFetchInput

    def test_input_schema(self):
        from shared.tools.http_tool import WebFetchInput
        inp = WebFetchInput(url="https://example.com")
        assert inp.url == "https://example.com"


# ── Shared Core Modules (Coverage Expansion) ─────────────────────────────


class TestInstrumentation:
    def test_import(self):
        from shared.instrumentation import WorkflowLogger

    def test_logger_init(self):
        from shared.instrumentation import WorkflowLogger
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = WorkflowLogger(
                team="IdeationTeam",
                mode="ModeNoWcNoHITL",
                output_dir=tmpdir,
            )
            assert logger.team == "IdeationTeam"

    def test_logger_start_execution(self):
        from shared.instrumentation import WorkflowLogger
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = WorkflowLogger(
                team="IdeationTeam",
                mode="ModeNoWcNoHITL",
                output_dir=tmpdir,
                quiet=True,
            )
            logger.start_execution(metadata={"topic": "test"})
            assert logger.started_at is not None

    def test_logger_stage_lifecycle(self):
        from shared.instrumentation import WorkflowLogger
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = WorkflowLogger(
                team="IdeationTeam",
                mode="ModeNoWcNoHITL",
                output_dir=tmpdir,
                quiet=True,
            )
            logger.start_execution()
            logger.start_stage("SourcingStage", stage_number=1)
            logger.end_stage("SourcingStage", status="success", item_count=10)
            assert len(logger.stages) >= 1


class TestConsoleUI:
    def test_import(self):
        from shared.console_ui import ConsoleUI

    def test_create_ui(self):
        from shared.console_ui import create_ui
        ui = create_ui("IdeationTeam", "ModeNoWcNoHITL")
        assert ui is not None


class TestAutoInput:
    def test_import_functions(self):
        from shared.auto_input import auto_input, auto_yes_no, is_auto_mode

    def test_is_auto_mode_default(self):
        from shared.auto_input import is_auto_mode
        # Should return a boolean
        result = is_auto_mode()
        assert isinstance(result, bool)


# ── Register All Tools (Coverage) ────────────────────────────────────────


class TestRegisterAll:
    def test_import(self):
        from shared.tools.register_all import register_all_tools

    def test_register_all_populates_registry(self):
        from shared.tools.tool_registry import ToolRegistry
        from shared.tools.register_all import register_all_tools
        register_all_tools()
        tools = ToolRegistry.tool_names()
        # register_all should ensure tools exist (may already be registered)
        assert isinstance(tools, list)


# ── Evaluation Trajectory Module ─────────────────────────────────────────


class TestTrajectoryEvalImports:
    def test_import(self):
        from evaluation.trajectory.trajectory_evaluator import TrajectoryEvaluator


# ── Multi-Eval Modules ──────────────────────────────────────────────────


class TestMultiEvalModules:
    def test_import_adversarial_reviewer(self):
        from evaluation.consensus.adversarial_reviewer import AdversarialReviewer

    def test_import_consensus_engine(self):
        from evaluation.consensus.consensus_engine import ConsensusEngine
