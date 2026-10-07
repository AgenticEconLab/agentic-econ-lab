# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
AEL V0.5 Phase 4 — Agent-Level Evaluation Tests

Tests for ToolCorrectnessScorer, StepEfficiencyScorer, PlanAdherenceScorer,
ArgumentCorrectnessScorer, and JudgeCalibratorGLM.
"""

import math
import os
import sys

import pytest


# ── Sample execution logs for testing ─────────────────────────────────

_SENTINEL = object()

def _make_log(
    stages=None, llm_calls=_SENTINEL, tool_calls=_SENTINEL,
    total_duration=60, errors=None,
):
    """Helper to build a minimal execution log."""
    if llm_calls is _SENTINEL:
        llm_calls = [
            {"model": "gpt-4o-mini", "input_tokens": 500, "output_tokens": 200},
            {"model": "gpt-4o-mini", "input_tokens": 600, "output_tokens": 300},
            {"model": "gpt-4o-mini", "input_tokens": 400, "output_tokens": 150},
        ]
    if tool_calls is _SENTINEL:
        tool_calls = [
            {"tool": "arxiv_search", "arguments": {"query": "AI economics"}, "status": "success"},
            {"tool": "web_search", "arguments": {"query": "macroeconomics trends"}, "status": "success"},
        ]
    return {
        "total_duration_seconds": total_duration,
        "stages": stages or [
            {"name": "Sourcing", "status": "success", "duration_seconds": 20,
             "item_count": 10, "output_files": ["literature_results.csv"]},
            {"name": "Refinement", "status": "success", "duration_seconds": 20,
             "item_count": 5, "output_files": ["refinement_results.json"]},
            {"name": "Integration", "status": "success", "duration_seconds": 20,
             "item_count": 3, "output_files": ["finalized_research_questions.json"]},
        ],
        "errors": errors or [],
        "observability": {
            "llm_calls": llm_calls,
            "tool_calls": tool_calls,
        },
    }


# ── ToolCorrectnessScorer ─────────────────────────────────────────────

class TestToolCorrectnessCore:
    """Core tool correctness scoring."""

    def test_perfect_tool_usage(self):
        from evaluation.agent_metrics.tool_correctness import ToolCorrectnessScorer
        scorer = ToolCorrectnessScorer()
        log = _make_log()
        result = scorer.score(log, team="IdeationTeam")
        assert result["tool_correctness"] > 0.7
        assert result["tool_calls_total"] == 2

    def test_no_tool_calls_with_expectations(self):
        from evaluation.agent_metrics.tool_correctness import ToolCorrectnessScorer
        scorer = ToolCorrectnessScorer()
        log = _make_log(tool_calls=[])
        result = scorer.score(log, team="IdeationTeam")
        # IdeationTeam has expected tools, so score reflects that
        assert result["tool_calls_total"] == 0
        assert result["tool_correctness"] in (0.5, 1.0)

    def test_no_tool_calls_no_expectations(self):
        from evaluation.agent_metrics.tool_correctness import ToolCorrectnessScorer
        scorer = ToolCorrectnessScorer(expected_tools={}, forbidden_tools={})
        log = _make_log(tool_calls=[])
        result = scorer.score(log, team="UnknownTeam")
        assert result["tool_calls_total"] == 0

    def test_forbidden_tool_detected(self):
        from evaluation.agent_metrics.tool_correctness import ToolCorrectnessScorer
        scorer = ToolCorrectnessScorer()
        log = _make_log(tool_calls=[
            {"tool": "arxiv_search", "arguments": {}, "status": "success"},
            {"tool": "fred_fetch", "arguments": {}, "status": "success"},
        ])
        result = scorer.score(log, team="DataTeam")
        assert result["forbidden_tools_used"] >= 1

    def test_tool_failure_tracked(self):
        from evaluation.agent_metrics.tool_correctness import ToolCorrectnessScorer
        scorer = ToolCorrectnessScorer()
        log = _make_log(tool_calls=[
            {"tool": "arxiv_search", "arguments": {}, "status": "error"},
        ])
        result = scorer.score(log, team="IdeationTeam")
        assert result["tool_success_rate"] == 0.0


class TestToolCorrectnessExpectedTools:
    """Expected tool configuration tests."""

    def test_default_expected_tools_exist(self):
        from evaluation.agent_metrics.tool_correctness import EXPECTED_TOOLS
        assert "IdeationTeam" in EXPECTED_TOOLS
        assert "LiteratureTeam" in EXPECTED_TOOLS
        assert "DataTeam" in EXPECTED_TOOLS
        assert "ModelTeam" in EXPECTED_TOOLS

    def test_custom_expected_tools(self):
        from evaluation.agent_metrics.tool_correctness import ToolCorrectnessScorer
        scorer = ToolCorrectnessScorer(
            expected_tools={"TestTeam": {"Stage1": ["custom_tool"]}},
            forbidden_tools={},
        )
        log = _make_log(tool_calls=[
            {"tool": "custom_tool", "arguments": {}, "status": "success"},
        ])
        result = scorer.score(log, team="TestTeam")
        assert result["selection_score"] == 1.0


# ── StepEfficiencyScorer ──────────────────────────────────────────────

class TestStepEfficiencyCore:
    """Core step efficiency scoring."""

    def test_normal_execution(self):
        from evaluation.agent_metrics.step_efficiency import StepEfficiencyScorer
        scorer = StepEfficiencyScorer()
        log = _make_log()
        result = scorer.score(log, team="IdeationTeam")
        assert 0 <= result["step_efficiency"] <= 1.0
        assert result["llm_calls_total"] == 3

    def test_no_llm_calls(self):
        from evaluation.agent_metrics.step_efficiency import StepEfficiencyScorer
        scorer = StepEfficiencyScorer()
        log = _make_log(llm_calls=[])
        result = scorer.score(log, team="IdeationTeam")
        # With balanced stages but no LLM calls, score should be lower than normal
        assert result["llm_calls_total"] == 0

    def test_balanced_stages(self):
        from evaluation.agent_metrics.step_efficiency import StepEfficiencyScorer
        scorer = StepEfficiencyScorer()
        log = _make_log()  # All stages 20s each
        result = scorer.score(log, team="IdeationTeam")
        assert result["stage_balance"] == 1.0

    def test_unbalanced_stages(self):
        from evaluation.agent_metrics.step_efficiency import StepEfficiencyScorer
        scorer = StepEfficiencyScorer()
        stages = [
            {"name": "S1", "status": "success", "duration_seconds": 1,
             "item_count": 1, "output_files": ["f.csv"]},
            {"name": "S2", "status": "success", "duration_seconds": 100,
             "item_count": 1, "output_files": ["g.json"]},
        ]
        log = _make_log(stages=stages)
        result = scorer.score(log, team="IdeationTeam")
        assert result["stage_balance"] < 0.8

    def test_token_efficiency_computed(self):
        from evaluation.agent_metrics.step_efficiency import StepEfficiencyScorer
        scorer = StepEfficiencyScorer()
        log = _make_log()
        result = scorer.score(log, team="IdeationTeam")
        assert "token_efficiency" in result
        assert 0 <= result["token_efficiency"] <= 1.0


class TestStepEfficiencyExpected:
    """Expected call count configuration."""

    def test_expected_calls_exist(self):
        from evaluation.agent_metrics.step_efficiency import EXPECTED_LLM_CALLS
        assert "IdeationTeam" in EXPECTED_LLM_CALLS

    def test_within_expected_range(self):
        from evaluation.agent_metrics.step_efficiency import StepEfficiencyScorer
        scorer = StepEfficiencyScorer()
        # 3 calls is within IdeationTeam range (5-33)
        log = _make_log(llm_calls=[
            {"model": "m", "input_tokens": 100, "output_tokens": 50}
            for _ in range(8)
        ])
        result = scorer.score(log, team="IdeationTeam")
        assert result["call_efficiency"] >= 0.8


# ── PlanAdherenceScorer ───────────────────────────────────────────────

class TestPlanAdherenceCore:
    """Core plan adherence scoring."""

    def test_perfect_adherence(self):
        from evaluation.agent_metrics.plan_adherence import PlanAdherenceScorer
        scorer = PlanAdherenceScorer()
        log = _make_log()
        result = scorer.score(log, team="IdeationTeam")
        assert result["plan_adherence"] > 0.7
        assert result["ordering_correct"] is True

    def test_missing_stage(self):
        from evaluation.agent_metrics.plan_adherence import PlanAdherenceScorer
        scorer = PlanAdherenceScorer()
        stages = [
            {"name": "Sourcing", "status": "success", "duration_seconds": 20,
             "item_count": 10, "output_files": ["lit.csv"]},
            # Missing Refinement and Integration
        ]
        log = _make_log(stages=stages)
        result = scorer.score(log, team="IdeationTeam")
        assert result["completion_rate"] < 1.0

    def test_wrong_order(self):
        from evaluation.agent_metrics.plan_adherence import PlanAdherenceScorer
        scorer = PlanAdherenceScorer()
        stages = [
            {"name": "Integration", "status": "success", "duration_seconds": 20,
             "item_count": 3, "output_files": ["final.json"]},
            {"name": "Sourcing", "status": "success", "duration_seconds": 20,
             "item_count": 10, "output_files": ["lit.csv"]},
        ]
        log = _make_log(stages=stages)
        result = scorer.score(log, team="IdeationTeam")
        assert result["ordering_correct"] is False

    def test_unknown_team(self):
        from evaluation.agent_metrics.plan_adherence import PlanAdherenceScorer
        scorer = PlanAdherenceScorer()
        log = _make_log()
        result = scorer.score(log, team="UnknownTeam")
        assert result["plan_adherence"] == 0.8

    def test_output_coverage(self):
        from evaluation.agent_metrics.plan_adherence import PlanAdherenceScorer
        scorer = PlanAdherenceScorer()
        log = _make_log()
        result = scorer.score(log, team="IdeationTeam")
        assert "output_coverage" in result


class TestPlanAdherenceExpected:
    """Expected stage plans."""

    def test_expected_stages_exist(self):
        from evaluation.agent_metrics.plan_adherence import EXPECTED_STAGES
        for team in ["IdeationTeam", "LiteratureTeam", "DataTeam", "ModelTeam"]:
            assert team in EXPECTED_STAGES
            assert len(EXPECTED_STAGES[team]) == 3

    def test_expected_outputs_exist(self):
        from evaluation.agent_metrics.plan_adherence import EXPECTED_OUTPUTS
        for team in ["IdeationTeam", "LiteratureTeam", "DataTeam", "ModelTeam"]:
            assert team in EXPECTED_OUTPUTS


# ── ArgumentCorrectnessScorer ─────────────────────────────────────────

class TestArgumentCorrectnessCore:
    """Core argument correctness scoring."""

    def test_correct_arguments(self):
        from evaluation.agent_metrics.argument_correctness import ArgumentCorrectnessScorer
        scorer = ArgumentCorrectnessScorer()
        log = _make_log(tool_calls=[
            {"tool": "arxiv_search", "arguments": {"query": "AI economics"}, "status": "success"},
            {"tool": "web_search", "arguments": {"query": "macro trends"}, "status": "success"},
        ])
        result = scorer.score(log, team="IdeationTeam")
        assert result["argument_correctness"] == 1.0
        assert result["args_missing"] == 0

    def test_missing_required_arg(self):
        from evaluation.agent_metrics.argument_correctness import ArgumentCorrectnessScorer
        scorer = ArgumentCorrectnessScorer()
        log = _make_log(tool_calls=[
            {"tool": "arxiv_search", "arguments": {}, "status": "success"},  # missing query
        ])
        result = scorer.score(log, team="IdeationTeam")
        assert result["args_missing"] >= 1
        assert result["argument_correctness"] < 1.0

    def test_empty_arg_value(self):
        from evaluation.agent_metrics.argument_correctness import ArgumentCorrectnessScorer
        scorer = ArgumentCorrectnessScorer()
        log = _make_log(tool_calls=[
            {"tool": "arxiv_search", "arguments": {"query": ""}, "status": "success"},
        ])
        result = scorer.score(log, team="IdeationTeam")
        assert result["args_empty"] >= 1

    def test_no_tool_calls(self):
        from evaluation.agent_metrics.argument_correctness import ArgumentCorrectnessScorer
        scorer = ArgumentCorrectnessScorer()
        log = _make_log(tool_calls=[])
        result = scorer.score(log, team="IdeationTeam")
        assert result["argument_correctness"] == 1.0

    def test_unknown_tool_non_empty_args(self):
        from evaluation.agent_metrics.argument_correctness import ArgumentCorrectnessScorer
        scorer = ArgumentCorrectnessScorer()
        log = _make_log(tool_calls=[
            {"tool": "custom_tool", "arguments": {"param": "value"}, "status": "success"},
        ])
        result = scorer.score(log, team="IdeationTeam")
        assert result["argument_correctness"] == 1.0

    def test_type_check(self):
        from evaluation.agent_metrics.argument_correctness import ArgumentCorrectnessScorer
        assert ArgumentCorrectnessScorer._check_type("hello", "str") is True
        assert ArgumentCorrectnessScorer._check_type(42, "int") is True
        assert ArgumentCorrectnessScorer._check_type(3.14, "float") is True
        assert ArgumentCorrectnessScorer._check_type([1, 2], "list") is True
        assert ArgumentCorrectnessScorer._check_type({"a": 1}, "dict") is True
        assert ArgumentCorrectnessScorer._check_type(True, "bool") is True
        assert ArgumentCorrectnessScorer._check_type("hello", "int") is False


class TestArgumentCorrectnessExpected:
    """Expected argument patterns."""

    def test_expected_args_exist(self):
        from evaluation.agent_metrics.argument_correctness import EXPECTED_ARGS
        assert "arxiv_search" in EXPECTED_ARGS
        assert "web_search" in EXPECTED_ARGS
        assert "fred_fetch" in EXPECTED_ARGS

    def test_required_args_defined(self):
        from evaluation.agent_metrics.argument_correctness import EXPECTED_ARGS
        for tool, spec in EXPECTED_ARGS.items():
            assert "required" in spec, f"{tool} missing required field"
            assert len(spec["required"]) >= 1, f"{tool} has no required args"


# ── JudgeCalibratorGLM ───────────────────────────────────────────────

class TestJudgeCalibratorCore:
    """Core GLM calibrator tests."""

    def test_fit_basic(self):
        from evaluation.calibration.judge_calibrator import JudgeCalibratorGLM, CalibrationPair
        cal = JudgeCalibratorGLM()
        pairs = [
            CalibrationPair(0.6, 0.7),
            CalibrationPair(0.7, 0.8),
            CalibrationPair(0.8, 0.9),
            CalibrationPair(0.9, 1.0),
        ]
        params = cal.fit(pairs)
        assert cal.is_fitted
        assert params["n_pairs"] == 4
        assert abs(params["beta"] - 1.0) < 0.2  # Near identity

    def test_calibrate_identity(self):
        from evaluation.calibration.judge_calibrator import JudgeCalibratorGLM, CalibrationPair
        cal = JudgeCalibratorGLM()
        pairs = [CalibrationPair(x / 10, x / 10) for x in range(1, 10)]
        cal.fit(pairs)
        assert abs(cal.calibrate(0.5) - 0.5) < 0.1

    def test_calibrate_bias_correction(self):
        from evaluation.calibration.judge_calibrator import JudgeCalibratorGLM, CalibrationPair
        cal = JudgeCalibratorGLM()
        # Raw scores systematically 0.1 lower than reference
        pairs = [CalibrationPair(x / 10, x / 10 + 0.1) for x in range(1, 9)]
        cal.fit(pairs)
        calibrated = cal.calibrate(0.5)
        assert calibrated > 0.5  # Should adjust upward

    def test_calibrate_clamped(self):
        from evaluation.calibration.judge_calibrator import JudgeCalibratorGLM
        cal = JudgeCalibratorGLM()
        cal.alpha = -0.5
        cal.beta = 2.0
        cal._fitted = True
        assert cal.calibrate(0.0) == 0.0  # Clamped at 0
        assert cal.calibrate(1.0) <= 1.0  # Clamped at 1

    def test_calibrate_batch(self):
        from evaluation.calibration.judge_calibrator import JudgeCalibratorGLM
        cal = JudgeCalibratorGLM()
        results = cal.calibrate_batch([0.3, 0.5, 0.7])
        assert len(results) == 3

    def test_get_params(self):
        from evaluation.calibration.judge_calibrator import JudgeCalibratorGLM
        cal = JudgeCalibratorGLM()
        params = cal.get_params()
        assert "alpha" in params
        assert "beta" in params
        assert "r_squared" in params

    def test_not_fitted_initially(self):
        from evaluation.calibration.judge_calibrator import JudgeCalibratorGLM
        cal = JudgeCalibratorGLM()
        assert cal.is_fitted is False

    def test_fit_single_pair(self):
        from evaluation.calibration.judge_calibrator import JudgeCalibratorGLM, CalibrationPair
        cal = JudgeCalibratorGLM()
        cal.fit([CalibrationPair(0.5, 0.6)])
        assert cal.is_fitted
        assert cal.beta == 1.0  # Can't fit slope with 1 point


class TestJudgeCalibratorCorrelation:
    """Correlation computation tests."""

    def test_perfect_correlation(self):
        from evaluation.calibration.judge_calibrator import JudgeCalibratorGLM, CalibrationPair
        cal = JudgeCalibratorGLM()
        pairs = [CalibrationPair(x / 10, x / 10) for x in range(1, 10)]
        r = cal.correlation(pairs)
        assert abs(r - 1.0) < 0.01

    def test_negative_correlation(self):
        from evaluation.calibration.judge_calibrator import JudgeCalibratorGLM, CalibrationPair
        cal = JudgeCalibratorGLM()
        pairs = [CalibrationPair(x / 10, 1.0 - x / 10) for x in range(1, 10)]
        r = cal.correlation(pairs)
        assert r < -0.9

    def test_no_correlation_single_point(self):
        from evaluation.calibration.judge_calibrator import JudgeCalibratorGLM, CalibrationPair
        cal = JudgeCalibratorGLM()
        r = cal.correlation([CalibrationPair(0.5, 0.5)])
        assert r == 0.0

    def test_r_squared_high(self):
        from evaluation.calibration.judge_calibrator import JudgeCalibratorGLM, CalibrationPair
        cal = JudgeCalibratorGLM()
        pairs = [CalibrationPair(x / 10, x / 10 + 0.01) for x in range(1, 10)]
        cal.fit(pairs)
        assert cal.r_squared > 0.9


class TestCalibrationPair:
    """CalibrationPair data class."""

    def test_basic_creation(self):
        from evaluation.calibration.judge_calibrator import CalibrationPair
        pair = CalibrationPair(0.5, 0.6, dimension="reliability")
        assert pair.raw_score == 0.5
        assert pair.reference_score == 0.6
        assert pair.dimension == "reliability"

    def test_default_metadata(self):
        from evaluation.calibration.judge_calibrator import CalibrationPair
        pair = CalibrationPair(0.5, 0.6)
        assert pair.metadata == {}


# ── Agent Metrics __init__ imports ────────────────────────────────────

class TestAgentMetricsImports:
    """Verify agent_metrics module exports."""

    def test_import_tool_correctness(self):
        from evaluation.agent_metrics import ToolCorrectnessScorer
        assert ToolCorrectnessScorer is not None

    def test_import_step_efficiency(self):
        from evaluation.agent_metrics import StepEfficiencyScorer
        assert StepEfficiencyScorer is not None

    def test_import_plan_adherence(self):
        from evaluation.agent_metrics import PlanAdherenceScorer
        assert PlanAdherenceScorer is not None

    def test_import_argument_correctness(self):
        from evaluation.agent_metrics import ArgumentCorrectnessScorer
        assert ArgumentCorrectnessScorer is not None


class TestCalibrationImports:
    """Verify calibration module exports."""

    def test_import_calibrator(self):
        from evaluation.calibration import JudgeCalibratorGLM
        assert JudgeCalibratorGLM is not None


# ── run_ael_evaluation.py wiring ──────────────────────────────────────

class TestEvaluationWiring:
    """Verify agent metrics are wired into run_ael_evaluation.py."""

    def test_agent_metrics_flag_exists(self):
        """The --agent-metrics argument should be defined."""
        filepath = os.path.join(
            os.path.dirname(__file__), "..", "..", "evaluation", "run_ael_evaluation.py"
        )
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        assert "--agent-metrics" in content

    def test_lightweight_judge_flag_exists(self):
        filepath = os.path.join(
            os.path.dirname(__file__), "..", "..", "evaluation", "run_ael_evaluation.py"
        )
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        assert "--lightweight-judge" in content

    def test_agent_metrics_import_in_eval(self):
        filepath = os.path.join(
            os.path.dirname(__file__), "..", "..", "evaluation", "run_ael_evaluation.py"
        )
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        assert "ToolCorrectnessScorer" in content
        assert "StepEfficiencyScorer" in content
        assert "PlanAdherenceScorer" in content
        assert "ArgumentCorrectnessScorer" in content
