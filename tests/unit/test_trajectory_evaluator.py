# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Tests for trajectory evaluator."""

import json
import pytest
from evaluation.trajectory.trajectory_evaluator import (
    TrajectoryEvaluator,
    TrajectoryScore,
    DecisionPoint,
    compute_trajectory_output_correlation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_log(
    stages=None,
    team="IdeationTeam",
    mode="ModeNoWcNoHITL",
    topic="AI economics",
):
    """Create a minimal execution log."""
    if stages is None:
        stages = [
            {
                "name": "SourcingStage",
                "number": 1,
                "status": "success",
                "duration_seconds": 30.0,
                "output_files": ["sourcing_output.json"],
                "item_count": 10,
            },
            {
                "name": "RefinementStage",
                "number": 2,
                "status": "success",
                "duration_seconds": 25.0,
                "output_files": ["refinement_output.json"],
                "item_count": 5,
            },
            {
                "name": "IntegrationStage",
                "number": 3,
                "status": "success",
                "duration_seconds": 20.0,
                "output_files": ["integration_output.json"],
                "item_count": 3,
            },
        ]
    return {
        "team": team,
        "mode": mode,
        "metadata": {"research_topic": topic},
        "stages": stages,
        "summary": {"total_stages": len(stages)},
    }


# ---------------------------------------------------------------------------
# DecisionPoint tests
# ---------------------------------------------------------------------------

class TestDecisionPoint:
    def test_create(self):
        dp = DecisionPoint(
            stage_name="SourcingStage",
            stage_number=1,
            success=True,
            item_count=10,
        )
        assert dp.stage_name == "SourcingStage"
        assert dp.success is True

    def test_default_values(self):
        dp = DecisionPoint(stage_name="test")
        assert dp.stage_number == 0
        assert dp.decision_type == "stage_execution"
        assert dp.duration_sec == 0.0


# ---------------------------------------------------------------------------
# TrajectoryScore tests
# ---------------------------------------------------------------------------

class TestTrajectoryScore:
    def test_to_dict(self):
        score = TrajectoryScore(
            team="T",
            mode="M",
            decision_coherence=0.8,
            information_utilization=0.7,
            scope_management=0.9,
            stage_transition_quality=0.85,
            overall_score=0.82,
        )
        d = score.to_dict()
        assert d["team"] == "T"
        assert d["overall_score"] == 0.82
        assert d["n_decision_points"] == 0


# ---------------------------------------------------------------------------
# TrajectoryEvaluator structural tests
# ---------------------------------------------------------------------------

class TestTrajectoryEvaluatorStructural:
    def test_basic_evaluation(self):
        evaluator = TrajectoryEvaluator()
        log = _make_log()
        score = evaluator.evaluate_trajectory(log)
        assert isinstance(score, TrajectoryScore)
        assert score.team == "IdeationTeam"
        assert score.mode == "ModeNoWcNoHITL"
        assert score.llm_evaluated is False
        assert 0.0 <= score.overall_score <= 1.0

    def test_all_stages_succeed(self):
        evaluator = TrajectoryEvaluator()
        log = _make_log()
        score = evaluator.evaluate_trajectory(log)
        # All succeed → high coherence and utilization
        assert score.decision_coherence > 0.7
        assert score.information_utilization > 0.7

    def test_stage_failure(self):
        stages = [
            {"name": "S1", "number": 1, "status": "success",
             "duration_seconds": 10, "output_files": ["a.json"], "item_count": 5},
            {"name": "S2", "number": 2, "status": "failed",
             "duration_seconds": 5, "output_files": [], "item_count": 0, "error": "timeout"},
            {"name": "S3", "number": 3, "status": "success",
             "duration_seconds": 10, "output_files": ["b.json"], "item_count": 3},
        ]
        evaluator = TrajectoryEvaluator()
        score = evaluator.evaluate_trajectory(_make_log(stages=stages))
        # One failure should lower coherence
        assert score.decision_coherence < 1.0

    def test_empty_stages(self):
        evaluator = TrajectoryEvaluator()
        score = evaluator.evaluate_trajectory(_make_log(stages=[]))
        assert score.overall_score == 0.0

    def test_single_stage(self):
        stages = [
            {"name": "S1", "number": 1, "status": "success",
             "duration_seconds": 10, "output_files": ["a.json"], "item_count": 5},
        ]
        evaluator = TrajectoryEvaluator()
        score = evaluator.evaluate_trajectory(_make_log(stages=stages))
        assert score.overall_score > 0.0

    def test_extract_decision_points(self):
        evaluator = TrajectoryEvaluator()
        log = _make_log()
        points = evaluator._extract_decision_points(log)
        assert len(points) == 3
        assert points[0].stage_name == "SourcingStage"
        assert points[0].item_count == 10
        assert points[2].stage_name == "IntegrationStage"

    def test_unordered_stages(self):
        stages = [
            {"name": "S3", "number": 3, "status": "success",
             "duration_seconds": 10, "output_files": ["c.json"], "item_count": 1},
            {"name": "S1", "number": 1, "status": "success",
             "duration_seconds": 10, "output_files": ["a.json"], "item_count": 5},
        ]
        evaluator = TrajectoryEvaluator()
        score = evaluator.evaluate_trajectory(_make_log(stages=stages))
        # Unordered should lower coherence
        assert score.decision_coherence < 1.0


# ---------------------------------------------------------------------------
# TrajectoryEvaluator with LLM tests
# ---------------------------------------------------------------------------

class TestTrajectoryEvaluatorLLM:
    def test_llm_evaluation(self):
        mock_response = json.dumps({
            "decision_coherence": 4,
            "information_utilization": 5,
            "scope_management": 4,
            "stage_transition_quality": 3,
            "justification": "Good trajectory overall.",
        })
        evaluator = TrajectoryEvaluator(llm_invoke_fn=lambda p: mock_response)
        log = _make_log()
        outputs = {"output.json": {"research_questions": ["q1", "q2"]}}
        score = evaluator.evaluate_trajectory(log, outputs)
        assert score.llm_evaluated is True
        assert score.overall_score > 0.5

    def test_llm_failure_falls_back_to_structural(self):
        evaluator = TrajectoryEvaluator(
            llm_invoke_fn=lambda p: "not json"
        )
        log = _make_log()
        outputs = {"output.json": {"data": "test"}}
        score = evaluator.evaluate_trajectory(log, outputs)
        # Should still produce a score from structural evaluation
        assert score.llm_evaluated is False
        assert score.overall_score > 0.0

    def test_no_outputs_skips_llm(self):
        call_count = {"n": 0}

        def mock_llm(prompt):
            call_count["n"] += 1
            return '{"decision_coherence": 4, "information_utilization": 4, "scope_management": 4, "stage_transition_quality": 4}'

        evaluator = TrajectoryEvaluator(llm_invoke_fn=mock_llm)
        log = _make_log()
        score = evaluator.evaluate_trajectory(log, outputs=None)
        assert call_count["n"] == 0  # No LLM call without outputs
        assert score.llm_evaluated is False


# ---------------------------------------------------------------------------
# Correlation function tests
# ---------------------------------------------------------------------------

class TestCorrelation:
    def test_perfect_positive_correlation(self):
        t = [0.1, 0.2, 0.3, 0.4, 0.5]
        o = [0.1, 0.2, 0.3, 0.4, 0.5]
        r = compute_trajectory_output_correlation(t, o)
        assert r == pytest.approx(1.0, abs=0.001)

    def test_perfect_negative_correlation(self):
        t = [0.1, 0.2, 0.3, 0.4, 0.5]
        o = [0.5, 0.4, 0.3, 0.2, 0.1]
        r = compute_trajectory_output_correlation(t, o)
        assert r == pytest.approx(-1.0, abs=0.001)

    def test_no_correlation(self):
        t = [1.0, 1.0, 1.0]
        o = [0.1, 0.5, 0.9]
        r = compute_trajectory_output_correlation(t, o)
        assert r == pytest.approx(0.0, abs=0.001)

    def test_too_few_points(self):
        assert compute_trajectory_output_correlation([0.5], [0.5]) == 0.0

    def test_mismatched_lengths(self):
        assert compute_trajectory_output_correlation([0.1, 0.2], [0.1]) == 0.0

    def test_empty(self):
        assert compute_trajectory_output_correlation([], []) == 0.0
