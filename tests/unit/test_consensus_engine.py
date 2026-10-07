# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Tests for multi-evaluator consensus engine."""

import pytest
from unittest.mock import MagicMock, patch
from evaluation.consensus.consensus_engine import (
    ConsensusEngine,
    CalibrationMethod,
    ConsensusResult,
    ModelCalibration,
    weighted_kappa,
    fleiss_kappa,
)
from evaluation.schemas.llm_scores import LLMDimensionScore, SubCriterionScore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_score(dim, team, mode, sub_scores, model="test-model"):
    """Create an LLMDimensionScore from a list of (name, score) tuples."""
    subs = [
        SubCriterionScore(criterion_name=n, score=s, justification="ok")
        for n, s in sub_scores
    ]
    return LLMDimensionScore.from_sub_criteria(
        dimension=dim, team=team, mode=mode,
        sub_criteria=subs, evaluator_model=model,
    )


# ---------------------------------------------------------------------------
# weighted_kappa tests
# ---------------------------------------------------------------------------

class TestWeightedKappa:
    def test_perfect_agreement(self):
        a = [1, 2, 3, 4, 5]
        b = [1, 2, 3, 4, 5]
        assert weighted_kappa(a, b, k=5) == pytest.approx(1.0)

    def test_empty_input(self):
        assert weighted_kappa([], [], k=5) == 0.0

    def test_moderate_agreement(self):
        a = [3, 4, 3, 4, 3]
        b = [3, 3, 4, 4, 3]
        kappa = weighted_kappa(a, b, k=5)
        assert -1.0 <= kappa <= 1.0
        assert kappa > 0  # some agreement

    def test_all_same_ratings(self):
        a = [3, 3, 3, 3]
        b = [3, 3, 3, 3]
        kappa = weighted_kappa(a, b, k=5)
        # Perfect agreement but no variance — kappa should be 1.0 or handle edge case
        assert kappa == pytest.approx(1.0) or kappa == pytest.approx(0.0)

    def test_near_perfect_ratings(self):
        # Ratings spread across full range with close agreement
        a = [1, 2, 3, 4, 5, 3, 4]
        b = [1, 2, 3, 5, 5, 3, 4]
        kappa = weighted_kappa(a, b, k=5)
        assert kappa > 0.5  # substantial agreement for close ratings


# ---------------------------------------------------------------------------
# fleiss_kappa tests
# ---------------------------------------------------------------------------

class TestFleissKappa:
    def test_perfect_agreement(self):
        # All raters agree on category 3
        subjects = [[3, 3, 3] for _ in range(5)]
        kappa = fleiss_kappa(subjects, k=5)
        assert kappa == pytest.approx(1.0)

    def test_empty_subjects(self):
        assert fleiss_kappa([], k=5) == 0.0

    def test_single_rater(self):
        subjects = [[3] for _ in range(5)]
        assert fleiss_kappa(subjects, k=5) == 0.0

    def test_moderate_agreement(self):
        subjects = [
            [3, 3, 4],
            [4, 4, 3],
            [3, 4, 3],
            [5, 4, 5],
            [2, 2, 3],
        ]
        kappa = fleiss_kappa(subjects, k=5)
        assert -1.0 <= kappa <= 1.0

    def test_three_raters(self):
        # Explicit 3-rater case
        subjects = [
            [1, 1, 2],
            [3, 3, 3],
            [4, 4, 5],
        ]
        kappa = fleiss_kappa(subjects, k=5)
        assert -1.0 <= kappa <= 1.0


# ---------------------------------------------------------------------------
# CalibrationMethod tests
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_no_calibration(self):
        engine = ConsensusEngine(
            model_names=["a", "b"],
            calibration=CalibrationMethod.NONE,
            evaluator_factory=lambda m: None,
        )
        raw = {
            "a": {"dim1": 0.7, "dim2": 0.5},
            "b": {"dim1": 0.9, "dim2": 0.6},
        }
        calibrated = engine._calibrate_scores(
            # Need to convert to LLMDimensionScore format for the actual method,
            # but _calibrate_scores works on extracted floats internally.
            raw  # This works since CalibrationMethod.NONE returns model_dim_scores directly
        )
        assert calibrated == raw

    def test_mean_shift_calibration(self):
        engine = ConsensusEngine(
            model_names=["a", "b"],
            calibration=CalibrationMethod.MEAN_SHIFT,
            evaluator_factory=lambda m: None,
        )
        # Model a scores consistently lower than model b
        raw = {
            "a": {"dim1": 0.3, "dim2": 0.4},
            "b": {"dim1": 0.7, "dim2": 0.8},
        }
        calibrated = engine._calibrate_scores(raw)
        # After mean-shift, both models should be closer to grand mean
        assert "a" in calibrated
        assert "b" in calibrated
        # Grand mean per dim: dim1=(0.3+0.7)/2=0.5, dim2=(0.4+0.8)/2=0.6
        # Model a mean = 0.35, model b mean = 0.75, grand_mean_overall = 0.55
        # a shift = 0.55-0.35 = 0.20, b shift = 0.55-0.75 = -0.20
        assert calibrated["a"]["dim1"] > 0.3  # shifted up
        assert calibrated["b"]["dim1"] < 0.7  # shifted down

    def test_z_score_calibration(self):
        engine = ConsensusEngine(
            model_names=["a"],
            calibration=CalibrationMethod.Z_SCORE,
            evaluator_factory=lambda m: None,
        )
        raw = {"a": {"dim1": 0.2, "dim2": 0.8}}
        calibrated = engine._z_score_calibrate(raw, {"dim1", "dim2"})
        assert "a" in calibrated
        # After z-score: centered around 0.5
        assert 0.0 <= calibrated["a"]["dim1"] <= 1.0
        assert 0.0 <= calibrated["a"]["dim2"] <= 1.0


# ---------------------------------------------------------------------------
# ConsensusEngine tests
# ---------------------------------------------------------------------------

class TestConsensusEngine:
    def test_init_defaults(self):
        engine = ConsensusEngine()
        assert len(engine.model_names) == 2
        assert engine.calibration == CalibrationMethod.MEAN_SHIFT

    def test_custom_models(self):
        engine = ConsensusEngine(model_names=["model-a", "model-b", "model-c"])
        assert len(engine.model_names) == 3

    def test_compute_consensus(self):
        engine = ConsensusEngine(evaluator_factory=lambda m: None)
        calibrated = {
            "a": {"dim1": 0.6, "dim2": 0.8},
            "b": {"dim1": 0.8, "dim2": 0.6},
        }
        consensus = engine._compute_consensus(calibrated)
        assert consensus["dim1"] == pytest.approx(0.7)
        assert consensus["dim2"] == pytest.approx(0.7)

    def test_consensus_single_model(self):
        engine = ConsensusEngine(evaluator_factory=lambda m: None)
        calibrated = {"a": {"dim1": 0.5}}
        consensus = engine._compute_consensus(calibrated)
        assert consensus["dim1"] == pytest.approx(0.5)

    def test_evaluate_with_no_evaluators(self):
        engine = ConsensusEngine(
            model_names=["nonexistent"],
            evaluator_factory=lambda m: None,
        )
        from pathlib import Path
        result = engine.evaluate_with_consensus("IdeationTeam", "ModeNoWcNoHITL", Path("."))
        assert isinstance(result, ConsensusResult)
        assert result.consensus_scores == {}

    def test_result_to_dict(self):
        result = ConsensusResult(
            team="IdeationTeam",
            mode="ModeNoWcNoHITL",
            models=["a", "b"],
            calibration_method="mean_shift",
            raw_scores={},
            calibrated_scores={},
            consensus_scores={"dim1": 0.75},
            fleiss_kappa=0.65,
        )
        d = result.to_dict()
        assert d["team"] == "IdeationTeam"
        assert d["consensus_scores"]["dim1"] == 0.75
        assert d["fleiss_kappa"] == 0.65

    def test_pairwise_kappas_with_mock_scores(self):
        engine = ConsensusEngine(evaluator_factory=lambda m: None)
        score_a = _make_score("correctness", "T", "M", [("c1", 4), ("c2", 3)], "a")
        score_b = _make_score("correctness", "T", "M", [("c1", 4), ("c2", 4)], "b")
        raw = {
            "a": {"correctness": score_a},
            "b": {"correctness": score_b},
        }
        kappas = engine._compute_pairwise_kappas(raw)
        assert "a_vs_b" in kappas

    def test_fleiss_kappa_with_mock_scores(self):
        engine = ConsensusEngine(evaluator_factory=lambda m: None)
        score_a = _make_score("dim", "T", "M", [("c1", 4), ("c2", 3)], "a")
        score_b = _make_score("dim", "T", "M", [("c1", 4), ("c2", 3)], "b")
        raw = {"a": {"dim": score_a}, "b": {"dim": score_b}}
        fk = engine._compute_fleiss_kappa(raw)
        assert -1.0 <= fk <= 1.0
