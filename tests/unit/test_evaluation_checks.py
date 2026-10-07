# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Evaluation integration tests — 14-dimension wiring."""

from __future__ import annotations

import pytest

from evaluation.core.dimensions import (
    DIMENSIONS,
    DIMENSION_CLUSTERS,
    DimensionCluster,
)


class TestFourteenDimensions:
    def test_dimensions_registry_has_simulation_fidelity(self):
        assert "simulation_fidelity" in DIMENSIONS
        assert DIMENSIONS["simulation_fidelity"].cluster == DimensionCluster.ECONOMICS

    def test_dimensions_registry_has_causal_validity(self):
        assert "causal_validity" in DIMENSIONS
        assert DIMENSIONS["causal_validity"].cluster == DimensionCluster.ECONOMICS

    def test_cluster_map_includes_economics(self):
        assert "Economics" in DIMENSION_CLUSTERS
        assert "simulation_fidelity" in DIMENSION_CLUSTERS["Economics"]
        assert "causal_validity" in DIMENSION_CLUSTERS["Economics"]

    def test_v06_clusters_unchanged(self):
        assert DIMENSION_CLUSTERS["Quality"] == ["reliability", "correctness", "soundness"]
        assert DIMENSION_CLUSTERS["Operational"] == ["efficiency", "scalability", "robustness"]
        assert DIMENSION_CLUSTERS["Domain"] == ["decision_quality", "economic_rigor"]

    def test_total_dimension_count_is_14(self):
        assert len(DIMENSIONS) == 14


class TestTier1Calculators:
    def test_simulation_fidelity_not_evaluated_when_absent(self):
        from evaluation.run_ael_evaluation import calculate_tier1_metrics
        log = {"total_duration_seconds": 1, "stages": [], "errors": []}
        m = calculate_tier1_metrics(log)
        assert "simulation_fidelity" in m
        assert m["simulation_fidelity"] is None      # not evaluated, not a neutral 0.5

    def test_causal_validity_not_evaluated_when_absent(self):
        from evaluation.run_ael_evaluation import calculate_tier1_metrics
        log = {"total_duration_seconds": 1, "stages": [], "errors": []}
        m = calculate_tier1_metrics(log)
        assert "causal_validity" in m
        assert m["causal_validity"] is None

    def test_simulation_fidelity_scored_when_present(self):
        from evaluation.run_ael_evaluation import calculate_tier1_metrics
        log = {
            "total_duration_seconds": 1, "stages": [], "errors": [],
            "simulation_result": {
                "scenario": "stackelberg_tax",
                "metrics": {"tau_saez_optimal": 2.0 / 3.0},
            },
        }
        m = calculate_tier1_metrics(log)
        assert m["simulation_fidelity"] == 1.0  # exact benchmark match

    def test_causal_validity_scored_when_true_ate(self):
        from evaluation.run_ael_evaluation import calculate_tier1_metrics
        log = {
            "total_duration_seconds": 1, "stages": [], "errors": [],
            "causal_estimate": {"ate": 2.0, "method": "back_door_ols", "true_ate": 2.0},
        }
        m = calculate_tier1_metrics(log)
        assert m["causal_validity"] == 1.0

    def test_causal_validity_not_evaluated_for_refusal_record(self):
        from evaluation.run_ael_evaluation import calculate_tier1_metrics
        log = {
            "total_duration_seconds": 1, "stages": [], "errors": [],
            "causal_estimate": {"status": "refused", "ate": None, "true_ate": 2.0,
                                "method": "back_door"},
        }
        assert calculate_tier1_metrics(log)["causal_validity"] is None

    def test_v06_tier1_fields_preserved(self):
        from evaluation.run_ael_evaluation import calculate_tier1_metrics
        log = {
            "total_duration_seconds": 1,
            "stages": [{"number": 1, "status": "success", "output_files": ["a.json"],
                        "item_count": 5}],
            "errors": [],
        }
        m = calculate_tier1_metrics(log)
        for dim in ("reliability", "correctness", "soundness",
                    "decision_quality", "economic_rigor"):
            assert dim in m



class TestEfficiencyNormalizationD8:
    """Efficiency normalization: max(0, 1 - T/3600) scored every run over an hour as 0 (a ~5 h
    chained run always scored 0)."""

    def _m(self, seconds, team="IdeationTeam", n_stages=3):
        from evaluation.run_ael_evaluation import calculate_tier1_metrics
        stages = [{"name": f"s{i}", "status": "success", "number": i} for i in range(n_stages)]
        return calculate_tier1_metrics({"team": team, "total_duration_seconds": seconds,
                                        "stages": stages, "errors": []})

    def test_multi_hour_pipeline_is_not_zero(self):
        m = self._m(14148.7, team="Pipeline", n_stages=7)
        assert m["efficiency_reference_seconds"] == 7 * 3600
        assert 0.5 < m["efficiency"] < 0.7                 # 25200 / (25200 + 14149)

    def test_monotone_and_half_at_reference(self):
        assert abs(self._m(3600)["efficiency"] - 0.5) < 1e-9
        assert self._m(1800)["efficiency"] > self._m(7200)["efficiency"] > 0.0
