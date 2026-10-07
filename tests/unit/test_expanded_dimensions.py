# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Tests for expanded evaluation dimensions (10 → 12) and new rubrics."""

import pytest
from evaluation.core.dimensions import (
    DIMENSIONS,
    DIMENSION_CLUSTERS,
    get_dimension,
    get_dimensions_by_cluster,
    get_all_metrics,
)
from evaluation.scoring.rubrics import (
    get_rubric,
    LLM_EVALUATED_DIMENSIONS,
    DECISION_QUALITY_RUBRIC,
    ECONOMIC_RIGOR_RUBRICS,
)


# ---------------------------------------------------------------------------
# Dimension definition tests
# ---------------------------------------------------------------------------

class TestExpandedDimensions:
    def test_twelve_dimensions_exist(self):
        # V0.7: framework grew to 14 dims (added SimulationFidelity,
        # CausalValidity). V0.6 baseline of 12 is preserved as a lower bound.
        assert len(DIMENSIONS) >= 12
        assert "decision_quality" in DIMENSIONS
        assert "economic_rigor" in DIMENSIONS

    def test_decision_quality_exists(self):
        dim = get_dimension("decision_quality")
        assert dim is not None
        assert dim.name == "Decision Quality"
        assert len(dim.metrics) >= 2

    def test_economic_rigor_exists(self):
        dim = get_dimension("economic_rigor")
        assert dim is not None
        assert dim.name == "Economic Rigor"
        assert len(dim.metrics) >= 2

    def test_domain_cluster_exists(self):
        assert "Domain" in DIMENSION_CLUSTERS
        assert "decision_quality" in DIMENSION_CLUSTERS["Domain"]
        assert "economic_rigor" in DIMENSION_CLUSTERS["Domain"]

    def test_original_ten_still_present(self):
        original = [
            "reliability", "correctness", "soundness",
            "efficiency", "scalability", "robustness",
            "transparency", "traceability", "reproducibility",
            "innovation_potential",
        ]
        for dim_id in original:
            assert dim_id in DIMENSIONS, f"Missing original dimension: {dim_id}"

    def test_domain_cluster_dimensions(self):
        domain_dims = get_dimensions_by_cluster("Domain")
        assert len(domain_dims) == 2
        names = {d.dimension_id for d in domain_dims}
        assert "decision_quality" in names
        assert "economic_rigor" in names

    def test_all_clusters_populated(self):
        for cluster_name, dim_ids in DIMENSION_CLUSTERS.items():
            assert len(dim_ids) > 0, f"Empty cluster: {cluster_name}"
            for dim_id in dim_ids:
                assert dim_id in DIMENSIONS, f"{dim_id} not in DIMENSIONS"


# ---------------------------------------------------------------------------
# Rubric tests
# ---------------------------------------------------------------------------

class TestExpandedRubrics:
    def test_decision_quality_rubric_exists(self):
        assert DECISION_QUALITY_RUBRIC is not None
        assert DECISION_QUALITY_RUBRIC.dimension == "decision_quality"
        assert len(DECISION_QUALITY_RUBRIC.sub_criteria) == 3

    def test_economic_rigor_rubrics_per_team(self):
        teams = ["IdeationTeam", "LiteratureTeam", "ModelTeam", "DataTeam"]
        for team in teams:
            assert team in ECONOMIC_RIGOR_RUBRICS, f"No economic_rigor rubric for {team}"
            rubric = ECONOMIC_RIGOR_RUBRICS[team]
            assert rubric.dimension == "economic_rigor"
            assert len(rubric.sub_criteria) >= 3

    def test_get_rubric_decision_quality(self):
        rubric = get_rubric("decision_quality", "IdeationTeam")
        assert rubric.dimension == "decision_quality"
        # Generic rubric, not team-specific
        assert rubric.team == "generic"

    def test_get_rubric_economic_rigor(self):
        for team in ["IdeationTeam", "LiteratureTeam", "ModelTeam", "DataTeam"]:
            rubric = get_rubric("economic_rigor", team)
            assert rubric.dimension == "economic_rigor"
            assert rubric.team == team

    def test_llm_evaluated_dimensions_includes_new(self):
        assert "decision_quality" in LLM_EVALUATED_DIMENSIONS
        assert "economic_rigor" in LLM_EVALUATED_DIMENSIONS
        assert len(LLM_EVALUATED_DIMENSIONS) == 6

    def test_rubric_to_prompt_text(self):
        rubric = get_rubric("decision_quality", "IdeationTeam")
        text = rubric.to_prompt_text()
        assert "DECISION_QUALITY" in text
        assert "Decision Coherence" in text
        assert "1 (Poor)" in text
        assert "5 (Excellent)" in text

    def test_economic_rigor_rubric_content(self):
        rubric = get_rubric("economic_rigor", "ModelTeam")
        criteria_names = {sc.name for sc in rubric.sub_criteria}
        assert "Model Specification Standards" in criteria_names
        assert "Parameter Calibration Rigor" in criteria_names

    def test_rubric_levels_complete(self):
        """All rubrics should have 5 levels per sub-criterion."""
        for dim in LLM_EVALUATED_DIMENSIONS:
            rubric = get_rubric(dim, "IdeationTeam")
            for sc in rubric.sub_criteria:
                assert len(sc.levels) == 5, (
                    f"{dim}/{sc.name} has {len(sc.levels)} levels, expected 5"
                )

    def test_unknown_dimension_raises(self):
        with pytest.raises(ValueError, match="No Tier-2 rubric"):
            get_rubric("nonexistent_dimension", "IdeationTeam")
