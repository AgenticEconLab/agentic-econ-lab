# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Tests for economics benchmarks."""

import pytest
from evaluation.benchmarks.econ_benchmarks import (
    BenchmarkComparator,
    BenchmarkResult,
    TeamBenchmark,
    DimensionAnchor,
    get_team_benchmark,
    get_all_benchmarks,
)


# ---------------------------------------------------------------------------
# Benchmark data tests
# ---------------------------------------------------------------------------

class TestBenchmarkData:
    def test_all_teams_have_benchmarks(self):
        teams = ["IdeationTeam", "LiteratureTeam", "ModelTeam", "DataTeam"]
        for team in teams:
            b = get_team_benchmark(team)
            assert b is not None, f"No benchmark for {team}"
            assert b.team == team

    def test_unknown_team_returns_none(self):
        assert get_team_benchmark("NonexistentTeam") is None

    def test_get_all_benchmarks(self):
        all_b = get_all_benchmarks()
        assert len(all_b) == 4

    def test_benchmarks_have_exemplars(self):
        for team, b in get_all_benchmarks().items():
            assert b.strong_exemplar, f"{team} missing strong exemplar"
            assert b.weak_exemplar, f"{team} missing weak exemplar"

    def test_benchmarks_have_anchors(self):
        for team, b in get_all_benchmarks().items():
            assert len(b.dimension_anchors) > 0, f"{team} missing anchors"
            for anchor in b.dimension_anchors:
                assert anchor.dimension
                assert anchor.strong_description
                assert anchor.weak_description

    def test_ideation_benchmark_detail(self):
        b = get_team_benchmark("IdeationTeam")
        assert "research_questions" in b.strong_exemplar
        assert "research_questions" in b.weak_exemplar
        # Strong has specific methodology
        strong_q = b.strong_exemplar["research_questions"][0]
        assert "difference-in-differences" in strong_q.get("methodology", "").lower() or \
               "methodology" in strong_q


# ---------------------------------------------------------------------------
# BenchmarkComparator tests
# ---------------------------------------------------------------------------

class TestBenchmarkComparator:
    def test_strong_output_scores_high(self):
        """Outputs with strong features should score closer to 1.0."""
        comparator = BenchmarkComparator()
        benchmark = get_team_benchmark("IdeationTeam")
        # Simulate outputs with strong economics content
        outputs = {
            "research_questions.json": {
                "questions": [
                    {
                        "question": "How does AI adoption affect labor market polarization?",
                        "theoretical_framework": "Skill-Biased Technical Change (SBTC, Autor et al. 2003)",
                        "methodology": "Difference-in-differences using FRED and BLS data",
                        "data_sources": ["FRED", "BLS", "Census"],
                        "elasticity": "labor supply elasticity = 0.5",
                        "equilibrium": "general equilibrium with heterogeneous agents",
                    }
                ]
            }
        }
        result = comparator.compare(outputs, benchmark)
        assert isinstance(result, BenchmarkResult)
        assert result.relative_quality > 0.3  # should have some features

    def test_weak_output_scores_low(self):
        """Outputs with weak features should score closer to 0.0."""
        comparator = BenchmarkComparator()
        benchmark = get_team_benchmark("IdeationTeam")
        outputs = {
            "research_questions.json": {
                "questions": [
                    {
                        "question": "How does AI affect things?",
                        "notes": "This is interesting",
                    }
                ]
            }
        }
        result = comparator.compare(outputs, benchmark)
        assert result.relative_quality < 0.5

    def test_empty_outputs(self):
        comparator = BenchmarkComparator()
        benchmark = get_team_benchmark("IdeationTeam")
        result = comparator.compare({}, benchmark)
        assert result.relative_quality <= 0.5

    def test_data_team_benchmark(self):
        comparator = BenchmarkComparator()
        benchmark = get_team_benchmark("DataTeam")
        outputs = {
            "data_requirements.json": {
                "variables": [
                    {"name": "GDPC1", "source": "FRED", "frequency": "quarterly"},
                    {"name": "unemployment_rate", "source": "BLS"},
                ],
                "time_period": "1990-2024",
                "quality_checks": ["stationarity_test"],
            }
        }
        result = comparator.compare(outputs, benchmark)
        assert result.relative_quality > 0.3

    def test_model_team_benchmark(self):
        comparator = BenchmarkComparator()
        benchmark = get_team_benchmark("ModelTeam")
        outputs = {
            "model_output.json": {
                "model_type": "DSGE",
                "assumptions": ["CES production function", "Following Smets and Wouters (2007)"],
                "equations": "Y = A * K^alpha * L^(1-alpha)",
                "calibration": "Calibrated to match US data",
                "equilibrium": "Steady-state equilibrium computed",
            }
        }
        result = comparator.compare(outputs, benchmark)
        assert result.relative_quality > 0.3

    def test_result_to_dict(self):
        result = BenchmarkResult(
            team="IdeationTeam",
            relative_quality=0.75,
            dimension_assessments={"correctness": 0.8},
            feature_matches={"named_theories_with_citations": True},
        )
        d = result.to_dict()
        assert d["team"] == "IdeationTeam"
        assert d["relative_quality"] == 0.75

    def test_literature_team_benchmark(self):
        comparator = BenchmarkComparator()
        benchmark = get_team_benchmark("LiteratureTeam")
        outputs = {
            "review.json": {
                "papers": [
                    {"title": "AI and Labor (2020)", "method": "empirical"},
                    {"title": "Macro Effects of AI (2021)", "method": "theoretical"},
                ],
                "themes": ["macroeconomics", "labor"],
                "gaps": ["underexplored interaction effects"],
                "systematic": True,
            }
        }
        result = comparator.compare(outputs, benchmark)
        assert result.relative_quality > 0.2


# ---------------------------------------------------------------------------
# DimensionAnchor tests
# ---------------------------------------------------------------------------

class TestDimensionAnchor:
    def test_create(self):
        anchor = DimensionAnchor(
            dimension="correctness",
            strong_description="Accurate",
            weak_description="Inaccurate",
            discriminating_features=["feature_a"],
        )
        assert anchor.dimension == "correctness"
        assert len(anchor.discriminating_features) == 1
