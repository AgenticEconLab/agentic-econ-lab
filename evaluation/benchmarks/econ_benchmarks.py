# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Economics-Specific Benchmarks — Reference outputs for evaluation calibration.

Provides curated reference outputs for each team so that LLM evaluators
can compare workflow outputs against known-quality exemplars.

Benchmark Types:
- Strong exemplar: A high-quality reference output (expected score 4-5)
- Weak exemplar: A low-quality reference output (expected score 1-2)
- Dimension anchors: Descriptions of what strong/weak looks like per dimension

Usage:
    from evaluation.benchmarks.econ_benchmarks import (
        BenchmarkComparator, get_team_benchmark
    )

    benchmark = get_team_benchmark("IdeationTeam")
    comparator = BenchmarkComparator()
    result = comparator.compare(workflow_outputs, benchmark)
    print(f"Relative quality: {result.relative_quality:.3f}")
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DimensionAnchor:
    """Anchor descriptions for a dimension (what strong/weak looks like)."""

    dimension: str
    strong_description: str
    weak_description: str
    discriminating_features: List[str] = field(default_factory=list)


@dataclass
class TeamBenchmark:
    """Complete benchmark specification for a team."""

    team: str
    strong_exemplar: Dict[str, Any]
    weak_exemplar: Dict[str, Any]
    dimension_anchors: List[DimensionAnchor] = field(default_factory=list)
    expected_strong_score: float = 0.8
    expected_weak_score: float = 0.3


@dataclass
class BenchmarkResult:
    """Result of comparing outputs against benchmarks."""

    team: str
    relative_quality: float = 0.0  # 0=closer to weak, 1=closer to strong
    dimension_assessments: Dict[str, float] = field(default_factory=dict)
    feature_matches: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "team": self.team,
            "relative_quality": round(self.relative_quality, 4),
            "dimension_assessments": {
                k: round(v, 4) for k, v in self.dimension_assessments.items()
            },
            "feature_matches": self.feature_matches,
        }


# ---------------------------------------------------------------------------
# Team-specific benchmarks
# ---------------------------------------------------------------------------

_IDEATION_BENCHMARK = TeamBenchmark(
    team="IdeationTeam",
    strong_exemplar={
        "research_questions": [
            {
                "question": "How does the adoption of generative AI tools affect labor market polarization across different skill levels in OECD economies?",
                "theoretical_framework": "Skill-Biased Technical Change (SBTC) extended with routine-biased models (Autor, Levy & Murnane 2003; Acemoglu & Restrepo 2020)",
                "methodology": "Difference-in-differences with staggered adoption across industries, using OECD PIAAC data and patent filings as AI exposure measures",
                "novelty": "Extends SBTC from automation to augmentation effects, distinguishing between AI that replaces vs. complements human tasks",
                "data_sources": ["OECD PIAAC", "USPTO patent data", "EU-LFS", "O*NET task descriptions"],
            }
        ],
        "key_qualities": [
            "Specific, testable research questions grounded in established theory",
            "Appropriate methodology for causal identification",
            "Multiple credible data sources identified",
            "Clear contribution to existing literature",
        ],
    },
    weak_exemplar={
        "research_questions": [
            {
                "question": "How does AI affect the economy?",
                "theoretical_framework": "General economic theory",
                "methodology": "Literature review and data analysis",
                "novelty": "New perspective on AI economics",
                "data_sources": ["various online sources"],
            }
        ],
        "key_weaknesses": [
            "Vague, non-testable question",
            "No specific theoretical framework",
            "No causal identification strategy",
            "No specific data sources",
        ],
    },
    dimension_anchors=[
        DimensionAnchor(
            dimension="correctness",
            strong_description="All theories correctly cited with proper attribution; methods match standard economics practice",
            weak_description="Theories mischaracterized or non-existent; methods inappropriate for the question",
            discriminating_features=[
                "named_theories_with_citations",
                "specific_methodology",
                "identified_data_sources",
                "causal_identification",
            ],
        ),
        DimensionAnchor(
            dimension="economic_rigor",
            strong_description="Uses precise economics terminology; proposes standard econometric methods; identifies appropriate data sources",
            weak_description="Vague terminology; no specific methodology; generic data references",
            discriminating_features=[
                "precise_terminology",
                "standard_econometric_method",
                "specific_data_provider",
            ],
        ),
        DimensionAnchor(
            dimension="innovation_potential",
            strong_description="Identifies genuine research gap; proposes novel cross-domain connection; actionable research design",
            weak_description="Restates known questions; no novel combination; too vague to implement",
            discriminating_features=[
                "identifies_gap",
                "cross_domain_connection",
                "actionable_design",
            ],
        ),
    ],
)


_LITERATURE_BENCHMARK = TeamBenchmark(
    team="LiteratureTeam",
    strong_exemplar={
        "literature_review": {
            "papers_reviewed": 25,
            "thematic_categories": [
                "AI and labor markets",
                "Skill-biased technical change",
                "Automation and employment",
                "AI adoption patterns",
            ],
            "gaps_identified": [
                "Limited evidence on AI augmentation effects (vs. automation)",
                "Cross-country comparative studies rare",
                "Interaction between AI adoption and labor market institutions understudied",
            ],
            "synthesis": "The literature establishes that automation displaces routine tasks, but emerging evidence suggests AI may augment rather than replace non-routine cognitive tasks...",
        },
        "key_qualities": [
            "Systematic coverage of relevant subfields",
            "Clear thematic organization",
            "Genuine gaps identified from evidence",
            "Synthesis builds coherent narrative",
        ],
    },
    weak_exemplar={
        "literature_review": {
            "papers_reviewed": 5,
            "thematic_categories": ["AI economics"],
            "gaps_identified": ["More research needed on AI"],
            "synthesis": "AI is an important topic in economics and more research is needed.",
        },
        "key_weaknesses": [
            "Insufficient coverage",
            "No meaningful thematic organization",
            "Trivial gap identification",
            "Synthesis adds no insight",
        ],
    },
    dimension_anchors=[
        DimensionAnchor(
            dimension="correctness",
            strong_description="Papers accurately cited; gap claims supported by systematic analysis",
            weak_description="Fabricated citations; gaps are trivial or already addressed",
            discriminating_features=[
                "verifiable_citations",
                "systematic_gap_analysis",
                "accurate_paper_summaries",
            ],
        ),
        DimensionAnchor(
            dimension="economic_rigor",
            strong_description="Correct methodological classification; appropriate field coverage; nuanced contribution assessment",
            weak_description="Misclassified methods; narrow coverage; superficial assessment",
            discriminating_features=[
                "correct_method_classification",
                "multi_subfield_coverage",
                "nuanced_contribution_assessment",
            ],
        ),
    ],
)


_MODEL_BENCHMARK = TeamBenchmark(
    team="ModelTeam",
    strong_exemplar={
        "model_specification": {
            "model_type": "DSGE with heterogeneous agents",
            "key_equations": 12,
            "assumptions": [
                "Representative firm with CES production function",
                "Heterogeneous households with idiosyncratic productivity shocks",
                "AI capital follows Acemoglu-Restrepo task-based framework",
            ],
            "calibration_targets": [
                "US labor share of income (BLS)",
                "AI adoption rates by industry (Census)",
                "Skill premium (CPS data)",
            ],
            "sensitivity_analysis": "Elasticity of substitution varied 0.5-1.5; key finding robust",
        },
        "key_qualities": [
            "Well-specified model with clear assumptions",
            "Calibration uses standard data sources",
            "Sensitivity analysis covers key parameters",
            "Theory-model alignment is explicit",
        ],
    },
    weak_exemplar={
        "model_specification": {
            "model_type": "Economic model",
            "key_equations": 2,
            "assumptions": ["Standard assumptions"],
            "calibration_targets": ["General economic data"],
            "sensitivity_analysis": "Not performed",
        },
        "key_weaknesses": [
            "Model type unspecified",
            "Assumptions not stated",
            "No specific calibration data",
            "No sensitivity analysis",
        ],
    },
    dimension_anchors=[
        DimensionAnchor(
            dimension="correctness",
            strong_description="Equations dimensionally consistent; assumptions justified with citations",
            weak_description="Mathematical errors; unjustified assumptions",
            discriminating_features=[
                "dimensional_consistency",
                "justified_assumptions",
                "complete_equation_system",
            ],
        ),
        DimensionAnchor(
            dimension="economic_rigor",
            strong_description="Follows standard DSGE/structural modeling conventions; calibration per economics practice",
            weak_description="Non-standard specification; ad hoc calibration",
            discriminating_features=[
                "standard_model_conventions",
                "rigorous_calibration",
                "appropriate_dynamics",
            ],
        ),
    ],
)


_DATA_BENCHMARK = TeamBenchmark(
    team="DataTeam",
    strong_exemplar={
        "data_requirements": {
            "variables": [
                {"name": "real_gdp", "source": "FRED", "series_id": "GDPC1", "frequency": "quarterly"},
                {"name": "unemployment_rate", "source": "BLS", "series_id": "LNS14000000", "frequency": "monthly"},
                {"name": "cpi_all_items", "source": "FRED", "series_id": "CPIAUCSL", "frequency": "monthly"},
            ],
            "time_period": "1990Q1-2024Q4",
            "cross_section": "US aggregate",
            "quality_checks": ["stationarity_test", "missing_value_imputation", "outlier_detection"],
        },
        "key_qualities": [
            "Specific series IDs and data sources",
            "Appropriate frequency and time period",
            "Comprehensive quality checks specified",
            "Standard economics data providers",
        ],
    },
    weak_exemplar={
        "data_requirements": {
            "variables": [
                {"name": "economic data", "source": "internet", "frequency": "unknown"},
            ],
            "time_period": "recent",
            "cross_section": "various countries",
            "quality_checks": [],
        },
        "key_weaknesses": [
            "No specific data series or sources",
            "Undefined time period and frequency",
            "No quality checks",
            "Non-standard data sources",
        ],
    },
    dimension_anchors=[
        DimensionAnchor(
            dimension="correctness",
            strong_description="Valid API endpoints; correct series IDs; appropriate variable specifications",
            weak_description="Invalid sources; wrong series IDs; vague specifications",
            discriminating_features=[
                "valid_series_ids",
                "correct_frequencies",
                "appropriate_time_periods",
            ],
        ),
        DimensionAnchor(
            dimension="economic_rigor",
            strong_description="Standard economics data providers; correct variable construction; appropriate temporal specs",
            weak_description="Non-standard sources; incorrect variable definitions; wrong temporal granularity",
            discriminating_features=[
                "standard_data_providers",
                "correct_variable_construction",
                "appropriate_temporal_specs",
            ],
        ),
    ],
)


_BENCHMARKS: Dict[str, TeamBenchmark] = {
    "IdeationTeam": _IDEATION_BENCHMARK,
    "LiteratureTeam": _LITERATURE_BENCHMARK,
    "ModelTeam": _MODEL_BENCHMARK,
    "DataTeam": _DATA_BENCHMARK,
}


def get_team_benchmark(team: str) -> Optional[TeamBenchmark]:
    """Get the benchmark for a specific team."""
    return _BENCHMARKS.get(team)


def get_all_benchmarks() -> Dict[str, TeamBenchmark]:
    """Get all team benchmarks."""
    return dict(_BENCHMARKS)


class BenchmarkComparator:
    """
    Compares workflow outputs against benchmark exemplars.

    Uses structural feature matching to assess whether outputs
    are closer to strong or weak exemplars, without requiring LLM calls.
    """

    def compare(
        self,
        outputs: Dict[str, Any],
        benchmark: TeamBenchmark,
    ) -> BenchmarkResult:
        """
        Compare outputs against a team benchmark.

        Args:
            outputs: Workflow output files {filename: content}.
            benchmark: TeamBenchmark with strong/weak exemplars.

        Returns:
            BenchmarkResult with relative quality assessment.
        """
        # Extract feature matches from outputs
        feature_matches = {}
        dimension_assessments = {}

        for anchor in benchmark.dimension_anchors:
            matches = 0
            total = len(anchor.discriminating_features)
            for feat in anchor.discriminating_features:
                present = self._check_feature(feat, outputs)
                feature_matches[feat] = present
                if present:
                    matches += 1
            if total > 0:
                dimension_assessments[anchor.dimension] = matches / total

        # Overall relative quality
        if dimension_assessments:
            relative_quality = sum(dimension_assessments.values()) / len(
                dimension_assessments
            )
        else:
            relative_quality = 0.5  # neutral if no anchors

        return BenchmarkResult(
            team=benchmark.team,
            relative_quality=relative_quality,
            dimension_assessments=dimension_assessments,
            feature_matches=feature_matches,
        )

    def _check_feature(self, feature: str, outputs: Dict[str, Any]) -> bool:
        """Check if a discriminating feature is present in outputs."""
        # Convert all outputs to a single text for searching
        text = self._flatten_outputs(outputs).lower()

        feature_checks = {
            "named_theories_with_citations": self._has_named_theories(text),
            "specific_methodology": self._has_specific_methodology(text),
            "identified_data_sources": self._has_data_sources(text),
            "causal_identification": self._has_causal_identification(text),
            "precise_terminology": self._has_precise_terminology(text),
            "standard_econometric_method": self._has_econometric_method(text),
            "specific_data_provider": self._has_data_provider(text),
            "identifies_gap": self._has_gap_identification(text),
            "cross_domain_connection": self._has_cross_domain(text),
            "actionable_design": self._has_actionable_design(text),
            "verifiable_citations": self._has_citations(text),
            "systematic_gap_analysis": self._has_systematic_analysis(text),
            "accurate_paper_summaries": len(text) > 500,
            "correct_method_classification": self._has_method_classification(text),
            "multi_subfield_coverage": self._has_multi_subfield(text),
            "nuanced_contribution_assessment": len(text) > 1000,
            "dimensional_consistency": self._has_equations(text),
            "justified_assumptions": "assumption" in text and ("because" in text or "following" in text),
            "complete_equation_system": self._has_equations(text),
            "standard_model_conventions": self._has_model_type(text),
            "rigorous_calibration": "calibrat" in text,
            "appropriate_dynamics": "equilibrium" in text or "dynamic" in text,
            "valid_series_ids": self._has_series_ids(text),
            "correct_frequencies": any(f in text for f in ["quarterly", "monthly", "annual", "daily"]),
            "appropriate_time_periods": any(str(y) in text for y in range(1990, 2027)),
            "standard_data_providers": self._has_data_provider(text),
            "correct_variable_construction": self._has_variable_specs(text),
            "appropriate_temporal_specs": any(f in text for f in ["frequency", "period", "time series"]),
        }

        return feature_checks.get(feature, False)

    def _flatten_outputs(self, outputs: Dict[str, Any]) -> str:
        """Flatten all outputs into a single text string."""
        import json

        parts = []
        for _, content in outputs.items():
            if isinstance(content, (dict, list)):
                parts.append(json.dumps(content, default=str))
            else:
                parts.append(str(content))
        return " ".join(parts)

    def _has_named_theories(self, text: str) -> bool:
        theories = [
            "keynesian", "monetar", "neoclassical", "behavioral",
            "game theory", "general equilibrium", "rational expectations",
            "sbtc", "skill-biased", "endogenous growth", "new trade",
            "dsge", "overlapping generations", "real business cycle",
        ]
        return sum(1 for t in theories if t in text) >= 2

    def _has_specific_methodology(self, text: str) -> bool:
        methods = [
            "difference-in-differences", "regression discontinuity",
            "instrumental variable", "panel data", "time series",
            "structural estimation", "gmm", "maximum likelihood",
            "bayesian", "var", "dsge", "calibration",
        ]
        return any(m in text for m in methods)

    def _has_data_sources(self, text: str) -> bool:
        sources = [
            "fred", "bls", "census", "world bank", "imf", "oecd",
            "eurostat", "compustat", "crsp", "penn world", "nber",
        ]
        return any(s in text for s in sources)

    def _has_causal_identification(self, text: str) -> bool:
        causal = [
            "causal", "identification", "endogeneity",
            "instrument", "natural experiment", "quasi-experiment",
        ]
        return any(c in text for c in causal)

    def _has_precise_terminology(self, text: str) -> bool:
        terms = [
            "elasticity", "marginal", "equilibrium", "welfare",
            "externality", "utility", "production function",
            "aggregate demand", "monetary policy", "fiscal",
        ]
        return sum(1 for t in terms if t in text) >= 3

    def _has_econometric_method(self, text: str) -> bool:
        methods = [
            "ols", "iv", "2sls", "gmm", "probit", "logit",
            "tobit", "panel", "fixed effect", "random effect",
            "var", "cointegration", "granger",
        ]
        return any(m in text for m in methods)

    def _has_data_provider(self, text: str) -> bool:
        providers = [
            "fred", "bls", "census", "world bank", "imf", "oecd",
            "bloomberg", "refinitiv", "compustat", "wrds",
        ]
        return any(p in text for p in providers)

    def _has_gap_identification(self, text: str) -> bool:
        return "gap" in text or "underexplored" in text or "limited research" in text

    def _has_cross_domain(self, text: str) -> bool:
        domains = [
            "macroeconomics", "microeconomics", "behavioral",
            "labor", "trade", "finance", "development",
            "industrial organization", "public economics",
        ]
        return sum(1 for d in domains if d in text) >= 2

    def _has_actionable_design(self, text: str) -> bool:
        return ("data" in text and "method" in text) or "testable" in text

    def _has_citations(self, text: str) -> bool:
        import re
        # Look for year citations like (Smith 2020) or (2020)
        return bool(re.search(r"\(\d{4}\)", text)) or bool(re.search(r"\b\d{4}\b", text))

    def _has_systematic_analysis(self, text: str) -> bool:
        return "systematic" in text or ("theme" in text and "gap" in text)

    def _has_method_classification(self, text: str) -> bool:
        return "empirical" in text or "theoretical" in text or "experimental" in text

    def _has_multi_subfield(self, text: str) -> bool:
        return self._has_cross_domain(text)

    def _has_equations(self, text: str) -> bool:
        import re
        return bool(re.search(r"[=+\-*/].*[a-zA-Z]", text))

    def _has_model_type(self, text: str) -> bool:
        types = ["dsge", "var", "structural", "reduced form", "calibration"]
        return any(t in text for t in types)

    def _has_series_ids(self, text: str) -> bool:
        import re
        # Look for FRED-style series IDs (uppercase letters + optional numbers)
        return bool(re.search(r"\b[A-Z]{3,}[0-9]*\b", text))

    def _has_variable_specs(self, text: str) -> bool:
        specs = ["gdp", "cpi", "unemployment", "inflation", "interest rate"]
        return sum(1 for s in specs if s in text) >= 2
