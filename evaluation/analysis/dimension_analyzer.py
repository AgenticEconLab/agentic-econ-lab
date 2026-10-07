# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Dimension-specific analysis for workflow evaluation.

Provides deep analysis of individual dimensions and clusters,
including statistical analysis and visualization data generation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import statistics

from ..schemas.execution import ExecutionTrace
from ..schemas.metrics import EvaluationResult, DimensionScore, MetricResult
from ..core.dimensions import (
    DIMENSIONS,
    DimensionCluster,
    DimensionDefinition,
    MetricDefinition,
    get_dimension,
    get_dimensions_by_cluster,
    DIMENSION_CLUSTERS,
)


@dataclass
class MetricAnalysis:
    """Analysis results for a single metric."""

    metric_id: str
    metric_name: str
    dimension: str
    values: List[float]
    normalized_values: List[float]
    mean: float
    std_dev: float
    min_value: float
    max_value: float
    median: float
    count: int


@dataclass
class DimensionAnalysis:
    """Analysis results for a single dimension."""

    dimension_id: str
    dimension_name: str
    cluster: str
    description: str
    metric_analyses: List[MetricAnalysis]
    overall_score: float
    score_std_dev: float
    sample_size: int
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class ClusterAnalysis:
    """Analysis results for a dimension cluster."""

    cluster_name: str
    dimensions: List[str]
    dimension_analyses: Dict[str, DimensionAnalysis]
    cluster_score: float
    cluster_std_dev: float


@dataclass
class RadarChartData:
    """Data structure for radar chart visualization."""

    dimensions: List[str]
    dimension_names: List[str]
    series: Dict[str, List[float]]  # Series name -> scores for each dimension


class DimensionAnalyzer:
    """
    Analyzer for dimension-level workflow evaluation.

    Provides detailed analysis of individual dimensions and clusters,
    statistical comparisons, and visualization data generation.
    """

    def __init__(self):
        """Initialize dimension analyzer."""
        pass

    def analyze_dimension(
        self,
        dimension_id: str,
        results: List[EvaluationResult],
    ) -> DimensionAnalysis:
        """
        Analyze a specific dimension across evaluation results.

        Args:
            dimension_id: ID of the dimension to analyze
            results: List of evaluation results

        Returns:
            DimensionAnalysis with detailed statistics
        """
        dim_def = get_dimension(dimension_id)
        if not dim_def:
            raise ValueError(f"Unknown dimension: {dimension_id}")

        # Collect scores for this dimension
        dim_scores = []
        metric_values = {m.metric_id: [] for m in dim_def.metrics}
        metric_normalized = {m.metric_id: [] for m in dim_def.metrics}

        for result in results:
            for ds in result.dimension_scores:
                if ds.dimension == dimension_id:
                    dim_scores.append(ds.score)
                    for metric in ds.metrics:
                        if metric.metric_id in metric_values:
                            metric_values[metric.metric_id].append(metric.value)
                            if metric.normalized_value is not None:
                                metric_normalized[metric.metric_id].append(metric.normalized_value)

        # Build metric analyses
        metric_analyses = []
        for metric_def in dim_def.metrics:
            values = metric_values[metric_def.metric_id]
            normalized = metric_normalized.get(metric_def.metric_id, [])

            if values:
                metric_analyses.append(MetricAnalysis(
                    metric_id=metric_def.metric_id,
                    metric_name=metric_def.name,
                    dimension=dimension_id,
                    values=values,
                    normalized_values=normalized,
                    mean=statistics.mean(values),
                    std_dev=statistics.stdev(values) if len(values) > 1 else 0.0,
                    min_value=min(values),
                    max_value=max(values),
                    median=statistics.median(values),
                    count=len(values),
                ))

        # Calculate overall statistics
        overall_score = statistics.mean(dim_scores) if dim_scores else 0.0
        score_std_dev = statistics.stdev(dim_scores) if len(dim_scores) > 1 else 0.0

        # Generate insights
        strengths, weaknesses, recommendations = self._generate_dimension_insights(
            dim_def, metric_analyses, overall_score
        )

        return DimensionAnalysis(
            dimension_id=dimension_id,
            dimension_name=dim_def.name,
            cluster=dim_def.cluster.value,
            description=dim_def.description,
            metric_analyses=metric_analyses,
            overall_score=overall_score,
            score_std_dev=score_std_dev,
            sample_size=len(dim_scores),
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
        )

    def _generate_dimension_insights(
        self,
        dim_def: DimensionDefinition,
        metric_analyses: List[MetricAnalysis],
        overall_score: float,
    ) -> Tuple[List[str], List[str], List[str]]:
        """Generate strengths, weaknesses, and recommendations."""
        strengths = []
        weaknesses = []
        recommendations = []

        # Threshold for good/bad scores
        GOOD_THRESHOLD = 0.7
        BAD_THRESHOLD = 0.4

        for analysis in metric_analyses:
            metric_def = dim_def.get_metric(analysis.metric_id)
            if not metric_def:
                continue

            # Get normalized score
            if analysis.normalized_values:
                norm_mean = statistics.mean(analysis.normalized_values)
            else:
                norm_mean = metric_def.normalize(analysis.mean)

            if norm_mean >= GOOD_THRESHOLD:
                strengths.append(f"{metric_def.name}: Strong performance ({norm_mean:.2f})")
            elif norm_mean <= BAD_THRESHOLD:
                weaknesses.append(f"{metric_def.name}: Needs improvement ({norm_mean:.2f})")
                # Add recommendation based on metric
                rec = self._get_metric_recommendation(metric_def)
                if rec:
                    recommendations.append(rec)

        # Add general recommendations based on design features
        if overall_score < 0.5:
            for feature in dim_def.design_features[:2]:  # Top 2 features
                recommendations.append(f"Consider implementing: {feature}")

        return strengths, weaknesses, recommendations

    def _get_metric_recommendation(self, metric_def: MetricDefinition) -> Optional[str]:
        """Get specific recommendation for improving a metric."""
        recommendations = {
            "error_rate": "Review error handling and add more validation checkpoints",
            "stage_completion_rate": "Add retry mechanisms and fallback strategies",
            "execution_time": "Optimize slow stages and consider parallelization",
            "throughput": "Increase parallelization or reduce per-item processing time",
            "error_recovery_rate": "Implement more robust error recovery mechanisms",
            "documentation_coverage": "Add more structured output documentation",
            "provenance_completeness": "Enhance metadata tracking in outputs",
            "output_consistency_cv": "Reduce sources of non-determinism",
            "output_diversity": "Increase diversity of sources and exploration strategies",
        }
        return recommendations.get(metric_def.metric_id)

    def analyze_cluster(
        self,
        cluster_name: str,
        results: List[EvaluationResult],
    ) -> ClusterAnalysis:
        """
        Analyze a dimension cluster.

        Args:
            cluster_name: Name of the cluster (Quality, Operational, Epistemic, Innovation)
            results: List of evaluation results

        Returns:
            ClusterAnalysis with cluster-level statistics
        """
        dimension_ids = DIMENSION_CLUSTERS.get(cluster_name, [])
        if not dimension_ids:
            raise ValueError(f"Unknown cluster: {cluster_name}")

        dimension_analyses = {}
        cluster_scores = []

        for dim_id in dimension_ids:
            analysis = self.analyze_dimension(dim_id, results)
            dimension_analyses[dim_id] = analysis
            cluster_scores.append(analysis.overall_score)

        cluster_score = statistics.mean(cluster_scores) if cluster_scores else 0.0
        cluster_std_dev = statistics.stdev(cluster_scores) if len(cluster_scores) > 1 else 0.0

        return ClusterAnalysis(
            cluster_name=cluster_name,
            dimensions=dimension_ids,
            dimension_analyses=dimension_analyses,
            cluster_score=cluster_score,
            cluster_std_dev=cluster_std_dev,
        )

    def analyze_all_dimensions(
        self,
        results: List[EvaluationResult],
    ) -> Dict[str, DimensionAnalysis]:
        """
        Analyze all dimensions.

        Args:
            results: List of evaluation results

        Returns:
            Dictionary mapping dimension ID to analysis
        """
        analyses = {}
        for dim_id in DIMENSIONS.keys():
            analyses[dim_id] = self.analyze_dimension(dim_id, results)
        return analyses

    def analyze_all_clusters(
        self,
        results: List[EvaluationResult],
    ) -> Dict[str, ClusterAnalysis]:
        """
        Analyze all clusters.

        Args:
            results: List of evaluation results

        Returns:
            Dictionary mapping cluster name to analysis
        """
        analyses = {}
        for cluster_name in DIMENSION_CLUSTERS.keys():
            analyses[cluster_name] = self.analyze_cluster(cluster_name, results)
        return analyses

    def generate_radar_data(
        self,
        results_by_series: Dict[str, List[EvaluationResult]],
    ) -> RadarChartData:
        """
        Generate data for radar chart visualization.

        Args:
            results_by_series: Dictionary mapping series name to results
                (e.g., {"ModeNoWcNoHITL": [...], "ModeWithWcNoHITL": [...]})

        Returns:
            RadarChartData for plotting
        """
        dimensions = list(DIMENSIONS.keys())
        dimension_names = [DIMENSIONS[d].name for d in dimensions]

        series = {}
        for series_name, results in results_by_series.items():
            scores = []
            for dim_id in dimensions:
                dim_scores = []
                for result in results:
                    for ds in result.dimension_scores:
                        if ds.dimension == dim_id:
                            dim_scores.append(ds.score)
                scores.append(statistics.mean(dim_scores) if dim_scores else 0.0)
            series[series_name] = scores

        return RadarChartData(
            dimensions=dimensions,
            dimension_names=dimension_names,
            series=series,
        )

    def generate_cluster_radar_data(
        self,
        results_by_series: Dict[str, List[EvaluationResult]],
    ) -> RadarChartData:
        """
        Generate cluster-level radar chart data.

        Args:
            results_by_series: Dictionary mapping series name to results

        Returns:
            RadarChartData with cluster scores
        """
        clusters = list(DIMENSION_CLUSTERS.keys())

        series = {}
        for series_name, results in results_by_series.items():
            cluster_scores = []
            for cluster_name in clusters:
                cluster_analysis = self.analyze_cluster(cluster_name, results)
                cluster_scores.append(cluster_analysis.cluster_score)
            series[series_name] = cluster_scores

        return RadarChartData(
            dimensions=clusters,
            dimension_names=clusters,
            series=series,
        )

    def compare_results(
        self,
        results_a: List[EvaluationResult],
        results_b: List[EvaluationResult],
        label_a: str = "A",
        label_b: str = "B",
    ) -> Dict[str, Any]:
        """
        Compare two sets of evaluation results.

        Args:
            results_a: First set of results
            results_b: Second set of results
            label_a: Label for first set
            label_b: Label for second set

        Returns:
            Comparison dictionary with differences and statistical tests
        """
        comparison = {
            "labels": [label_a, label_b],
            "dimension_differences": {},
            "cluster_differences": {},
            "significant_differences": [],
            "winner_by_dimension": {},
            "overall_winner": None,
        }

        # Analyze all dimensions for both sets
        analyses_a = self.analyze_all_dimensions(results_a)
        analyses_b = self.analyze_all_dimensions(results_b)

        overall_a = []
        overall_b = []

        for dim_id in DIMENSIONS.keys():
            score_a = analyses_a[dim_id].overall_score
            score_b = analyses_b[dim_id].overall_score
            diff = score_b - score_a

            comparison["dimension_differences"][dim_id] = {
                label_a: score_a,
                label_b: score_b,
                "difference": diff,
                "percent_change": (diff / score_a * 100) if score_a > 0 else 0,
            }

            overall_a.append(score_a)
            overall_b.append(score_b)

            # Determine winner
            if abs(diff) > 0.05:  # Significant difference threshold
                winner = label_b if diff > 0 else label_a
                comparison["winner_by_dimension"][dim_id] = winner
                comparison["significant_differences"].append({
                    "dimension": dim_id,
                    "winner": winner,
                    "margin": abs(diff),
                })

        # Cluster-level comparison
        clusters_a = self.analyze_all_clusters(results_a)
        clusters_b = self.analyze_all_clusters(results_b)

        for cluster_name in DIMENSION_CLUSTERS.keys():
            score_a = clusters_a[cluster_name].cluster_score
            score_b = clusters_b[cluster_name].cluster_score
            comparison["cluster_differences"][cluster_name] = {
                label_a: score_a,
                label_b: score_b,
                "difference": score_b - score_a,
            }

        # Overall winner
        mean_a = statistics.mean(overall_a) if overall_a else 0
        mean_b = statistics.mean(overall_b) if overall_b else 0

        if abs(mean_b - mean_a) > 0.03:
            comparison["overall_winner"] = label_b if mean_b > mean_a else label_a

        return comparison

    def get_top_dimensions(
        self,
        results: List[EvaluationResult],
        n: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Get top performing dimensions.

        Args:
            results: Evaluation results
            n: Number of dimensions to return

        Returns:
            List of (dimension_id, score) tuples sorted by score descending
        """
        analyses = self.analyze_all_dimensions(results)
        sorted_dims = sorted(
            analyses.items(),
            key=lambda x: x[1].overall_score,
            reverse=True
        )
        return [(d[0], d[1].overall_score) for d in sorted_dims[:n]]

    def get_bottom_dimensions(
        self,
        results: List[EvaluationResult],
        n: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Get lowest performing dimensions.

        Args:
            results: Evaluation results
            n: Number of dimensions to return

        Returns:
            List of (dimension_id, score) tuples sorted by score ascending
        """
        analyses = self.analyze_all_dimensions(results)
        sorted_dims = sorted(
            analyses.items(),
            key=lambda x: x[1].overall_score,
        )
        return [(d[0], d[1].overall_score) for d in sorted_dims[:n]]

    def generate_summary_report(
        self,
        results: List[EvaluationResult],
        team: str = "",
        mode: str = "",
    ) -> Dict[str, Any]:
        """
        Generate a summary report for evaluation results.

        Args:
            results: Evaluation results
            team: Team name (for context)
            mode: Mode name (for context)

        Returns:
            Dictionary with summary information
        """
        analyses = self.analyze_all_dimensions(results)
        cluster_analyses = self.analyze_all_clusters(results)

        # Collect all strengths and weaknesses
        all_strengths = []
        all_weaknesses = []
        all_recommendations = []

        for analysis in analyses.values():
            all_strengths.extend(analysis.strengths)
            all_weaknesses.extend(analysis.weaknesses)
            all_recommendations.extend(analysis.recommendations)

        # Overall score
        overall_scores = [a.overall_score for a in analyses.values()]
        overall_score = statistics.mean(overall_scores) if overall_scores else 0.0

        return {
            "team": team,
            "mode": mode,
            "sample_size": len(results),
            "overall_score": overall_score,
            "top_dimensions": self.get_top_dimensions(results),
            "bottom_dimensions": self.get_bottom_dimensions(results),
            "cluster_scores": {
                name: ca.cluster_score
                for name, ca in cluster_analyses.items()
            },
            "strengths": all_strengths[:5],  # Top 5
            "weaknesses": all_weaknesses[:5],  # Top 5
            "recommendations": list(set(all_recommendations))[:5],  # Unique top 5
            "dimension_scores": {
                dim_id: analysis.overall_score
                for dim_id, analysis in analyses.items()
            },
        }
