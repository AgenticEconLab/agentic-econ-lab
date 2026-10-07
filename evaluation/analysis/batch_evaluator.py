# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Batch evaluator for comparing workflows across teams and modes.

Supports:
- Evaluating all modes of a team
- Comparing teams using the same mode
- Running full cross-team, cross-mode comparisons
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

from ..schemas.execution import ExecutionTrace
from ..schemas.metrics import EvaluationResult, DimensionScore, MetricResult
from ..core.dimensions import DIMENSIONS, DimensionCluster, get_dimensions_by_cluster
from ..core.metrics import MetricCalculator
from ..parsers.ael_parser import AELParser
from ..runners.ael_runner import AELRunner, RunConfig, ExecutionResult


@dataclass
class TeamEvaluation:
    """Evaluation results for a single team across all modes."""

    team: str
    framework: str
    modes_evaluated: List[str]
    n_runs_per_mode: int
    evaluation_timestamp: datetime
    mode_results: Dict[str, List[EvaluationResult]]
    summary: Dict[str, Any] = field(default_factory=dict)

    def get_mode_averages(self, mode: str) -> Dict[str, float]:
        """Get average scores for a mode across all runs."""
        results = self.mode_results.get(mode, [])
        if not results:
            return {}

        dim_scores = {}
        for dim_id in DIMENSIONS.keys():
            scores = []
            for result in results:
                for ds in result.dimension_scores:
                    if ds.dimension == dim_id:
                        scores.append(ds.score)
            if scores:
                dim_scores[dim_id] = sum(scores) / len(scores)

        return dim_scores

    def get_best_mode(self, dimension: str = None) -> str:
        """Get the best performing mode overall or for a dimension."""
        mode_avg = {}
        for mode in self.modes_evaluated:
            averages = self.get_mode_averages(mode)
            if dimension:
                mode_avg[mode] = averages.get(dimension, 0.0)
            else:
                mode_avg[mode] = sum(averages.values()) / len(averages) if averages else 0.0

        if not mode_avg:
            return ""
        return max(mode_avg, key=mode_avg.get)


@dataclass
class ModeComparison:
    """Comparison results between modes."""

    team: str
    modes: List[str]
    comparison_type: str  # "fc_vs_no_fc", "hitl_vs_no_hitl", "full"
    dimension_comparisons: Dict[str, Dict[str, float]]  # dimension -> mode -> score
    statistical_tests: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class FullEvaluation:
    """Complete evaluation across all teams and modes."""

    framework: str
    evaluation_timestamp: datetime
    teams_evaluated: List[str]
    team_evaluations: Dict[str, TeamEvaluation]
    cross_team_comparison: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)


class BatchEvaluator:
    """
    Batch evaluator for comprehensive workflow comparison.

    Supports evaluating workflows from existing outputs or by running workflows.
    """

    def __init__(
        self,
        base_path: Optional[Path] = None,
        framework: str = "ael",
    ):
        """
        Initialize batch evaluator.

        Args:
            base_path: Base path to workflow codebase
            framework: Framework to evaluate (default: ael)
        """
        self.base_path = base_path or Path.cwd()
        self.framework = framework

        # Initialize components
        if framework == "ael":
            self.parser = AELParser(base_path)
            self.runner = AELRunner(base_path)
        else:
            raise ValueError(f"Unsupported framework: {framework}")

        self.calculator = MetricCalculator()

    def evaluate_from_outputs(
        self,
        team: str,
        mode: str,
        output_dirs: List[Path],
        log_files: Optional[List[Path]] = None,
    ) -> List[EvaluationResult]:
        """
        Evaluate from existing workflow outputs.

        Args:
            team: Team name
            mode: Mode name
            output_dirs: List of output directories from different runs
            log_files: Optional list of log files corresponding to each run

        Returns:
            List of evaluation results, one per output directory
        """
        results = []
        traces = []

        # Build execution traces from outputs
        for i, output_dir in enumerate(output_dirs):
            log_file = log_files[i] if log_files and i < len(log_files) else None
            trace = self.parser.build_execution_trace(
                team=team,
                mode=mode,
                output_dir=output_dir,
                log_file=log_file,
            )
            traces.append(trace)

        # Calculate metrics - single trace metrics for each run
        for trace in traces:
            single_metrics = self.calculator.calculate_single_trace_metrics(trace)

            # Build dimension scores
            dimension_scores = self._build_dimension_scores(single_metrics, {})
            overall_score = sum(ds.score for ds in dimension_scores) / len(dimension_scores) if dimension_scores else 0.0

            results.append(EvaluationResult(
                team=team,
                mode=mode,
                run_id=trace.run_id,
                framework=self.framework,
                dimension_scores=dimension_scores,
                overall_score=overall_score,
                timestamp=datetime.now(),
            ))

        # Calculate multi-trace metrics if we have multiple runs
        if len(traces) > 1:
            multi_metrics = self.calculator.calculate_multi_trace_metrics(traces)

            # Update results with multi-trace metrics
            for result in results:
                for ds in result.dimension_scores:
                    for metric_id, value in multi_metrics.items():
                        # Check if this metric belongs to this dimension
                        dim_def = DIMENSIONS.get(ds.dimension)
                        if dim_def:
                            for metric_def in dim_def.metrics:
                                if metric_def.metric_id == metric_id:
                                    ds.metrics.append(MetricResult(
                                        metric_id=metric_id,
                                        metric_name=metric_def.name,
                                        dimension=ds.dimension,
                                        value=value,
                                        normalized_value=metric_def.normalize(value),
                                    ))

        return results

    def _build_dimension_scores(
        self,
        single_metrics: Dict[str, float],
        multi_metrics: Dict[str, float],
    ) -> List[DimensionScore]:
        """Build dimension scores from calculated metrics."""
        dimension_scores = []

        for dim_id, dim_def in DIMENSIONS.items():
            metrics = []
            scores = []

            for metric_def in dim_def.metrics:
                value = single_metrics.get(metric_def.metric_id) or multi_metrics.get(metric_def.metric_id)
                if value is not None:
                    normalized = metric_def.normalize(value)
                    scores.append(normalized)
                    metrics.append(MetricResult(
                        metric_id=metric_def.metric_id,
                        metric_name=metric_def.name,
                        dimension=dim_id,
                        value=value,
                        normalized_value=normalized,
                    ))

            # Calculate dimension score as average of metric scores
            score = sum(scores) / len(scores) if scores else 0.0

            dimension_scores.append(DimensionScore(
                dimension=dim_id,
                cluster=dim_def.cluster.value,
                score=score,
                metrics=metrics,
            ))

        return dimension_scores

    def evaluate_team(
        self,
        team: str,
        modes: Optional[List[str]] = None,
        n_runs: int = 1,
        inputs: Optional[Dict[str, Any]] = None,
        run_workflows: bool = False,
        output_base_dir: Optional[Path] = None,
    ) -> TeamEvaluation:
        """
        Evaluate all modes of a team.

        Args:
            team: Team name
            modes: List of modes to evaluate (default: all supported)
            n_runs: Number of runs per mode
            inputs: Input parameters for workflows
            run_workflows: If True, run workflows; if False, use existing outputs
            output_base_dir: Base directory for outputs

        Returns:
            TeamEvaluation with results for all modes
        """
        if modes is None:
            modes = self.runner.get_supported_modes(team)

        mode_results = {}
        timestamp = datetime.now()

        for mode in modes:
            if run_workflows:
                # Run workflows and collect results
                results = self._run_and_evaluate(
                    team=team,
                    mode=mode,
                    n_runs=n_runs,
                    inputs=inputs or {},
                    output_base_dir=output_base_dir,
                )
            else:
                # Find existing outputs
                output_dirs = self._find_existing_outputs(team, mode, output_base_dir)
                if output_dirs:
                    results = self.evaluate_from_outputs(team, mode, output_dirs)
                else:
                    results = []

            mode_results[mode] = results

        evaluation = TeamEvaluation(
            team=team,
            framework=self.framework,
            modes_evaluated=modes,
            n_runs_per_mode=n_runs,
            evaluation_timestamp=timestamp,
            mode_results=mode_results,
        )

        # Compute summary statistics
        evaluation.summary = self._compute_team_summary(evaluation)

        return evaluation

    def _run_and_evaluate(
        self,
        team: str,
        mode: str,
        n_runs: int,
        inputs: Dict[str, Any],
        output_base_dir: Optional[Path],
    ) -> List[EvaluationResult]:
        """Run workflows and evaluate results."""
        config = RunConfig(
            team=team,
            mode=mode,
            inputs=inputs,
        )

        # Run multiple times
        execution_results = self.runner.run_repeated(config, n_runs)

        # Collect output directories
        output_dirs = [r.output_dir for r in execution_results if r.output_dir and r.output_dir.exists()]
        log_files = [r.log_file for r in execution_results if r.log_file and r.log_file.exists()]

        if not output_dirs:
            return []

        return self.evaluate_from_outputs(team, mode, output_dirs, log_files)

    def _find_existing_outputs(
        self,
        team: str,
        mode: str,
        output_base_dir: Optional[Path],
    ) -> List[Path]:
        """Find existing output directories for a team/mode."""
        if output_base_dir is None:
            output_base_dir = self.base_path / "outputs"

        search_dir = output_base_dir / team / mode
        if not search_dir.exists():
            return []

        # Find all run directories
        output_dirs = [d for d in search_dir.iterdir() if d.is_dir()]
        return sorted(output_dirs)

    def _compute_team_summary(self, evaluation: TeamEvaluation) -> Dict[str, Any]:
        """Compute summary statistics for a team evaluation."""
        summary = {
            "total_runs": 0,
            "successful_runs": 0,
            "dimension_averages": {},
            "cluster_averages": {},
            "best_mode": {},
        }

        # Count runs
        for mode, results in evaluation.mode_results.items():
            summary["total_runs"] += len(results)
            summary["successful_runs"] += sum(1 for r in results if r.overall_score > 0)

        # Calculate dimension averages across all modes
        all_dim_scores = {dim_id: [] for dim_id in DIMENSIONS.keys()}
        for mode, results in evaluation.mode_results.items():
            for result in results:
                for ds in result.dimension_scores:
                    all_dim_scores[ds.dimension].append(ds.score)

        for dim_id, scores in all_dim_scores.items():
            if scores:
                summary["dimension_averages"][dim_id] = sum(scores) / len(scores)

        # Calculate cluster averages
        for cluster_name, dim_ids in {
            "Quality": ["reliability", "correctness", "soundness"],
            "Operational": ["efficiency", "scalability", "robustness"],
            "Epistemic": ["transparency", "traceability", "reproducibility"],
            "Innovation": ["innovation_potential"],
        }.items():
            cluster_scores = [
                summary["dimension_averages"].get(d, 0)
                for d in dim_ids
                if d in summary["dimension_averages"]
            ]
            if cluster_scores:
                summary["cluster_averages"][cluster_name] = sum(cluster_scores) / len(cluster_scores)

        # Find best mode for each dimension
        for dim_id in DIMENSIONS.keys():
            best_mode = evaluation.get_best_mode(dim_id)
            if best_mode:
                summary["best_mode"][dim_id] = best_mode

        return summary

    def compare_modes(
        self,
        team: str,
        comparison_type: str = "full",
    ) -> ModeComparison:
        """
        Compare modes within a team.

        Args:
            team: Team name
            comparison_type: Type of comparison:
                - "fc_vs_no_fc": Compare Firecrawl modes
                - "hitl_vs_no_hitl": Compare HITL modes
                - "full": Full pairwise comparison

        Returns:
            ModeComparison with detailed comparison results
        """
        modes = self.runner.get_supported_modes(team)

        # Filter modes based on comparison type
        if comparison_type == "fc_vs_no_fc":
            mode_pairs = [
                (m, m.replace("NoFc", "WithFc"))
                for m in modes if "NoFc" in m
            ]
            modes = list(set(m for pair in mode_pairs for m in pair if m in modes))
        elif comparison_type == "hitl_vs_no_hitl":
            mode_pairs = [
                (m, m.replace("NoHITL", "WithHITL"))
                for m in modes if "NoHITL" in m
            ]
            modes = list(set(m for pair in mode_pairs for m in pair if m in modes))

        # Find existing outputs and evaluate
        dimension_comparisons = {dim_id: {} for dim_id in DIMENSIONS.keys()}

        for mode in modes:
            output_dirs = self._find_existing_outputs(team, mode, None)
            if output_dirs:
                results = self.evaluate_from_outputs(team, mode, output_dirs)
                for dim_id in DIMENSIONS.keys():
                    scores = []
                    for result in results:
                        for ds in result.dimension_scores:
                            if ds.dimension == dim_id:
                                scores.append(ds.score)
                    if scores:
                        dimension_comparisons[dim_id][mode] = sum(scores) / len(scores)

        # Generate recommendations
        recommendations = self._generate_recommendations(dimension_comparisons, comparison_type)

        return ModeComparison(
            team=team,
            modes=modes,
            comparison_type=comparison_type,
            dimension_comparisons=dimension_comparisons,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        dimension_comparisons: Dict[str, Dict[str, float]],
        comparison_type: str,
    ) -> List[str]:
        """Generate recommendations based on comparison results."""
        recommendations = []

        for dim_id, mode_scores in dimension_comparisons.items():
            if not mode_scores:
                continue

            best_mode = max(mode_scores, key=mode_scores.get)
            best_score = mode_scores[best_mode]

            dim_def = DIMENSIONS.get(dim_id)
            if dim_def:
                recommendations.append(
                    f"For {dim_def.name}: {best_mode} performs best (score: {best_score:.3f})"
                )

        return recommendations

    def evaluate_all_teams(
        self,
        teams: Optional[List[str]] = None,
        n_runs: int = 1,
        run_workflows: bool = False,
    ) -> FullEvaluation:
        """
        Run full evaluation across all teams and modes.

        Args:
            teams: List of teams to evaluate (default: all supported)
            n_runs: Number of runs per mode
            run_workflows: If True, run workflows; if False, use existing outputs

        Returns:
            FullEvaluation with complete results
        """
        if teams is None:
            teams = self.runner.SUPPORTED_TEAMS

        timestamp = datetime.now()
        team_evaluations = {}

        for team in teams:
            evaluation = self.evaluate_team(
                team=team,
                n_runs=n_runs,
                run_workflows=run_workflows,
            )
            team_evaluations[team] = evaluation

        full_evaluation = FullEvaluation(
            framework=self.framework,
            evaluation_timestamp=timestamp,
            teams_evaluated=teams,
            team_evaluations=team_evaluations,
        )

        # Compute cross-team comparison
        full_evaluation.cross_team_comparison = self._compute_cross_team_comparison(team_evaluations)

        # Compute overall summary
        full_evaluation.summary = self._compute_full_summary(full_evaluation)

        return full_evaluation

    def _compute_cross_team_comparison(
        self,
        team_evaluations: Dict[str, TeamEvaluation],
    ) -> Dict[str, Any]:
        """Compute cross-team comparison statistics."""
        comparison = {
            "dimension_by_team": {},
            "cluster_by_team": {},
            "overall_by_team": {},
        }

        for team, evaluation in team_evaluations.items():
            if evaluation.summary:
                comparison["dimension_by_team"][team] = evaluation.summary.get("dimension_averages", {})
                comparison["cluster_by_team"][team] = evaluation.summary.get("cluster_averages", {})

                # Overall score
                dim_avgs = evaluation.summary.get("dimension_averages", {})
                if dim_avgs:
                    comparison["overall_by_team"][team] = sum(dim_avgs.values()) / len(dim_avgs)

        return comparison

    def _compute_full_summary(self, full_evaluation: FullEvaluation) -> Dict[str, Any]:
        """Compute summary for full evaluation."""
        summary = {
            "total_teams": len(full_evaluation.teams_evaluated),
            "total_modes": 0,
            "total_runs": 0,
            "best_team_by_dimension": {},
            "best_team_overall": None,
        }

        for team, evaluation in full_evaluation.team_evaluations.items():
            summary["total_modes"] += len(evaluation.modes_evaluated)
            summary["total_runs"] += evaluation.summary.get("total_runs", 0)

        # Find best team by dimension
        cross_team = full_evaluation.cross_team_comparison
        for dim_id in DIMENSIONS.keys():
            best_team = None
            best_score = 0
            for team, dim_scores in cross_team.get("dimension_by_team", {}).items():
                score = dim_scores.get(dim_id, 0)
                if score > best_score:
                    best_score = score
                    best_team = team
            if best_team:
                summary["best_team_by_dimension"][dim_id] = best_team

        # Find best team overall
        overall_scores = cross_team.get("overall_by_team", {})
        if overall_scores:
            summary["best_team_overall"] = max(overall_scores, key=overall_scores.get)

        return summary

    def save_results(
        self,
        evaluation: FullEvaluation,
        output_dir: Path,
        format: str = "json",
    ) -> Path:
        """
        Save evaluation results to file.

        Args:
            evaluation: Evaluation results
            output_dir: Directory to save to
            format: Output format (json, csv, markdown)

        Returns:
            Path to saved file
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = evaluation.evaluation_timestamp.strftime("%Y%m%d_%H%M%S")

        if format == "json":
            filepath = output_dir / f"evaluation_{timestamp}.json"
            data = self._evaluation_to_dict(evaluation)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            return filepath

        raise ValueError(f"Unsupported format: {format}")

    def _evaluation_to_dict(self, evaluation: FullEvaluation) -> Dict[str, Any]:
        """Convert evaluation to dictionary for JSON serialization."""
        return {
            "framework": evaluation.framework,
            "evaluation_timestamp": evaluation.evaluation_timestamp.isoformat(),
            "teams_evaluated": evaluation.teams_evaluated,
            "team_evaluations": {
                team: {
                    "team": te.team,
                    "framework": te.framework,
                    "modes_evaluated": te.modes_evaluated,
                    "n_runs_per_mode": te.n_runs_per_mode,
                    "summary": te.summary,
                }
                for team, te in evaluation.team_evaluations.items()
            },
            "cross_team_comparison": evaluation.cross_team_comparison,
            "summary": evaluation.summary,
        }
