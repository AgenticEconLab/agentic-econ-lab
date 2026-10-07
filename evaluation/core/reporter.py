# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Report generation for workflow evaluation.

Generates reports in multiple formats:
- JSON: Machine-readable structured data
- CSV: Spreadsheet-compatible tabular data
- Markdown: Human-readable documentation
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..schemas.metrics import EvaluationResult, DimensionScore
from ..analysis.batch_evaluator import FullEvaluation, TeamEvaluation, ModeComparison
from ..analysis.dimension_analyzer import DimensionAnalysis, ClusterAnalysis, RadarChartData
from .dimensions import DIMENSIONS, DIMENSION_CLUSTERS


class EvaluationReporter:
    """
    Report generator for evaluation results.

    Supports multiple output formats and report types.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize reporter.

        Args:
            output_dir: Default output directory for reports
        """
        self.output_dir = output_dir or Path("reports")

    def generate_json_report(
        self,
        evaluation: FullEvaluation,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Generate machine-readable JSON report.

        Args:
            evaluation: Full evaluation results
            output_path: Output file path

        Returns:
            Path to generated report
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"evaluation_report_{timestamp}.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "framework": evaluation.framework,
                "evaluation_timestamp": evaluation.evaluation_timestamp.isoformat(),
            },
            "summary": evaluation.summary,
            "teams": {},
            "cross_team_comparison": evaluation.cross_team_comparison,
        }

        for team, team_eval in evaluation.team_evaluations.items():
            report["teams"][team] = {
                "modes_evaluated": team_eval.modes_evaluated,
                "n_runs_per_mode": team_eval.n_runs_per_mode,
                "summary": team_eval.summary,
                "mode_results": {
                    mode: [self._result_to_dict(r) for r in results]
                    for mode, results in team_eval.mode_results.items()
                },
            }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        return output_path

    def _result_to_dict(self, result: EvaluationResult) -> Dict[str, Any]:
        """Convert evaluation result to dictionary."""
        return {
            "run_id": result.run_id,
            "team": result.team,
            "mode": result.mode,
            "framework": result.framework,
            "overall_score": result.overall_score,
            "timestamp": result.timestamp.isoformat() if result.timestamp else None,
            "dimension_scores": [
                {
                    "dimension": ds.dimension,
                    "cluster": ds.cluster,
                    "score": ds.score,
                    "metrics": [
                        {
                            "metric_id": m.metric_id,
                            "metric_name": m.metric_name,
                            "value": m.value,
                            "normalized_value": m.normalized_value,
                        }
                        for m in ds.metrics
                    ],
                }
                for ds in result.dimension_scores
            ],
        }

    def generate_csv_report(
        self,
        evaluation: FullEvaluation,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Generate CSV report for spreadsheet analysis.

        Args:
            evaluation: Full evaluation results
            output_path: Output file path

        Returns:
            Path to generated report
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"evaluation_report_{timestamp}.csv"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build rows
        rows = []
        headers = [
            "team", "mode", "run_id", "overall_score", "timestamp"
        ] + list(DIMENSIONS.keys())

        for team, team_eval in evaluation.team_evaluations.items():
            for mode, results in team_eval.mode_results.items():
                for result in results:
                    row = {
                        "team": team,
                        "mode": mode,
                        "run_id": result.run_id,
                        "overall_score": result.overall_score,
                        "timestamp": result.timestamp.isoformat() if result.timestamp else "",
                    }
                    # Add dimension scores
                    for ds in result.dimension_scores:
                        row[ds.dimension] = ds.score
                    rows.append(row)

        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

        return output_path

    def generate_markdown_report(
        self,
        evaluation: FullEvaluation,
        output_path: Optional[Path] = None,
        include_details: bool = True,
    ) -> Path:
        """
        Generate human-readable Markdown report.

        Args:
            evaluation: Full evaluation results
            output_path: Output file path
            include_details: Include detailed per-run results

        Returns:
            Path to generated report
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"evaluation_report_{timestamp}.md"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []

        # Header
        lines.append("# Workflow Evaluation Report")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Framework:** {evaluation.framework}")
        lines.append(f"**Evaluation Date:** {evaluation.evaluation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")

        summary = evaluation.summary
        lines.append(f"- **Teams Evaluated:** {summary.get('total_teams', 0)}")
        lines.append(f"- **Total Modes:** {summary.get('total_modes', 0)}")
        lines.append(f"- **Total Runs:** {summary.get('total_runs', 0)}")

        if summary.get("best_team_overall"):
            lines.append(f"- **Best Team Overall:** {summary['best_team_overall']}")
        lines.append("")

        # Cross-Team Comparison
        lines.append("## Cross-Team Comparison")
        lines.append("")

        cross_team = evaluation.cross_team_comparison
        if cross_team.get("overall_by_team"):
            lines.append("### Overall Scores by Team")
            lines.append("")
            lines.append("| Team | Overall Score |")
            lines.append("|------|--------------|")
            for team, score in sorted(cross_team["overall_by_team"].items(), key=lambda x: -x[1]):
                lines.append(f"| {team} | {score:.3f} |")
            lines.append("")

        # Cluster comparison
        if cross_team.get("cluster_by_team"):
            lines.append("### Cluster Scores by Team")
            lines.append("")
            clusters = list(DIMENSION_CLUSTERS.keys())
            header = "| Team | " + " | ".join(clusters) + " |"
            separator = "|------|" + "|".join(["------" for _ in clusters]) + "|"
            lines.append(header)
            lines.append(separator)
            for team, cluster_scores in cross_team["cluster_by_team"].items():
                scores = [f"{cluster_scores.get(c, 0):.3f}" for c in clusters]
                lines.append(f"| {team} | " + " | ".join(scores) + " |")
            lines.append("")

        # Team Details
        lines.append("## Team Details")
        lines.append("")

        for team, team_eval in evaluation.team_evaluations.items():
            lines.append(f"### {team}")
            lines.append("")
            lines.append(f"**Modes Evaluated:** {', '.join(team_eval.modes_evaluated)}")
            lines.append(f"**Runs per Mode:** {team_eval.n_runs_per_mode}")
            lines.append("")

            team_summary = team_eval.summary
            if team_summary.get("dimension_averages"):
                lines.append("#### Dimension Averages")
                lines.append("")
                lines.append("| Dimension | Score |")
                lines.append("|-----------|-------|")
                for dim_id, score in sorted(team_summary["dimension_averages"].items()):
                    dim_name = DIMENSIONS[dim_id].name if dim_id in DIMENSIONS else dim_id
                    lines.append(f"| {dim_name} | {score:.3f} |")
                lines.append("")

            if team_summary.get("best_mode"):
                lines.append("#### Best Mode by Dimension")
                lines.append("")
                lines.append("| Dimension | Best Mode |")
                lines.append("|-----------|-----------|")
                for dim_id, mode in team_summary["best_mode"].items():
                    dim_name = DIMENSIONS[dim_id].name if dim_id in DIMENSIONS else dim_id
                    lines.append(f"| {dim_name} | {mode} |")
                lines.append("")

            if include_details:
                lines.append("#### Mode Results")
                lines.append("")
                for mode, results in team_eval.mode_results.items():
                    if results:
                        avg_score = sum(r.overall_score for r in results) / len(results)
                        lines.append(f"**{mode}:** {len(results)} runs, avg score: {avg_score:.3f}")
                lines.append("")

        # Dimension Reference
        lines.append("## Dimension Reference")
        lines.append("")
        for cluster_name, dim_ids in DIMENSION_CLUSTERS.items():
            lines.append(f"### {cluster_name} Cluster")
            lines.append("")
            for dim_id in dim_ids:
                dim_def = DIMENSIONS[dim_id]
                lines.append(f"**{dim_def.name}:** {dim_def.description}")
                lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*Report generated by the AEL Evaluation Framework*")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return output_path

    def generate_dimension_report(
        self,
        analyses: Dict[str, DimensionAnalysis],
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Generate detailed dimension analysis report.

        Args:
            analyses: Dictionary of dimension analyses
            output_path: Output file path

        Returns:
            Path to generated report
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"dimension_report_{timestamp}.md"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        lines.append("# Dimension Analysis Report")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Summary table
        lines.append("## Summary")
        lines.append("")
        lines.append("| Dimension | Cluster | Score | Std Dev | Sample Size |")
        lines.append("|-----------|---------|-------|---------|-------------|")
        for dim_id, analysis in sorted(analyses.items(), key=lambda x: -x[1].overall_score):
            lines.append(
                f"| {analysis.dimension_name} | {analysis.cluster} | "
                f"{analysis.overall_score:.3f} | {analysis.score_std_dev:.3f} | "
                f"{analysis.sample_size} |"
            )
        lines.append("")

        # Detailed analysis for each dimension
        for dim_id, analysis in analyses.items():
            lines.append(f"## {analysis.dimension_name}")
            lines.append("")
            lines.append(f"**Cluster:** {analysis.cluster}")
            lines.append(f"**Description:** {analysis.description}")
            lines.append(f"**Overall Score:** {analysis.overall_score:.3f} (±{analysis.score_std_dev:.3f})")
            lines.append("")

            # Metrics
            if analysis.metric_analyses:
                lines.append("### Metrics")
                lines.append("")
                lines.append("| Metric | Mean | Std Dev | Min | Max | Median |")
                lines.append("|--------|------|---------|-----|-----|--------|")
                for ma in analysis.metric_analyses:
                    lines.append(
                        f"| {ma.metric_name} | {ma.mean:.3f} | {ma.std_dev:.3f} | "
                        f"{ma.min_value:.3f} | {ma.max_value:.3f} | {ma.median:.3f} |"
                    )
                lines.append("")

            # Strengths and Weaknesses
            if analysis.strengths:
                lines.append("### Strengths")
                lines.append("")
                for strength in analysis.strengths:
                    lines.append(f"- {strength}")
                lines.append("")

            if analysis.weaknesses:
                lines.append("### Weaknesses")
                lines.append("")
                for weakness in analysis.weaknesses:
                    lines.append(f"- {weakness}")
                lines.append("")

            if analysis.recommendations:
                lines.append("### Recommendations")
                lines.append("")
                for rec in analysis.recommendations:
                    lines.append(f"- {rec}")
                lines.append("")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return output_path

    def generate_comparison_report(
        self,
        comparison: ModeComparison,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Generate mode comparison report.

        Args:
            comparison: Mode comparison results
            output_path: Output file path

        Returns:
            Path to generated report
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"comparison_report_{timestamp}.md"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        lines.append("# Mode Comparison Report")
        lines.append("")
        lines.append(f"**Team:** {comparison.team}")
        lines.append(f"**Comparison Type:** {comparison.comparison_type}")
        lines.append(f"**Modes Compared:** {', '.join(comparison.modes)}")
        lines.append("")

        # Dimension comparison table
        lines.append("## Dimension Scores by Mode")
        lines.append("")

        modes = comparison.modes
        header = "| Dimension | " + " | ".join(modes) + " |"
        separator = "|-----------|" + "|".join(["------" for _ in modes]) + "|"
        lines.append(header)
        lines.append(separator)

        for dim_id, mode_scores in comparison.dimension_comparisons.items():
            dim_name = DIMENSIONS[dim_id].name if dim_id in DIMENSIONS else dim_id
            scores = [f"{mode_scores.get(m, 0):.3f}" for m in modes]
            lines.append(f"| {dim_name} | " + " | ".join(scores) + " |")
        lines.append("")

        # Recommendations
        if comparison.recommendations:
            lines.append("## Recommendations")
            lines.append("")
            for rec in comparison.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return output_path

    def generate_latex_table(
        self,
        evaluation: FullEvaluation,
        output_path: Optional[Path] = None,
        table_type: str = "dimension",
    ) -> Path:
        """
        Generate LaTeX table for academic publication.

        Args:
            evaluation: Full evaluation results
            output_path: Output file path
            table_type: Type of table ("dimension", "cluster", "team")

        Returns:
            Path to generated LaTeX file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"table_{table_type}_{timestamp}.tex"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []

        if table_type == "dimension":
            lines.append(r"\begin{table}[htbp]")
            lines.append(r"\centering")
            lines.append(r"\caption{Evaluation Results by Dimension}")
            lines.append(r"\label{tab:dimension_results}")

            teams = list(evaluation.team_evaluations.keys())
            lines.append(r"\begin{tabular}{l" + "c" * len(teams) + "}")
            lines.append(r"\toprule")
            lines.append(r"Dimension & " + " & ".join(teams) + r" \\")
            lines.append(r"\midrule")

            cross_team = evaluation.cross_team_comparison
            for dim_id in DIMENSIONS.keys():
                dim_name = DIMENSIONS[dim_id].name
                scores = []
                for team in teams:
                    score = cross_team.get("dimension_by_team", {}).get(team, {}).get(dim_id, 0)
                    scores.append(f"{score:.3f}")
                lines.append(f"{dim_name} & " + " & ".join(scores) + r" \\")

            lines.append(r"\bottomrule")
            lines.append(r"\end{tabular}")
            lines.append(r"\end{table}")

        elif table_type == "cluster":
            lines.append(r"\begin{table}[htbp]")
            lines.append(r"\centering")
            lines.append(r"\caption{Evaluation Results by Cluster}")
            lines.append(r"\label{tab:cluster_results}")

            teams = list(evaluation.team_evaluations.keys())
            lines.append(r"\begin{tabular}{l" + "c" * len(teams) + "}")
            lines.append(r"\toprule")
            lines.append(r"Cluster & " + " & ".join(teams) + r" \\")
            lines.append(r"\midrule")

            cross_team = evaluation.cross_team_comparison
            for cluster_name in DIMENSION_CLUSTERS.keys():
                scores = []
                for team in teams:
                    score = cross_team.get("cluster_by_team", {}).get(team, {}).get(cluster_name, 0)
                    scores.append(f"{score:.3f}")
                lines.append(f"{cluster_name} & " + " & ".join(scores) + r" \\")

            lines.append(r"\bottomrule")
            lines.append(r"\end{tabular}")
            lines.append(r"\end{table}")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return output_path
