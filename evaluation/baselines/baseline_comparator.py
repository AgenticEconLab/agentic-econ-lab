# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Baseline comparison framework.

Compares agentic workflow evaluation scores against single-prompt
baseline scores to quantify the value-added of multi-agent orchestration.

Produces dimension-by-dimension relative improvements and generates
LaTeX comparison tables for the manuscript.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DimensionComparison:
    """Comparison result for a single evaluation dimension."""
    dimension: str
    workflow_score: float
    baseline_score: float
    absolute_diff: float = 0.0
    relative_improvement: float = 0.0

    def __post_init__(self):
        self.absolute_diff = self.workflow_score - self.baseline_score
        if self.baseline_score > 0:
            self.relative_improvement = (
                (self.workflow_score - self.baseline_score) / self.baseline_score
            )
        elif self.workflow_score > 0:
            self.relative_improvement = 1.0
        else:
            self.relative_improvement = 0.0


@dataclass
class ComparisonResult:
    """Full comparison result across all dimensions for a team."""
    team: str
    dimensions: List[DimensionComparison]
    workflow_mode: str = ""
    baseline_model: str = ""

    @property
    def avg_workflow_score(self) -> float:
        if not self.dimensions:
            return 0.0
        return sum(d.workflow_score for d in self.dimensions) / len(self.dimensions)

    @property
    def avg_baseline_score(self) -> float:
        if not self.dimensions:
            return 0.0
        return sum(d.baseline_score for d in self.dimensions) / len(self.dimensions)

    @property
    def avg_improvement(self) -> float:
        if not self.dimensions:
            return 0.0
        return sum(d.absolute_diff for d in self.dimensions) / len(self.dimensions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "team": self.team,
            "workflow_mode": self.workflow_mode,
            "baseline_model": self.baseline_model,
            "avg_workflow_score": round(self.avg_workflow_score, 4),
            "avg_baseline_score": round(self.avg_baseline_score, 4),
            "avg_improvement": round(self.avg_improvement, 4),
            "dimensions": [
                {
                    "dimension": d.dimension,
                    "workflow": round(d.workflow_score, 4),
                    "baseline": round(d.baseline_score, 4),
                    "absolute_diff": round(d.absolute_diff, 4),
                    "relative_improvement": round(d.relative_improvement, 4),
                }
                for d in self.dimensions
            ],
        }


class BaselineComparator:
    """
    Compares agentic workflow evaluation scores against baseline scores.

    Produces dimension-by-dimension comparisons and generates formatted
    output for analysis and manuscript inclusion.
    """

    def compare_against_baseline(
        self,
        workflow_scores: Dict[str, float],
        baseline_scores: Dict[str, float],
        team: str = "",
        workflow_mode: str = "",
        baseline_model: str = "vllm/mistral-small-3.2-24b-fp8",
    ) -> ComparisonResult:
        """
        Compare workflow scores against baseline scores dimension by dimension.

        Args:
            workflow_scores: {dimension_name: score} from workflow evaluation.
            baseline_scores: {dimension_name: score} from baseline evaluation.
            team: Team name for labeling.
            workflow_mode: Workflow mode (e.g., ModeNoWcNoHITL).
            baseline_model: Model used for baseline generation.

        Returns:
            ComparisonResult with per-dimension comparisons.
        """
        all_dimensions = sorted(
            set(list(workflow_scores.keys()) + list(baseline_scores.keys()))
        )

        dimensions = []
        for dim in all_dimensions:
            wf_score = workflow_scores.get(dim, 0.0)
            bl_score = baseline_scores.get(dim, 0.0)
            dimensions.append(
                DimensionComparison(
                    dimension=dim,
                    workflow_score=wf_score,
                    baseline_score=bl_score,
                )
            )

        return ComparisonResult(
            team=team,
            dimensions=dimensions,
            workflow_mode=workflow_mode,
            baseline_model=baseline_model,
        )

    def compare_from_files(
        self,
        workflow_results_path: Path,
        baseline_results_path: Path,
        team: str,
        workflow_mode: str = "",
    ) -> ComparisonResult:
        """
        Compare from saved evaluation result JSON files.

        Expects files in the format produced by run_ael_evaluation.py,
        containing dimension scores as {dimension: {overall_score: float}}.

        Args:
            workflow_results_path: Path to workflow evaluation results JSON.
            baseline_results_path: Path to baseline evaluation results JSON.
            team: Team name.
            workflow_mode: Workflow mode.

        Returns:
            ComparisonResult.
        """
        with open(workflow_results_path, "r", encoding="utf-8") as f:
            workflow_data = json.load(f)
        with open(baseline_results_path, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)

        workflow_scores = self._extract_scores(workflow_data, team, workflow_mode)
        baseline_scores = self._extract_scores(baseline_data, team, "SinglePromptBaseline")
        baseline_model = self._extract_baseline_model(baseline_data)

        return self.compare_against_baseline(
            workflow_scores=workflow_scores,
            baseline_scores=baseline_scores,
            team=team,
            workflow_mode=workflow_mode,
            baseline_model=baseline_model,
        )

    def compare_all_teams(
        self,
        workflow_results: Dict[str, Dict[str, float]],
        baseline_results: Dict[str, Dict[str, float]],
    ) -> Dict[str, ComparisonResult]:
        """
        Compare all teams at once.

        Args:
            workflow_results: {team: {dimension: score}}.
            baseline_results: {team: {dimension: score}}.

        Returns:
            {team: ComparisonResult}.
        """
        results = {}
        all_teams = sorted(
            set(list(workflow_results.keys()) + list(baseline_results.keys()))
        )
        for team in all_teams:
            wf = workflow_results.get(team, {})
            bl = baseline_results.get(team, {})
            results[team] = self.compare_against_baseline(
                workflow_scores=wf,
                baseline_scores=bl,
                team=team,
            )
        return results

    def generate_comparison_table(
        self,
        comparisons: Dict[str, ComparisonResult],
        caption: str = "Agentic Workflow vs.~Single-Prompt Baseline",
        label: str = "tab:baseline-comparison",
    ) -> str:
        """
        Generate a LaTeX comparison table.

        Produces a table with columns:
        Team | Dimension | Agentic | Baseline | Delta | Rel. Improvement

        Args:
            comparisons: {team: ComparisonResult} from compare_all_teams.
            caption: LaTeX table caption.
            label: LaTeX table label.

        Returns:
            LaTeX table string.
        """
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\small",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            r"\begin{tabular}{llcccc}",
            r"\toprule",
            r"Team & Dimension & Agentic & Baseline & $\Delta$ & Rel.\ Impr. \\",
            r"\midrule",
        ]

        for team, result in sorted(comparisons.items()):
            first_row = True
            for dc in result.dimensions:
                team_label = team if first_row else ""
                delta_sign = "+" if dc.absolute_diff >= 0 else ""
                rel_pct = f"{dc.relative_improvement * 100:+.1f}\\%"

                lines.append(
                    f"{team_label} & {dc.dimension} & "
                    f"{dc.workflow_score:.3f} & {dc.baseline_score:.3f} & "
                    f"{delta_sign}{dc.absolute_diff:.3f} & {rel_pct} \\\\"
                )
                first_row = False

            # Add team average row
            lines.append(
                f" & \\textbf{{Average}} & "
                f"\\textbf{{{result.avg_workflow_score:.3f}}} & "
                f"\\textbf{{{result.avg_baseline_score:.3f}}} & "
                f"\\textbf{{{'+' if result.avg_improvement >= 0 else ''}"
                f"{result.avg_improvement:.3f}}} & \\\\"
            )
            lines.append(r"\midrule")

        # Remove last midrule and replace with bottomrule
        if lines[-1] == r"\midrule":
            lines[-1] = r"\bottomrule"

        lines.extend([
            r"\end{tabular}",
            r"\end{table}",
        ])

        return "\n".join(lines)

    def generate_summary_table(
        self,
        comparisons: Dict[str, ComparisonResult],
        caption: str = "Agentic vs.~Baseline: Summary by Team",
        label: str = "tab:baseline-summary",
    ) -> str:
        """
        Generate a compact summary table (one row per team).

        Args:
            comparisons: {team: ComparisonResult}.
            caption: LaTeX table caption.
            label: LaTeX table label.

        Returns:
            LaTeX table string.
        """
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Team & Agentic (avg) & Baseline (avg) & $\Delta$ (avg) \\",
            r"\midrule",
        ]

        for team, result in sorted(comparisons.items()):
            delta_sign = "+" if result.avg_improvement >= 0 else ""
            lines.append(
                f"{team} & {result.avg_workflow_score:.3f} & "
                f"{result.avg_baseline_score:.3f} & "
                f"{delta_sign}{result.avg_improvement:.3f} \\\\"
            )

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

        return "\n".join(lines)

    def save_comparison(
        self,
        comparisons: Dict[str, ComparisonResult],
        output_path: Path,
    ) -> None:
        """
        Save comparison results to JSON.

        Args:
            comparisons: {team: ComparisonResult}.
            output_path: Path to save JSON file.
        """
        data = {
            team: result.to_dict()
            for team, result in comparisons.items()
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_scores(
        self,
        data: Dict[str, Any],
        team: str,
        mode: str,
    ) -> Dict[str, float]:
        """Extract dimension scores from evaluation results JSON."""
        scores = {}

        # Try format: {team: {mode: {dimension: {overall_score: float}}}}
        team_data = data.get(team, data)
        if isinstance(team_data, dict) and mode in team_data:
            mode_data = team_data[mode]
            if isinstance(mode_data, dict):
                for dim, val in mode_data.items():
                    if isinstance(val, dict) and "overall_score" in val:
                        scores[dim] = val["overall_score"]
                    elif isinstance(val, (int, float)):
                        scores[dim] = float(val)

        # Try flat format: {dimension: score} or {dimension: {overall_score: float}}
        if not scores:
            for dim, val in data.items():
                if dim.startswith("_"):
                    continue
                if isinstance(val, dict) and "overall_score" in val:
                    scores[dim] = val["overall_score"]
                elif isinstance(val, (int, float)):
                    scores[dim] = float(val)

        return scores

    def _extract_baseline_model(self, data: Dict[str, Any]) -> str:
        """Extract baseline model name from metadata."""
        if "_baseline_metadata" in data:
            return data["_baseline_metadata"].get("model", "vllm/mistral-small-3.2-24b-fp8")
        for val in data.values():
            if isinstance(val, dict) and "_baseline_metadata" in val:
                return val["_baseline_metadata"].get("model", "vllm/mistral-small-3.2-24b-fp8")
        return "vllm/mistral-small-3.2-24b-fp8"
