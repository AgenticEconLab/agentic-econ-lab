# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Metric result schemas for workflow evaluation.

These models capture evaluation results across the 10 dimensions.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import uuid


class MetricResult(BaseModel):
    """Result of a single metric calculation."""

    metric_id: str = Field(..., description="Unique metric identifier")
    metric_name: str = Field(..., description="Human-readable metric name")
    dimension: str = Field(..., description="Dimension this metric belongs to")
    value: float = Field(..., description="Calculated metric value")
    normalized_value: Optional[float] = Field(
        None, description="Value normalized to 0-1 scale"
    )
    raw_data: Optional[Dict[str, Any]] = Field(
        None, description="Raw data used in calculation"
    )
    notes: Optional[str] = Field(None, description="Additional notes or caveats")
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        json_schema_extra = {
            "example": {
                "metric_id": "reliability_output_cv",
                "metric_name": "Output Consistency (CV)",
                "dimension": "reliability",
                "value": 0.15,
                "normalized_value": 0.85
            }
        }


class DimensionScore(BaseModel):
    """Aggregated score for a single dimension."""

    dimension: str = Field(..., description="Dimension name")
    cluster: str = Field(..., description="Cluster (Quality, Operational, Epistemic, Innovation)")
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized score (0-1)")
    metrics: List[MetricResult] = Field(default_factory=list)
    weight: float = Field(1.0, description="Weight for overall score calculation")
    notes: Optional[str] = Field(None)

    @property
    def metric_count(self) -> int:
        """Number of metrics in this dimension."""
        return len(self.metrics)

    class Config:
        json_schema_extra = {
            "example": {
                "dimension": "reliability",
                "cluster": "Quality",
                "score": 0.82,
                "weight": 1.0
            }
        }


class EvaluationResult(BaseModel):
    """Complete evaluation result for a single workflow run."""

    # Identification
    evaluation_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    run_id: str = Field(..., description="ID of the execution trace")
    team: str = Field(..., description="Team name")
    mode: str = Field(..., description="Mode name")
    framework: str = Field("ael", description="Framework used")

    # Scores
    dimension_scores: List[DimensionScore] = Field(default_factory=list)
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Weighted overall score")

    # Cluster scores
    quality_score: Optional[float] = Field(None)
    operational_score: Optional[float] = Field(None)
    epistemic_score: Optional[float] = Field(None)
    innovation_score: Optional[float] = Field(None)

    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now)
    schema_version: str = Field("1.0.0")
    notes: Optional[str] = Field(None)

    def compute_cluster_scores(self) -> None:
        """Compute cluster-level scores from dimension scores."""
        cluster_dims = {
            "Quality": [],
            "Operational": [],
            "Epistemic": [],
            "Innovation": []
        }

        for dim_score in self.dimension_scores:
            if dim_score.cluster in cluster_dims:
                cluster_dims[dim_score.cluster].append(dim_score.score)

        # Compute averages
        if cluster_dims["Quality"]:
            self.quality_score = sum(cluster_dims["Quality"]) / len(cluster_dims["Quality"])
        if cluster_dims["Operational"]:
            self.operational_score = sum(cluster_dims["Operational"]) / len(cluster_dims["Operational"])
        if cluster_dims["Epistemic"]:
            self.epistemic_score = sum(cluster_dims["Epistemic"]) / len(cluster_dims["Epistemic"])
        if cluster_dims["Innovation"]:
            self.innovation_score = sum(cluster_dims["Innovation"]) / len(cluster_dims["Innovation"])

    def get_dimension_score(self, dimension: str) -> Optional[DimensionScore]:
        """Get score for a specific dimension."""
        for dim_score in self.dimension_scores:
            if dim_score.dimension == dimension:
                return dim_score
        return None

    class Config:
        json_schema_extra = {
            "example": {
                "team": "IdeationTeam",
                "mode": "ModeNoWcNoHITL",
                "overall_score": 0.78
            }
        }


class TeamEvaluation(BaseModel):
    """Evaluation results for all modes of a team."""

    team: str = Field(..., description="Team name")
    framework: str = Field("ael")
    evaluations: List[EvaluationResult] = Field(default_factory=list)

    # Aggregated statistics
    mean_overall_score: Optional[float] = Field(None)
    std_overall_score: Optional[float] = Field(None)
    best_mode: Optional[str] = Field(None)
    worst_mode: Optional[str] = Field(None)

    # By mode
    mode_scores: Dict[str, List[float]] = Field(default_factory=dict)

    timestamp: datetime = Field(default_factory=datetime.now)

    def compute_statistics(self) -> None:
        """Compute aggregated statistics."""
        if not self.evaluations:
            return

        # Group by mode
        for eval_result in self.evaluations:
            if eval_result.mode not in self.mode_scores:
                self.mode_scores[eval_result.mode] = []
            self.mode_scores[eval_result.mode].append(eval_result.overall_score)

        # Overall statistics
        all_scores = [e.overall_score for e in self.evaluations]
        self.mean_overall_score = sum(all_scores) / len(all_scores)

        if len(all_scores) > 1:
            variance = sum((s - self.mean_overall_score) ** 2 for s in all_scores) / len(all_scores)
            self.std_overall_score = variance ** 0.5

        # Best/worst modes (by mean)
        mode_means = {
            mode: sum(scores) / len(scores)
            for mode, scores in self.mode_scores.items()
        }
        if mode_means:
            self.best_mode = max(mode_means, key=mode_means.get)
            self.worst_mode = min(mode_means, key=mode_means.get)


class FullEvaluation(BaseModel):
    """Complete evaluation across all teams and modes."""

    evaluation_name: str = Field("full_evaluation")
    teams: List[TeamEvaluation] = Field(default_factory=list)

    # Cross-team comparison
    team_rankings: Dict[str, int] = Field(default_factory=dict)
    dimension_leaders: Dict[str, str] = Field(
        default_factory=dict,
        description="Best team for each dimension"
    )

    # Metadata
    total_runs: int = Field(0)
    timestamp: datetime = Field(default_factory=datetime.now)
    schema_version: str = Field("1.0.0")

    def compute_rankings(self) -> None:
        """Compute team rankings based on overall scores."""
        team_scores = {}
        for team_eval in self.teams:
            if team_eval.mean_overall_score is not None:
                team_scores[team_eval.team] = team_eval.mean_overall_score

        # Sort and rank
        sorted_teams = sorted(team_scores.items(), key=lambda x: x[1], reverse=True)
        self.team_rankings = {team: rank + 1 for rank, (team, _) in enumerate(sorted_teams)}

        # Total runs
        self.total_runs = sum(len(t.evaluations) for t in self.teams)

    def get_team_evaluation(self, team: str) -> Optional[TeamEvaluation]:
        """Get evaluation for a specific team."""
        for team_eval in self.teams:
            if team_eval.team == team:
                return team_eval
        return None
