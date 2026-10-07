# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
LLM-as-Reviewer evaluation schemas.

Pydantic models for capturing LLM-based content quality assessments
as part of the two-tier evaluation framework (Tier 2).
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SubCriterionScore(BaseModel):
    """Score for a single sub-criterion within a dimension."""

    criterion_name: str = Field(..., description="Name of the sub-criterion")
    score: int = Field(..., ge=1, le=5, description="Score on 1-5 Likert scale")
    justification: str = Field(..., description="LLM's reasoning for the score")

    @property
    def normalized_score(self) -> float:
        """Normalize 1-5 score to 0-1 scale."""
        return (self.score - 1) / 4.0


class LLMDimensionScore(BaseModel):
    """LLM-evaluated score for a single dimension of a single workflow."""

    dimension: str = Field(..., description="Dimension evaluated")
    team: str = Field(..., description="Team name")
    mode: str = Field(..., description="Mode name")
    sub_criteria: List[SubCriterionScore] = Field(default_factory=list)
    overall_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Normalized 0-1 score (average of sub-criteria)"
    )
    raw_average: float = Field(
        ..., ge=1.0, le=5.0,
        description="Average of sub-criteria on 1-5 scale"
    )
    evaluator_model: str = Field("vllm/mistral-small-3.2-24b-fp8", description="Model used for evaluation")
    timestamp: datetime = Field(default_factory=datetime.now)
    notes: Optional[str] = Field(None)

    @classmethod
    def from_sub_criteria(
        cls,
        dimension: str,
        team: str,
        mode: str,
        sub_criteria: List[SubCriterionScore],
        evaluator_model: str = "vllm/mistral-small-3.2-24b-fp8",
        notes: Optional[str] = None,
    ) -> "LLMDimensionScore":
        """Construct from a list of sub-criterion scores."""
        if not sub_criteria:
            raise ValueError("At least one sub-criterion score is required")
        raw_avg = sum(sc.score for sc in sub_criteria) / len(sub_criteria)
        normalized = (raw_avg - 1) / 4.0
        return cls(
            dimension=dimension,
            team=team,
            mode=mode,
            sub_criteria=sub_criteria,
            overall_score=round(normalized, 4),
            raw_average=round(raw_avg, 4),
            evaluator_model=evaluator_model,
            notes=notes,
        )


class TwoTierResult(BaseModel):
    """Combined two-tier evaluation result for a single workflow configuration."""

    team: str = Field(..., description="Team name")
    mode: str = Field(..., description="Mode name")
    tier1_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Tier 1 structural scores (all 10 dimensions)"
    )
    tier2_scores: Dict[str, LLMDimensionScore] = Field(
        default_factory=dict,
        description="Tier 2 LLM content-quality scores (4 dimensions)"
    )
    combined_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Final combined scores (all 10 dimensions)"
    )
    timestamp: datetime = Field(default_factory=datetime.now)

    def compute_combined_scores(
        self, tier1_weight: float = 0.3, tier2_weight: float = 0.7
    ) -> None:
        """Compute combined scores using weighted average for Tier-2 dimensions."""
        for dim, t1_score in self.tier1_scores.items():
            if dim in self.tier2_scores:
                t2_score = self.tier2_scores[dim].overall_score
                self.combined_scores[dim] = round(
                    tier1_weight * t1_score + tier2_weight * t2_score, 4
                )
            else:
                self.combined_scores[dim] = t1_score


class EnsemblePassResult(BaseModel):
    """Result from a single evaluator pass within an ensemble."""

    evaluator_model: str = Field(..., description="Model used for this pass")
    pass_index: int = Field(..., description="Pass index (0-based) within this model")
    dimension_scores: Dict[str, LLMDimensionScore] = Field(
        default_factory=dict,
        description="Scores keyed by dimension name"
    )


class EnsembleMetadata(BaseModel):
    """Metadata for a multi-model ensemble Tier-2 evaluation."""

    models: List[str] = Field(
        default_factory=list,
        description="List of evaluator model names used"
    )
    repeats_per_model: int = Field(2, description="Number of passes per model")
    total_passes: int = Field(0, description="Total evaluator passes (models × repeats)")
    per_pass_results: List[EnsemblePassResult] = Field(
        default_factory=list,
        description="Individual pass results for transparency"
    )
    within_model_std: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Std dev within each model, keyed by model then dimension"
    )
    between_model_gap: Dict[str, float] = Field(
        default_factory=dict,
        description="Mean score difference (model_A - model_B) per dimension"
    )

    def to_summary_dict(self) -> Dict[str, Any]:
        """Export ensemble metadata for JSON serialization."""
        return {
            "models": self.models,
            "repeats_per_model": self.repeats_per_model,
            "total_passes": self.total_passes,
            "within_model_std": self.within_model_std,
            "between_model_gap": self.between_model_gap,
            "per_pass_scores": [
                {
                    "evaluator_model": p.evaluator_model,
                    "pass_index": p.pass_index,
                    "dimensions": {
                        dim: {
                            "overall_score": s.overall_score,
                            "raw_average": s.raw_average,
                        }
                        for dim, s in p.dimension_scores.items()
                    },
                }
                for p in self.per_pass_results
            ],
        }


class TwoTierEvaluationReport(BaseModel):
    """Full two-tier evaluation report across all workflow configurations."""

    generated_at: datetime = Field(default_factory=datetime.now)
    evaluator_model: str = Field("vllm/mistral-small-3.2-24b-fp8")
    tier1_weight: float = Field(0.3)
    tier2_weight: float = Field(0.7)
    tier2_dimensions: List[str] = Field(
        default_factory=lambda: [
            "correctness", "soundness", "innovation_potential", "transparency"
        ]
    )
    results: Dict[str, TwoTierResult] = Field(
        default_factory=dict,
        description="Keyed by '{team}_{mode}'"
    )
    total_configurations: int = Field(0)

    def to_summary_dict(self) -> Dict[str, Any]:
        """Export summary for JSON serialization."""
        return {
            "generated_at": self.generated_at.isoformat(),
            "evaluator_model": self.evaluator_model,
            "tier1_weight": self.tier1_weight,
            "tier2_weight": self.tier2_weight,
            "tier2_dimensions": self.tier2_dimensions,
            "total_configurations": self.total_configurations,
            "results": {
                key: {
                    "team": r.team,
                    "mode": r.mode,
                    "tier1": r.tier1_scores,
                    "tier2": {
                        dim: {
                            "overall_score": s.overall_score,
                            "raw_average": s.raw_average,
                            "sub_criteria": [
                                {
                                    "criterion_name": sc.criterion_name,
                                    "score": sc.score,
                                    "justification": sc.justification,
                                }
                                for sc in s.sub_criteria
                            ],
                        }
                        for dim, s in r.tier2_scores.items()
                    },
                    "combined": r.combined_scores,
                }
                for key, r in self.results.items()
            },
        }
