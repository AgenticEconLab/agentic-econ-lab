# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Multi-Evaluator Consensus Engine — Aggregates scores from multiple LLM evaluators.

Supports:
- Multiple LLM evaluators (open-weight vLLM judges by default; commercial fallback)
- Score calibration (per-model bias correction)
- Consensus methods (calibrated mean, median)
- Fleiss' kappa for multi-evaluator agreement (>2 raters)
- Cohen's weighted kappa for pairwise agreement

Cost-optimized: defaults to 1 pass per model (no repeats).

Usage:
    from evaluation.consensus.consensus_engine import ConsensusEngine

    engine = ConsensusEngine(model_names=["vllm/mistral-small-3.2-24b-fp8", "vllm/gemma-3-27b-it-fp8"])
    result = engine.evaluate_with_consensus(team, mode, base_path)
    print(f"Consensus κ: {result.fleiss_kappa:.3f}")
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from evaluation.schemas.llm_scores import LLMDimensionScore, SubCriterionScore


class CalibrationMethod(str, Enum):
    """Score calibration strategies."""

    NONE = "none"
    MEAN_SHIFT = "mean_shift"
    Z_SCORE = "z_score"


@dataclass
class ModelCalibration:
    """Calibration parameters for a single evaluator model."""

    model: str
    mean_bias: Dict[str, float] = field(default_factory=dict)
    std_scale: Dict[str, float] = field(default_factory=dict)


@dataclass
class ConsensusResult:
    """Result from multi-evaluator consensus evaluation."""

    team: str
    mode: str
    models: List[str]
    calibration_method: str
    raw_scores: Dict[str, Dict[str, LLMDimensionScore]]  # model -> dim -> score
    calibrated_scores: Dict[str, Dict[str, float]]  # model -> dim -> calibrated
    consensus_scores: Dict[str, float]  # dim -> final consensus score
    fleiss_kappa: float = 0.0
    pairwise_kappas: Dict[str, float] = field(default_factory=dict)
    calibrations: Dict[str, ModelCalibration] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "team": self.team,
            "mode": self.mode,
            "models": self.models,
            "calibration_method": self.calibration_method,
            "consensus_scores": {
                k: round(v, 4) for k, v in self.consensus_scores.items()
            },
            "fleiss_kappa": round(self.fleiss_kappa, 4),
            "pairwise_kappas": {
                k: round(v, 4) for k, v in self.pairwise_kappas.items()
            },
            "per_model_scores": {},
        }
        for model, dim_scores in self.raw_scores.items():
            result["per_model_scores"][model] = {
                dim: {
                    "overall_score": round(score.overall_score, 4),
                    "raw_average": round(score.raw_average, 4),
                }
                for dim, score in dim_scores.items()
            }
        return result


class ConsensusEngine:
    """
    Multi-evaluator consensus engine with score calibration.

    Evaluates workflow outputs with multiple LLM models and produces
    calibrated consensus scores with inter-evaluator agreement metrics.

    Cost-optimized: 1 pass per model by default.
    """

    def __init__(
        self,
        model_names: Optional[List[str]] = None,
        calibration: CalibrationMethod = CalibrationMethod.MEAN_SHIFT,
        reference_scores: Optional[Dict[str, Dict[str, float]]] = None,
        evaluator_factory: Optional[Callable] = None,
    ):
        """
        Args:
            model_names: LLM models to use as evaluators.
            calibration: Calibration strategy for cross-model alignment.
            reference_scores: Optional pre-computed reference scores for
                calibration, keyed by model -> dimension -> reference score.
            evaluator_factory: Optional factory to create LLMEvaluator instances.
                Signature: (model_name: str) -> LLMEvaluator.
                If None, creates evaluators from evaluation.scoring.llm_evaluator.
        """
        self.model_names = model_names or [
            "vllm/mistral-small-3.2-24b-fp8",
            "vllm/gemma-3-27b-it-fp8",
        ]
        self.calibration = calibration
        self.reference_scores = reference_scores or {}
        self._evaluator_factory = evaluator_factory
        self._calibrations: Dict[str, ModelCalibration] = {}

    def evaluate_with_consensus(
        self,
        team: str,
        mode: str,
        base_path: Path,
        collector: Optional[Any] = None,
    ) -> ConsensusResult:
        """
        Run evaluation with multiple models and produce consensus.

        Args:
            team: Team name.
            mode: Mode name.
            base_path: Base path to workflow outputs.
            collector: Optional MetricsCollector for observability.

        Returns:
            ConsensusResult with consensus scores and agreement metrics.
        """
        # Evaluate with each model (1 pass per model for cost optimization)
        raw_scores: Dict[str, Dict[str, LLMDimensionScore]] = {}

        for model_name in self.model_names:
            evaluator = self._create_evaluator(model_name, collector)
            if evaluator is None:
                continue
            scores = evaluator.evaluate_workflow(team, mode, base_path)
            if scores:
                raw_scores[model_name] = scores

        if not raw_scores:
            return ConsensusResult(
                team=team,
                mode=mode,
                models=self.model_names,
                calibration_method=self.calibration.value,
                raw_scores={},
                calibrated_scores={},
                consensus_scores={},
            )

        # Calibrate scores
        calibrated = self._calibrate_scores(raw_scores)

        # Compute consensus
        consensus = self._compute_consensus(calibrated)

        # Compute agreement metrics
        fleiss_kappa = self._compute_fleiss_kappa(raw_scores)
        pairwise_kappas = self._compute_pairwise_kappas(raw_scores)

        return ConsensusResult(
            team=team,
            mode=mode,
            models=list(raw_scores.keys()),
            calibration_method=self.calibration.value,
            raw_scores=raw_scores,
            calibrated_scores=calibrated,
            consensus_scores=consensus,
            fleiss_kappa=fleiss_kappa,
            pairwise_kappas=pairwise_kappas,
            calibrations=self._calibrations,
        )

    def _create_evaluator(self, model_name: str, collector: Any = None):
        """Create an LLMEvaluator for the given model."""
        if self._evaluator_factory is not None:
            return self._evaluator_factory(model_name)
        try:
            from evaluation.scoring.llm_evaluator import LLMEvaluator

            return LLMEvaluator(
                model=model_name,
                temperature=0.0,
                collector=collector,
            )
        except Exception:
            return None

    def _calibrate_scores(
        self,
        raw_scores: Dict[str, Dict[str, LLMDimensionScore]],
    ) -> Dict[str, Dict[str, float]]:
        """Apply calibration to raw scores."""
        # Extract overall scores (handle both LLMDimensionScore and raw float)
        model_dim_scores: Dict[str, Dict[str, float]] = {}
        for model, dim_scores in raw_scores.items():
            model_dim_scores[model] = {}
            for dim, score in dim_scores.items():
                if hasattr(score, "overall_score"):
                    model_dim_scores[model][dim] = score.overall_score
                else:
                    model_dim_scores[model][dim] = float(score)

        if self.calibration == CalibrationMethod.NONE:
            return model_dim_scores

        # Collect all dimensions
        all_dims = set()
        for scores in model_dim_scores.values():
            all_dims.update(scores.keys())

        if self.calibration == CalibrationMethod.MEAN_SHIFT:
            return self._mean_shift_calibrate(model_dim_scores, all_dims)
        elif self.calibration == CalibrationMethod.Z_SCORE:
            return self._z_score_calibrate(model_dim_scores, all_dims)

        return model_dim_scores

    def _mean_shift_calibrate(
        self,
        model_scores: Dict[str, Dict[str, float]],
        all_dims: set,
    ) -> Dict[str, Dict[str, float]]:
        """Calibrate via mean-shift: adjust each model to the grand mean."""
        calibrated = {}

        for dim in all_dims:
            dim_values = [
                model_scores[m].get(dim, 0.0)
                for m in model_scores
                if dim in model_scores[m]
            ]
            if not dim_values:
                continue
            grand_mean = sum(dim_values) / len(dim_values)

            for model in model_scores:
                if model not in calibrated:
                    calibrated[model] = {}
                raw = model_scores[model].get(dim)
                if raw is None:
                    continue
                model_vals = [
                    model_scores[model].get(d, 0.0)
                    for d in all_dims
                    if d in model_scores[model]
                ]
                model_mean = sum(model_vals) / len(model_vals) if model_vals else 0.0
                shift = grand_mean - model_mean if model_vals else 0.0
                calibrated[model][dim] = max(0.0, min(1.0, raw + shift))

                # Record calibration
                if model not in self._calibrations:
                    self._calibrations[model] = ModelCalibration(model=model)
                self._calibrations[model].mean_bias[dim] = round(shift, 4)

        return calibrated

    def _z_score_calibrate(
        self,
        model_scores: Dict[str, Dict[str, float]],
        all_dims: set,
    ) -> Dict[str, Dict[str, float]]:
        """Calibrate via z-score normalization per model, then rescale."""
        calibrated = {}

        for model, scores in model_scores.items():
            vals = list(scores.values())
            if len(vals) < 2:
                calibrated[model] = dict(scores)
                continue
            mean = sum(vals) / len(vals)
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
            if std == 0:
                calibrated[model] = dict(scores)
                continue
            calibrated[model] = {
                dim: max(0.0, min(1.0, 0.5 + (v - mean) / (2 * std)))
                for dim, v in scores.items()
            }

            if model not in self._calibrations:
                self._calibrations[model] = ModelCalibration(model=model)
            for dim in scores:
                self._calibrations[model].std_scale[dim] = round(std, 4)

        return calibrated

    def _compute_consensus(
        self,
        calibrated: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        """Compute consensus scores as the mean of calibrated scores per dimension."""
        all_dims = set()
        for scores in calibrated.values():
            all_dims.update(scores.keys())

        consensus = {}
        for dim in all_dims:
            values = [
                calibrated[m][dim]
                for m in calibrated
                if dim in calibrated[m]
            ]
            if values:
                consensus[dim] = sum(values) / len(values)

        return consensus

    def _compute_fleiss_kappa(
        self,
        raw_scores: Dict[str, Dict[str, LLMDimensionScore]],
    ) -> float:
        """
        Compute Fleiss' kappa for multi-rater agreement.

        Discretizes normalized scores (0-1) into 5 bins matching
        the 1-5 Likert scale, then applies Fleiss' kappa formula.
        """
        models = list(raw_scores.keys())
        if len(models) < 2:
            return 0.0

        # Collect all dimensions
        all_dims = set()
        for scores in raw_scores.values():
            all_dims.update(scores.keys())

        # Build rating matrix: each sub-criterion is a "subject"
        # Each model's sub-criterion score (1-5) is a rating
        subjects: List[List[int]] = []  # subjects × raters
        for dim in sorted(all_dims):
            # Get sub-criteria names from first model that has this dimension
            criteria_names = set()
            for model_scores in raw_scores.values():
                if dim in model_scores:
                    for sc in model_scores[dim].sub_criteria:
                        criteria_names.add(sc.criterion_name)

            for crit_name in sorted(criteria_names):
                ratings = []
                for model in models:
                    if dim in raw_scores[model]:
                        sc_dict = {
                            sc.criterion_name: sc.score
                            for sc in raw_scores[model][dim].sub_criteria
                        }
                        ratings.append(sc_dict.get(crit_name, 3))
                    else:
                        ratings.append(3)  # neutral default
                subjects.append(ratings)

        return fleiss_kappa(subjects, k=5)

    def _compute_pairwise_kappas(
        self,
        raw_scores: Dict[str, Dict[str, LLMDimensionScore]],
    ) -> Dict[str, float]:
        """Compute pairwise weighted Cohen's kappa between all model pairs."""
        models = list(raw_scores.keys())
        if len(models) < 2:
            return {}

        kappas = {}
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                m_a, m_b = models[i], models[j]
                ratings_a, ratings_b = [], []

                shared_dims = set(raw_scores[m_a].keys()) & set(
                    raw_scores[m_b].keys()
                )
                for dim in shared_dims:
                    subs_a = {
                        sc.criterion_name: sc.score
                        for sc in raw_scores[m_a][dim].sub_criteria
                    }
                    subs_b = {
                        sc.criterion_name: sc.score
                        for sc in raw_scores[m_b][dim].sub_criteria
                    }
                    for crit in set(subs_a.keys()) & set(subs_b.keys()):
                        ratings_a.append(subs_a[crit])
                        ratings_b.append(subs_b[crit])

                if len(ratings_a) >= 2:
                    kappas[f"{m_a}_vs_{m_b}"] = weighted_kappa(
                        ratings_a, ratings_b, k=5
                    )

        return kappas


# ---------------------------------------------------------------------------
# Agreement statistics (module-level, reusable)
# ---------------------------------------------------------------------------


def weighted_kappa(
    ratings_a: List[int],
    ratings_b: List[int],
    k: int = 5,
) -> float:
    """
    Compute quadratic-weighted Cohen's kappa for ordinal ratings.

    Args:
        ratings_a: Ratings from evaluator A (1-indexed, 1..k).
        ratings_b: Ratings from evaluator B (1-indexed, 1..k).
        k: Number of rating categories.

    Returns:
        Weighted kappa in [-1, 1].
    """
    n = len(ratings_a)
    if n == 0:
        return 0.0

    observed = [[0] * k for _ in range(k)]
    for a, b in zip(ratings_a, ratings_b):
        observed[max(0, min(k - 1, a - 1))][max(0, min(k - 1, b - 1))] += 1

    row_sums = [sum(observed[i]) for i in range(k)]
    col_sums = [sum(observed[i][j] for i in range(k)) for j in range(k)]

    expected = [[0.0] * k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            expected[i][j] = (row_sums[i] * col_sums[j]) / n if n > 0 else 0

    weights = [[0.0] * k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            weights[i][j] = 1.0 - ((i - j) / (k - 1)) ** 2

    po = sum(
        weights[i][j] * observed[i][j] for i in range(k) for j in range(k)
    ) / n if n > 0 else 0

    pe = sum(
        weights[i][j] * expected[i][j] for i in range(k) for j in range(k)
    ) / n if n > 0 else 0

    if pe == 1.0:
        return 1.0

    return (po - pe) / (1.0 - pe) if (1.0 - pe) != 0 else 0.0


def fleiss_kappa(subjects: List[List[int]], k: int = 5) -> float:
    """
    Compute Fleiss' kappa for multi-rater agreement.

    Args:
        subjects: List of [rater_1_score, rater_2_score, ...] per subject.
            Each score is 1-indexed (1..k).
        k: Number of categories.

    Returns:
        Fleiss' kappa in [-1, 1].
    """
    n_subjects = len(subjects)
    if n_subjects == 0:
        return 0.0
    n_raters = len(subjects[0])
    if n_raters < 2:
        return 0.0

    # Build category count matrix: n_subjects × k
    counts = [[0] * k for _ in range(n_subjects)]
    for i, ratings in enumerate(subjects):
        for r in ratings:
            cat = max(0, min(k - 1, r - 1))
            counts[i][cat] += 1

    # P_i = proportion of agreement for each subject
    p_i = []
    for i in range(n_subjects):
        sum_sq = sum(c * c for c in counts[i])
        p_i.append((sum_sq - n_raters) / (n_raters * (n_raters - 1)))

    p_bar = sum(p_i) / n_subjects if n_subjects > 0 else 0

    # P_j = proportion of all ratings in category j
    p_j = []
    total_ratings = n_subjects * n_raters
    for j in range(k):
        col_sum = sum(counts[i][j] for i in range(n_subjects))
        p_j.append(col_sum / total_ratings if total_ratings > 0 else 0)

    pe_bar = sum(pj * pj for pj in p_j)

    if pe_bar == 1.0:
        return 1.0

    return (p_bar - pe_bar) / (1.0 - pe_bar) if (1.0 - pe_bar) != 0 else 0.0
