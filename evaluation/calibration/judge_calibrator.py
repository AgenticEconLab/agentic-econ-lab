# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Judge Calibrator — GLM-based score calibration for LLM-as-judge evaluators.

Calibrates raw LLM judge scores against paired reference scores
(e.g., an open-weight judge vs. a structural/reference baseline) to reduce
systematic bias.

Uses a simple linear model (GLM): calibrated = alpha + beta * raw_score
Parameters estimated via ordinary least squares on paired calibration data.
"""

import math
from typing import Any, Dict, List, Optional, Tuple


class CalibrationPair:
    """A single calibration pair: raw score from judge + reference score."""

    __slots__ = ("raw_score", "reference_score", "dimension", "metadata")

    def __init__(
        self,
        raw_score: float,
        reference_score: float,
        dimension: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.raw_score = raw_score
        self.reference_score = reference_score
        self.dimension = dimension
        self.metadata = metadata or {}


class JudgeCalibratorGLM:
    """
    GLM-based score calibration for LLM-as-judge evaluators.

    Fits a simple linear model: calibrated = alpha + beta * raw_score
    using ordinary least squares on paired calibration data.
    """

    def __init__(self):
        self.alpha: float = 0.0
        self.beta: float = 1.0
        self.r_squared: float = 0.0
        self.n_pairs: int = 0
        self._fitted: bool = False

    def fit(self, pairs: List[CalibrationPair]) -> Dict[str, float]:
        """
        Fit the calibration model on paired scores.

        Args:
            pairs: List of CalibrationPair (raw_score, reference_score).

        Returns:
            Dict with alpha, beta, r_squared, n_pairs.
        """
        if len(pairs) < 2:
            self.alpha = 0.0
            self.beta = 1.0
            self.r_squared = 0.0
            self.n_pairs = len(pairs)
            self._fitted = True
            return self.get_params()

        xs = [p.raw_score for p in pairs]
        ys = [p.reference_score for p in pairs]
        n = len(xs)

        x_mean = sum(xs) / n
        y_mean = sum(ys) / n

        # OLS: beta = cov(x, y) / var(x)
        cov_xy = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(xs, ys)) / n
        var_x = sum((xi - x_mean) ** 2 for xi in xs) / n

        if var_x < 1e-12:
            # All raw scores are identical — can't fit slope
            self.beta = 1.0
            self.alpha = y_mean - x_mean
        else:
            self.beta = cov_xy / var_x
            self.alpha = y_mean - self.beta * x_mean

        # R²
        ss_res = sum((yi - (self.alpha + self.beta * xi)) ** 2 for xi, yi in zip(xs, ys))
        ss_tot = sum((yi - y_mean) ** 2 for yi in ys)
        self.r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

        self.n_pairs = n
        self._fitted = True

        return self.get_params()

    def calibrate(self, raw_score: float) -> float:
        """
        Calibrate a raw score using the fitted model.

        Args:
            raw_score: Raw score from an LLM judge.

        Returns:
            Calibrated score (clamped to [0, 1] range).
        """
        calibrated = self.alpha + self.beta * raw_score
        return max(0.0, min(1.0, calibrated))

    def calibrate_batch(self, raw_scores: List[float]) -> List[float]:
        """Calibrate a batch of raw scores."""
        return [self.calibrate(s) for s in raw_scores]

    def get_params(self) -> Dict[str, float]:
        """Return current calibration parameters."""
        return {
            "alpha": round(self.alpha, 6),
            "beta": round(self.beta, 6),
            "r_squared": round(self.r_squared, 6),
            "n_pairs": self.n_pairs,
        }

    @property
    def is_fitted(self) -> bool:
        """Check if the calibrator has been fitted."""
        return self._fitted

    def correlation(self, pairs: List[CalibrationPair]) -> float:
        """
        Compute Pearson correlation between raw and reference scores.

        Args:
            pairs: List of CalibrationPair.

        Returns:
            Pearson r coefficient (-1 to 1).
        """
        if len(pairs) < 2:
            return 0.0

        xs = [p.raw_score for p in pairs]
        ys = [p.reference_score for p in pairs]
        n = len(xs)

        x_mean = sum(xs) / n
        y_mean = sum(ys) / n

        cov_xy = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(xs, ys))
        std_x = math.sqrt(sum((xi - x_mean) ** 2 for xi in xs))
        std_y = math.sqrt(sum((yi - y_mean) ** 2 for yi in ys))

        if std_x < 1e-12 or std_y < 1e-12:
            return 0.0

        return cov_xy / (std_x * std_y)
