# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
CausalValidity evaluation dimension (V0.7).

Scores a CausalFM / DoWhy / OLS estimate against a known true ATE (on
synthetic benchmarks) or against a human-provided prior (on empirical
studies). Returns a score in [0, 1].

For synthetic benchmarks:
    score = 1.0 if |estimate - true_ate| / |true_ate| <= tolerance
            scaled linearly below tolerance

For empirical studies where only a CI is provided:
    score = 1.0 if posterior contains the prior mean, else scaled by overlap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class CausalValidityScore:
    score: Optional[float]           # None = no ground truth or prior to compare against
    ate_estimated: float
    ate_true: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    relative_error: Optional[float] = None
    method: str = ""
    notes: str = ""


class CausalValidityEvaluator:
    def __init__(self, *, tolerance: float = 0.15) -> None:
        self.tolerance = tolerance

    def evaluate(
        self,
        causal_estimate: Dict[str, Any],
        *,
        true_ate: Optional[float] = None,
        prior_mean: Optional[float] = None,
    ) -> CausalValidityScore:
        ate_est = float(causal_estimate.get("ate", 0.0))
        ci_low = causal_estimate.get("ci_low")
        ci_high = causal_estimate.get("ci_high")
        method = str(causal_estimate.get("method", ""))

        if true_ate is not None:
            denom = abs(true_ate) if abs(true_ate) > 1e-6 else 1.0
            rel_err = abs(ate_est - true_ate) / denom
            score = max(0.0, 1.0 - rel_err / self.tolerance)
            score = min(1.0, score)
            notes = f"synthetic truth: |err|/|truth|={rel_err:.4f} vs tol {self.tolerance}"
        elif prior_mean is not None and ci_low is not None and ci_high is not None:
            contained = ci_low <= prior_mean <= ci_high
            if contained:
                score = 1.0
                notes = "prior mean contained in CI"
            else:
                # Linear falloff by relative distance to CI
                width = max(float(ci_high) - float(ci_low), 1e-6)
                dist = min(abs(prior_mean - float(ci_low)), abs(prior_mean - float(ci_high)))
                score = max(0.0, 1.0 - dist / width)
                notes = f"prior {prior_mean:.4f} outside CI [{ci_low:.4f}, {ci_high:.4f}]"
            rel_err = None
        else:
            # No ground truth or prior: nothing to score against. Report "not evaluated"
            # (None) rather than a neutral 0.5 that reads like a measurement.
            score = None
            rel_err = None
            notes = "no ground-truth or prior supplied; not evaluated"

        return CausalValidityScore(
            score=(round(score, 4) if score is not None else None),
            ate_estimated=ate_est,
            ate_true=true_ate,
            ci_low=ci_low,
            ci_high=ci_high,
            relative_error=rel_err,
            method=method,
            notes=notes,
        )
