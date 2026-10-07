# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Hypothesis testing + robustness sweeps (§4.6 HypothesisTester + RobustnessAnalyzer).

Hypotheses are theory-derived sign/zero restrictions carried in the EstimationSpec; each is
tested deterministically on the stored analysis panel. Robustness re-estimates deterministic
variants (drop-one-regressor, split-sample, covariance swap) and reports whether the focus
coefficient's sign and significance survive. All numerics here; narration stays in the stage.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import pandas as pd

from .harness import run_estimation
from .types import (
    VERDICT_INESTIMABLE,
    EstimationOutcome,
    EstimationSpec,
    HypothesisResult,
    InferenceReport,
    RobustnessCheck,
)


def _coef(outcome: EstimationOutcome, name: str):
    for c in outcome.coefficients:
        if c.name == name:
            return c
    return None


def run_hypotheses(outcome: EstimationOutcome) -> List[HypothesisResult]:
    spec = outcome.spec
    if spec is None or outcome.verdict == VERDICT_INESTIMABLE:
        return []
    results: List[HypothesisResult] = []
    for h in spec.hypotheses:
        c = _coef(outcome, h.param)
        if c is None:
            results.append(HypothesisResult(
                name=h.name, param=h.param, restriction=h.restriction,
                detail=f"parameter '{h.param}' not in the estimated specification"))
            continue
        if h.restriction == "=0":
            p, supported = c.p_value, c.p_value >= 0.05
            h_outcome = "consistent" if supported else "rejected"
            detail = "two-sided t: fail to reject b=0" if supported else "two-sided t: reject b=0"
        else:
            sign_ok = c.estimate < 0 if h.restriction == "<0" else c.estimate > 0
            one_sided = c.p_value / 2.0 if sign_ok else 1.0 - c.p_value / 2.0
            p, supported = one_sided, sign_ok and one_sided < 0.05
            # A failed sign prediction is not a rejection unless the estimate is significantly
            # on the other side: an imprecise estimate (e.g. b = 0.0001, se = 0.042) is
            # inconclusive, not evidence against the prediction.
            if supported:
                h_outcome = "supported"
            elif not sign_ok and c.p_value < 0.05:
                h_outcome = "contradicted"
            else:
                h_outcome = "inconclusive"
            detail = (f"one-sided t for b{h.restriction}: estimate={c.estimate:.4g}, "
                      f"{'predicted sign, ' if sign_ok else 'opposite sign, '}"
                      f"one-sided p={one_sided:.4f}, two-sided p={c.p_value:.4f} -> {h_outcome}")
        results.append(HypothesisResult(
            name=h.name, param=h.param, restriction=h.restriction,
            estimate=c.estimate, p_value=float(p), supported=bool(supported),
            outcome=h_outcome, detail=detail))
    return results


def _panel_from_outcome(outcome: EstimationOutcome) -> Optional[pd.DataFrame]:
    if not outcome.analysis_data:
        return None
    frame = pd.DataFrame(outcome.analysis_data)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.set_index("date").astype(float)


def _rerun(spec: EstimationSpec, frame: pd.DataFrame, min_obs: int) -> EstimationOutcome:
    """Re-estimate a variant on the ALREADY-TRANSFORMED analysis panel: every variable's
    transform is neutralized to 'level' so the stored columns are used as-is."""
    variant = spec.model_copy(deep=True)
    variant.dependent.transform, variant.dependent.lag = "level", 0
    variant.dependent.series_ref = variant.dependent.name
    for v in list(variant.regressors) + list(variant.instruments):
        v.transform, v.lag = "level", 0
        v.series_ref = v.name
        # The stored column IS already the product/difference — a
        # live interact_with/subtract_ref would look for the raw second series (absent
        # from the analysis panel) and fail every variant.
        v.interact_with = None
        v.subtract_ref = None
    variant.sample_start = variant.sample_end = None
    return run_estimation(variant, frame, alias_map={}, min_obs=min_obs)


def run_robustness(outcome: EstimationOutcome, focus_param: Optional[str] = None,
                   min_obs: int = 20) -> List[RobustnessCheck]:
    spec = outcome.spec
    if spec is None or outcome.verdict == VERDICT_INESTIMABLE:
        return []
    frame = _panel_from_outcome(outcome)
    if frame is None:
        return []

    if focus_param is None:
        focus_param = spec.hypotheses[0].param if spec.hypotheses else spec.regressors[0].name
    base = _coef(outcome, focus_param)
    if base is None:
        return []

    checks: List[RobustnessCheck] = []

    def record(name: str, description: str, variant: EstimationOutcome):
        c = _coef(variant, focus_param)
        if variant.verdict == VERDICT_INESTIMABLE or c is None:
            checks.append(RobustnessCheck(
                name=name, description=description, focus_param=focus_param,
                baseline_estimate=base.estimate,
                detail=f"variant not estimable ({variant.reason}) — check inconclusive"))
            return
        sign_stable = (c.estimate > 0) == (base.estimate > 0)
        sig_stable = (c.p_value < 0.05) == (base.p_value < 0.05)
        checks.append(RobustnessCheck(
            name=name, description=description, focus_param=focus_param,
            baseline_estimate=base.estimate, variant_estimate=c.estimate,
            sign_stable=sign_stable, significance_stable=sig_stable,
            detail=f"baseline={base.estimate:.4g} (p={base.p_value:.3f}) -> "
                   f"variant={c.estimate:.4g} (p={c.p_value:.3f})"))

    # (a) leave-one-regressor-out for every non-focus regressor
    for v in spec.regressors:
        if v.name == focus_param or len(spec.regressors) < 2:
            continue
        variant_spec = spec.model_copy(deep=True)
        variant_spec.regressors = [r for r in variant_spec.regressors if r.name != v.name]
        record(f"drop_{v.name}", f"omit regressor '{v.name}'", _rerun(variant_spec, frame, min_obs))

    # (b) split-sample halves
    mid = len(frame) // 2
    if mid >= min_obs:
        for label, sub in (("first_half", frame.iloc[:mid]), ("second_half", frame.iloc[mid:])):
            record(f"subsample_{label}", f"re-estimate on the {label.replace('_', ' ')}",
                   _rerun(spec, sub, min_obs))

    # (c) covariance-estimator swap
    alt_cov = "HC1" if spec.cov_type == "HAC" else "HAC"
    cov_spec = spec.model_copy(deep=True)
    cov_spec.cov_type = alt_cov
    record(f"cov_{alt_cov.lower()}", f"swap covariance estimator to {alt_cov}",
           _rerun(cov_spec, frame, min_obs))

    return checks


def run_inference(outcome: EstimationOutcome, focus_param: Optional[str] = None) -> InferenceReport:
    hypotheses = run_hypotheses(outcome)
    robustness = run_robustness(outcome, focus_param=focus_param)
    judged = [c for c in robustness if c.sign_stable is not None]
    stability = (sum(1 for c in judged if c.sign_stable) / len(judged)) if judged else None
    return InferenceReport(hypotheses=hypotheses, robustness=robustness,
                           stability_score=stability)
