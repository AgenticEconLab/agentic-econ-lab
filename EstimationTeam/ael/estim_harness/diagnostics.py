# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Deterministic diagnostics battery (§4.6 Diagnostic + Validator agents' numeric core).

Every statistic is computed by statsmodels on the stored analysis panel; the LLM may narrate
the results but can neither add nor remove a verdict. Severity is calibrated to what the
chosen covariance estimator already absorbs: with HAC/HC1 standard errors, heteroskedasticity
and MODERATE residual autocorrelation degrade to warnings (inference is already robustified).
Two residual conditions stay SEVERE whatever the transforms (a guard that ran only for
level/log dependents would let regressions with DW = 0.02 be reported `estimated`):

* a unit root left in the residuals (ADF p >= 0.10) — no covariance choice rescues a
  spurious regression, and a growth-rate regression can be spurious too;
* severe first-order residual autocorrelation: AR(1) rho >= 0.9 or Durbin-Watson < 0.5
  (DW ~ 2(1 - rho), so DW < 0.5 corresponds to rho above about 0.75). HAC with a bandwidth
  chosen from the residual persistence (harness.hac_maxlags) is unreliable at that
  persistence in samples of the size these regressions use.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .harness import refit_from_outcome, residual_ar1
from .types import (
    VERDICT_ESTIMATED,
    VERDICT_FRAGILE,
    DiagnosticResult,
    DiagnosticsReport,
    EstimationOutcome,
)

SEVERE_RHO = 0.9       # residual AR(1) coefficient at or above which the fit is fragile
SEVERE_DW = 0.5        # Durbin-Watson below which the fit is fragile
WARN_RHO = 0.5


def _bp_test(res, exog) -> DiagnosticResult:
    from statsmodels.stats.diagnostic import het_breuschpagan
    try:
        _, p, _, _ = het_breuschpagan(res.resid, np.asarray(exog, dtype=float))
        verdict = "pass" if p >= 0.05 else "warn"
        return DiagnosticResult(name="breusch_pagan", statistic=None, p_value=float(p),
                                verdict=verdict,
                                detail="heteroskedasticity" + ("" if verdict == "pass" else
                                       " detected; robust covariance already applied"))
    except Exception as e:
        return DiagnosticResult(name="breusch_pagan", verdict="not_applicable", detail=str(e))


def _dw_test(res) -> DiagnosticResult:
    from statsmodels.stats.stattools import durbin_watson
    dw = float(durbin_watson(res.resid))
    if dw < SEVERE_DW:
        verdict = "fail"
    else:
        verdict = "pass" if 1.5 <= dw <= 2.5 else "warn"
    return DiagnosticResult(name="durbin_watson", statistic=dw, verdict=verdict,
                            detail=f"DW={dw:.2f} (2=no first-order autocorrelation"
                                   + (f"; below {SEVERE_DW}: severe residual autocorrelation, "
                                      "possibly spurious)" if verdict == "fail" else ")"))


def _residual_ar1(res) -> DiagnosticResult:
    rho = residual_ar1(res.resid)
    verdict = "fail" if rho >= SEVERE_RHO else ("warn" if rho >= WARN_RHO else "pass")
    return DiagnosticResult(name="residual_ar1", statistic=rho, verdict=verdict,
                            detail=f"residual AR(1) rho={rho:.2f}"
                                   + (f" (>= {SEVERE_RHO}: residuals nearly a random walk)"
                                      if verdict == "fail" else ""))


def _ljung_box(res) -> DiagnosticResult:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    n = len(res.resid)
    lags = max(1, min(10, n // 5))
    try:
        lb = acorr_ljungbox(res.resid, lags=[lags], return_df=True)
        p = float(lb["lb_pvalue"].iloc[0])
        return DiagnosticResult(name="ljung_box", p_value=p,
                                verdict="pass" if p >= 0.05 else "warn",
                                detail=f"residual autocorrelation up to lag {lags}")
    except Exception as e:
        return DiagnosticResult(name="ljung_box", verdict="not_applicable", detail=str(e))


def _jarque_bera(res) -> DiagnosticResult:
    from statsmodels.stats.stattools import jarque_bera
    try:
        stat, p, _, _ = jarque_bera(res.resid)
        return DiagnosticResult(name="jarque_bera", statistic=float(stat), p_value=float(p),
                                verdict="pass" if p >= 0.05 else "warn",
                                detail="residual normality (matters mainly for small samples)")
    except Exception as e:
        return DiagnosticResult(name="jarque_bera", verdict="not_applicable", detail=str(e))


def _residual_adf(res, outcome: EstimationOutcome) -> DiagnosticResult:
    """Spurious-regression guard, run for EVERY transform."""
    spec = outcome.spec
    in_levels = spec is not None and spec.dependent.transform in ("level", "log") and any(
        v.transform in ("level", "log") for v in spec.regressors)
    from statsmodels.tsa.stattools import adfuller
    try:
        p = float(adfuller(res.resid, autolag="AIC")[1])
        verdict = "pass" if p < 0.05 else ("warn" if p < 0.10 else "fail")
        kind = "levels regression" if in_levels else "regression"
        return DiagnosticResult(name="residual_adf", p_value=p, verdict=verdict,
                                detail=f"{kind}: non-stationary residuals mean a spurious "
                                       "regression" if verdict == "fail" else
                                       "residuals stationary at the 5% level" if verdict == "pass"
                                       else "residual stationarity borderline (5% < p < 10%)")
    except Exception as e:
        return DiagnosticResult(name="residual_adf", verdict="not_applicable", detail=str(e))


def _reset_test(res) -> DiagnosticResult:
    from statsmodels.stats.diagnostic import linear_reset
    try:
        r = linear_reset(res, power=2, use_f=True)
        p = float(r.pvalue)
        return DiagnosticResult(name="ramsey_reset", p_value=p,
                                verdict="pass" if p >= 0.05 else "warn",
                                detail="functional-form misspecification (squared fitted values)")
    except Exception as e:
        return DiagnosticResult(name="ramsey_reset", verdict="not_applicable", detail=str(e))


def _vif(exog) -> DiagnosticResult:
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        arr = np.asarray(exog, dtype=float)
        cols = list(exog.columns) if isinstance(exog, pd.DataFrame) else []
        vifs = []
        for i in range(arr.shape[1]):
            if cols and cols[i] == "const":
                continue
            vifs.append(float(variance_inflation_factor(arr, i)))
        if not vifs:
            return DiagnosticResult(name="vif", verdict="not_applicable", detail="no regressors")
        mx = max(vifs)
        return DiagnosticResult(name="vif", statistic=mx,
                                verdict="pass" if mx < 10 else "warn",
                                detail=f"max VIF={mx:.1f} (multicollinearity inflates SEs)")
    except Exception as e:
        return DiagnosticResult(name="vif", verdict="not_applicable", detail=str(e))


def _small_sample(outcome: EstimationOutcome) -> DiagnosticResult:
    n = outcome.n_obs
    return DiagnosticResult(name="sample_size", statistic=float(n),
                            verdict="pass" if n >= 50 else "warn",
                            detail=f"n={n}" + ("" if n >= 50 else
                                   " — small sample; asymptotic inference is approximate"))


def run_diagnostics(outcome: EstimationOutcome) -> DiagnosticsReport:
    """Run the battery on an estimated outcome; inestimable outcomes get an empty report."""
    refit = refit_from_outcome(outcome)
    if refit is None:
        return DiagnosticsReport(results=[], overall="fragile",
                                 severe_failures=["not refittable (inestimable or missing data)"])
    res, _y, exog = refit

    results: List[DiagnosticResult] = [
        _bp_test(res, exog),
        _dw_test(res),
        _residual_ar1(res),
        _ljung_box(res),
        _jarque_bera(res),
        _residual_adf(res, outcome),
        _reset_test(res),
        _vif(exog),
        _small_sample(outcome),
    ]

    severe = [r.name for r in results if r.verdict == "fail"]
    warns = [r.name for r in results if r.verdict == "warn"]
    overall = "fragile" if severe else ("caveats" if warns else "clean")
    return DiagnosticsReport(results=results, overall=overall, severe_failures=severe)


def apply_diagnostics_verdict(outcome: EstimationOutcome,
                              report: DiagnosticsReport) -> EstimationOutcome:
    """Downgrade 'estimated' to 'fragile' on severe failures — the harness's verdict, not
    narration, is what downstream consumers read."""
    if outcome.verdict == VERDICT_ESTIMATED and report.overall == "fragile":
        outcome = outcome.model_copy(update={
            "verdict": VERDICT_FRAGILE,
            "notes": outcome.notes + [
                "downgraded to fragile by diagnostics: " + ", ".join(report.severe_failures)],
        })
    return outcome
