# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Data contracts for the deterministic estimation harness.

Mirror of ModelTeam's ``calib_harness/types.py`` philosophy: the LLM *proposes* an
``EstimationSpec``; the harness validates it against the actual dataset, estimates it with
statsmodels, and issues the honest verdict. The verdict the pipeline reports is always the
harness's, never the LLM's narration.

Verdicts:
    estimated    -- converged fit, no severe diagnostic failures
    fragile      -- estimated, but the diagnostics battery found severe violations
    inestimable  -- the spec cannot be honestly estimated on this data; ``reason`` is
                    machine-readable (missing_variable | insufficient_observations |
                    no_overlap | singular_design | degenerate_dependent | method_unavailable |
                    invalid_spec)
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

VERDICT_ESTIMATED = "estimated"
VERDICT_FRAGILE = "fragile"
VERDICT_INESTIMABLE = "inestimable"

TRANSFORMS = ("level", "log", "diff", "log_diff", "pct_change", "yoy_pct_change")


class VariableSpec(BaseModel):
    """One variable in the regression, mapped to a column of the loaded dataset."""
    name: str = Field(description="Role name used in the regression, e.g. 'inflation'")
    series_ref: str = Field(description="Column in the loaded dataset (series_id or series_name)")
    transform: Literal["level", "log", "diff", "log_diff", "pct_change", "yoy_pct_change"] = "level"
    lag: int = Field(default=0, ge=0, le=8, description="Lags applied AFTER the transform")
    # Interaction terms: without this field the LLM would encode
    # "Fed Funds x Debt Service" as a duplicate FEDFUNDS column -> singular design.
    interact_with: Optional[str] = Field(
        default=None,
        description="series_ref of a SECOND series: the regressor column becomes the product "
                    "of this variable (fully built: series, minus subtract_ref when set) and "
                    "the second series under this variable's `transform` (an interaction "
                    "term): (X - W) * Z; `lag` then applies to the product")
    # Derived differences: without this field "real interest rate" would be proxied by
    # the NOMINAL rate because r = i - pi would have no spec form.
    subtract_ref: Optional[str] = Field(
        default=None,
        description="series_ref of a series to SUBTRACT: the column becomes the difference "
                    "of the two transformed series (e.g. real rate = nominal rate minus "
                    "inflation)")
    # A real rate needs MIXED transforms — level(FEDFUNDS) minus
    # yoy_pct_change(CPI); forcing a common transform would subtract the price LEVEL and
    # produce a degenerate regressor the harness could only report as an exact zero.
    subtract_transform: Optional[
        Literal["level", "log", "diff", "log_diff", "pct_change", "yoy_pct_change"]] = Field(
        default=None,
        description="transform applied to subtract_ref's series (None = same as `transform`); "
                    "e.g. real rate: transform='level' on the nominal rate, "
                    "subtract_transform='yoy_pct_change' on the price index")
    # Which variable of the CodeTeam's executable model this series measures. Recorded
    # and checked against the module (estim_harness.model_link); the estimate ignores it.
    model_variable: Optional[str] = Field(
        default=None,
        description="symbol of the executable model's variable this series measures, if any")


class HypothesisSpec(BaseModel):
    """A theory-derived restriction on one coefficient, tested deterministically."""
    name: str
    param: str = Field(description="Regressor name the restriction applies to")
    restriction: Literal["<0", ">0", "=0"] = Field(
        description="Sign/zero restriction; one-sided t for </> and two-sided for =0")
    rationale: str = ""


class EstimationSpec(BaseModel):
    """What the Estimator agent proposes and the harness disposes."""
    dependent: VariableSpec
    regressors: List[VariableSpec] = Field(min_length=1)
    method: Literal["ols", "iv2sls", "panel_fe"] = "ols"
    # iv2sls only: which regressors are endogenous, and the excluded instruments.
    # Order condition len(instruments) >= len(endogenous) is enforced by the harness.
    endogenous: List[str] = Field(default_factory=list)
    instruments: List[VariableSpec] = Field(default_factory=list)
    cov_type: Literal["HAC", "HC1", "nonrobust"] = "HAC"
    add_constant: bool = True
    # A long levels regression invites common-trend confounding
    # (wrong-sign significants). When True, a deterministic linear trend joins the regressors.
    include_trend: bool = False
    sample_start: Optional[str] = None   # ISO date; None = full overlap
    sample_end: Optional[str] = None
    hypotheses: List[HypothesisSpec] = Field(default_factory=list)
    rationale: str = ""
    fallback_spec: bool = Field(
        default=False,
        description="True when the harness substituted a deterministic default because the "
                    "LLM proposal was unusable (disclosed, never silent)")


class CoefficientResult(BaseModel):
    name: str
    estimate: float
    std_error: float
    t_stat: float
    p_value: float
    ci_low: float
    ci_high: float


class EstimationOutcome(BaseModel):
    """The harness's honest verdict for ONE estimated specification."""
    verdict: Literal["estimated", "fragile", "inestimable"]
    reason: Optional[str] = None          # machine-readable, only for inestimable
    method: str = "ols"
    cov_type: str = "HAC"
    coefficients: List[CoefficientResult] = Field(default_factory=list)
    n_obs: int = 0
    r_squared: Optional[float] = None
    adj_r_squared: Optional[float] = None
    f_pvalue: Optional[float] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    sample_start: str = ""
    sample_end: str = ""
    frequency: str = ""                   # frequency the panel was aligned to
    dependent_name: str = ""
    notes: List[str] = Field(default_factory=list)
    # The aligned analysis panel (records of {date, <var>: value, ...}) so downstream stages
    # re-run diagnostics/robustness DETERMINISTICALLY without touching the network again.
    analysis_data: List[Dict] = Field(default_factory=list)
    spec: Optional[EstimationSpec] = None


class DiagnosticResult(BaseModel):
    name: str
    statistic: Optional[float] = None
    p_value: Optional[float] = None
    verdict: Literal["pass", "warn", "fail", "not_applicable"]
    detail: str = ""


class DiagnosticsReport(BaseModel):
    """Deterministic battery outcome; ``overall`` feeds the verdict upgrade/downgrade."""
    results: List[DiagnosticResult] = Field(default_factory=list)
    overall: Literal["clean", "caveats", "fragile"] = "clean"
    severe_failures: List[str] = Field(default_factory=list)
    interpretation: str = ""              # LLM narration of the COMPUTED stats (never a verdict)


class HypothesisResult(BaseModel):
    name: str
    param: str
    restriction: str
    estimate: Optional[float] = None
    p_value: Optional[float] = None       # one-sided for </>, two-sided for =0
    supported: Optional[bool] = None      # None when the parameter was absent
    # Finer verdict. Sign restrictions (<0, >0):
    # "supported" (predicted sign, one-sided p < 0.05) | "inconclusive" (not significant either
    # way) | "contradicted" (opposite sign, two-sided p < 0.05). Zero restriction (=0):
    # "consistent" (fail to reject b = 0) | "rejected" (reject b = 0). None when absent.
    outcome: Optional[str] = None
    detail: str = ""


class RobustnessCheck(BaseModel):
    name: str
    description: str
    focus_param: str
    baseline_estimate: Optional[float] = None
    variant_estimate: Optional[float] = None
    sign_stable: Optional[bool] = None
    significance_stable: Optional[bool] = None
    detail: str = ""


class InferenceReport(BaseModel):
    hypotheses: List[HypothesisResult] = Field(default_factory=list)
    robustness: List[RobustnessCheck] = Field(default_factory=list)
    stability_score: Optional[float] = None   # share of robustness checks with stable sign
    interpretation: str = ""                  # LLM narration only


class PerformanceProfile(BaseModel):
    """§4.6 Optimizer: real wall-clock timing across the 3-stage workflow, split into
    deterministic-compute vs. LLM proposal/narration time, so the researcher can see
    where the econometric bottleneck actually sits. Profiles the harness; rewrites
    nothing (an Optimizer that edited estimation code would forfeit the harness's
    'the verdict is always the harness's, never the LLM's' guarantee)."""
    stage_timing_sec: Dict[str, float] = Field(default_factory=dict)
    deterministic_sec: float = 0.0
    llm_sec: float = 0.0
    total_sec: float = 0.0
    bottleneck_step: str = ""
    notes: List[str] = Field(default_factory=list)
