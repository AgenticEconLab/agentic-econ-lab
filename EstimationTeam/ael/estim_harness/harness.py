# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Deterministic estimation core: EstimationSpec + panel -> honest EstimationOutcome.

The LLM proposed the spec; everything here is executed numerics. Refusals are first-class:
a spec this data cannot support returns ``inestimable`` with a machine-readable reason,
never a degraded fit passed off as evidence.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .transforms import align_panel, apply_transform, infer_frequency
from .types import (
    VERDICT_ESTIMATED,
    VERDICT_INESTIMABLE,
    CoefficientResult,
    EstimationOutcome,
    EstimationSpec,
    VariableSpec,
)

MIN_OBS = 30            # absolute floor for a time-series regression we are willing to report
MIN_OBS_PER_PARAM = 8   # and at least this many observations per estimated parameter


def _inestimable(spec: EstimationSpec, reason: str, note: str) -> EstimationOutcome:
    return EstimationOutcome(
        verdict=VERDICT_INESTIMABLE, reason=reason, method=spec.method,
        cov_type=spec.cov_type, dependent_name=spec.dependent.name,
        notes=[note], spec=spec,
    )


def resolve_ref(ref: str, columns, alias_map: Dict[str, str]) -> Optional[str]:
    """Resolve a spec's series_ref to a column/key (exact, alias, case-insensitive)."""
    columns = list(columns)
    if ref in columns:
        return ref
    hit = alias_map.get(ref.strip().lower())
    if hit in columns:
        return hit
    lower_cols = {str(c).lower(): c for c in columns}
    return lower_cols.get(ref.strip().lower())


def build_design(
    spec: EstimationSpec,
    data,
    alias_map: Dict[str, str],
) -> Tuple[Optional[pd.DataFrame], str, List[str]]:
    """Transform + lag every variable and inner-join into the analysis frame.

    ``data`` is either an UNJOINED {series_id: pd.Series} map (the loader's output — only the
    variables this spec uses are aligned, so one sparse unrelated series cannot empty the
    sample) or an already-aligned DataFrame. Returns (design, frequency, notes); design is
    None with notes[-1] holding the failure when a reference is missing or a transform is
    invalid for its series.
    """
    notes: List[str] = []
    keys = list(data.keys()) if isinstance(data, dict) else list(data.columns)

    resolved: Dict[str, str] = {}
    interact_resolved: Dict[str, str] = {}
    subtract_resolved: Dict[str, str] = {}
    for var in [spec.dependent] + list(spec.regressors) + list(spec.instruments):
        col = resolve_ref(var.series_ref, keys, alias_map)
        if col is None:
            notes.append(f"missing_variable: '{var.series_ref}' (for '{var.name}') is not "
                         f"among the loaded series ({keys})")
            return None, "annual", notes
        resolved[var.name] = col
        other = getattr(var, "interact_with", None)
        if other:
            col2 = resolve_ref(other, keys, alias_map)
            if col2 is None:
                notes.append(f"missing_variable: interaction series '{other}' (for "
                             f"'{var.name}') is not among the loaded series ({keys})")
                return None, "annual", notes
            interact_resolved[var.name] = col2
        minus = getattr(var, "subtract_ref", None)
        if minus:
            col3 = resolve_ref(minus, keys, alias_map)
            if col3 is None:
                notes.append(f"missing_variable: subtraction series '{minus}' (for "
                             f"'{var.name}') is not among the loaded series ({keys})")
                return None, "annual", notes
            subtract_resolved[var.name] = col3

    if isinstance(data, dict):
        spec_cols = (set(resolved.values()) | set(interact_resolved.values())
                     | set(subtract_resolved.values()))
        panel, frequency = align_panel({c: data[c] for c in spec_cols})
        notes.append(f"aligned {len(spec_cols)} spec series at {frequency} "
                     f"frequency: {len(panel)} overlapping observations")
    else:
        panel = data
        frequency = infer_frequency(panel.index) if len(panel) >= 3 else "annual"

    columns: Dict[str, pd.Series] = {}
    for var in [spec.dependent] + list(spec.regressors) + list(spec.instruments):
        tf_notes: List[str] = []
        try:
            s = apply_transform(panel[resolved[var.name]].astype(float), var.transform,
                                frequency, tf_notes)
            if var.name in subtract_resolved:
                # Derived difference (real rate = nominal - inflation); the
                # subtrahend may need its OWN transform (yoy inflation from a price index).
                sub_tf = getattr(var, "subtract_transform", None) or var.transform
                s = s - apply_transform(
                    panel[subtract_resolved[var.name]].astype(float), sub_tf, frequency,
                    tf_notes)
            if var.name in interact_resolved:
                # Interaction term: the base variable is built FULLY
                # first (series minus its subtrahend), THEN multiplied, so the column is
                # (X - W) * Z, not X * Z - W. The partner Z takes the
                # term's own `transform` (not `subtract_transform`, which belongs to W).
                s = s * apply_transform(
                    panel[interact_resolved[var.name]].astype(float), var.transform, frequency,
                    tf_notes)
        except ValueError as e:
            notes.append(f"invalid transform for '{var.name}' ({resolved[var.name]}): {e}")
            return None, frequency, notes
        notes.extend(f"'{var.name}': {n}" for n in tf_notes)
        if var.lag:
            s = s.shift(var.lag)
        columns[var.name] = s

    design = pd.DataFrame(columns)
    if spec.sample_start:
        design = design[design.index >= pd.Timestamp(spec.sample_start)]
    if spec.sample_end:
        design = design[design.index <= pd.Timestamp(spec.sample_end)]
    design = design.dropna(how="any")
    if spec.include_trend:
        design["trend"] = np.arange(len(design), dtype=float)
        notes.append("linear trend included (common-trend guard)")
    return design, frequency, notes


def residual_ar1(resid) -> float:
    """First-order autocorrelation of a residual series (OLS of e_t on e_{t-1}, no constant)."""
    e = np.asarray(resid, dtype=float)
    e = e[np.isfinite(e)]
    if len(e) < 3:
        return 0.0
    den = float(np.dot(e[:-1], e[:-1]))
    return float(np.dot(e[1:], e[:-1]) / den) if den > 0 else 0.0


def hac_maxlags(resid) -> Tuple[int, float]:
    """HAC bandwidth that adapts to residual persistence (floor(0.75 n^(1/3)) gives only
    3 lags for n = 61 even with DW = 0.02). It is the larger of the Newey-West
    (1994) rule floor(4 (n/100)^(2/9)) and the Andrews (1991) AR(1) plug-in for the Bartlett
    kernel, S = 1.1447 (alpha n)^(1/3) with alpha = 4 rho^2 / ((1 - rho)^6 (1 + rho)^2),
    rho the residuals' AR(1) coefficient (clipped to |rho| <= 0.97), capped at
    min(n // 4, ceil(sqrt(n))): near rho = 0.97 the plug-in reaches about n / 4 (1,319 lags
    for n = 5,279), where the HAC estimate is itself unreliable; such residuals already make
    the verdict 'fragile'. Returns (maxlags, rho)."""
    n = len(resid)
    rho = residual_ar1(resid)
    r = max(-0.97, min(0.97, rho))
    nw = int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    alpha = 4.0 * r * r / (((1.0 - r) ** 6) * ((1.0 + r) ** 2))
    andrews = int(math.ceil(1.1447 * (alpha * n) ** (1.0 / 3.0))) if alpha > 0 else 0
    cap = max(1, min(n // 4, int(math.ceil(math.sqrt(n)))))
    return max(1, min(cap, max(nw, andrews))), rho


def required_observations(spec: EstimationSpec, min_obs: int = MIN_OBS) -> Tuple[int, int]:
    """(parameters counted, observations required) — the rule run_estimation applies."""
    k = len(spec.regressors) + (1 if spec.add_constant else 0)
    return k, max(min_obs, k * MIN_OBS_PER_PARAM)


def _regressor_cols(spec: EstimationSpec) -> List[str]:
    """Regressor column names in the design frame, incl. the optional trend."""
    cols = [v.name for v in spec.regressors]
    if spec.include_trend:
        cols.append("trend")
    return cols


def _dedup_regressors(spec: EstimationSpec) -> Tuple[EstimationSpec, List[str]]:
    """A spec may carry an 'interaction' regressor with the SAME (series_ref, transform,
    lag) as another — a duplicate column that makes the whole regression rank-deficient. Identical columns are dropped with a disclosed note so
    estimation proceeds on the distinct regressors instead of refusing everything."""
    seen: Dict[tuple, str] = {}
    kept, dropped_notes = [], []
    for v in spec.regressors:
        key = (v.series_ref.strip().lower(), v.transform, v.lag,
               (getattr(v, "interact_with", None) or "").strip().lower(),
               (getattr(v, "subtract_ref", None) or "").strip().lower(),
               getattr(v, "subtract_transform", None) or "")
        if key in seen:
            dropped_notes.append(
                f"dropped duplicate regressor '{v.name}': identical (series, transform, lag) "
                f"as '{seen[key]}' — a genuine interaction needs `interact_with`")
        else:
            seen[key] = v.name
            kept.append(v)
    if not dropped_notes:
        return spec, []
    return spec.model_copy(update={"regressors": kept}), dropped_notes


def missing_main_effects(spec: EstimationSpec) -> List[str]:
    """Interactions entered without their main effects (e.g. labour share x Gini alone).
    For each interaction regressor, the constituent terms absent from
    the other regressors are listed — the base term (series, minus subtract_ref when set)
    and the partner series. A coefficient on X*Z without X and Z is not a moderation effect
    around the main effects. Returns one note per interaction with a missing main effect."""
    def key(ref):
        return (ref or "").strip().lower()

    plain = [v for v in spec.regressors if not getattr(v, "interact_with", None)]
    out = []
    for v in spec.regressors:
        partner = getattr(v, "interact_with", None)
        if not partner:
            continue
        missing = []
        # the main effect must be the same construction as the interaction's base term
        # (series, transform, lag, subtracted series and its transform), not just the series
        def same_base(p):
            return (key(p.series_ref) == key(v.series_ref) and p.transform == v.transform
                    and p.lag == v.lag
                    and key(getattr(p, "subtract_ref", None)) == key(getattr(v, "subtract_ref", None))
                    and (getattr(p, "subtract_transform", None) or p.transform)
                    == (getattr(v, "subtract_transform", None) or v.transform))
        base_present = any(same_base(p) for p in plain)
        if not base_present:
            base = v.series_ref + (f" - {v.subtract_ref}" if getattr(v, "subtract_ref", None)
                                   else "")
            missing.append(f"'{base}'")
        # the partner enters the product with the term's own transform and lag
        if not any(key(p.series_ref) == key(partner) and not getattr(p, "subtract_ref", None)
                   and p.transform == v.transform and p.lag == v.lag for p in plain):
            missing.append(f"'{partner}'")
        if missing:
            out.append(f"interaction without main effects: '{v.name}' enters without the main "
                       f"effect(s) of {' and '.join(missing)}; its coefficient is not a "
                       "moderation effect conditional on the main effects")
    return out


def dependent_in_regressors(spec: EstimationSpec, panel, alias_map: Dict[str, str]) -> List[str]:
    """A regressor 'real rate = FEDFUNDS - yoy CPI inflation' with annual CPI inflation as
    the dependent regresses the dependent on itself (R^2 = 0.996, coefficient ~ -1). Any regressor or instrument that uses
    the dependent's series - as its own series, the subtracted series, or the interaction
    partner - at the dependent's lag puts the dependent on both sides. Lagged use is allowed.

    The dependent is itself a construction - its series_ref, its subtract_ref and its
    interact_with are ALL constituents (regressing (A - B) on contemporaneous B gives
    R^2 0.9999 if only A is checked). Every constituent of the dependent is compared with every constituent of
    each regressor and instrument at the same lag."""
    try:
        columns = list(panel.keys()) if hasattr(panel, "keys") else list(panel.columns)
    except Exception:
        columns = []

    def canon(ref):
        if not ref:
            return None
        hit = resolve_ref(ref, columns, alias_map) if columns else None
        return str(hit if hit is not None else ref).strip().lower()

    fields = ("series_ref", "subtract_ref", "interact_with")
    dep_parts = {canon(getattr(spec.dependent, f, None)) for f in fields} - {None}
    hits = []
    for v in list(spec.regressors) + list(spec.instruments):
        if v.lag != spec.dependent.lag:
            continue
        for field in fields:
            if canon(getattr(v, field, None)) in dep_parts:
                hits.append(f"'{v.name}' ({field})")
                break
    return hits


def repair_departures(approved: EstimationSpec, repair: EstimationSpec, columns,
                      alias_map: Dict[str, str]) -> List[str]:
    """A repair proposed after a harness refusal must repair the
    APPROVED specification, not replace it. Returns the departures (empty = acceptable):

    - the dependent must have the same construction (series, transform, lag, subtracted
      series and its transform, interaction partner);
    - every approved hypothesis must be kept with the same parameter name and restriction,
      and that parameter's regressor must measure the same construction (series, transform,
      subtracted series and its transform, interaction partner). Its lag may change: moving
      a regressor to a lag is the standard repair of a same-period overlap with the dependent.
    Other regressors, the sample window and the method may change; with a reviewer, those
    changes are voted on before the repair is fitted."""
    columns = list(columns or [])

    def canon(ref):
        if not ref:
            return None
        hit = resolve_ref(ref, columns, alias_map) if columns else None
        return str(hit if hit is not None else ref).strip().lower()

    def construction(v: VariableSpec, with_lag: bool) -> tuple:
        sub = canon(getattr(v, "subtract_ref", None))
        key = (canon(v.series_ref), v.transform, sub,
               (getattr(v, "subtract_transform", None) or v.transform) if sub else None,
               canon(getattr(v, "interact_with", None)))
        return key + ((v.lag,) if with_lag else ())

    out: List[str] = []
    if construction(repair.dependent, True) != construction(approved.dependent, True):
        out.append(f"the repair changes the dependent variable ('{approved.dependent.name}' "
                   f"on {approved.dependent.series_ref} -> '{repair.dependent.name}' on "
                   f"{repair.dependent.series_ref}, or its transform, lag or construction)")
    approved_regs = {v.name: v for v in approved.regressors}
    repair_regs = {v.name: v for v in repair.regressors}
    for h in approved.hypotheses:
        if not any(r.param == h.param and r.restriction == h.restriction
                   for r in repair.hypotheses):
            out.append(f"the repair drops hypothesis '{h.name}' ({h.param} {h.restriction})")
            continue
        old, new = approved_regs.get(h.param), repair_regs.get(h.param)
        if old is None:
            continue
        if new is None:
            out.append(f"the repair removes regressor '{h.param}', the parameter of "
                       f"hypothesis '{h.name}'")
        elif construction(new, False) != construction(old, False):
            out.append(f"the repair changes what regressor '{h.param}' (hypothesis "
                       f"'{h.name}') measures")
    return out


def run_estimation(
    spec: EstimationSpec,
    panel,  # {series_id: pd.Series} from the loader, or an aligned pd.DataFrame
    alias_map: Optional[Dict[str, str]] = None,
    min_obs: int = MIN_OBS,
) -> EstimationOutcome:
    """Estimate the spec on the loaded series and return the harness's verdict."""
    alias_map = alias_map or {}
    spec, dedup_notes = _dedup_regressors(spec)

    if spec.method == "panel_fe":
        return _inestimable(spec, "method_unavailable",
                            "panel_fe needs an entity dimension the single-entity time-series "
                            "loader does not carry — honest decline")
    if spec.method == "iv2sls":
        reg_names = {v.name for v in spec.regressors}
        if not spec.endogenous or not spec.instruments:
            return _inestimable(spec, "invalid_spec",
                                "iv2sls requires both `endogenous` regressor names and "
                                "`instruments`")
        if not set(spec.endogenous) <= reg_names:
            return _inestimable(spec, "invalid_spec",
                                f"endogenous names {spec.endogenous} must be regressors")
        if len(spec.instruments) < len(spec.endogenous):
            return _inestimable(spec, "under_identified",
                                f"{len(spec.endogenous)} endogenous regressor(s) but only "
                                f"{len(spec.instruments)} instrument(s) — order condition fails")

    dup = [v.name for v in spec.regressors if v.name == spec.dependent.name]
    all_names = ([v.name for v in spec.regressors] + [v.name for v in spec.instruments])
    if dup or len(set(all_names)) != len(all_names):
        return _inestimable(spec, "invalid_spec", "duplicate variable names in the specification")

    same_period = dependent_in_regressors(spec, panel, alias_map)
    if same_period:
        return _inestimable(
            spec, "dependent_in_regressor",
            "a series that constructs the dependent (its series, subtracted series or "
            "interaction partner) enters " + ", ".join(same_period) + " at the dependent's own "
            "lag, so it appears on both sides of the regression; use a lagged value instead")

    design, frequency, notes = build_design(spec, panel, alias_map)
    if design is None:
        reason = "missing_variable" if notes and notes[-1].startswith("missing_variable") else "invalid_spec"
        return _inestimable(spec, reason, notes[-1] if notes else "design construction failed")
    notes = dedup_notes + notes   # disclose any dropped duplicate columns
    notes.extend(missing_main_effects(spec))

    k, needed = required_observations(spec, min_obs)
    if len(design) < needed:
        return _inestimable(
            spec, "insufficient_observations",
            f"{len(design)} overlapping observations after transforms/lags; "
            f"{needed} required for {k} parameters")

    y = design[spec.dependent.name]
    X = design[_regressor_cols(spec)]
    if float(y.std()) < 1e-12 or not math.isfinite(float(y.std())):
        return _inestimable(spec, "degenerate_dependent",
                            f"dependent '{spec.dependent.name}' has (near-)zero variance")

    import statsmodels.api as sm
    exog = sm.add_constant(X, has_constant="add") if spec.add_constant else X
    if np.linalg.matrix_rank(np.asarray(exog, dtype=float)) < exog.shape[1]:
        return _inestimable(spec, "singular_design",
                            "regressor matrix is rank-deficient (perfectly collinear regressors)")

    if spec.method == "iv2sls":
        res, names, err = _fit_iv(spec, design, notes)
        if err:
            return _inestimable(spec, err[0], err[1])
        coefficients = _iv_coefficients(res, names)
        r_squared = float(res.rsquared)
        adj_r_squared = None
        f_pvalue = None
        aic = bic = None
    else:
        model = sm.OLS(np.asarray(y, dtype=float), np.asarray(exog, dtype=float))
        if spec.cov_type == "HAC":
            maxlags, rho = hac_maxlags(model.fit().resid)
            res = model.fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
            notes.append(f"HAC (Newey-West) covariance, maxlags={maxlags} (bandwidth from "
                         f"residual persistence: AR(1) rho={rho:.2f})")
        elif spec.cov_type == "HC1":
            res = model.fit(cov_type="HC1")
        else:
            res = model.fit()

        names = (["const"] if spec.add_constant else []) + _regressor_cols(spec)
        ci = res.conf_int()
        coefficients = [
            CoefficientResult(
                name=names[i],
                estimate=float(res.params[i]),
                std_error=float(res.bse[i]),
                t_stat=float(res.tvalues[i]),
                p_value=float(res.pvalues[i]),
                ci_low=float(ci[i][0]),
                ci_high=float(ci[i][1]),
            )
            for i in range(len(names))
        ]
        r_squared = float(res.rsquared)
        adj_r_squared = float(res.rsquared_adj)
        _f = getattr(res, "f_pvalue", None)
        f_pvalue = None if _f is None or not math.isfinite(float(_f)) else float(_f)
        aic, bic = float(res.aic), float(res.bic)

    # The identification statement is part of the ARTIFACT, not the
    # narration — a reader of the output alone must see what the design does and does not
    # claim. Method-derived, discipline-free; proxy disclosure keys on the spec's own
    # naming, not on any topic list.
    if spec.method == "iv2sls":
        notes.append("identification: instrumental variables — validity rests on instrument "
                     "relevance and exogeneity; exogeneity is an assumption, not a test result")
    else:
        notes.append("identification: least squares on observational time series — "
                     "coefficients are associations conditional on the included regressors; "
                     "no causal identification strategy is claimed")
    _proxies = [v.name for v in spec.regressors if "proxy" in (v.name or "").lower()]
    if _proxies:
        notes.append("proxy disclosure: " + ", ".join(_proxies) + " — proxy measurement(s) "
                     "of the target construct; estimates inherit the proxy's measurement error")

    records = design.reset_index().rename(columns={design.index.name or "index": "date"})
    records["date"] = records["date"].astype(str)

    return EstimationOutcome(
        verdict=VERDICT_ESTIMATED,     # diagnostics (stage 2) may downgrade to 'fragile'
        method=spec.method,
        cov_type=spec.cov_type,
        coefficients=coefficients,
        n_obs=int(len(design)),
        r_squared=r_squared,
        adj_r_squared=adj_r_squared,
        f_pvalue=f_pvalue,
        aic=aic,
        bic=bic,
        sample_start=str(design.index.min().date()),
        sample_end=str(design.index.max().date()),
        frequency=frequency,
        dependent_name=spec.dependent.name,
        notes=notes,
        analysis_data=records.to_dict(orient="records"),
        spec=spec,
    )


def _fit_iv(spec: EstimationSpec, design: pd.DataFrame, notes: List[str]):
    """linearmodels IV2SLS fit. Returns (results, coefficient_names, error_or_None)."""
    try:
        from linearmodels.iv import IV2SLS
    except ImportError:
        return None, [], ("method_unavailable", "linearmodels is not installed")
    dep = design[spec.dependent.name]
    endog_names = list(spec.endogenous)
    exog_names = [c for c in _regressor_cols(spec) if c not in spec.endogenous]
    instr_names = [v.name for v in spec.instruments]
    exog = design[exog_names] if exog_names else None
    if spec.add_constant:
        import statsmodels.api as sm
        exog = sm.add_constant(exog if exog is not None else pd.DataFrame(index=design.index),
                               has_constant="add")
    try:
        res = IV2SLS(dep, exog, design[endog_names], design[instr_names]).fit(
            cov_type="robust" if spec.cov_type in ("HAC", "HC1") else "unadjusted")
        notes.append(f"IV2SLS: endogenous={endog_names}, instruments={instr_names}, "
                     f"robust covariance")
    except Exception as e:
        return None, [], ("invalid_spec", f"IV2SLS fit failed: {e}")
    names = list(res.params.index)
    return res, names, None


def _iv_coefficients(res, names: List[str]) -> List[CoefficientResult]:
    ci = res.conf_int()
    return [
        CoefficientResult(
            name=str(n),
            estimate=float(res.params[n]),
            std_error=float(res.std_errors[n]),
            t_stat=float(res.tstats[n]),
            p_value=float(res.pvalues[n]),
            ci_low=float(ci.loc[n, "lower"]),
            ci_high=float(ci.loc[n, "upper"]),
        )
        for n in names
    ]


def refit_from_outcome(outcome: EstimationOutcome):
    """Deterministically re-fit the stored analysis panel (stages 2/3 never re-touch the
    network). Returns (results, y, exog DataFrame) or None when not refittable. For iv2sls
    the results object is a minimal shim exposing ``.resid`` — enough for the residual-based
    diagnostics; statsmodels-specific tests (RESET) degrade to not_applicable."""
    if outcome.verdict == VERDICT_INESTIMABLE or not outcome.analysis_data or outcome.spec is None:
        return None
    import statsmodels.api as sm
    frame = pd.DataFrame(outcome.analysis_data)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.set_index("date").astype(float)
    y = frame[outcome.spec.dependent.name]
    X = frame[[c for c in _regressor_cols(outcome.spec) if c in frame.columns]]
    exog = sm.add_constant(X, has_constant="add") if outcome.spec.add_constant else X

    if outcome.method == "iv2sls":
        iv_res, _names, err = _fit_iv(outcome.spec, frame, [])
        if err:
            return None
        from types import SimpleNamespace
        shim = SimpleNamespace(resid=np.asarray(iv_res.resids, dtype=float))
        return shim, y, exog

    if outcome.cov_type == "HAC":
        maxlags, _rho = hac_maxlags(sm.OLS(y, exog).fit().resid)
        res = sm.OLS(y, exog).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    elif outcome.cov_type == "HC1":
        res = sm.OLS(y, exog).fit(cov_type="HC1")
    else:
        res = sm.OLS(y, exog).fit()
    return res, y, exog
