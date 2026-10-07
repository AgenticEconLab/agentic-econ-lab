# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Deterministic calibration harness — orchestration.

``run(formal_model, calibrated_parameters)`` scores an LLM-PROPOSED parameter vector against
EXTERNAL empirical targets using deterministic, sympy-derived model moments. No LLM call, no
fabricated target, no tautological "fit". A model that cannot be honestly scored returns
``calibration_status='uncalibratable'`` with ``fit_score=None`` — never a fabricated number.

Pipeline:  parse equations -> map parameter roles -> detect archetype moments (by structural
           signature) -> compute each moment from the calibrated values + ANNUALIZE ->
           match external targets + guard (tautology / identification) -> GMM score over the
           held-out over-identifying moments -> honest verdict + coverage report.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .parser import parse_equation
from .types import FitOutcome
from . import archetypes, targets as targets_mod, guards, gmm, estimate


def _calibrated_values(calibrated_parameters: List[Any]) -> Dict[str, float]:
    """Extract {parameter_symbol: float} from CalibratedParameter dicts/objects (best-effort)."""
    out: Dict[str, float] = {}
    for p in calibrated_parameters or []:
        get = (lambda k: p.get(k)) if isinstance(p, dict) else (lambda k: getattr(p, k, None))
        sym = get("parameter_symbol") or get("symbol")
        val = get("calibrated_value")
        if val is None:
            val = get("value")
        if not sym:
            continue
        try:
            out[archetypes.canonical_symbol(sym)] = float(val)
        except (TypeError, ValueError):
            continue
    return out


def _compute(moment, values: Dict[str, float]) -> Optional[float]:
    """Evaluate a moment recipe on the calibrated values and annualize to match the targets."""
    try:
        raw = float(moment.value_fn(values))
    except Exception:
        return None
    periods = {"quarterly": 4, "monthly": 12}.get(getattr(moment, "frequency", "annual"), 1)
    try:
        return float(archetypes.annualize(moment.moment_key, raw, periods))
    except Exception:
        return raw


def run(formal_model: Dict[str, Any], calibrated_parameters: List[Any]) -> FitOutcome:
    """Score a model. Closed-form moment matching first; if that is ``uncalibratable`` (the model
    has no closed-form structural moments, e.g. an agent-based / RL model), fall back to
    simulation-based estimation (SMM), which returns either a genuine ``simulation_calibrated`` fit
    or a MORE PRECISE ``uncalibratable`` reason (e.g. learned components need training). Never
    fabricates a fit."""
    outcome = _run_closed_form(formal_model, calibrated_parameters)
    if outcome.calibration_status != "uncalibratable":
        return outcome
    try:
        from .sim_calib import try_simulation_calibrate
        sim = try_simulation_calibrate(formal_model, calibrated_parameters)
    except Exception:
        return outcome  # SMM path unavailable -> keep the honest closed-form verdict
    # Prefer the SMM outcome: a real simulation fit, or a sharper uncalibratable reason.
    return sim if sim is not None else outcome


def _run_closed_form(formal_model: Dict[str, Any], calibrated_parameters: List[Any]) -> FitOutcome:
    equations = formal_model.get("equations") or []
    parameters = formal_model.get("parameters") or []

    parsed = [parse_equation(e if isinstance(e, dict) else dict(e)) for e in equations]
    n_ok = sum(1 for p in parsed if p.status == "ok")
    n_refused = sum(1 for p in parsed if p.status.startswith("refused"))
    n_parse_fail = sum(1 for p in parsed if p.status.startswith("parse_fail"))
    # every equation is accounted for: parseable + refused + parse_fail (+ no_equation) = total
    eqc = {"parse_fail_eqs": n_parse_fail, "total_eqs": len(equations)}

    # Role mapping + archetype detection run even when NO equation parsed: signatures read each
    # equation's RAW text (refused ones included) and role-based moments (e.g. the HANK
    # hand-to-mouth MPC pin) need only the parameter declarations. The old n_ok==0 early return
    # short-circuited this, contradicting the signature design and dropping real HANK models.
    roles = archetypes.map_param_roles(parameters)
    values = _calibrated_values(calibrated_parameters)
    moments = archetypes.detect_moments(parsed, roles, archetypes.model_period(formal_model, roles))

    if not moments:
        # R1: nothing algebraic AND nothing recognized.
        if n_ok == 0:
            return FitOutcome(
                calibration_status="uncalibratable", fit_score=None,
                uncalibratable_reason="non_algebraic: no equation parsed to closed form (refused/parse_fail)",
                coverage={"parseable_eqs": 0, "refused_eqs": n_refused, **eqc},
            )
        # R5: no archetype matched (structure unrecognized / roles unresolved).
        return FitOutcome(
            calibration_status="uncalibratable", fit_score=None,
            uncalibratable_reason="no_archetype_match: model structure not recognized by any v1 archetype",
            coverage={"parseable_eqs": n_ok, "refused_eqs": n_refused, "moments_detected": 0, **eqc},
        )

    target_book = targets_mod.load_targets()
    # ESTIMATE: optimize the free structural parameters (within their literature bounds) to
    # best-match the external targets, instead of scoring the LLM's asserted (typically
    # non-matching) values. Turns the degenerate all-zero fit into a genuine calibration: a
    # just-identified model becomes point-calibrated (exact), an over-identified one yields a
    # real testable fit.
    est_values, estimated, opt, bound_active = _estimate_params(moments, values, parameters, target_book)
    use_values = est_values if estimated else values
    # every optimized parameter off its bounds is an estimate, including one that happens to equal
    # the proposal (a "differs from the proposal" filter would report 0 estimated)
    est_params = {k: x for k, x in opt.items() if k not in bound_active}
    eqc["bound_active_params"] = len(bound_active)

    computed_values = {m.moment_key: _compute(m, use_values) for m in moments}
    results = guards.partition_moments(moments, computed_values, target_book)

    match_score = _match_score(results)
    over_id = [r for r in results if r.role == "over_id"]
    n_pin = sum(1 for r in results if r.role == "pin")

    # R2/R3: no over-identifying moment. After estimation a pin-only model is a valid POINT
    # calibration (its estimated params reproduce the targets) even though there is no
    # over-identifying restriction to TEST — report that honestly rather than "uncalibratable".
    if not over_id:
        all_taut = bool(results) and all(r.tautological for r in results if r.role != "unmatched" and r.computed is not None)
        point = bool(estimated and n_pin and est_params and not bound_active and pins_hit(results))
        return FitOutcome(
            calibration_status=("point_calibrated" if point else "partially_calibrated" if n_pin else "uncalibratable"),
            fit_score=None, moment_match_score=match_score,
            uncalibratable_reason=(None if point else _shortfall(bound_active, results) or
                                   ("all_tautological" if all_taut
                                    else "no_overidentifying_moment_with_external_target")),
            moments=results, coverage={**_coverage(n_ok, n_refused, results), **eqc},
            estimated=estimated, estimated_params=est_params,
        )

    fit_score, j_stat, df = gmm.score(results)

    # R4: df<1 -> no over-identifying restriction to test. With estimation this is a point
    # calibration (params estimated to hit the moments); reported via moment_match_score.
    if fit_score is None or (df is not None and df < 1):
        point = bool(estimated and est_params and not bound_active and pins_hit(results))
        return FitOutcome(
            calibration_status=("point_calibrated" if point else "partially_calibrated"),
            fit_score=None, moment_match_score=match_score,
            uncalibratable_reason=(None if point else _shortfall(bound_active, results) or
                                   f"under_identified: degrees_of_freedom={df} (<1, untestable)"),
            moments=results, j_stat=j_stat, degrees_of_freedom=df,
            coverage={**_coverage(n_ok, n_refused, results), **eqc},
            estimated=estimated, estimated_params=est_params,
        )

    # 'calibrated' also needs the over-identifying restrictions not rejected (J test, 5%), no
    # parameter stuck on a bound, and pinned moments that hit their targets (a fit with all
    # estimates on their bounds is not 'calibrated').
    rejected = _j_rejected(j_stat, df)
    ok_fit = not rejected and not bound_active and (pins_hit(results) or not n_pin)
    status = "calibrated" if (n_ok >= 3 and len(over_id) >= 2 and ok_fit) else "partially_calibrated"
    reason = None
    if status != "calibrated":
        reason = (_shortfall(bound_active, results)
                  or ("overidentifying_restrictions_rejected: J test p < 0.05" if rejected else None))
    return FitOutcome(
        calibration_status=status, fit_score=fit_score, moment_match_score=match_score,
        uncalibratable_reason=reason,
        moments=results, j_stat=j_stat, degrees_of_freedom=df,
        coverage={**_coverage(n_ok, n_refused, results), **eqc},
        estimated=estimated, estimated_params=est_params,
    )


def _shortfall(bound_active, results) -> Optional[str]:
    if bound_active:
        return ("target_outside_declared_range: " + ", ".join(bound_active)
                + " stopped at a bound of the declared range without reproducing the target")
    if any(r.role == "pin" for r in results) and not pins_hit(results):
        return "target_not_reproduced: a pinned moment misses its target by more than two standard errors"
    return None


def _j_rejected(j_stat, df) -> bool:
    if j_stat is None or not df or df < 1:
        return False
    try:
        from scipy.stats import chi2
        return float(chi2.sf(j_stat, df)) < 0.05
    except Exception:
        return False


def _parse_range(rng):
    """Parse a typical_range like '[0.3, 0.7]' / '0.95, 0.999' -> (lo, hi) or None."""
    if rng is None:
        return None
    import re as _re
    nums = _re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", str(rng))
    if len(nums) >= 2:
        lo, hi = float(nums[0]), float(nums[1])
        if lo < hi:
            return (lo, hi)
    return None


def _param_bounds(parameters) -> Dict[str, tuple]:
    out: Dict[str, tuple] = {}
    for p in parameters or []:
        get = (lambda k: p.get(k)) if isinstance(p, dict) else (lambda k: getattr(p, k, None))
        sym = get("parameter_symbol") or get("symbol")
        rng = _parse_range(get("typical_range"))
        if sym and rng:
            # keyed like the moments' free_params: a raw '\\beta' key never matched 'beta', so
            # the parameter would have no bounds and be silently left at the LLM's value
            out[archetypes.canonical_symbol(sym)] = rng
    return out


def _estimate_params(moments, values, parameters, target_book):
    """Optimize the free params (within literature bounds) to best-match matched targets.
    Returns (values_with_estimates, estimated_bool). Falls back to the asserted values on failure."""
    matched = [(m, target_book[m.moment_key]) for m in moments if m.moment_key in target_book]
    if not matched:
        return values, False, {}, []
    bounds = _param_bounds(parameters)
    free = sorted({p for m, _ in matched for p in m.free_params if p in bounds})
    if not free:
        return values, False, {}, []
    x0 = [values.get(p, sum(bounds[p]) / 2.0) for p in free]
    lo = [bounds[p][0] for p in free]
    hi = [bounds[p][1] for p in free]

    def resid(x):
        v = dict(values)
        v.update({p: x[i] for i, p in enumerate(free)})
        out = []
        for m, tr in matched:
            cm = _compute(m, v)
            if cm is None:
                out.append(1e3)
                continue
            se = tr.std_error if (tr.std_error and tr.std_error > 0) else max(abs(tr.value) * 0.1, 1e-3)
            out.append((cm - tr.value) / se)
        return out

    opt, ok = estimate.estimate(free, x0, lo, hi, resid)
    if not ok:
        return values, False, {}, []
    v = dict(values)
    v.update(opt)
    return v, True, {p: opt[p] for p in free}, at_bound(opt, bounds)


def at_bound(opt: Dict[str, float], bounds: Dict[str, tuple]) -> List[str]:
    """Parameters whose estimate sits on a bound of its admissible range. Such a value is where
    the optimizer stopped, not an estimate that reproduces the target (e.g. beta = 0.99, the
    declared upper bound, implying r* = 4.1% against a 2.0% target)."""
    out = []
    for p, x in opt.items():
        lo, hi = bounds[p]
        tol = 1e-4 * max(hi - lo, 1e-12)
        if x <= lo + tol or x >= hi - tol:
            out.append(p)
    return sorted(out)


def pins_hit(results) -> bool:
    """Every pinning moment reproduces its target within two standard errors."""
    pins = [r for r in results if r.role == "pin"]
    if not pins:
        return False
    for r in pins:
        if r.computed is None or r.target is None:
            return False
        se = r.std_error if (r.std_error and r.std_error > 0) else max(abs(r.target) * 0.1, 1e-3)
        if abs(r.computed - r.target) > 2.0 * se:
            return False
    return True


def _match_score(results):
    """Descriptive (NOT a formal over-id test): 1 - mean relative error over matched, identified
    moments (pin + over_id). Informative because the LLM proposed params WITHOUT solving
    identification, so its moments genuinely miss the external targets."""
    rels = [r.rel_error for r in results if r.role in ("pin", "over_id") and r.rel_error is not None]
    if not rels:
        return None
    return round(max(0.0, 1.0 - sum(rels) / len(rels)), 4)


def _coverage(n_ok: int, n_refused: int, results) -> Dict[str, int]:
    return {
        "parseable_eqs": n_ok,
        "refused_eqs": n_refused,
        "moments_detected": len(results),
        "moments_with_target": sum(1 for r in results if r.target is not None),
        "over_id_moments": sum(1 for r in results if r.role == "over_id"),
        "pin_moments": sum(1 for r in results if r.role == "pin"),
        "tautological_moments": sum(1 for r in results if r.tautological),
    }
