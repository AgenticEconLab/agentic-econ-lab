# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Deterministic economic-magnitude computation from the estimation artifact (§4.7
ResultsInterpreter's numeric core).

Everything is computed from the stored ``analysis_data`` panel + coefficients — pure
pandas/numpy, no LLM. The interpretation KIND follows the (dependent transform, regressor
transform) pair (treating growth-rate dependents as levels would give meaningless
"elasticities at means", and trend/interaction rows get no one-SD effects):

    dependent \\ regressor   level / diff          log                 growth (log_diff, pct)
    level                    marginal_effect       level_on_log        level_on_growth
    log                      semi_elasticity       elasticity          semi_elasticity
    diff                     change_effect         change_on_log       change_on_growth
    growth                   growth_effect         growth_on_log       growth_on_growth

(a ``diff`` regressor reads as "per unit change"; log_diff on log_diff is an elasticity.)
The elasticity at means b * mean(x) / mean(y) is computed only for true level-on-level (both
transforms 'level', neither variable built by subtraction), where both means are levels of
the variables; a 'diff' regressor gets none; for log-log the coefficient is itself the
elasticity.
Trend rows report neither a one-SD effect nor an elasticity. Interaction rows report the
marginal effect of the base variable conditional on the moderator at its sample mean,
b_base + b_interaction * mean(moderator), when both main effects are in the specification,
the base main effect has the same construction as the interaction's base term (series,
transform, lag, subtracted series and its transform) and the moderator's column is built
with the same transform and lag; otherwise n/a with a note. Point values only: the artifact holds no coefficient covariance.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from .types import EffectSize, InterpretationResult

_GROWTH = ("log_diff", "pct_change", "yoy_pct_change")


def _cls(transform: str) -> str:
    t = transform or "level"
    if t in _GROWTH:
        return "growth"
    return t if t in ("level", "log", "diff") else "level"


def _kind(dep_transform: str, reg_transform: str) -> str:
    dep, reg = _cls(dep_transform), _cls(reg_transform)
    if dep_transform == "log_diff" and reg_transform == "log_diff":
        return "elasticity"
    if reg == "diff":
        reg = "level"
    table = {
        ("level", "level"): "marginal_effect", ("level", "log"): "level_on_log",
        ("level", "growth"): "level_on_growth",
        ("log", "level"): "semi_elasticity", ("log", "log"): "elasticity",
        ("log", "growth"): "semi_elasticity",
        ("diff", "level"): "change_effect", ("diff", "log"): "change_on_log",
        ("diff", "growth"): "change_on_growth",
        ("growth", "level"): "growth_effect", ("growth", "log"): "growth_on_log",
        ("growth", "growth"): "growth_on_growth",
    }
    return table[(dep, reg)]


def _key(ref) -> str:
    return (ref or "").strip().lower()


def _construction(r: Dict) -> tuple:
    """How a regressor column is built: (series, transform, lag, subtracted series, the
    subtrahend's effective transform). Two terms are the same variable only when all match."""
    tf = r.get("transform") or "level"
    sub = _key(r.get("subtract_ref"))
    return (_key(r.get("series_ref")), tf, int(r.get("lag") or 0), sub,
            (r.get("subtract_transform") or tf) if sub else "")


def _interaction_conditional(c_name: str, reg: Dict, regs: List[Dict], coefs: Dict[str, float],
                             frame: pd.DataFrame):
    """(value, note) for the base variable's marginal effect at the moderator's mean.

    Matching the main effect on the series alone would, for b1 log(X) + b2 X*Z, give
    b1 + b2 mean(Z), while dy/dX = b1/X + b2 Z. The main effect therefore matches only when its full construction (series, transform, lag, subtracted
    series and its transform) equals the interaction's base term."""
    partner = reg.get("interact_with")
    plain = [r for r in regs if not r.get("interact_with")]
    base = next((r for r in plain if _construction(r) == _construction(reg)), None)
    mod = next((r for r in plain if _key(r.get("series_ref")) == _key(partner)
                and not r.get("subtract_ref")), None)
    if base is None and any(_key(r.get("series_ref")) == _key(reg.get("series_ref"))
                            for r in plain):
        return None, ("the main effect of the interaction's base series enters with a different "
                      "construction (transform, lag or subtraction) than inside the "
                      "interaction, so the marginal effect is not b(main) + b(interaction) x "
                      "mean(moderator); no conditional marginal effect is reported")
    if base is None or mod is None:
        return None, ("interaction entered without its main effect(s); no conditional "
                      "marginal effect is reported")
    if (mod.get("transform", "level") != reg.get("transform", "level")
            or int(mod.get("lag") or 0) != int(reg.get("lag") or 0)
            or mod.get("name") not in frame.columns):
        return None, ("the moderator's own column is built with a different transform or lag "
                      "than inside the interaction; no conditional marginal effect is reported")
    if base.get("name") not in coefs or c_name not in coefs:
        return None, "main-effect coefficient missing from the estimates"
    mean_z = float(frame[mod["name"]].mean())
    value = coefs[base["name"]] + coefs[c_name] * mean_z
    return value, (f"marginal effect of {base['name']} at the sample mean of {mod['name']} "
                   f"= b({base['name']}) + b({c_name}) × mean({mod['name']}) = "
                   f"{value:.4g} (point value; no standard error)")


def interpret_estimation(estimation_results: Dict) -> InterpretationResult:
    """Compute effect sizes + descriptives from an InferenceStageOutput-shaped dict.

    Honest passthrough: an inestimable upstream yields an empty result carrying the verdict —
    downstream sections then report the refusal instead of inventing magnitudes."""
    outcome = (estimation_results or {}).get("outcome") or {}
    verdict = outcome.get("verdict", "")
    result = InterpretationResult(
        verdict=verdict,
        dependent=outcome.get("dependent_name", ""),
        n_obs=int(outcome.get("n_obs") or 0),
        r_squared=outcome.get("r_squared"),
    )
    if verdict == "inestimable" or not outcome.get("analysis_data"):
        result.notes.append(f"no magnitudes: upstream estimation verdict is "
                            f"'{verdict or 'missing'}'"
                            + (f" ({outcome.get('reason')})" if outcome.get("reason") else ""))
        return result

    frame = pd.DataFrame(outcome["analysis_data"])
    if "date" in frame.columns:
        frame = frame.drop(columns=["date"])
    frame = frame.astype(float)

    for col in frame.columns:
        s = frame[col]
        result.descriptives[col] = {
            "mean": float(s.mean()), "sd": float(s.std()),
            "min": float(s.min()), "max": float(s.max()),
        }

    spec = outcome.get("spec") or {}
    dep_name = (spec.get("dependent") or {}).get("name", result.dependent)
    dep_transform = (spec.get("dependent") or {}).get("transform", "level")
    dep_level = ((dep_transform or "level") == "level"
                 and not (spec.get("dependent") or {}).get("subtract_ref"))
    regs = [r or {} for r in spec.get("regressors") or []]
    reg_by_name = {r.get("name"): r for r in regs}
    sd_y = result.descriptives.get(dep_name, {}).get("sd")
    mean_y = result.descriptives.get(dep_name, {}).get("mean")
    coefs = {c.get("name"): float(c.get("estimate", 0.0))
             for c in outcome.get("coefficients") or []}

    for c in outcome.get("coefficients") or []:
        name = c.get("name")
        if name == "const" or name not in frame.columns:
            continue
        b = float(c.get("estimate", 0.0))
        sd_x = result.descriptives[name]["sd"]
        mean_x = result.descriptives[name]["mean"]
        reg = reg_by_name.get(name, {})
        effect = dict(param=name, estimate=b, p_value=float(c.get("p_value", 1.0)), sd_x=sd_x)

        if name == "trend" and name not in reg_by_name:
            result.effects.append(EffectSize(
                **effect, interpretation_kind="trend",
                note="deterministic linear trend: the coefficient is the change in the "
                     "dependent per period; no one-SD effect or elasticity applies"))
            continue
        if reg.get("interact_with"):
            value, note = _interaction_conditional(name, reg, regs, coefs, frame)
            result.effects.append(EffectSize(
                **effect, interpretation_kind="interaction",
                conditional_marginal_effect=value, note=note))
            continue

        kind = _kind(dep_transform, reg.get("transform", "level"))
        elasticity: Optional[float] = None
        # _kind reads a 'diff' regressor as level-like, which would let an elasticity at
        # means use mean(delta X). It is computed only for true
        # level-on-level: both transforms 'level' and neither side built by subtraction.
        if (dep_level and (reg.get("transform") or "level") == "level"
                and not reg.get("subtract_ref")
                and mean_y is not None and abs(mean_y) > 1e-12):
            elasticity = b * mean_x / mean_y
        result.effects.append(EffectSize(
            **effect,
            one_sd_effect=b * sd_x,
            standardized_beta=(b * sd_x / sd_y) if sd_y and abs(sd_y) > 1e-12 else None,
            elasticity_at_means=elasticity,
            interpretation_kind=kind,
        ))
    return result
