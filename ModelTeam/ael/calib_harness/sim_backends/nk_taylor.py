# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Runnable archetype: backward-looking New-Keynesian economy under a Taylor rule.

Three equations, simulated as a stochastic linear recursion (pure numpy — no Mesa needed):

    IS:       x_t  = a·x_{t-1} − b·(i_t − π_t) + ε^d_t
    Phillips: π_t  = c·π_{t-1} + κ·x_t + ε^s_t
    Taylor:   i_t  = φ_π·π_t + φ_x·x_t

The within-period simultaneity (x_t, π_t) solves in closed form each period. The identifying
moment is **inflation volatility** (std of year-over-year inflation): a stronger Taylor response
φ_π damps inflation fluctuations, so std(π) is monotonically DECREASING in φ_π — SMM can recover
φ_π from the cited US inflation-volatility target (`inflation_volatility` in macro_targets.yaml).

This is the second entry in the runnable-archetype library: it makes monetary-policy /
Taylor-rule models — otherwise `partially_calibrated` on a single moment — genuinely simulation-calibratable. ``build`` matches conservatively (Taylor/monetary
-policy wording + an inflation-response parameter) and declines otherwise."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# --- structural signature: a monetary-policy model with a Taylor-type inflation response ---
_POLICY_RE = re.compile(
    r"taylor rule|monetary policy|policy rate|central bank|interest rate rule|inflation target",
    re.IGNORECASE,
)
# The role is the policy response to INFLATION; a generic "policy ... response" also
# names responses to output, credit or shocks.
_PHI_PI_RE = re.compile(
    r"phi_?\{?\\?_?\{?\\?(pi|π)|taylor.{0,30}(coefficient|response|parameter)|"
    r"inflation.{0,20}(response|reaction)|(response|reaction) to inflation",
    re.IGNORECASE,
)

# Fixed non-estimated structure (standard backward-looking NK parametrization; the SMM free
# parameter is the Taylor inflation response). Shock scales chosen so std(yoy π) spans the
# empirical target range (~0.8–2.5pp) over φ_π ∈ [1.05, 3.5].
_A, _B, _C, _KAPPA, _PHI_X = 0.7, 0.4, 0.6, 0.15, 0.125
_SIG_D, _SIG_S = 0.007, 0.0033


def simulate_moments(params: Dict[str, float], seed: int) -> Dict[str, float]:
    """Simulate the NK-Taylor economy; return std of year-over-year inflation (decimal).

    ``params['phi_pi']`` is the Taylor inflation response (>1 = active policy)."""
    import numpy as np

    phi_pi = float(params.get("phi_pi", 1.5))
    rng = np.random.default_rng(int(seed))
    T, burn = 1200, 200
    eps_d = rng.standard_normal(T) * _SIG_D
    eps_s = rng.standard_normal(T) * _SIG_S

    x = pi = 0.0
    pis = np.zeros(T)
    # substitute Taylor into IS and Phillips into IS -> closed-form x_t each period
    denom = 1.0 + _B * _PHI_X + _B * (phi_pi - 1.0) * _KAPPA
    for t in range(T):
        x = (_A * x - _B * (phi_pi - 1.0) * (_C * pi + eps_s[t]) + eps_d[t]) / denom
        pi = _C * pi + _KAPPA * x + eps_s[t]
        pis[t] = pi
    yoy = pis[burn:] + pis[burn - 1:-1] + pis[burn - 2:-2] + pis[burn - 3:-3]  # 4-quarter sum
    return {"inflation_volatility": float(np.std(yoy))}


def _model_text(formal_model: Dict[str, Any]) -> str:
    parts = [str(formal_model.get("model_title", "")), str(formal_model.get("model_summary", ""))]
    for coll in ("variables", "parameters", "equations"):
        for it in (formal_model.get(coll) or []):
            g = (lambda k: it.get(k)) if isinstance(it, dict) else (lambda k: getattr(it, k, None))
            for k in ("variable_name", "parameter_name", "equation_name", "description"):
                v = g(k)
                if v:
                    parts.append(str(v))
    return " ".join(parts)


def _phi_pi_symbol(formal_model: Dict[str, Any]) -> Optional[str]:
    """The parameter acting as the Taylor inflation-response coefficient, if any."""
    for p in (formal_model.get("parameters") or []):
        g = (lambda k: p.get(k)) if isinstance(p, dict) else (lambda k: getattr(p, k, None))
        # symbol and NAME only (an "Inflation Premium Weight" or a loss-function "Inflation
        # Stabilization Weight" was taken for the Taylor coefficient via description/'weight')
        text = f"{g('parameter_symbol') or ''} {g('parameter_name') or ''}"
        if _PHI_PI_RE.search(text):
            sym = g("parameter_symbol") or g("symbol")
            if sym:
                return str(sym).strip()
    return None


def build(formal_model: Dict[str, Any], calibrated_parameters: List[Any]):
    """SMM simulator spec for a monetary-policy model with a Taylor inflation response; else None."""
    try:
        import numpy  # noqa: F401
    except Exception:
        return None
    if not _POLICY_RE.search(_model_text(formal_model)):
        return None
    sym = _phi_pi_symbol(formal_model)
    if not sym:
        return None

    def _sim(params: Dict[str, float], seed: int) -> Dict[str, float]:
        return simulate_moments({"phi_pi": params.get(sym, 1.5)}, seed)

    # Taylor-principle region: active policy (>1); x0 at the canonical 1.5.
    return (_sim, ["inflation_volatility"], [sym], [1.5], {sym: (1.05, 3.5)})
