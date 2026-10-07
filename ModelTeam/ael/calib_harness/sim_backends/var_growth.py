# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Runnable archetype: reduced-form output-growth autoregression (VAR/BVAR family core).

Time-series macro models the pipeline often produces (regime-switching BVARs, VAR-based policy
frameworks) were `requires_training` solely for their LLM-derived inputs — but their econometric
CORE is a reduced-form autoregression, which is trivially simulable. This archetype simulates

    g_t = (1 − ρ)·μ + ρ·g_{t−1} + σ·ε_t          (annual output growth, μ fixed at 2.5%)

and matches TWO cited moments — the volatility and the first-order persistence of US real GDP
growth — so SMM just-identifies (ρ, σ). ``build`` matches conservatively: VAR/autoregressive
wording plus a confirmed persistence parameter AND a shock-scale parameter; otherwise None."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

_MU = 0.025   # unconditional mean growth (documented constant, not estimated)

_VAR_RE = re.compile(
    r"\bvar\b|vector autoregress|autoregress|regime.switching|\bbvar\b|markov.switching",
    re.IGNORECASE,
)
_PERSIST_RE = re.compile(r"persist|autoregressive (coefficient|parameter)|ar\(1\)|serial correlation",
                         re.IGNORECASE)
# A bare "std"/"dispersion" also names choice-error or cross-sectional scales; the role is
# the scale of the growth/shock process.
_SHOCK_SCALE_RE = re.compile(
    r"(shock|innovation|disturbance|growth).{0,25}(volatility|std|standard deviation|scale|size|"
    r"magnitude|dispersion)|(volatility|std|standard deviation|dispersion) of (the )?"
    r"(shock|innovation|disturbance|growth|output)",
    re.IGNORECASE)
# The stand-in simulates AGGREGATE output growth, so both roles must name an
# aggregate object; "Income Share Persistence" or an individual income-shock s.d. would match
# the role words alone and be fit to output-growth moments.
_AGGREGATE_RE = re.compile(r"\b(output|gdp|growth|aggregate|productivity|tfp)\b", re.IGNORECASE)


def simulate_moments(params: Dict[str, float], seed: int) -> Dict[str, float]:
    """Simulate the growth AR(1); return std + lag-1 autocorrelation of annual growth."""
    import numpy as np

    rho = min(max(float(params.get("rho", 0.3)), -0.95), 0.95)
    sigma = max(float(params.get("sigma", 0.015)), 1e-6)
    rng = np.random.default_rng(int(seed))
    T, burn = 800, 100
    g = np.zeros(T)
    eps = rng.standard_normal(T)
    for t in range(1, T):
        g[t] = (1.0 - rho) * _MU + rho * g[t - 1] + sigma * eps[t]
    g = g[burn:]
    return {"output_growth_volatility": float(np.std(g)),
            "output_growth_persistence": float(np.corrcoef(g[:-1], g[1:])[0, 1])}


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


def _param_matching(formal_model: Dict[str, Any], pattern: re.Pattern) -> Optional[str]:
    for p in (formal_model.get("parameters") or []):
        g = (lambda k: p.get(k)) if isinstance(p, dict) else (lambda k: getattr(p, k, None))
        text = f"{g('parameter_name') or ''}"
        if pattern.search(text) and _AGGREGATE_RE.search(text):
            sym = g("parameter_symbol") or g("symbol")
            if sym:
                return str(sym).strip()
    return None


def build(formal_model: Dict[str, Any], calibrated_parameters: List[Any]):
    """SMM spec for a VAR-family model with persistence + shock-scale parameters; else None."""
    try:
        import numpy  # noqa: F401
    except Exception:
        return None
    if not _VAR_RE.search(_model_text(formal_model)):
        return None
    rho_sym = _param_matching(formal_model, _PERSIST_RE)
    sig_sym = _param_matching(formal_model, _SHOCK_SCALE_RE)
    if not rho_sym or not sig_sym or rho_sym == sig_sym:
        return None

    def _sim(params: Dict[str, float], seed: int) -> Dict[str, float]:
        return simulate_moments({"rho": params.get(rho_sym, 0.3),
                                 "sigma": params.get(sig_sym, 0.015)}, seed)

    return (_sim, ["output_growth_volatility", "output_growth_persistence"],
            [rho_sym, sig_sym], [0.3, 0.015],
            {rho_sym: (-0.5, 0.9), sig_sym: (0.002, 0.05)})
