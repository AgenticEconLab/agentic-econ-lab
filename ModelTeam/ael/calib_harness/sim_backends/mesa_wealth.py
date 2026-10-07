# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Runnable ABM archetype: the kinetic wealth-exchange model (Chakraborti & Chakrabarti 2000).

A canonical agent-based economic model: N agents each start with unit wealth and repeatedly meet
pairwise; in each encounter both save a fraction ``lambda`` of their wealth and randomly split the
pooled remainder. The stationary wealth distribution's inequality (Gini) is a monotone function of
the savings propensity ``lambda`` (higher savings -> lower inequality), so the Gini is a moment SMM
can match to recover ``lambda``. Built on **Mesa 3** (seeded, reproducible).

This is one entry in the runnable-archetype library (``sim_backends``): the SMM counterpart of the
closed-form ``archetypes``. It calibrates wealth-inequality ABMs — a class the closed-form harness
refuses (no closed-form Gini). Mesa is an OPTIONAL dependency; ``build`` returns None if Mesa is
absent or the model doesn't match the archetype, yielding an honest verdict rather than a crash."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# --- signatures that identify a wealth-inequality exchange model ---
_WEALTH_RE = re.compile(r"wealth|inequal|gini|income distribution|distribution of (wealth|income)", re.I)
# A bare "propensity" would capture a propensity SCORE and a case-insensitive
# "lambda" an error-distribution "\Lambda" in a description, so unrelated
# parameters would be "calibrated" as a savings rate. Match the economic role by phrase, never a
# bare word or a Greek letter.
_SAVINGS_RE = re.compile(
    r"\bsav(e|es|ed|ing|ings)\b|\bthrift|propensity to (save|hoard)|"
    r"retention (rate|share|ratio|propensity)|(wealth|income) retention",
    re.I)


def _gini(values) -> float:
    import numpy as np
    x = np.sort(np.asarray(list(values), dtype=float))
    n = x.size
    s = x.sum()
    if n == 0 or s <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * x) / (n * s)) - (n + 1.0) / n)


def simulate_moments(params: Dict[str, float], seed: int) -> Dict[str, float]:
    """Simulate the kinetic wealth-exchange model at the given savings propensity and return the
    stationary wealth-Gini. ``params['savings']`` (aka lambda) in [0, 1). Deterministic given seed."""
    import mesa

    lam = float(params.get("savings", params.get("lambda", 0.3)))
    lam = min(max(lam, 0.0), 0.999)
    n_agents, sweeps = 150, 160

    class _Trader(mesa.Agent):
        def __init__(self, model, wealth):
            super().__init__(model)          # Mesa 3: auto-assigns unique_id
            self.wealth = wealth

    class _Model(mesa.Model):
        def __init__(self):
            super().__init__(seed=int(seed))
            for _ in range(n_agents):
                _Trader(self, 1.0)

        def step(self):
            ags = list(self.agents)
            for _ in range(len(ags) // 2):   # one sweep = N/2 pairwise encounters
                a = self.random.choice(ags)
                b = self.random.choice(ags)
                if a is b:
                    continue
                eps = self.random.random()
                pool = (1.0 - lam) * (a.wealth + b.wealth)
                a.wealth = lam * a.wealth + eps * pool
                b.wealth = lam * b.wealth + (1.0 - eps) * pool

    m = _Model()
    for _ in range(sweeps):
        m.step()
    return {"income_gini": _gini(a.wealth for a in m.agents)}


def _model_text(formal_model: Dict[str, Any]) -> str:
    """Concatenate the model's title/summary + variable & parameter names for signature matching."""
    parts = [str(formal_model.get("model_title", "")), str(formal_model.get("model_summary", ""))]
    for coll in ("variables", "parameters"):
        for it in (formal_model.get(coll) or []):
            g = (lambda k: it.get(k)) if isinstance(it, dict) else (lambda k: getattr(it, k, None))
            parts += [str(g("variable_name") or ""), str(g("parameter_name") or ""),
                      str(g("description") or "")]
    return " ".join(parts)


def _savings_param_symbol(formal_model: Dict[str, Any]) -> Optional[str]:
    """The parameter symbol acting as the savings propensity (the SMM free parameter), if any."""
    for p in (formal_model.get("parameters") or []):
        g = (lambda k: p.get(k)) if isinstance(p, dict) else (lambda k: getattr(p, k, None))
        # the role is read from the parameter's NAME only: descriptions such as "consume rather
        # than save" or "consumption and savings decisions" can describe an MPC or a
        # risk-aversion coefficient
        name = f"{g('parameter_name') or ''}"
        if _SAVINGS_RE.search(name):
            sym = g("parameter_symbol") or g("symbol")
            if sym:
                return str(sym).strip()
    return None


def build(formal_model: Dict[str, Any], calibrated_parameters: List[Any]):
    """Return an SMM simulator spec if this is a wealth-inequality exchange model, else None.

    Conservative match: the model must be about wealth/inequality AND expose a savings-propensity
    parameter (that becomes the SMM free parameter). Also requires Mesa to be importable."""
    try:
        import mesa  # noqa: F401
    except Exception:
        return None

    text = _model_text(formal_model)
    if not (_WEALTH_RE.search(text) and _SAVINGS_RE.search(text)):
        return None
    sym = _savings_param_symbol(formal_model)
    if not sym:
        return None

    # The SMM engine optimizes the model's own savings-propensity symbol; simulate_moments reads
    # 'savings', so remap the symbol -> 'savings' before each simulation.
    def _sim(params: Dict[str, float], seed: int) -> Dict[str, float]:
        return simulate_moments({"savings": params.get(sym, 0.3)}, seed)

    return (_sim, ["income_gini"], [sym], [0.3], {sym: (0.05, 0.95)})
