# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
DSGE synthetic trajectory generator (V0.7).

Reference: Chib & Tan, "Learning the Macroeconomic Language" (arXiv 2512.21031,
Dec 2025) — pre-train transformers on 10M DSGE-simulated trajectories + 10%
real data, fine-tune on empirical data.

Ships reference implementations for:
    * RBC (real-business-cycle)
    * NK3eq (3-equation New Keynesian: IS + Phillips + Taylor rule)

Outputs are column-dicts of float lists to avoid a pandas dependency; callers
can convert to DataFrames trivially (``pd.DataFrame(result)``).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Tuple


ModelFamily = Literal["rbc", "nk3eq", "custom"]


@dataclass
class DSGESpec:
    model_family: ModelFamily
    n_trajectories: int = 10
    n_periods: int = 80
    # parameter -> ("uniform" | "normal", {...})
    parameter_distribution: Dict[str, Tuple[str, Dict[str, float]]] = field(default_factory=dict)
    shocks: List[str] = field(default_factory=list)
    burn_in: int = 20
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_trajectories <= 0:
            raise ValueError("n_trajectories must be positive")
        if self.n_periods <= 0:
            raise ValueError("n_periods must be positive")
        if self.burn_in < 0:
            raise ValueError("burn_in must be non-negative")


class DSGEGenerator:
    """Generate synthetic DSGE trajectories for theory-guided pre-training."""

    def generate(self, spec: DSGESpec) -> Dict[str, List[float]]:
        if spec.model_family == "rbc":
            return self._rbc(spec)
        if spec.model_family == "nk3eq":
            return self._nk3eq(spec)
        raise ValueError(f"Unsupported model_family: {spec.model_family!r}")

    # ------------------------------------------------------------------
    # RBC model (stochastic, log-linearised around steady state)
    # ------------------------------------------------------------------

    def _rbc(self, spec: DSGESpec) -> Dict[str, List[float]]:
        rng = random.Random(spec.seed)
        out = {"traj_id": [], "t": [], "y": [], "c": [], "k": [], "z": []}
        for tid in range(spec.n_trajectories):
            beta   = _sample(rng, spec.parameter_distribution.get("beta", ("uniform", {"low": 0.95, "high": 0.99})))
            alpha  = _sample(rng, spec.parameter_distribution.get("alpha", ("uniform", {"low": 0.30, "high": 0.40})))
            delta  = _sample(rng, spec.parameter_distribution.get("delta", ("uniform", {"low": 0.02, "high": 0.05})))
            rho    = _sample(rng, spec.parameter_distribution.get("rho", ("uniform", {"low": 0.85, "high": 0.99})))
            sigma  = _sample(rng, spec.parameter_distribution.get("sigma", ("uniform", {"low": 0.005, "high": 0.02})))

            k = 1.0
            z = 0.0
            # Steady-state savings rate for log-utility RBC:
            #   s* = alpha * beta, i.e. capital share × discount factor.
            # This keeps beta and alpha meaningful across the parameter grid,
            # unlike the earlier saturating expression which clipped to 0.5
            # for every value in the prior.
            s_rate = max(0.01, min(0.95, alpha * beta))
            for t in range(-spec.burn_in, spec.n_periods):
                eps = rng.gauss(0, sigma)
                z = rho * z + eps
                y = math.exp(z) * (k ** alpha)
                c = max(1e-6, (1 - s_rate) * y)
                k_new = (1 - delta) * k + max(0.0, y - c)
                if t >= 0:
                    out["traj_id"].append(tid)
                    out["t"].append(t)
                    out["y"].append(y)
                    out["c"].append(c)
                    out["k"].append(k)
                    out["z"].append(z)
                k = max(1e-4, k_new)
        return out

    # ------------------------------------------------------------------
    # 3-equation New Keynesian (IS + Phillips + Taylor rule)
    # ------------------------------------------------------------------

    def _nk3eq(self, spec: DSGESpec) -> Dict[str, List[float]]:
        rng = random.Random(spec.seed)
        out = {"traj_id": [], "t": [], "y_gap": [], "pi": [], "r": []}
        for tid in range(spec.n_trajectories):
            kappa  = _sample(rng, spec.parameter_distribution.get("kappa", ("uniform", {"low": 0.05, "high": 0.20})))
            sigma_is = _sample(rng, spec.parameter_distribution.get("sigma_is", ("uniform", {"low": 0.5, "high": 1.5})))
            phi_pi = _sample(rng, spec.parameter_distribution.get("phi_pi", ("uniform", {"low": 1.2, "high": 2.0})))
            phi_y  = _sample(rng, spec.parameter_distribution.get("phi_y",  ("uniform", {"low": 0.0, "high": 1.0})))
            sd_d   = _sample(rng, spec.parameter_distribution.get("sd_d",   ("uniform", {"low": 0.005, "high": 0.02})))
            sd_s   = _sample(rng, spec.parameter_distribution.get("sd_s",   ("uniform", {"low": 0.005, "high": 0.02})))

            y, pi = 0.0, 0.0
            for t in range(-spec.burn_in, spec.n_periods):
                eps_d = rng.gauss(0, sd_d)
                eps_s = rng.gauss(0, sd_s)
                r = phi_pi * pi + phi_y * y
                y_next = 0.9 * y - (1.0 / sigma_is) * (r - pi) + eps_d
                pi_next = 0.7 * pi + kappa * y_next + eps_s
                y, pi = y_next, pi_next
                if t >= 0:
                    out["traj_id"].append(tid)
                    out["t"].append(t)
                    out["y_gap"].append(y)
                    out["pi"].append(pi)
                    out["r"].append(r)
        return out


def _sample(rng: random.Random, dist: Tuple[str, Dict[str, float]]) -> float:
    kind, params = dist
    if kind == "uniform":
        return rng.uniform(params["low"], params["high"])
    if kind == "normal":
        return rng.gauss(params.get("mean", 0.0), params.get("std", 1.0))
    if kind == "constant":
        return params.get("value", 0.0)
    raise ValueError(f"Unknown distribution kind: {kind!r}")


def flatten_to_records(columns: Dict[str, List[float]]) -> List[Dict[str, float]]:
    """Convert column-dict output to a list of row-dicts (e.g. for pandas)."""
    n = len(next(iter(columns.values()))) if columns else 0
    keys = list(columns.keys())
    return [{k: columns[k][i] for k in keys} for i in range(n)]
