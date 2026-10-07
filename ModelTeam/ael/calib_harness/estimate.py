# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Deterministic parameter ESTIMATION for the calibration harness.

Only SCORING the LLM's asserted parameters against external targets produces a degenerate all-zero fit, because the LLM asserts values that do not actually
reproduce the data. This module closes that gap: it OPTIMIZES the free structural parameters (within
their literature bounds) to best-match the external empirical targets via bounded weighted
least-squares (a minimum-distance / GMM-style estimator). The harness then reports the ESTIMATED
parameters and the fit they achieve, so a just-identified model is point-calibrated (exact match)
and an over-identified model yields a genuine, testable fit — instead of fit ~ 0 everywhere.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares


def estimate(
    free_params: List[str],
    x0: Sequence[float],
    lo: Sequence[float],
    hi: Sequence[float],
    residual_fn: Callable[[np.ndarray], Sequence[float]],
    max_nfev: int = 300,
) -> Tuple[Dict[str, float], bool]:
    """Bounded least-squares minimize of ``residual_fn`` over ``free_params``.

    ``residual_fn(x)`` returns the per-moment weighted residuals ``(computed - target)/se``.
    Returns ``(optimized {param: value}, success)``. Never raises — a failed solve returns
    ``({}, False)`` so the caller falls back to the asserted parameters.
    """
    if not free_params:
        return {}, False
    x0a = np.clip(np.asarray(x0, dtype=float), np.asarray(lo, dtype=float), np.asarray(hi, dtype=float))
    try:
        sol = least_squares(
            lambda x: np.asarray(residual_fn(x), dtype=float),
            x0a,
            bounds=(np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)),
            max_nfev=max_nfev,
            method="trf",
        )
    except Exception:
        return {}, False
    if not np.all(np.isfinite(sol.x)):
        return {}, False
    return {p: float(sol.x[i]) for i, p in enumerate(free_params)}, bool(getattr(sol, "success", False) or sol.status > 0)
