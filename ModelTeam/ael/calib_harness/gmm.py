# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""GMM-style over-identification scoring for the calibration harness.

Scoring happens over HELD-OUT (over-identifying) moments ONLY -- moments that were used to
*pin* parameters are excluded so a model cannot "fit" the very data that set its parameters.

For the over-identifying set the J-statistic is the standard distance between model-implied
moments and external targets, weighted by the targets' standard errors::

    J(theta) = sum_j  (computed_j - target_j)^2 / se_j^2

The degrees of freedom is the number of over-identifying restrictions::

    df = (# scored over_id moments) - (# free params they depend on that are NOT pinned)

If ``df < 1`` the configuration is *just-identified or under-identified* and the J-statistic
carries no testable information -> we return ``fit_score=None`` (the caller marks the model
``uncalibratable``; an honest N/A, NEVER a fabricated 0.0 or ~1.0).

The bounded, monotone map ``fit = exp(-J/df)`` sends a perfect match (J=0) to 1.0 and degrades
smoothly toward 0 as the normalized misfit grows. ``df`` normalizes J so that adding more
over-identifying restrictions does not mechanically penalize the score.

ZERO LLM calls.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from .types import MomentResult
from .guards import ROLE_OVER_ID, ROLE_PIN


def _is_finite(x: Optional[float]) -> bool:
    return x is not None and math.isfinite(x)


def score(
    moment_results: List[MomentResult],
) -> Tuple[Optional[float], Optional[float], int]:
    """Score a model over its over-identifying moments.

    Returns ``(fit_score, j_stat, df)``:

    * ``fit_score`` in ``[0, 1]`` (``exp(-J/df)``), or ``None`` when ``df < 1``;
    * ``j_stat`` the GMM J-statistic, or ``None`` when ``df < 1``;
    * ``df`` the integer degrees of freedom (always returned, may be <1).
    """
    # Pinning moments contribute identified parameters that cost no degree of freedom.
    pinned_params = set()
    for m in moment_results:
        if m.role == ROLE_PIN:
            pinned_params.update(m.free_params or [])

    # Only over_id moments with a usable computed/target/se enter J and df.
    usable: List[MomentResult] = [
        m
        for m in moment_results
        if m.role == ROLE_OVER_ID
        and _is_finite(m.computed)
        and _is_finite(m.target)
        and m.std_error is not None
        and m.std_error > 0
    ]

    over_id_params = set()
    for m in usable:
        over_id_params.update(m.free_params or [])

    free_params_not_pinned = over_id_params - pinned_params
    df = len(usable) - len(free_params_not_pinned)

    if df < 1:
        # Under-/just-identified: the J-statistic is not a testable over-identification.
        return (None, None, df)

    j_stat = 0.0
    for m in usable:
        resid = (m.computed - m.target) / m.std_error
        j_stat += resid * resid

    fit_score = math.exp(-j_stat / df)
    # Guard against floating-point excursions outside [0, 1].
    fit_score = max(0.0, min(1.0, fit_score))

    return (fit_score, j_stat, df)
