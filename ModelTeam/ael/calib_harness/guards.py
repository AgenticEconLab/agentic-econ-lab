# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Validity guards for the deterministic calibration harness.

These are the *validity core* of the redesign. The old LLM-self-validation path scored
fits against moments that were either tautological (``computed == a calibrated parameter``,
e.g. ``labor_share 0.65 == alpha 0.65``) or against LLM-invented targets. The guards here
make those failure modes structurally impossible:

* :func:`is_tautological` — a moment is a real over-identifying restriction ONLY if it
  genuinely depends on >=2 free parameters. A 1-parameter identity (``labor_share = alpha``,
  ``r_star = 1/beta - 1``) carries no independent information once that single parameter is
  set, so it cannot *score*; it may only *pin* a parameter.
* :func:`partition_moments` — split externally-targeted moments into roles
  (``pin`` / ``over_id`` / ``tautological`` / ``unmatched`` / ``compute_error``) so the GMM
  step scores HELD-OUT (over-identifying) moments only.
* :func:`provenance_ok` — assert the external target literal never entered the moment
  computation, and that the computed value is genuinely a function of parameters.

ZERO LLM calls. Pure structural / numeric checks over the locked dataclasses in ``types.py``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .types import Moment, MomentResult, TargetRow

# Role constants (mirror MomentResult.role vocabulary in types.py).
ROLE_PIN = "pin"
ROLE_OVER_ID = "over_id"
ROLE_TAUTOLOGICAL = "tautological"
ROLE_UNMATCHED = "unmatched"
ROLE_COMPUTE_ERROR = "compute_error"


def is_tautological(moment: Moment) -> bool:
    """True if the moment cannot serve as an over-identifying restriction.

    A moment is structurally tautological when its ``free_params`` set has fewer than two
    entries: a 0-param moment is a pure constant and a 1-param moment is an identity in that
    single parameter (knowing the parameter determines the moment exactly, so it carries no
    independent, over-identifying information). Such moments may legitimately *pin* a
    parameter, but they must never be scored.
    """
    return len(moment.free_params) < 2


def _rel_error(computed: Optional[float], target: Optional[float]) -> Optional[float]:
    """Relative error |computed - target| / |target|, or None if not computable."""
    if computed is None or target is None:
        return None
    if target == 0:
        # fall back to absolute error when the target is exactly zero
        return abs(computed)
    return abs(computed - target) / abs(target)


def partition_moments(
    moments: List[Moment],
    computed_values: Dict[str, Optional[float]],
    targets: Optional[Dict[str, TargetRow]] = None,
) -> List[MomentResult]:
    """Tag each moment with an identification role and build its :class:`MomentResult`.

    Parameters
    ----------
    moments:
        The moment *recipes* (from ``archetypes``).
    computed_values:
        Map ``moment_key -> computed float`` (the harness already evaluated each recipe's
        ``value_fn`` on the proposed parameters). A missing key or ``None`` value means the
        computation failed -> role ``compute_error``.
    targets:
        Map ``moment_key -> TargetRow`` of *matched external* targets. A moment with no row
        here is ``unmatched`` (never back-filled).

    Role assignment (priority order, mutually exclusive):

    1. ``compute_error`` — the recipe produced no numeric value.
    2. ``unmatched``     — no external target row matched this ``moment_key``.
    3. ``pin``           — a just-identifying moment (``moment.pins`` set); held out of scoring
                            because it is *used to set* a parameter, not to test the model.
    4. ``tautological``  — structurally tautological (<2 free params) and not pinning anything;
                            dropped (cannot over-identify even though a target exists).
    5. ``over_id``       — >=2 free params, matched, not pinning -> HELD-OUT scorable moment.
    """
    targets = targets or {}
    results: List[MomentResult] = []

    for m in moments:
        computed = computed_values.get(m.moment_key)
        trow = targets.get(m.moment_key)
        target = trow.value if trow is not None else None
        se = trow.std_error if trow is not None else None
        citation = trow.source_citation if trow is not None else ""
        taut = is_tautological(m)
        rel = _rel_error(computed, target)
        free = sorted(m.free_params)

        if computed is None:
            role, note = ROLE_COMPUTE_ERROR, "value_fn produced no numeric value"
        elif target is None:
            role, note = ROLE_UNMATCHED, "no_external_target_match"
        elif m.pins is not None:
            role = ROLE_PIN
            note = f"just-identifies '{m.pins}' (held out of scoring)"
        elif taut:
            role = ROLE_TAUTOLOGICAL
            note = f"single-param identity in {free or '<const>'}; not over-identifying"
        else:
            role, note = ROLE_OVER_ID, "over-identifying restriction (scored)"

        results.append(
            MomentResult(
                moment_key=m.moment_key,
                computed=computed,
                target=target,
                std_error=se,
                rel_error=rel,
                free_params=free,
                role=role,
                source_citation=citation,
                tautological=taut,
                note=note,
            )
        )

    return results


def provenance_ok(moment_result: MomentResult) -> bool:
    """Assert the moment was computed from parameters, NOT copied from its target.

    Returns ``False`` (provenance violated) when any of these hold:

    * ``computed is None`` — nothing was computed, so provenance cannot be established;
    * ``free_params`` is empty — the value does not depend on any parameter, so it is not a
      genuine model-implied moment ``m(theta)``;
    * ``computed == target`` exactly — in a deterministic harness the computed value comes from
      ``value_fn(theta)`` and the target comes from the curated YAML; bit-identical floats are
      the signature of the external target literal leaking into the computation (the original
      tautology bug: ``computed`` was set equal to the empirical target).

    A clean over-identifying moment depends on >=1 parameter and differs from its target.
    """
    if moment_result.computed is None:
        return False
    if not moment_result.free_params:
        return False
    if moment_result.target is not None and moment_result.computed == moment_result.target:
        return False
    return True
