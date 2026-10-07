# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Deterministic calibration harness for AEL ModelTeam.

Replaces the LLM-self-validation calibration (which produced tautological/invented-target/
non-reproducible "fits") with a deterministic, externally-grounded moment-matching scorer.

Public API:
    run(formal_model, calibrated_parameters) -> FitOutcome

"""

from .types import FitOutcome, Moment, MomentResult, ParamRole, TargetRow  # noqa: F401

__all__ = ["run", "FitOutcome", "Moment", "MomentResult", "ParamRole", "TargetRow"]


def run(formal_model, calibrated_parameters):
    # Imported lazily so importing the package (e.g. for types) doesn't require sympy.
    from .harness import run as _run
    return _run(formal_model, calibrated_parameters)
