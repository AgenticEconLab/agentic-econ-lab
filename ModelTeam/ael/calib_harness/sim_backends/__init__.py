# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Runnable-spine registry for simulation-based (SMM) calibration — Tier 1c/2 instances.

These are canonical WORKED INSTANCES of
the simulation spine (reduced-form dynamics, agent-based exchange, policy recursions), each
admitted under the Tier-2 contract (canonical family + cited targets + conservative
match-or-decline + unit tests). They are not a coverage claim, and per-run instance accretion is
policy-stopped: coverage grows mechanism-first (Tier 0 derivation from the model's own equations;
SimulationStage executing the model's own spec), with new instances only by deliberate decision
against the contract.

Each backend exposes ``build(formal_model, calibrated_parameters)`` -> a simulator spec
``(simulate_moments, moment_keys, free_params, x0, bounds)`` if the model matches its instance,
else None (decline is the default). ``build_from_registry`` returns the first match. Heavy deps
(e.g. Mesa) are OPTIONAL: a backend that can't import them declines, so the pipeline degrades to
an honest ``uncalibratable`` verdict rather than crashing."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def match_registry(formal_model: Dict[str, Any], calibrated_parameters: List[Any]):
    """Try each runnable archetype; return ``(archetype_name, spec)`` for the first match, or None.

    Every registry entry is a canonical STAND-IN: it simulates its own fixed equations, not the
    model's, and estimates only the parameters it shares with the model. Callers must report a
    registry fit as ``archetype_calibrated``, never as a simulation of the model itself."""
    from . import mesa_wealth, nk_taylor, var_growth
    for name, backend in (("mesa_wealth", mesa_wealth.build), ("nk_taylor", nk_taylor.build),
                          ("var_growth", var_growth.build)):
        try:
            spec = backend(formal_model, calibrated_parameters)
        except Exception:
            spec = None
        if spec is not None:
            return name, spec
    return None


def build_from_registry(formal_model: Dict[str, Any], calibrated_parameters: List[Any]):
    """Try each runnable archetype; return the first matching simulator spec, or None."""
    hit = match_registry(formal_model, calibrated_parameters)
    return hit[1] if hit else None
