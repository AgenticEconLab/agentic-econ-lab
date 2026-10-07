# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Shared data contracts for the deterministic calibration harness.

These dataclasses are the FIXED interface between modules:
    parser  -> ParsedEquation         (already defined in parser.py)
    archetypes.detect(...)            -> List[Moment]      (recipes: how to compute a moment from params)
    harness evaluates each Moment     -> MomentResult      (computed value + matched external target)
    guards / gmm consume MomentResult -> FitOutcome        (score + honest verdict)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set


@dataclass
class ParamRole:
    """A calibrated parameter mapped to a canonical economic role (CONFIRM, never guess)."""
    symbol: str                      # e.g. "alpha"
    role: Optional[str]              # canonical role e.g. "capital_share" | "labor_share" | "discount" | None
    value: Optional[float]           # the LLM-proposed calibrated value
    range: Optional[tuple] = None    # typical_range, used to CONFIRM the role
    description: str = ""


@dataclass
class Moment:
    """A RECIPE: how to compute one model-implied moment deterministically from parameter values.

    ``value_fn`` maps {param_symbol: float} -> float. ``free_params`` is the set of parameter symbols
    the moment genuinely depends on (>=2 required to be scorable — see guards: a 1-param identity is
    a tautology). ``pins`` names the single parameter a just-identifying moment pins (or None).
    """
    moment_key: str                  # canonical key joining to macro_targets.yaml, e.g. "capital_output_ratio"
    value_fn: Callable[[Dict[str, float]], float]
    free_params: Set[str]
    frequency: str = "annual"        # "annual" | "quarterly" — drives annualization vs targets
    pins: Optional[str] = None       # param this moment just-identifies (excluded from over-id scoring)
    archetype: str = ""              # which archetype produced it (provenance)


@dataclass
class TargetRow:
    """An external, cited empirical target. NEVER LLM-invented."""
    moment_key: str
    value: float
    std_error: float
    source_citation: str
    geography: str = "US"
    sample_period: str = ""
    vintage: str = ""
    frequency: str = "annual"


@dataclass
class MomentResult:
    moment_key: str
    computed: Optional[float]
    target: Optional[float]
    std_error: Optional[float]
    rel_error: Optional[float]
    free_params: List[str]
    role: str                        # "pin" | "over_id" | "unmatched" | "tautological" | "compute_error"
    source_citation: str = ""
    tautological: bool = False
    note: str = ""


@dataclass
class FitOutcome:
    """The harness's honest verdict for ONE model."""
    # "calibrated" | "point_calibrated" | "partially_calibrated" | "simulation_calibrated"
    # (SMM on a simulator of the model's OWN equations) | "archetype_calibrated" (SMM on a registered
    # canonical stand-in that shares a mechanism with the model; only ``estimated_params`` are
    # harness-estimated, every other parameter remains the LLM's proposal) | "uncalibratable"
    calibration_status: str
    fit_score: Optional[float]       # FORMAL over-identification GMM fit; None when df<1 (untestable). NEVER 0.0-coerced.
    uncalibratable_reason: Optional[str] = None
    # DESCRIPTIVE (not a formal over-id test): how well the LLM-proposed params reproduce the matched
    # external targets — mean over matched, non-tautological moments. Informative even when
    # just-identified (the LLM did NOT solve identification, so its moments genuinely miss targets).
    moment_match_score: Optional[float] = None
    moments: List[MomentResult] = field(default_factory=list)
    coverage: Dict[str, int] = field(default_factory=dict)  # parseable_eqs, refused_eqs, over_id_moments, ...
    j_stat: Optional[float] = None
    degrees_of_freedom: Optional[int] = None
    # Parameters the harness ESTIMATED (optimized within literature bounds to match targets), and
    # whether estimation was applied. When estimated, fit_score/moment_match reflect the estimated
    # params — a genuine calibration, not a score of the LLM's (typically non-matching) assertions.
    estimated: bool = False
    estimated_params: Dict[str, float] = field(default_factory=dict)
    # Which simulator produced a simulation-based fit, and whether it is a registered canonical
    # stand-in (archetype) rather than the model's own equations. Empty for closed-form paths.
    simulator: str = ""
    simulator_is_archetype: bool = False
