# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Data contracts for the deterministic code harness (§4.5, Code Implementation).

Design position: the module is DERIVED from the model's parsed sympy system —
the same parser the calibration harness trusts — via templated ``sympy.pycode`` printing.
No LLM text ever enters the generated module, which is why executing it in-process is
acceptable; LLM-freeform code generation (the paper's full Coder) stays deferred behind
sandbox hardening. Refusal is first-class: a model whose equations don't parse
gets `ungenerable` with the parser's reasons, never a hallucinated implementation."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class RefusedEquation(BaseModel):
    equation_id: str
    status: str            # parser status: refused_nonalgebraic | parse_fail:<Err> | no_equation
    raw: str = ""


class GenerationResult(BaseModel):
    """Verdict of deriving a runnable module from one model's equations."""
    verdict: Literal["generated", "partial", "ungenerable"]
    reason: Optional[str] = None
    model_title: str = ""
    module_text: str = ""
    variables: List[str] = Field(default_factory=list)      # unknowns, solve order
    parameters: Dict[str, float] = Field(default_factory=dict)   # calibrated values
    unset_parameters: List[str] = Field(default_factory=list)    # declared, no value — honest
    n_equations: int = 0
    n_parseable: int = 0
    refused: List[RefusedEquation] = Field(default_factory=list)
    square: bool = False                                     # n_eqs == n_unknowns
    # Exogenous innovations set to 0 in the steady state (never a calibrated constant)
    zeroed_shocks: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class ValidationCheck(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class ValidationResult(BaseModel):
    """Deterministic checks executed against the derived module.

    ``validated`` is granted only to a COMPLETE module (generation verdict ``generated``).
    A module derived from a subset of the model's equations whose fragment nonetheless
    solves gets ``fragment_solves`` (a module from 1 of 18 equations must not read as
    'validated 5/5') — its steady state describes the fragment, not the model."""
    verdict: Literal["validated", "fragment_solves", "partial", "failed", "not_applicable"]
    checks: List[ValidationCheck] = Field(default_factory=list)
    steady_state: Dict[str, float] = Field(default_factory=dict)
    max_residual: Optional[float] = None
    notes: List[str] = Field(default_factory=list)


class ComparativeStatic(BaseModel):
    parameter: str
    delta_pct: float                                          # +10 / -10
    converged: bool = False
    variable_changes_pct: Dict[str, float] = Field(default_factory=dict)


class ExperimentResult(BaseModel):
    """Comparative statics around the calibrated parameter point."""
    verdict: Literal["completed", "partial", "not_applicable"]
    baseline: Dict[str, float] = Field(default_factory=dict)
    statics: List[ComparativeStatic] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
