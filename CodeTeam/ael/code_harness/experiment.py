# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Comparative statics around the calibrated point (§4.5 BatchRunner's numeric core).

For each calibrated parameter, re-solve the steady state at ±delta and report the percent
change of every steady-state variable. Deterministic; a variant that fails to converge is
reported as such, never interpolated."""

from __future__ import annotations

from typing import Dict

from .types import ComparativeStatic, ExperimentResult, GenerationResult, ValidationResult
from .validate import _load_module


def comparative_statics(generation: GenerationResult, validation: ValidationResult,
                        workdir: str, delta: float = 0.10) -> ExperimentResult:
    if validation.verdict == "fragment_solves" or (validation.steady_state
                                                   and generation.verdict != "generated"):
        # Statics of a fragment (e.g. 1 of 18 equations) would
        # read as the model's comparative statics — skipped, never reported as model results
        return ExperimentResult(
            verdict="not_applicable",
            notes=[f"module is a fragment ({generation.n_parseable}/{generation.n_equations} "
                   "equations); comparative statics require the complete system"])
    if not validation.steady_state or not generation.square:
        return ExperimentResult(
            verdict="not_applicable",
            notes=["no validated steady state — comparative statics require one"])
    result = ExperimentResult(verdict="completed", baseline=dict(validation.steady_state))
    try:
        mod, _ = _load_module(generation.module_text, workdir)
    except Exception as e:                                     # pragma: no cover - defensive
        return ExperimentResult(verdict="not_applicable", notes=[f"module reload failed: {e}"])

    base_x0 = [validation.steady_state[v] for v in generation.variables]
    any_failed = False
    for pname, pval in generation.parameters.items():
        for sign in (+1.0, -1.0):
            new_val = pval * (1.0 + sign * delta) if pval else sign * delta
            cs = ComparativeStatic(parameter=pname, delta_pct=sign * delta * 100.0)
            try:
                sol = mod.solve_steady_state(x0=base_x0, params={pname: new_val})
                cs.converged = bool(sol.get("converged"))
                if cs.converged:
                    for var, base in result.baseline.items():
                        new = sol["solution"].get(var)
                        if new is not None and abs(base) > 1e-12:
                            cs.variable_changes_pct[var] = (new - base) / abs(base) * 100.0
                else:
                    any_failed = True
            except Exception as e:
                any_failed = True
                result.notes.append(f"{pname} {sign * delta:+.0%}: {str(e)[:120]}")
            result.statics.append(cs)
    if any_failed:
        result.verdict = "partial"
    return result
