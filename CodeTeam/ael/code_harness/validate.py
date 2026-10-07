# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Deterministic validation of the derived module.

The module contains only sympy-derived expressions in a fixed template (see generate.py), so
in-process execution is bounded; each check is executed and reported honestly. A failing
derived module is a REFUSAL to certify, not a debug loop — there is no LLM code to 'fix'."""

from __future__ import annotations

import importlib.util
import os
import uuid
from typing import Dict, Optional

import numpy as np

from .types import GenerationResult, ValidationCheck, ValidationResult


def _load_module(module_text: str, workdir: str):
    # The throwaway execution copy must not persist beside the canonical model_N.py
    # files (plus a __pycache__), which would double every module in the published
    # output. Execute from a scratch subdir; the caller removes it.
    scratch = os.path.join(workdir, "_exec_scratch")
    os.makedirs(scratch, exist_ok=True)
    path = os.path.join(scratch, f"generated_model_{uuid.uuid4().hex[:8]}.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(module_text)
    spec = importlib.util.spec_from_file_location(f"ael_generated_{uuid.uuid4().hex[:8]}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path


def cleanup_workdir(workdir: str) -> None:
    """Remove execution scratch + bytecode caches; canonical model files stay."""
    import shutil
    for sub in ("_exec_scratch", "__pycache__"):
        shutil.rmtree(os.path.join(workdir, sub), ignore_errors=True)


def validate_module(generation: GenerationResult, workdir: str,
                    x0: Optional[Dict[str, float]] = None) -> ValidationResult:
    """Execute the check battery against a generated/partial module."""
    if generation.verdict == "ungenerable" or not generation.module_text:
        return ValidationResult(verdict="not_applicable",
                                notes=[f"nothing to validate: {generation.reason}"])
    result = ValidationResult(verdict="failed")
    checks = result.checks

    try:
        mod, path = _load_module(generation.module_text, workdir)
        checks.append(ValidationCheck(name="module_compiles", passed=True,
                                      detail=os.path.basename(path)))
        result.notes.append(f"module file: {path}")
    except Exception as e:
        checks.append(ValidationCheck(name="module_compiles", passed=False, detail=str(e)[:200]))
        return result

    if generation.unset_parameters:
        checks.append(ValidationCheck(
            name="parameters_complete", passed=False,
            detail=f"unset parameters: {generation.unset_parameters} — steady-state checks "
                   "require values (honest refusal, not a default guess)"))
        result.verdict = "partial"
        return result
    checks.append(ValidationCheck(name="parameters_complete", passed=True))

    try:
        start = [x0.get(v, 0.5) if x0 else 0.5 for v in generation.variables]
        res = mod.residuals(start)
        finite = bool(np.all(np.isfinite(res)))
        checks.append(ValidationCheck(name="residuals_finite_at_start", passed=finite,
                                      detail=f"max |r| = {float(np.max(np.abs(res))):.4g}"))
    except Exception as e:
        checks.append(ValidationCheck(name="residuals_finite_at_start", passed=False,
                                      detail=str(e)[:200]))
        return result

    if not generation.square:
        checks.append(ValidationCheck(
            name="steady_state_solves", passed=False,
            detail="non-square system — solver honestly declines"))
        result.verdict = "partial"
        return result

    try:
        sol = mod.solve_steady_state(x0=start)
        converged = bool(sol.get("converged"))
        max_res = sol.get("max_residual")
        ok = converged and max_res is not None and max_res < 1e-6
        checks.append(ValidationCheck(
            name="steady_state_solves", passed=ok,
            detail=f"converged={converged}, max_residual={max_res}"))
        if ok:
            result.steady_state = sol.get("solution") or {}
            result.max_residual = float(max_res)
    except Exception as e:
        checks.append(ValidationCheck(name="steady_state_solves", passed=False,
                                      detail=str(e)[:200]))
        return result

    # non-degeneracy: perturbing a parameter must move the residuals at the solution
    if result.steady_state and generation.parameters:
        pname, pval = next(iter(generation.parameters.items()))
        try:
            at = [result.steady_state[v] for v in generation.variables]
            base = mod.residuals(at)
            pert = mod.residuals(at, params={pname: pval * 1.1 if pval else 0.1})
            moved = bool(np.max(np.abs(pert - base)) > 1e-12)
            checks.append(ValidationCheck(
                name="parameter_sensitivity", passed=moved,
                detail=f"perturbing {pname} moves residuals: {moved}"))
        except Exception as e:
            checks.append(ValidationCheck(name="parameter_sensitivity", passed=False,
                                          detail=str(e)[:200]))

    # A module built from 1 of 18 equations can pass every executed check and would be
    # reported 'validated 5/5'. Completeness is a check of its own: only a module
    # derived from the whole equation system (generation verdict 'generated') validates.
    complete = generation.verdict == "generated"
    checks.append(ValidationCheck(
        name="system_complete", passed=complete,
        detail=(f"{generation.n_parseable}/{generation.n_equations} equations in the module"
                + ("" if complete else f"; {generation.reason or 'partial generation'}"))))
    failed = [c for c in checks if not c.passed]
    if not failed:
        result.verdict = "validated"
    elif [c.name for c in failed] == ["system_complete"]:
        result.verdict = "fragment_solves"
        result.notes.append("the solved steady state is that of an equation-system fragment, "
                            "not of the model")
    else:
        result.verdict = "partial" if result.steady_state else "failed"
    return result
