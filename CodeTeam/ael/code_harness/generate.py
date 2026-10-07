# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Derive a runnable Python module from the model's parsed sympy system.

Reuses ``calib_harness.parser`` (the same trusted path the calibration verdicts run on):
each parseable equation ``lhs = rhs`` becomes the residual ``lhs - rhs``; residuals are
printed with ``sympy.pycode`` into a fixed template exposing ``residuals()`` and
``solve_steady_state()`` (scipy fsolve, square systems only). Python-keyword symbols
(``lambda``) are sanitized with a recorded SYMBOL_MAP."""

from __future__ import annotations

import keyword
import re as _re
from typing import Dict, List, Optional

import sympy as sp
from sympy.printing.pycode import pycode

from ModelTeam.ael.calib_harness.parser import canonical_name, parse_equation
from .types import GenerationResult, RefusedEquation

_HEADER = '''"""AUTO-GENERATED economic model module (AEL CodeTeam).

Derived deterministically from the model's parsed equation system via sympy.pycode —
no LLM-written code. Residual i corresponds to equation `lhs = rhs` as `lhs - rhs`.
"""
import math

import numpy as np
from scipy.optimize import fsolve

'''

_SOLVER = '''

def residuals(values, params=None):
    """Residual vector at `values` (order: VARIABLES). `params` overrides PARAMETERS.

    All values are np.float64 so a negative base under a fractional exponent yields NaN
    (real) instead of a Python complex — fsolve then backs away from the invalid region."""
    p = dict(PARAMETERS)
    p.update(params or {})
    missing = [k for k in UNSET_PARAMETERS if k not in p]
    if missing:
        raise ValueError(f"unset parameters need values: {missing}")
    env = {k: np.float64(v) for k, v in p.items()}
    env.update({name: np.float64(v) for name, v in zip(VARIABLES, values)})
    with np.errstate(all="ignore"):
        out = [_f(env) for _f in _RESIDUAL_FNS]
    return np.array([np.float64(np.real(r)) if np.isfinite(np.real(r)) and abs(np.imag(r)) < 1e-300
                     else np.float64("nan") if not np.isreal(r) else np.float64(r)
                     for r in out], dtype=np.float64)


_STARTS = (0.5, 1.0, 2.0, 0.1)   # deterministic multi-start ladder


def solve_steady_state(x0=None, params=None):
    """Solve residuals == 0 (square systems). Returns a dict with an honest verdict."""
    if len(_RESIDUAL_FNS) != len(VARIABLES):
        return {"converged": False,
                "reason": f"non-square system: {len(_RESIDUAL_FNS)} equations, "
                          f"{len(VARIABLES)} unknowns"}
    starts = [np.asarray(x0, dtype=float)] if x0 is not None else []
    starts += [np.full(len(VARIABLES), s, dtype=float) for s in _STARTS]
    best = None
    with np.errstate(all="ignore"):
        for start in starts:
            sol, _info, ier, msg = fsolve(lambda v: residuals(v, params), start,
                                          full_output=True)
            res = residuals(sol, params)
            ok = bool(ier == 1) and bool(np.all(np.isfinite(res)))
            max_res = float(np.max(np.abs(res))) if len(res) else 0.0
            cand = {"converged": ok,
                    "solution": {name: float(v) for name, v in zip(VARIABLES, sol)},
                    "max_residual": max_res, "message": str(msg)}
            if ok and max_res < 1e-8:
                return cand
            if best is None or (ok and not best["converged"]):
                best = cand
    return best


if __name__ == "__main__":
    _r = solve_steady_state()
    print(f"MODEL: {MODEL_TITLE}")
    if _r.get("converged"):
        print(f"steady state (max residual {_r['max_residual']:.2e}):")
        for _name, _val in sorted(_r["solution"].items()):
            print(f"  {_name} = {_val:.6g}")
    else:
        print(f"no certified steady state: {_r.get('reason') or _r.get('message')}")
'''


def _safe_name(name: str) -> str:
    out = name if name.isidentifier() else "sym_" + "".join(
        c if c.isalnum() else "_" for c in name)
    if keyword.iskeyword(out):
        out += "_"
    return out


def _calibrated_values(calibration: Optional[Dict], model_title: str) -> Dict[str, float]:
    """Scalar calibrated parameter values for (preferably) the matching model."""
    values: Dict[str, float] = {}
    models = (calibration or {}).get("calibrated_models") or []
    ranked = sorted(
        (m for m in models if isinstance(m, dict)),
        key=lambda m: 0 if model_title and model_title[:40] in str(m.get("model_title", "")
                                                                   ) + str(m.get("based_on_model", "")) else 1)
    for cm in ranked:
        for p in cm.get("calibrated_parameters") or []:
            if not isinstance(p, dict):
                continue
            sym = canonical_name(str(p.get("parameter_symbol")
                                     or p.get("parameter_name") or ""))
            val = p.get("calibrated_value")
            if sym and isinstance(val, (int, float)) and sym not in values:
                values[sym] = float(val)
        if values:
            break
    return values


# A declared variable whose name/description says it is an exogenous innovation
_SHOCK_WORDS = _re.compile(r"\b(shocks?|innovations?|disturbances?|white[- ]noise|error terms?|noise)\b",
                           _re.IGNORECASE)
# an UNDECLARED time-indexed epsilon is an innovation by notation (e.g. eps_{A,t})
_SHOCK_BASES = {"eps", "epsilon", "varepsilon"}


def _raw_key(name) -> str:
    return _re.sub(r"[\\{}\s$]", "", str(name or ""))


def _as_list(value) -> List:
    if isinstance(value, str):
        try:
            value = eval(value)  # the artifacts store these as a stringified list
        except Exception:
            return []
    return list(value or [])


def _declared_variables(model: Dict) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for v in model.get("variables") or []:
        if isinstance(v, dict):
            c = canonical_name(str(v.get("variable_symbol") or ""))
            if c:
                out.setdefault(c, v)
    return out


def _steady_state_shocks(parsed_ok, model_vars: Dict[str, Dict], params: set) -> List[str]:
    """Exogenous innovations that are zero in the deterministic steady state.

    A symbol is an innovation when the model declares it as a variable NAMED a
    shock/innovation/disturbance/noise, or when it is an UNDECLARED, time-indexed epsilon
    (eps_{A,t} in an AR(1)). A symbol with its own law of motion (it is the LHS of a parsed
    equation, e.g. nu_t = rho*nu_{t-1} + eps_t) is solved by the system, not set to zero;
    a declared parameter is never a shock."""
    lhs = set()
    for p in parsed_ok:
        lhs_expr = getattr(p, "lhs_expr", None)
        if lhs_expr is None:
            lhs.add(p.lhs_symbol)
        elif len(lhs_expr.free_symbols) == 1:        # ln(y_t) = ... is y's law of motion
            lhs.add(str(next(iter(lhs_expr.free_symbols))))
    used: set = set()
    timed: set = set()
    for p in parsed_ok:
        used |= set(p.free_symbols or [])
        timed |= set(getattr(p, "time_indexed", None) or [])
    shocks = []
    for name in sorted(used - lhs - params):
        decl = model_vars.get(name)
        if decl is not None:
            # the NAME, not the description: 'Individual Labor Income' is described as
            # "subject to idiosyncratic shocks" and is not itself an innovation
            if _SHOCK_WORDS.search(str(decl.get("variable_name") or "")):
                shocks.append(name)
        elif name in timed and name.split("_")[0] in _SHOCK_BASES:
            shocks.append(name)
    return shocks


def generate_module(model: Dict, calibration: Optional[Dict] = None) -> GenerationResult:
    """Derive the module for ONE formal model (a `formal_models` entry)."""
    title = str(model.get("model_title", ""))
    equations = [e for e in model.get("equations") or [] if isinstance(e, dict)]
    result = GenerationResult(verdict="ungenerable", model_title=title,
                              n_equations=len(equations))
    if not equations:
        result.reason = "no_equations"
        return result

    parsed = [parse_equation(eq) for eq in equations]
    model_params: Dict[str, set] = {}
    for prm in model.get("parameters") or []:
        if isinstance(prm, dict):
            raw = prm.get("parameter_symbol") or prm.get("parameter_name") or ""
            c = canonical_name(str(raw))
            if c:
                model_params.setdefault(c, set()).add(_raw_key(raw))
    # An equation VARIABLE that maps onto a model PARAMETER's name under a different
    # declared symbol (variable sigma_{i,t} vs parameter sigma) would be fed the parameter's
    # calibrated value — refuse the equation instead of merging two symbols
    for p, eq in zip(parsed, equations):
        if p.status != "ok":
            continue
        for v in _as_list(eq.get("variables_used")):
            c = canonical_name(str(v))
            if (c in model_params and c not in (p.declared_parameters or [])
                    and _raw_key(v) not in model_params[c]):
                p.status = f"refused_symbol_collision:{c}"
                break
    ok = [p for p in parsed if p.status == "ok" and p.lhs_symbol and p.expr is not None]
    result.refused = [RefusedEquation(equation_id=p.equation_id, status=p.status,
                                      raw=p.raw[:120]) for p in parsed if p.status != "ok"]
    result.n_parseable = len(ok)
    if not ok:
        statuses = sorted({p.status for p in parsed})
        result.reason = f"no_parseable_equations ({', '.join(statuses)})"
        return result

    # parameters: the model's list plus each parsed equation's own parameters_used
    # (rho_A / sigma_A declared only on the equation are parameters, not unknowns)
    declared_params = set(model_params)
    for p in ok:
        declared_params |= set(p.declared_parameters or [])
    values = _calibrated_values(calibration, title)
    shocks = _steady_state_shocks(ok, _declared_variables(model), declared_params)
    shock_subs = {sp.Symbol(s): 0 for s in shocks}
    if shocks:
        result.zeroed_shocks = shocks
        result.notes.append("exogenous innovations set to 0 in the steady state: "
                            + ", ".join(shocks))

    # Per-residual provenance (equation id, name, plain form) for the emitted comments
    eq_meta = {str(e.get("equation_id", "")): (str(e.get("equation_name") or ""),
                                               str(e.get("equation_plain") or ""))
               for e in equations}

    residual_exprs: List[sp.Expr] = []
    provenance: List[str] = []
    all_symbols: set = set()
    for p in ok:
        # A compound LHS is an expression — 'P*c + a = ...' becomes
        # residual (P*c + a) - rhs with P, c, a as the unknowns, never a pseudo-symbol.
        lhs = p.lhs_expr if getattr(p, "lhs_expr", None) is not None else sp.Symbol(p.lhs_symbol)
        expr = lhs - p.expr
        if shock_subs:
            expr = expr.subs(shock_subs)
            if expr == 0:
                result.notes.append(f"{p.equation_id}: identically zero once innovations are "
                                    "zero; not a steady-state restriction, dropped")
                continue
        residual_exprs.append(expr)
        provenance.append(p.equation_id)
        all_symbols |= {str(s) for s in expr.free_symbols}

    variables = sorted(s for s in all_symbols if s not in declared_params)
    used_params = sorted(s for s in all_symbols if s in declared_params)
    if not variables:
        result.reason = "no_unknowns (every symbol is a declared parameter)"
        return result

    # sanitize python-keyword / non-identifier symbols for the emitted code
    rename = {s: _safe_name(s) for s in sorted(all_symbols) if _safe_name(s) != s}
    sub_map = {sp.Symbol(old): sp.Symbol(new) for old, new in rename.items()}
    emitted = []                     # (code, equation_id) pairs — kept aligned on failure
    for expr, eid in zip(residual_exprs, provenance):
        try:
            emitted.append((pycode(expr.subs(sub_map)), eid))
        except Exception as e:                                  # pragma: no cover - defensive
            result.notes.append(f"pycode failed for one residual ({e}); equation dropped")
    if not emitted:
        result.reason = "pycode_failed_for_all_residuals"
        return result

    var_names = [rename.get(v, v) for v in variables]
    parameters = {rename.get(k, k): v for k, v in values.items() if k in used_params}
    unset = [rename.get(s, s) for s in used_params if s not in values]

    # emit residual functions reading from the env dict explicitly, each carrying its
    # equation provenance (id, name, plain form) so the module reads as a model, not a blob
    fn_lines = []
    for i, (code, eid) in enumerate(emitted):
        expr_code = code.replace("math.", "np.")
        for name in sorted(all_symbols, key=len, reverse=True):
            safe = rename.get(name, name)
            expr_code = _sub_token(expr_code, safe, f'env["{safe}"]')
        eq_name, eq_plain = eq_meta.get(eid, ("", ""))
        header = f"# {eid}" + (f" — {eq_name}" if eq_name else "")
        if eq_plain:
            header += f"\n#   {eq_plain[:100]}"
        fn_lines.append(f"{header}\ndef _residual_{i}(env):\n    return {expr_code}\n")
    fn_lines.append("_RESIDUAL_FNS = [" + ", ".join(f"_residual_{i}"
                                                    for i in range(len(emitted))) + "]\n")

    module_text = (_HEADER
                   + f"MODEL_TITLE = {title!r}\n"
                   + f"SYMBOL_MAP = {rename!r}  # original -> emitted name\n"
                   + f"PARAMETERS = {parameters!r}\n"
                   + f"UNSET_PARAMETERS = {unset!r}\n"
                   + f"VARIABLES = {var_names!r}\n\n"
                   + "\n".join(fn_lines)
                   + _SOLVER)

    result.module_text = module_text
    result.variables = var_names
    result.parameters = parameters
    result.unset_parameters = unset
    result.square = len(emitted) == len(var_names)
    if result.refused or not result.square or unset:
        result.verdict = "partial"
        why = []
        if result.refused:
            why.append(f"{len(result.refused)} equation(s) refused by the parser")
        if not result.square:
            why.append(f"non-square ({len(emitted)} eqs, {len(var_names)} unknowns)")
        if unset:
            why.append(f"{len(unset)} parameter(s) without calibrated values")
        result.reason = "; ".join(why)
    else:
        result.verdict = "generated"
    return result


def _sub_token(code: str, token: str, replacement: str) -> str:
    # the dot in the lookbehind protects attribute access: a variable named 'e' must not
    # rewrite sympy's Euler constant 'np.e' into 'np.env["e"]'
    return _re.sub(rf"(?<![\w\".]){_re.escape(token)}(?![\w\"])", replacement, code)
