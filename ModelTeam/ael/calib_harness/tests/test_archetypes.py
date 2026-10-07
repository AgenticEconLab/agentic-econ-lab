# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Unit + grounding tests for calib_harness.archetypes (run with the calib venv, no pytest needed).

    PYTHONPATH=.../ModelTeam/ael <venv>/bin/python calib_harness/test_archetypes.py
"""
from __future__ import annotations

import json
import math
import os

from calib_harness.archetypes import (
    annualize,
    capital_output_ratio,
    detect_moments,
    investment_output_ratio,
    labor_share_from_capital,
    map_param_roles,
    real_interest_rate,
)
from calib_harness.parser import parse_equation

# Sections [2]-[3] ground the role-mapper on two real ModelTeam artifacts. Point
# AEL_CALIB_RUNS_ROOT at a run tree containing ModelTeam/<mode>/run_001/model_design_output.json;
# without it those sections are skipped.
_RUNS_ROOT = os.environ.get("AEL_CALIB_RUNS_ROOT", "<runs_root>")
NOWC = os.path.join(_RUNS_ROOT, "ModelTeam/ModeNoWcNoHITL/run_001/model_design_output.json")
WITHWC = os.path.join(_RUNS_ROOT, "ModelTeam/ModeWithWcWithHITL/run_001/model_design_output.json")

_P = 0  # pass counter
_F = 0  # fail counter


def check(label, cond, detail=""):
    global _P, _F
    if cond:
        _P += 1
        print(f"  PASS  {label}" + (f"   [{detail}]" if detail else ""))
    else:
        _F += 1
        print(f"  FAIL  {label}" + (f"   [{detail}]" if detail else ""))


def close(a, b, tol):
    return a is not None and abs(a - b) <= tol


def load_model(path, i):
    d = json.load(open(path))
    fm = d["formal_models"][i]
    return fm["parameters"], [parse_equation(e) for e in fm["equations"]], fm.get("model_title", "")


def roles_dict(roles):
    return {r.symbol: r.role for r in roles}


def moment_keys(ms):
    return sorted(m.moment_key for m in ms)


# =================================================================================================
print("=" * 90)
print("[1] CLOSED-FORM ARCHETYPES vs TEXTBOOK VALUES")
print("=" * 90)

ky_q = capital_output_ratio(0.33, 0.99, 0.025)
print(f"  K/Y quarterly (alpha=.33, beta=.99, delta=.025) = {ky_q:.4f}   (textbook 9.4014)")
check("K/Y quarterly == 9.4014", close(ky_q, 9.4014, 1e-3), f"{ky_q:.4f}")

ky_a = annualize("capital_output_ratio", ky_q)
print(f"  K/Y annualized (/4)                            = {ky_a:.4f}   (textbook ~2.35)")
check("K/Y annual ~= 2.3503", close(ky_a, 2.3503, 1e-3), f"{ky_a:.4f}")

iy_q = investment_output_ratio(0.33, 0.99, 0.025)
print(f"  I/Y = delta*K/Y                                = {iy_q:.5f}   (= .025*9.4014 = .23504)")
check("I/Y quarterly == 0.23504", close(iy_q, 0.025 * 9.4014, 1e-4), f"{iy_q:.5f}")
check("I/Y frequency-invariant", close(annualize("investment_output_ratio", iy_q), iy_q, 1e-12))

r_q = real_interest_rate(0.99)
print(f"  r* quarterly (beta=.99)                        = {r_q:.6f}   (textbook .0101)")
check("r* quarterly == 0.010101", close(r_q, 0.0101010, 1e-5), f"{r_q:.6f}")
r_a = annualize("real_interest_rate", r_q)
print(f"  r* annualized ((1+r)^4-1)                      = {r_a:.6f}   (~.0410)")
check("r* annual ~= 0.041012", close(r_a, (1.010101010101) ** 4 - 1.0, 1e-6), f"{r_a:.6f}")

ls = labor_share_from_capital(0.33)
print(f"  labor_share = 1 - alpha_capital(.33)           = {ls:.4f}   (textbook .67)")
check("labor_share(1-.33) == 0.67", close(ls, 0.67, 1e-9), f"{ls:.4f}")

if not (os.path.exists(NOWC) and os.path.exists(WITHWC)):
    print("\nSKIP [2]-[3]: real-artifact grounding needs AEL_CALIB_RUNS_ROOT (see top of file)")
    print(f"RESULT: {_P} passed, {_F} failed")
    raise SystemExit(1 if _F else 0)

# =================================================================================================
print("\n" + "=" * 90)
print("[2] PARAM ROLE-MAPPER -- CONFIRM (don't guess); the real-Model-0 'alpha is LABOR share' trap")
print("=" * 90)

# --- NoWc Model 0 (ACSP): alpha is LABOR share, NOT capital share ---------------------------------
p0, eq0, title0 = load_model(NOWC, 0)
r0 = map_param_roles(p0)
rd0 = roles_dict(r0)
print(f"  NoWc Model 0: {title0[:60]}")
for r in r0:
    print(f"     {r.symbol:<10} range={str(r.range):<14} -> role={r.role}")
check("NoWc M0 alpha -> labor_share (NOT capital_share)", rd0.get("alpha") == "labor_share", rd0.get("alpha"))
check("NoWc M0 rho   -> substitution_elasticity", rd0.get("rho") == "substitution_elasticity", rd0.get("rho"))
check("NoWc M0 delta -> None (learning-by-doing, hi=0.25)", rd0.get("delta") is None, rd0.get("delta"))
check("NoWc M0 delta_K -> None (knowledge-depr hi=0.15 > .05)", rd0.get("delta_K") is None, rd0.get("delta_K"))
check("NoWc M0 theta_t -> None (cost deflator, not Calvo)", rd0.get("theta_t") is None, rd0.get("theta_t"))
check("NoWc M0 has NO discount role (no beta param)", "discount" not in rd0.values())

# --- WithWc Model 2 (NK): the clean capital_share + discount + depreciation + calvo case ----------
p2w, eq2w, title2w = load_model(WITHWC, 2)
r2w = map_param_roles(p2w)
rd2w = roles_dict(r2w)
print(f"\n  WithWc Model 2: {title2w[:60]}")
for r in r2w:
    print(f"     {r.symbol:<10} range={str(r.range):<14} -> role={r.role}")
check("WithWc M2 alpha -> capital_share", rd2w.get("alpha") == "capital_share", rd2w.get("alpha"))
check("WithWc M2 beta  -> discount", rd2w.get("beta") == "discount", rd2w.get("beta"))
check("WithWc M2 delta -> depreciation (range [.02,.025])", rd2w.get("delta") == "depreciation", rd2w.get("delta"))
check("WithWc M2 theta -> calvo (price stickiness)", rd2w.get("theta") == "calvo", rd2w.get("theta"))
check("WithWc M2 sigma -> None (IES, not production subst)", rd2w.get("sigma") is None, rd2w.get("sigma"))

# --- WithWc Model 0 (ACRT): beta is 'Opacity Sensitivity' [0.5,2.0] -> NOT discount ---------------
p0w, eq0w, title0w = load_model(WITHWC, 0)
r0w = map_param_roles(p0w)
rd0w = roles_dict(r0w)
check("WithWc M0 alpha -> capital_share (AI capital elasticity)", rd0w.get("alpha") == "capital_share", rd0w.get("alpha"))
check("WithWc M0 beta  -> None (opacity sensitivity, range [.5,2])", rd0w.get("beta") is None, rd0w.get("beta"))

# =================================================================================================
print("\n" + "=" * 90)
print("[3] detect_moments ON THE 2 REAL MODELS (which moments fire)")
print("=" * 90)


def report(tag, parsed_eqs, roles):
    ms = detect_moments(parsed_eqs, roles)
    print(f"\n  {tag}: {len(ms)} moment(s)")
    for m in ms:
        kind = "OVER-ID" if len(m.free_params) >= 2 else f"pin({m.pins})"
        print(f"     {m.moment_key:<24} free={sorted(m.free_params)!s:<24} {kind:<12} [{m.archetype}]")
    return ms


# NoWc Model 0 -- alpha is labor share, no beta: the capital/labor-share & K/Y moments must NOT fire
m_nowc0 = report("NoWc Model 0 (ACSP)", eq0, r0)
keys0 = moment_keys(m_nowc0)
check("NoWc M0 fires labor_share (= alpha, the labor param)", "labor_share" in keys0, str(keys0))
check("NoWc M0 fires substitution_elasticity (CES rho)", "substitution_elasticity" in keys0, str(keys0))
check("NoWc M0 does NOT fire capital_output_ratio", "capital_output_ratio" not in keys0)
check("NoWc M0 does NOT fire real_interest_rate (no discount)", "real_interest_rate" not in keys0)
check("NoWc M0 has ZERO over-identifying moments", all(len(m.free_params) < 2 for m in m_nowc0))

# WithWc Model 2 -- the clean RBC/NK core: K/Y & I/Y over-id moments MUST fire
m_withwc2 = report("WithWc Model 2 (NK)", eq2w, r2w)
keys2 = moment_keys(m_withwc2)
check("WithWc M2 fires capital_output_ratio (over-id)", "capital_output_ratio" in keys2, str(keys2))
check("WithWc M2 fires investment_output_ratio (over-id)", "investment_output_ratio" in keys2, str(keys2))
check("WithWc M2 fires real_interest_rate (pins beta)", "real_interest_rate" in keys2, str(keys2))
over_id_2 = [m for m in m_withwc2 if len(m.free_params) >= 2]
check("WithWc M2 has >=1 over-identifying moment (=> calibratable)", len(over_id_2) >= 1, f"{len(over_id_2)} over-id")

# Demonstrate a real numeric K/Y from the over-id recipe (range midpoints used ONLY for display).
ky_moment = next(m for m in m_withwc2 if m.moment_key == "capital_output_ratio")
mids = {r.symbol: (r.range[0] + r.range[1]) / 2 for r in r2w if r.range}
ky_val_q = ky_moment.value_fn(mids)
print(f"\n  WithWc M2 K/Y recipe at range-midpoints alpha={mids['alpha']}, beta={mids['beta']}, "
      f"delta={mids['delta']}: K/Y_q={ky_val_q:.4f} -> annual={annualize('capital_output_ratio', ky_val_q):.4f}")
check("WithWc M2 K/Y recipe computes a finite positive value", math.isfinite(ky_val_q) and ky_val_q > 0, f"{ky_val_q:.4f}")
check("WithWc M2 K/Y free_params == {alpha,beta,delta}", ky_moment.free_params == {"alpha", "beta", "delta"}, str(sorted(ky_moment.free_params)))

# =================================================================================================
print("\n" + "=" * 90)
print(f"RESULT: {_P} passed, {_F} failed")
print("=" * 90)
raise SystemExit(1 if _F else 0)
