# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Unit tests for guards.py + gmm.py (deterministic, no LLM, no network).

Run:
    PYTHONPATH=ModelTeam/ael  python ModelTeam/ael/calib_harness/tests/test_guards_gmm.py
"""

from __future__ import annotations

import math

from calib_harness.types import Moment, MomentResult, TargetRow
from calib_harness.guards import (
    is_tautological,
    partition_moments,
    provenance_ok,
)
from calib_harness.gmm import score

_PASS = 0
_FAIL = 0


def check(cond, msg):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  PASS  {msg}")
    else:
        _FAIL += 1
        print(f"  FAIL  {msg}")


def mr(key, role, computed, target, se, free, pins_taut=False):
    """Helper to build a MomentResult."""
    rel = None
    if computed is not None and target not in (None, 0):
        rel = abs(computed - target) / abs(target)
    return MomentResult(
        moment_key=key,
        computed=computed,
        target=target,
        std_error=se,
        rel_error=rel,
        free_params=list(free),
        role=role,
        tautological=pins_taut,
    )


# --------------------------------------------------------------------------------------
print("\n[1] is_tautological — structural >=2 free-param rule (Model-0 alpha role trap)")
# Ground against the REAL Model-0 artifact: alpha is the LABOR-SHARE parameter [0.3,0.7].
# labor_share = alpha is a 1-parameter identity -> tautological, can never over-identify.
labor_share = Moment(
    moment_key="labor_share",
    value_fn=lambda p: p["alpha"],            # labor share == alpha (Model 0 role)
    free_params={"alpha"},
    archetype="ces_production",
)
# A genuine 2-param moment (depends on alpha AND rho) IS scorable.
elasticity_ratio = Moment(
    moment_key="labor_ai_substitution",
    value_fn=lambda p: (1 - p["alpha"]) / p["alpha"] * p["rho"],
    free_params={"alpha", "rho"},
    archetype="ces_production",
)
const_moment = Moment(moment_key="const", value_fn=lambda p: 1.0, free_params=set())
check(is_tautological(labor_share) is True, "labor_share=alpha (1 free param) -> tautological")
check(is_tautological(const_moment) is True, "0 free params -> tautological")
check(is_tautological(elasticity_ratio) is False, "alpha&rho (2 free params) -> NOT tautological")

# --------------------------------------------------------------------------------------
print("\n[2] partition_moments — role tagging (pin / over_id / tautological / unmatched)")
moments = [
    Moment("k_over", lambda p: p["a"] + p["b"], {"a", "b"}),                 # over_id
    Moment("k_pin", lambda p: p["a"], {"a"}, pins="a"),                      # pin
    Moment("k_taut", lambda p: p["b"], {"b"}),                              # tautological (1 param, no pin)
    Moment("k_unm", lambda p: p["a"] * p["b"], {"a", "b"}),                  # unmatched (no target)
    Moment("k_err", lambda p: p["a"], {"a", "b"}),                          # compute_error (computed None)
]
computed = {"k_over": 2.4, "k_pin": 0.65, "k_taut": 0.35, "k_unm": 0.8, "k_err": None}
targets = {
    "k_over": TargetRow("k_over", 2.0, 0.2, "Cooley&Prescott 1995"),
    "k_pin": TargetRow("k_pin", 0.65, 0.02, "BLS labor share"),
    "k_taut": TargetRow("k_taut", 0.30, 0.05, "fake"),
    # k_unm intentionally absent -> unmatched
    "k_err": TargetRow("k_err", 1.0, 0.1, "fake"),
}
parts = partition_moments(moments, computed, targets)
roles = {p.moment_key: p.role for p in parts}
print("    roles:", roles)
check(roles["k_over"] == "over_id", "2-param + target + no-pin -> over_id")
check(roles["k_pin"] == "pin", "pins set + target -> pin")
check(roles["k_taut"] == "tautological", "1-param + target + no-pin -> tautological")
check(roles["k_unm"] == "unmatched", "no target row -> unmatched")
check(roles["k_err"] == "compute_error", "computed None -> compute_error")
over = next(p for p in parts if p.moment_key == "k_over")
check(abs(over.rel_error - 0.2) < 1e-9, f"k_over rel_error computed = {over.rel_error:.3f} (=0.2)")

# --------------------------------------------------------------------------------------
print("\n[3] provenance_ok — target literal must not leak into computed")
clean = mr("m", "over_id", computed=2.4, target=2.0, se=0.2, free={"a", "b"})
leak = mr("m", "over_id", computed=2.0, target=2.0, se=0.2, free={"a", "b"})   # computed==target
no_param = mr("m", "over_id", computed=2.0, target=2.05, se=0.2, free=set())   # not f(params)
no_comp = mr("m", "compute_error", computed=None, target=2.0, se=0.2, free={"a"})
check(provenance_ok(clean) is True, "computed!=target & depends on params -> ok")
check(provenance_ok(leak) is False, "computed==target (literal leaked) -> NOT ok")
check(provenance_ok(no_param) is False, "no free params (not f(theta)) -> NOT ok")
check(provenance_ok(no_comp) is False, "computed is None -> NOT ok")

# --------------------------------------------------------------------------------------
print("\n[4] gmm.score — perfect match -> fit ~ 1.0")
# Two over_id moments; their params a,b are pinned by two pin moments -> df = 2 - 0 = 2.
perfect = [
    mr("m1", "over_id", computed=2.0, target=2.0, se=0.1, free={"a", "b"}),
    mr("m2", "over_id", computed=0.5, target=0.5, se=0.1, free={"a", "b"}),
    mr("p_a", "pin", computed=0.65, target=0.65, se=0.02, free={"a"}),
    mr("p_b", "pin", computed=0.30, target=0.30, se=0.02, free={"b"}),
]
fit, j, df = score(perfect)
print(f"    fit={fit!r}  J={j!r}  df={df}")
check(df == 2, "df = #over_id(2) - #not_pinned(0) = 2")
check(j == 0.0, "perfect match -> J = 0")
check(fit is not None and abs(fit - 1.0) < 1e-12, f"perfect match -> fit ~ 1.0 (got {fit})")

# --------------------------------------------------------------------------------------
print("\n[5] gmm.score — 50% misfit -> strictly lower than perfect")
# Each over_id moment off by 0.5*se -> resid 0.5, sq 0.25; two moments -> J=0.5, df=2.
half = [
    mr("m1", "over_id", computed=2.0 + 0.5, target=2.0, se=1.0, free={"a", "b"}),
    mr("m2", "over_id", computed=0.5 + 0.5, target=0.5, se=1.0, free={"a", "b"}),
    mr("p_a", "pin", computed=0.65, target=0.65, se=0.02, free={"a"}),
    mr("p_b", "pin", computed=0.30, target=0.30, se=0.02, free={"b"}),
]
fit2, j2, df2 = score(half)
print(f"    fit={fit2:.4f}  J={j2:.4f}  df={df2}  (expected J=0.5, fit=exp(-0.25)={math.exp(-0.25):.4f})")
check(abs(j2 - 0.5) < 1e-9, "J = 2 * (0.5)^2 = 0.5")
check(fit2 is not None and fit2 < fit, f"50% misfit fit ({fit2:.4f}) < perfect fit ({fit:.4f})")
check(0.0 < fit2 < 1.0, "fit stays in (0,1)")
check(abs(fit2 - math.exp(-0.25)) < 1e-9, "fit == exp(-J/df) == exp(-0.25)")

# --------------------------------------------------------------------------------------
print("\n[6] gmm.score — df < 1 -> (None, None, df)  [untestable -> uncalibratable]")
under = [
    # single over_id moment with 2 UN-pinned params -> df = 1 - 2 = -1
    mr("m1", "over_id", computed=3.0, target=2.0, se=1.0, free={"a", "b"}),
]
fit3, j3, df3 = score(under)
print(f"    fit={fit3!r}  J={j3!r}  df={df3}")
check(fit3 is None and j3 is None, "df<1 -> fit_score=None and j_stat=None (honest N/A)")
check(df3 == -1, "df reported as -1 (1 over_id - 2 free unpinned)")

# --------------------------------------------------------------------------------------
print("\n[7] gmm.score — tautological & unmatched moments are EXCLUDED from J/df")
with_junk = perfect + [
    mr("taut", "tautological", computed=0.65, target=0.65, se=0.02, free={"a"}),
    mr("unm", "unmatched", computed=9.9, target=None, se=None, free={"a", "b"}),
]
fit4, j4, df4 = score(with_junk)
print(f"    fit={fit4!r}  J={j4!r}  df={df4}")
check((fit4, j4, df4) == (fit, j, df), "adding tautological+unmatched does NOT change (fit,J,df)")

# --------------------------------------------------------------------------------------
print("\n[8] integration — partition -> score (only over_id reaches GMM)")
# Build via partition so role assignment + scoring are exercised end-to-end.
i_moments = [
    Moment("k_over1", lambda p: p["a"] + p["b"], {"a", "b"}),
    Moment("k_over2", lambda p: p["a"] * p["b"], {"a", "b"}),
    Moment("k_pin_a", lambda p: p["a"], {"a"}, pins="a"),
    Moment("k_pin_b", lambda p: p["b"], {"b"}, pins="b"),
    Moment("labor_share", lambda p: p["alpha"], {"alpha"}),   # tautological (Model-0 trap)
]
# computed values are model-implied (NOT bit-identical to targets) -> good but honest fit
i_computed = {"k_over1": 1.02, "k_over2": 0.24, "k_pin_a": 0.5, "k_pin_b": 0.5,
              "labor_share": 0.65}
i_targets = {
    "k_over1": TargetRow("k_over1", 1.0, 0.5, "src"),
    "k_over2": TargetRow("k_over2", 0.25, 0.5, "src"),
    "k_pin_a": TargetRow("k_pin_a", 0.5, 0.05, "src"),
    "k_pin_b": TargetRow("k_pin_b", 0.5, 0.05, "src"),
    "labor_share": TargetRow("labor_share", 0.65, 0.02, "BLS"),
}
i_parts = partition_moments(i_moments, i_computed, i_targets)
i_roles = {p.moment_key: p.role for p in i_parts}
print("    roles:", i_roles)
check(i_roles["labor_share"] == "tautological", "labor_share=alpha tagged tautological (not scored)")
fit5, j5, df5 = score(i_parts)
print(f"    fit={fit5!r}  J={j5!r}  df={df5}")
# 2 over_id moments, params a,b both pinned -> df=2; near-perfect match -> fit close to 1.0
check(df5 == 2, "df = 2 over_id - 0 unpinned (a,b pinned) = 2")
check(fit5 is not None and fit5 > 0.99, f"good (non-leaked) over_id match -> fit ~ 1.0 (got {fit5:.4f})")
# provenance: all over_id moments are clean f(theta) and NOT copied from the target literal
check(all(provenance_ok(p) for p in i_parts if p.role == "over_id"),
      "all over_id moments pass provenance (computed != target leak guard)")

# --------------------------------------------------------------------------------------
print(f"\n==== RESULT: {_PASS} passed, {_FAIL} failed ====")
raise SystemExit(1 if _FAIL else 0)
