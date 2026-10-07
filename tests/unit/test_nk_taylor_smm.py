# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""The NK-Taylor runnable archetype — monetary-policy models become simulation-calibratable. std(yoy inflation) is monotone in the Taylor response phi_pi, SMM
recovers a known phi_pi, and a Taylor-rule model that the closed-form harness refuses now comes
back `archetype_calibrated` against the CITED US inflation-volatility target."""

import pytest

pytest.importorskip("scipy")
np = pytest.importorskip("numpy")

from ModelTeam.ael.calib_harness import sim_calib
from ModelTeam.ael.calib_harness.sim_backends import nk_taylor as nk
from ModelTeam.ael.calib_harness.types import TargetRow


def test_inflation_volatility_monotone_in_phi_pi():
    """Stronger Taylor response damps inflation — the identifying gradient."""
    v_weak = nk.simulate_moments({"phi_pi": 1.1}, seed=1)["inflation_volatility"]
    v_strong = nk.simulate_moments({"phi_pi": 3.0}, seed=1)["inflation_volatility"]
    assert 0.0 < v_strong < v_weak < 0.05
    assert (v_weak - v_strong) / v_weak > 0.15          # a clear, identifying gradient


def test_empirical_target_is_reachable():
    """The cited US target (0.012) must be interior to the simulable range, not a boundary fit."""
    lo = nk.simulate_moments({"phi_pi": 3.5}, seed=1)["inflation_volatility"]
    hi = nk.simulate_moments({"phi_pi": 1.05}, seed=1)["inflation_volatility"]
    assert lo < 0.012 < hi


def test_smm_recovers_known_phi_pi():
    phi_true = 2.0
    tgt = float(np.mean([nk.simulate_moments({"phi_pi": phi_true}, s)["inflation_volatility"]
                         for s in (1, 2, 3)]))
    tb = {"inflation_volatility": TargetRow("inflation_volatility", tgt, 0.001, "self-consistency")}
    opt, match = sim_calib.smm_estimate(
        nk.simulate_moments, ["inflation_volatility"], tb,
        free_params=["phi_pi"], x0=[1.3], bounds={"phi_pi": (1.05, 3.5)},
    )
    assert opt is not None
    assert abs(opt["phi_pi"] - phi_true) < 0.3, opt     # recovered (just-identified, MC noise)
    assert match is not None and match > 0.9


def test_archetype_matches_taylor_model_and_declines_others():
    taylor_model = {
        "model_title": "HANK model with Taylor rule monetary policy",
        "parameters": [
            {"parameter_symbol": "phi_pi", "parameter_name": "Taylor rule inflation response coefficient"},
            {"parameter_symbol": "beta", "parameter_name": "discount factor"},
        ],
    }
    spec = nk.build(taylor_model, [])
    assert spec is not None
    _, keys, free, x0, bounds = spec
    assert keys == ["inflation_volatility"] and free == ["phi_pi"]
    assert bounds["phi_pi"][0] > 1.0                     # Taylor principle: active policy only

    # no monetary-policy wording -> decline
    assert nk.build({"model_title": "Kinetic wealth exchange", "parameters": [
        {"parameter_symbol": "lam", "parameter_name": "savings propensity"}]}, []) is None
    # policy wording but no inflation-response parameter -> decline
    assert nk.build({"model_title": "Monetary policy under uncertainty", "parameters": [
        {"parameter_symbol": "beta", "parameter_name": "discount factor"}]}, []) is None


def test_harness_calibrates_refused_taylor_model_end_to_end():
    """A Taylor-rule model with no closed-form moments now gets a genuine simulation fit against
    the cited target book (previously `partially_calibrated` on a single Poor moment)."""
    pytest.importorskip("sympy")
    from ModelTeam.ael.calib_harness import run as harness_run
    model = {
        "model_title": "DRL-augmented monetary policy: central bank Taylor rule at the ZLB",
        "model_summary": "Monetary policy with an inflation-targeting central bank.",
        "parameters": [
            {"parameter_symbol": "phi_pi", "parameter_name": "policy response to inflation"},
        ],
        "variables": [{"variable_name": "inflation"}],
        "equations": [],                                  # nothing closed-form -> SMM fallback
    }
    out = harness_run(model, [])
    # the NK-Taylor backend simulates its own three equations, not the model's: archetype verdict
    assert out.calibration_status == "archetype_calibrated"
    assert out.simulator == "nk_taylor" and out.simulator_is_archetype
    assert "phi_pi" in out.estimated_params
    assert 1.05 <= out.estimated_params["phi_pi"] <= 3.5
    assert out.fit_score is not None and out.fit_score > 0.5