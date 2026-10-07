# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""A real simulator (Mesa) drives the SMM path.

The runnable ABM-archetype library (calib_harness/sim_backends) is the SMM counterpart of the
closed-form archetypes. Its first archetype is the kinetic income/wealth-exchange model (Mesa 3),
whose savings propensity drives the income-Gini moment. These tests verify: the model simulates a
sensible Gini, SMM recovers a known savings parameter, the archetype matches conservatively, and an
inequality model the closed-form harness refuses is now calibrated end-to-end via simulation."""

import pytest

pytest.importorskip("mesa")
pytest.importorskip("scipy")
np = pytest.importorskip("numpy")

from ModelTeam.ael.calib_harness import sim_calib
from ModelTeam.ael.calib_harness.sim_backends import mesa_wealth as mw
from ModelTeam.ael.calib_harness.types import TargetRow


def test_gini_is_monotone_in_savings():
    """Higher savings propensity -> lower income inequality (the identifying relationship)."""
    g_low = mw.simulate_moments({"savings": 0.1}, seed=1)["income_gini"]
    g_high = mw.simulate_moments({"savings": 0.8}, seed=1)["income_gini"]
    assert 0.0 < g_high < g_low < 0.6
    assert g_low - g_high > 0.1          # a clear, identifying gradient


def test_smm_recovers_known_savings_parameter():
    """Self-consistency: make a known lambda's Gini the target; SMM must recover that lambda."""
    lam_true = 0.35
    tgt = np.mean([mw.simulate_moments({"savings": lam_true}, s)["income_gini"] for s in (1, 2, 3)])
    tb = {"income_gini": TargetRow("income_gini", float(tgt), 0.01, "self-consistency")}
    opt, match = sim_calib.smm_estimate(
        mw.simulate_moments, ["income_gini"], tb,
        free_params=["savings"], x0=[0.6], bounds={"savings": (0.05, 0.95)},
    )
    assert opt is not None
    assert abs(opt["savings"] - lam_true) < 0.08, opt      # recovered, not fabricated
    assert match is not None and match > 0.9


def test_archetype_matches_inequality_model_with_savings_param():
    model = {
        "model_title": "Kinetic income inequality exchange model",
        "parameters": [{"parameter_symbol": "lambda", "parameter_name": "savings propensity"}],
        "variables": [{"variable_name": "household income"}],
    }
    spec = mw.build(model, [])
    assert spec is not None
    sim, keys, free, x0, bounds = spec
    assert keys == ["income_gini"] and free == ["lambda"]


def test_archetype_declines_unrelated_or_paramless_models():
    # unrelated model -> no match
    assert mw.build({"model_title": "New Keynesian DSGE", "parameters": [
        {"parameter_symbol": "beta", "parameter_name": "discount factor"}]}, []) is None
    # inequality wording but NO savings-type parameter -> no match (can't identify the free param)
    assert mw.build({"model_title": "Wealth inequality accounting identity",
                     "parameters": [{"parameter_symbol": "r", "parameter_name": "return on capital"}]}, []) is None


def test_try_sim_calibrate_calibrates_inequality_model_via_mesa():
    """End to end through the registry + the real cited income_gini target."""
    model = {
        "model_title": "Agent-based income inequality via kinetic wealth exchange",
        "parameters": [{"parameter_symbol": "lambda", "parameter_name": "savings propensity"}],
        "variables": [{"variable_name": "agent wealth"}],
        "equations": [],
    }
    out = sim_calib.try_simulation_calibrate(model, [])
    assert out.calibration_status == "archetype_calibrated"
    assert out.simulator == "mesa_wealth"
    assert "lambda" in out.estimated_params
    assert out.fit_score is not None and out.fit_score > 0.5


def test_harness_run_routes_refused_inequality_model_to_mesa_smm():
    pytest.importorskip("sympy")
    from ModelTeam.ael.calib_harness import run as harness_run
    model = {
        "model_title": "Kinetic wealth-exchange inequality model",
        "parameters": [{"parameter_symbol": "lam", "parameter_name": "savings/retention rate"}],
        "variables": [{"variable_name": "household wealth share"}],
        "equations": [],                    # closed-form path -> uncalibratable -> SMM fallback
    }
    out = harness_run(model, [])
    assert out.calibration_status == "archetype_calibrated"
    assert out.coverage.get("method") == "SMM"
