# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Calibration-harness fixes found by auditing v0.7.1 runs. Parameter texts are copied from those
runs."""

import pytest

pytest.importorskip("numpy")
pytest.importorskip("sympy")

from ModelTeam.ael.calib_harness import archetypes, harness, sim_calib
from ModelTeam.ael.calib_harness.parser import parse_equation
from ModelTeam.ael.calib_harness.sim_backends import match_registry


def _p(sym, name, desc="", rng=None):
    d = {"parameter_symbol": sym, "parameter_name": name, "description": desc}
    if rng:
        d["typical_range"] = rng
    return d


def test_stand_ins_read_the_parameter_name_not_its_description():
    rep04 = {"model_title": "Informality, monetary policy and the central bank's Taylor rule",
             "parameters": [_p("gamma", "Inflation Premium Weight",
                               "extent to which high inflation drives agents to the informal sector",
                               "[0.1, 0.8]")]}
    rep08 = {"model_title": "Wealth inequality with informal savings",
             "parameters": [_p("beta_1", "Marginal Propensity to Consume",
                               "fraction of additional wealth consumed rather than save or invest",
                               "[0.3, 0.8]")]}
    rep09 = {"model_title": "Wealth inequality and risk", "parameters": [
        _p("\\sigma", "Relative Risk Aversion",
           "affecting consumption and savings decisions", "[1.0, 5.0]")]}
    for m in (rep04, rep08, rep09):
        assert match_registry(m, []) is None


def test_stand_in_optimum_outside_declared_range_is_not_an_estimate():
    demo = {"model_title": "Behavioral NK model with a Taylor rule for monetary policy",
            "equations": [],
            "parameters": [_p("\\phi_{\\pi}", "Taylor Rule Inflation Response",
                              "coefficient on inflation deviations in the monetary policy rule",
                              "[1.2, 2.0]")]}
    out = sim_calib.try_simulation_calibrate(demo, [])
    assert out.calibration_status == "uncalibratable"
    assert out.uncalibratable_reason.startswith("target_outside_declared_range")


def test_htm_role_needs_the_share_itself():
    roles = archetypes.map_param_roles([
        _p("kappa", "Collateral Coefficient",
           "fraction of assets pledgeable by liquidity-constrained households", "[0.0, 0.5]"),
        _p("lambda", "Hand-to-mouth share", "share of hand-to-mouth households", "[0.2, 0.5]")])
    assert [r.role for r in roles] == [None, "hand_to_mouth_share"]


def test_matching_elasticity_is_not_a_labor_share():
    roles = archetypes.map_param_roles([
        _p("alpha", "Matching Elasticity",
           "elasticity of the labor market matching function with respect to unemployment",
           "[0.3, 0.7]")])
    assert roles[0].role is None


def test_model_period_from_wording_and_from_beta_range():
    assert archetypes.model_period({"model_summary": "annual model, one period is a year"}) == "annual"
    assert archetypes.model_period({"model_summary": "quarterly calibration"}) == "quarterly"
    roles = archetypes.map_param_roles([_p("beta", "Discount factor", "time preference", "[0.94, 0.97]")])
    assert archetypes.model_period({}, roles) == "annual"


def test_latex_symbols_keep_their_bounds():
    assert harness._param_bounds([_p("\\beta", "Discount factor", "", "[0.95, 0.99]")]) == \
        {"beta": (0.95, 0.99)}
    assert harness._calibrated_values([{"parameter_symbol": "\\beta", "calibrated_value": 0.98}]) == \
        {"beta": 0.98}


def test_bound_active_estimate_is_flagged():
    assert harness.at_bound({"beta": 0.99, "alpha": 0.33}, {"beta": (0.95, 0.99), "alpha": (0.25, 0.4)}) == ["beta"]


def test_superscript_label_versus_exponent():
    ok = parse_equation({"equation_plain": "y = A * L_pri,t^{1-alpha} + B",
                         "variables_used": ["y", "A", "L", "B"], "parameters_used": ["alpha"]})
    assert ok.status == "ok" and "alpha" in str(ok.expr)
    lab = parse_equation({"equation_plain": "c = theta^{risk} * w",
                          "variables_used": ["c", "w"], "parameters_used": ["theta"]})
    # an undeclared braced label is part of the symbol's identity: dropping it parsed
    # theta^{risk} as theta; with no declared theta_risk the equation refuses
    assert lab.status == "refused_superscript_label:risk"
