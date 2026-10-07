# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Registry stand-ins must match a parameter's
economic ROLE, not a bare word or a Greek letter. In a live run mesa_wealth "calibrated" a
propensity SCORE and a choice-error scale (its description mentions "\\Lambda") as a savings
rate, and after that was closed var_growth would have fit an individual income-share AR(1) to
aggregate output-growth moments. Parameter texts below are copied from that run.
"""

import pytest

pytest.importorskip("numpy")

from ModelTeam.ael.calib_harness.sim_backends import match_registry, mesa_wealth, nk_taylor, var_growth


def _p(sym, name, desc=""):
    return {"parameter_symbol": sym, "parameter_name": name, "description": desc}


NEYMAN = {
    "model_title": "Formal Model: Neyman-Orthogonal High-Dimensional Macro-Financial Welfare "
                   "Estimation Framework",
    "model_summary": "welfare, wealth and inequality under policy interventions",
    "parameters": [
        _p("\\pi", "Propensity Score",
           "The probability of receiving the policy intervention given the high-dimensional "
           "covariate vector X, ensuring overlap between treated and control groups."),
        _p("\\lambda", "Regularization Parameter",
           "The penalty strength applied to the high-dimensional nuisance estimators."),
        _p("\\sigma^2", "Error Variance",
           "The variance of the idiosyncratic error term in the welfare outcome equation."),
    ],
}

JOINT = {
    "model_title": "Formal Model: Joint Structural Identification of Behavioral Biases and "
                   "Income Distribution Dynamics",
    "model_summary": "income distribution dynamics with present bias",
    "parameters": [
        _p("\\rho", "Income Share Persistence",
           "The autoregressive coefficient governing the persistence of individual income "
           "shares over time in the dynamic income process."),
        _p("\\sigma_u", "Income Shock Standard Deviation",
           "The standard deviation of the idiosyncratic error term in the dynamic income-share "
           "process, capturing transitory income volatility."),
        _p("\\sigma_\\epsilon", "Choice Error Std Dev",
           "Scale parameter of the error distribution \\Lambda (e.g., Logit or Probit) in the "
           "structural choice probability equation, capturing unobserved utility shocks."),
    ],
}


@pytest.mark.parametrize("model", [NEYMAN, JOINT], ids=["propensity_score", "income_share_ar1"])
def test_recorded_models_match_no_stand_in(model):
    assert mesa_wealth.build(model, []) is None
    assert var_growth.build(model, []) is None
    assert nk_taylor.build(model, []) is None
    assert match_registry(model, []) is None


def test_role_phrases_still_match():
    assert mesa_wealth._SAVINGS_RE.search("savings propensity")
    assert mesa_wealth._SAVINGS_RE.search("savings/retention rate")
    assert not mesa_wealth._SAVINGS_RE.search("Propensity Score")
    assert not mesa_wealth._SAVINGS_RE.search("error distribution \\Lambda")

    for s in ("phi_pi", "\\phi_\\pi", "\\phi_{\\pi}", "Taylor rule inflation response",
              "policy response to inflation"):
        assert nk_taylor._PHI_PI_RE.search(s), s
    assert not nk_taylor._PHI_PI_RE.search("policy response to output")

    assert var_growth._SHOCK_SCALE_RE.search("growth shock volatility")
    assert not var_growth._SHOCK_SCALE_RE.search("Choice Error Std Dev")
    assert not var_growth._SHOCK_SCALE_RE.search("cross-sectional dispersion of income")


def test_aggregate_growth_var_still_matches():
    bvar = {"model_title": "Regime-Switching Bayesian VAR", "parameters": [
        _p("rho_g", "growth persistence (AR(1)) coefficient"),
        _p("sigma_g", "growth shock volatility")]}
    spec = var_growth.build(bvar, [])
    assert spec is not None and spec[2] == ["rho_g", "sigma_g"]
