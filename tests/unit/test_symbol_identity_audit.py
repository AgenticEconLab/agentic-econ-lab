# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Symbol identity in the equation parser shared by calibration and CodeTeam.

Stripping every subscript merged distinct symbols in 21/21 runs — the demo Taylor rule's
phi_pi and phi_y became one `phi`, one run's `ln A_t = rho_A ln A_{t-1} + sigma_A eps_{A,t}` used
the discount rate, the CRRA and an emissions parameter, and `dW`, `d(theta)/dt`, `Delta p`,
`pi_t^f` were parsed as products or powers."""

import pytest

from CodeTeam.ael.code_harness import generate_module, validate_module
from ModelTeam.ael.calib_harness.parser import canonical_name, parse_equation


def _eq(eid, plain, variables, parameters):
    return {"equation_id": eid, "equation_plain": plain,
            "variables_used": variables, "parameters_used": parameters}


def _parse(plain, variables, parameters):
    return parse_equation(_eq("E", plain, variables, parameters))


class TestCanonicalNames:
    @pytest.mark.parametrize("raw,name", [
        ("\\phi_{\\pi}", "phi_pi"), ("\\phi_{y}", "phi_y"), ("\\hat{y}_t", "y_hat"),
        ("y_hat_t", "y_hat"), ("\\epsilon_{t}^{mp}", "epsilon_mp"), ("rho_A", "rho_A"),
        ("\\pi_t^{public}", "pi_public"), ("r^*", "r_star"), ("c_{i,t+1}", "c"),
        ("K_i,t-1", "K"), ("\\sigma_{\\pi,ij,t}", "sigma_pi"), ("\\sigma_{\\beta}^2", "sigma_beta"),
        ("\\gamma_0", "gamma_0"), ("tau_{c,t}", "tau_c"), ("\\Delta K_i", "Delta_K"),
    ])
    def test_index_subscripts_drop_identifying_ones_stay(self, raw, name):
        assert canonical_name(raw) == name


class TestDistinctSymbols:
    def test_taylor_rule_keeps_phi_pi_phi_y_and_y_hat(self):
        p = _parse("r_t = rho_r * r_t-1 + (1 - rho_r) * ( phi_pi * pi_t + phi_y * y_hat_t ) "
                   "+ epsilon_t^mp",
                   ["r_t", "r_{t-1}", "\\pi_t", "\\hat{y}_t", "\\epsilon_{t}^{mp}"],
                   ["\\rho_r", "\\phi_{\\pi}", "\\phi_{y}"])
        assert p.status == "ok", p.status
        assert {"phi_pi", "phi_y", "y_hat", "rho_r", "epsilon_mp"} <= set(p.free_symbols)
        assert "phi" not in p.free_symbols and "y" not in p.free_symbols

    def test_rep10_productivity_ar1_uses_its_own_parameters(self):
        p = _parse("ln(A_t) = rho_A * ln(A_{t-1}) + sigma_A * epsilon_{A,t}",
                   ["A_t", "A_{t-1}"], ["rho_A", "sigma_A"])
        assert p.status == "ok", p.status
        assert set(p.free_symbols) == {"A", "rho_A", "sigma_A", "epsilon_A"}
        assert "epsilon_A" in p.time_indexed

    def test_numeric_subscripts_are_distinct_coefficients(self):
        p = _parse("mu_i = gamma_0 + gamma_1 * delta_i + gamma_2 * c_i",
                   ["mu_i", "delta_i", "c_i"], ["gamma_0", "gamma_1", "gamma_2"])
        assert p.status == "ok"
        assert {"gamma_0", "gamma_1", "gamma_2"} <= set(p.free_symbols)

    def test_declared_superscript_label_is_part_of_the_name(self):
        p = _parse("y_hat_t = ln(Y_t) - ln(Y_t^natural)",
                   ["\\hat{y}_t", "Y_t", "Y_t^{natural}"], [])
        assert p.status == "ok"
        assert p.lhs_symbol == "y_hat" and set(p.free_symbols) == {"Y", "Y_natural"}

    def test_declared_subscripted_exponent(self):
        p = _parse("Y_t = A_t * K_t^alpha_prod", ["Y_t", "A_t", "K_t"], ["alpha_{prod}"])
        assert p.status == "ok", p.status
        assert p.lhs_symbol == "Y" and set(p.free_symbols) == {"A", "K", "alpha_prod"}

    def test_parameter_and_variable_mapping_to_one_name_refuse(self):
        p = _parse("z = sigma * sigma_i,t", ["z", "\\sigma_{i,t}"], ["\\sigma"])
        assert p.status == "refused_symbol_collision:sigma"

    def test_undeclared_braced_label_next_to_the_bare_symbol_refuses(self):
        p = _parse("x = pi_t^{f} + pi_t", ["x", "pi_t"], [])
        assert p.status == "refused_superscript_label:f"

    @pytest.mark.parametrize("plain", [
        "y_t = x_t^{home} - x_t^{foreign}",        # parsed as y = 0 when labels were dropped
        "y_t = x_t^{home} + x_t^{foreign}",        # parsed as y = 2*x
        "y_t = x_t^{home} * z_t",
        "y_t = (z_t + x_t)^{home}",                 # label on a group: not an exponent
    ])
    def test_two_undeclared_braced_labels_never_merge(self, plain):
        """Dropping undeclared braced labels made two DIFFERENT labelled
        symbols one; the collision check only saw a dropped label next to the bare symbol."""
        p = _parse(plain, ["y_t", "x_t", "z_t"], [])
        assert p.status.startswith("refused_superscript_label:"), (plain, p.status, p.expr)

    def test_declared_labelled_composites_keep_their_identity(self):
        p = _parse("y_t = x_t^{home} - x_t^{foreign}", ["y_t", "x_t^{home}", "x_t^{foreign}"], [])
        assert p.status == "ok"
        assert set(p.free_symbols) == {"x_home", "x_foreign"}
        p = _parse("y_t = x_t^home - x_foreign_t", ["y_t", "x_home", "x_t^{foreign}"], [])
        assert p.status == "ok" and set(p.free_symbols) == {"x_home", "x_foreign"}

    @pytest.mark.parametrize("plain,expr", [
        ("y_t = x_t^{1-alpha}", "x**(1 - alpha)"), ("y_t = x_t^{alpha}", "x**alpha"),
        ("y_t = x_t^2", "x**2"), ("y_t = (z_t + x_t)^{alpha}", "(x + z)**alpha"),
        ("y_t = x_t^{i}", "x"),
    ])
    def test_exponents_and_superscript_indices_unchanged(self, plain, expr):
        p = _parse(plain, ["y_t", "x_t", "z_t"], ["alpha"])
        assert p.status == "ok" and str(p.expr) == expr


class TestOperatorsRefused:
    @pytest.mark.parametrize("plain,variables,parameters", [
        ("y_i,t = rho_y * y_i,t-1 + sigma_id * dW_i,t", ["y_{i,t}", "W_{i,t}"], ["rho_y", "sigma_id"]),
        ("d(theta_t)/dt = a * theta_t", ["theta_t"], ["a"]),
        ("dx_t/dt = a * x_t", ["x_t"], ["a"]),
        ("d_epsilon_i,t = sigma * dW_t", ["\\epsilon_{i,t}", "W_t"], ["sigma"]),
        ("S_t = S_{t-1} + lambda * S_{t-1} * dt", ["S_t"], ["lambda"]),
    ])
    def test_differentials(self, plain, variables, parameters):
        assert _parse(plain, variables, parameters).status == "refused_differential"

    def test_difference_operator_refused_unless_declared(self):
        assert _parse("Delta p_t = pi_t", ["p_t", "\\pi_t"], []).status == \
            "refused_difference_operator"
        ok = _parse("K_req = K + Delta K_i", ["K_req", "K", "\\Delta K_i"], [])
        assert ok.status == "ok" and "Delta_K" in ok.free_symbols

    def test_undeclared_unbraced_superscript_is_a_label_not_a_power(self):
        assert _parse("pi_t = (1 - theta) * pi_t^f + theta * x", ["pi_t", "x"], ["theta"]
                      ).status == "refused_superscript_label:f"

    def test_undefined_function_refused(self):
        assert _parse("Sigma_t = (1 - eta) * Sigma_{t-1} + eta * Cov(theta_{t-1})",
                      ["\\Sigma_t", "\\theta"], ["eta"]).status == "refused_undefined_function"

    def test_prose_refused(self):
        assert _parse("delta_hat_i = average of delta_i samples over S iterations",
                      ["\\delta_i"], []).status == "refused_prose"


class TestCodeTeamSymbolIdentity:
    def test_rep10_productivity_steady_state_is_one_not_a_calibrated_constant(self, tmp_path):
        # the model declares a discount rate rho, a CRRA sigma and an emissions intensity
        # epsilon; the productivity process has its own rho_A / sigma_A and an innovation
        model = {"model_title": "tfp", "equations": [
            _eq("EQ1", "ln(A_t) = rho_A * ln(A_{t-1}) + sigma_A * epsilon_{A,t}",
                ["A_t", "A_{t-1}"], ["rho_A", "sigma_A"]),
            _eq("EQ2", "Y_t = A_t * N_t^alpha", ["Y_t", "A_t", "N_t"], ["alpha"]),
            _eq("EQ3", "N_t = 2", ["N_t"], []),
        ], "parameters": [{"parameter_symbol": s} for s in
                          ("rho", "sigma", "epsilon", "alpha", "rho_A", "sigma_A")],
            "variables": [{"variable_symbol": "A_t"}, {"variable_symbol": "Y_t"},
                          {"variable_symbol": "N_t"}]}
        calib = {"calibrated_models": [{"model_title": "tfp", "calibrated_parameters": [
            {"parameter_symbol": "rho", "calibrated_value": 0.04},
            {"parameter_symbol": "sigma", "calibrated_value": 2.0},
            {"parameter_symbol": "epsilon", "calibrated_value": 0.05},
            {"parameter_symbol": "alpha", "calibrated_value": 0.33},
            {"parameter_symbol": "rho_A", "calibrated_value": 0.9},
            {"parameter_symbol": "sigma_A", "calibrated_value": 0.01}]}]}
        gen = generate_module(model, calib)
        assert gen.verdict == "generated", gen.reason
        assert gen.zeroed_shocks == ["epsilon_A"]
        assert not {"rho", "sigma", "epsilon"} & set(gen.parameters)
        val = validate_module(gen, str(tmp_path))
        assert val.verdict == "validated", [c.model_dump() for c in val.checks]
        assert val.steady_state["A"] == pytest.approx(1.0, abs=1e-8)

    def test_taylor_rule_module_has_both_coefficients(self):
        model = {"model_title": "taylor", "equations": [
            _eq("EQ1", "r_t = r_star + phi_pi * (pi_t - pi_star) + phi_y * y_hat_t",
                ["r_t", "\\pi_t", "\\hat{y}_t"], ["r^*", "\\pi^*", "\\phi_{\\pi}", "\\phi_{y}"]),
        ], "parameters": [{"parameter_symbol": s} for s in
                          ("r^*", "\\pi^*", "\\phi_{\\pi}", "\\phi_{y}")]}
        calib = {"calibrated_models": [{"model_title": "taylor", "calibrated_parameters": [
            {"parameter_symbol": "\\phi_{\\pi}", "calibrated_value": 1.5},
            {"parameter_symbol": "\\phi_{y}", "calibrated_value": 0.25}]}]}
        gen = generate_module(model, calib)
        assert gen.parameters["phi_pi"] == 1.5 and gen.parameters["phi_y"] == 0.25
        assert "y_hat" in gen.variables

    def test_variable_colliding_with_a_model_parameter_is_refused(self):
        model = {"model_title": "coll", "equations": [
            _eq("EQ1", "z_t = 2 * sigma_t", ["z_t", "\\sigma_{t}"], []),
            _eq("EQ2", "c = sigma * w", ["c", "w"], ["\\sigma"]),
        ], "parameters": [{"parameter_symbol": "\\sigma"}]}
        gen = generate_module(model)
        assert any(r.equation_id == "EQ1" and r.status == "refused_symbol_collision:sigma"
                   for r in gen.refused)
