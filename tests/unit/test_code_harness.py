# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""CodeTeam deterministic harness: derived-module generation, validation, comparative statics.

The non-negotiable test: a Solow-style system whose steady state has a closed form —
the derived module's fsolve solution must match k* = (s/δ)^(1/(1-α)) to 1e-6."""

import math
import re

import pytest

from CodeTeam.ael.code_harness import (
    comparative_statics,
    generate_module,
    validate_module,
)


def _eq(eid, plain, variables, parameters):
    return {"equation_id": eid, "equation_plain": plain,
            "variables_used": variables, "parameters_used": parameters}


def _solow_model():
    return {
        "model_title": "Solow steady state",
        "equations": [
            _eq("E1", "y = k**alpha", ["y", "k"], ["alpha"]),
            _eq("E2", "k = s*y/delta", ["k", "y"], ["s", "delta"]),
        ],
        "parameters": [{"parameter_symbol": "alpha"}, {"parameter_symbol": "s"},
                       {"parameter_symbol": "delta"}],
        "variables": [{"variable_symbol": "y"}, {"variable_symbol": "k"}],
    }


def _solow_calibration(alpha=0.33, s=0.2, delta=0.1):
    return {"calibrated_models": [{
        "model_title": "Solow steady state",
        "calibrated_parameters": [
            {"parameter_symbol": "alpha", "calibrated_value": alpha},
            {"parameter_symbol": "s", "calibrated_value": s},
            {"parameter_symbol": "delta", "calibrated_value": delta},
        ]}]}


class TestGeneration:
    def test_square_calibrated_system_is_generated(self):
        gen = generate_module(_solow_model(), _solow_calibration())
        assert gen.verdict == "generated", gen.reason
        assert gen.square and gen.n_parseable == 2
        assert set(gen.variables) == {"y", "k"}
        assert gen.parameters == {"alpha": 0.33, "s": 0.2, "delta": 0.1}
        assert "def residuals" in gen.module_text and "def solve_steady_state" in gen.module_text

    def test_missing_calibration_is_partial_not_guessed(self):
        gen = generate_module(_solow_model(), calibration=None)
        assert gen.verdict == "partial"
        assert set(gen.unset_parameters) == {"alpha", "s", "delta"}

    def test_unparseable_equations_are_ungenerable_with_reason(self):
        model = {"model_title": "opaque", "equations": [
            _eq("E1", "no equals sign here", ["x"], []),
        ], "parameters": [], "variables": [{"variable_symbol": "x"}]}
        gen = generate_module(model)
        assert gen.verdict == "ungenerable"
        assert "no_parseable_equations" in gen.reason

    def test_lambda_keyword_symbol_is_sanitized(self):
        model = {"model_title": "lam", "equations": [
            _eq("E1", "y = lambda*k", ["y", "k"], ["lambda"]),
            _eq("E2", "k = y/2 + 1", ["k", "y"], []),
        ], "parameters": [{"parameter_symbol": "lambda"}],
            "variables": [{"variable_symbol": "y"}, {"variable_symbol": "k"}]}
        calib = {"calibrated_models": [{"calibrated_parameters": [
            {"parameter_symbol": "lambda", "calibrated_value": 0.5}]}]}
        gen = generate_module(model, calib)
        assert gen.verdict == "generated", gen.reason
        assert "lambda_" in gen.parameters and "lambda_" in gen.module_text
        compile(gen.module_text, "<generated>", "exec")   # keyword never breaks the module


class TestValidation:
    def test_solow_steady_state_matches_closed_form(self, tmp_path):
        alpha, s, delta = 0.33, 0.2, 0.1
        gen = generate_module(_solow_model(), _solow_calibration(alpha, s, delta))
        val = validate_module(gen, str(tmp_path))
        assert val.verdict == "validated", [c.model_dump() for c in val.checks]
        k_star = (s / delta) ** (1.0 / (1.0 - alpha))
        y_star = k_star ** alpha
        assert val.steady_state["k"] == pytest.approx(k_star, abs=1e-6)
        assert val.steady_state["y"] == pytest.approx(y_star, abs=1e-6)
        assert val.max_residual < 1e-8

    def test_unset_parameters_block_certification_honestly(self, tmp_path):
        gen = generate_module(_solow_model(), calibration=None)
        val = validate_module(gen, str(tmp_path))
        assert val.verdict == "partial"
        blocked = next(c for c in val.checks if c.name == "parameters_complete")
        assert not blocked.passed and "unset parameters" in blocked.detail

    def test_non_square_system_declines_solving(self, tmp_path):
        model = {"model_title": "nonsquare", "equations": [
            _eq("E1", "y = k**alpha", ["y", "k"], ["alpha"]),
        ], "parameters": [{"parameter_symbol": "alpha"}],
            "variables": [{"variable_symbol": "y"}, {"variable_symbol": "k"}]}
        calib = {"calibrated_models": [{"calibrated_parameters": [
            {"parameter_symbol": "alpha", "calibrated_value": 0.33}]}]}
        gen = generate_module(model, calib)
        assert gen.verdict == "partial" and not gen.square
        val = validate_module(gen, str(tmp_path))
        assert val.verdict == "partial"
        solve_check = next(c for c in val.checks if c.name == "steady_state_solves")
        assert not solve_check.passed and "non-square" in solve_check.detail


class TestComparativeStatics:
    def test_solow_alpha_raises_steady_capital(self, tmp_path):
        """With s/delta = 2 > 1, dk*/dalpha > 0 — the sweep must reproduce the sign."""
        gen = generate_module(_solow_model(), _solow_calibration())
        val = validate_module(gen, str(tmp_path))
        exp = comparative_statics(gen, val, str(tmp_path))
        assert exp.verdict == "completed"
        alpha_up = next(cs for cs in exp.statics
                        if cs.parameter == "alpha" and cs.delta_pct > 0)
        assert alpha_up.converged
        assert alpha_up.variable_changes_pct["k"] > 0
        # and the magnitude matches the closed form at alpha*1.1
        a2 = 0.33 * 1.1
        k2 = (0.2 / 0.1) ** (1.0 / (1.0 - a2))
        k1 = (0.2 / 0.1) ** (1.0 / (1.0 - 0.33))
        expected_pct = (k2 - k1) / k1 * 100.0
        assert alpha_up.variable_changes_pct["k"] == pytest.approx(expected_pct, rel=1e-4)

    def test_not_applicable_without_validated_steady_state(self, tmp_path):
        gen = generate_module(_solow_model(), calibration=None)
        val = validate_module(gen, str(tmp_path))
        exp = comparative_statics(gen, val, str(tmp_path))
        assert exp.verdict == "not_applicable"


class TestCompoundSubscripts:
    def test_bare_compound_subscripts_parse_and_generate(self):
        """K_i,t-style subscripts left a dangling ',t' that parsed
        as a tuple — 11 refusals were this class. Steady-state accumulation now emerges."""
        model = {
            "model_title": "accumulation",
            "equations": [
                _eq("E1", "C_i,t = (1-s)*Y_i,t", ["C_i,t", "Y_i,t"], ["s"]),
                _eq("E2", "K_i,t = K_i,t-1 + I_i,t - delta*K_i,t-1", ["K_i,t", "I_i,t"],
                    ["delta"]),
            ],
            "parameters": [{"parameter_symbol": "s"}, {"parameter_symbol": "delta"}],
            "variables": [{"variable_symbol": "C"}, {"variable_symbol": "Y"},
                          {"variable_symbol": "K"}, {"variable_symbol": "I"}],
        }
        gen = generate_module(model, {"calibrated_models": [{"calibrated_parameters": [
            {"parameter_symbol": "s", "calibrated_value": 0.2},
            {"parameter_symbol": "delta", "calibrated_value": 0.1}]}]})
        assert gen.n_parseable == 2, [r.model_dump() for r in gen.refused]
        assert gen.verdict in ("generated", "partial")


class TestReservedSymbolSafety:
    """An equation with UNDECLARED bare 'E' parsed as Euler's constant,
    pycode printed 'np.e', and env-substitution mangled it to np.env["e"] -> module crash.
    Layer 1: undeclared E/I refuse honestly. Layer 2: attribute access survives substitution
    even when a variable is named 'e'."""

    def test_undeclared_E_is_refused_not_euler(self):
        model = {"model_title": "exp", "equations": [
            _eq("E1", "K = E*R*a**2 + D", ["K", "R", "a", "D"], []),
            _eq("E2", "D = K/2", ["D", "K"], []),
        ], "parameters": [], "variables": [
            {"variable_symbol": "K"}, {"variable_symbol": "R"},
            {"variable_symbol": "a"}, {"variable_symbol": "D"}]}
        gen = generate_module(model)
        assert any("refused_undeclared_reserved_symbol:E" in r.status for r in gen.refused)
        # no bare Euler constant in emitted code (np.errstate in the template is fine)
        assert not re.search(r"np\.e(?![a-zA-Z_])", gen.module_text or "")

    def test_declared_E_and_I_stay_symbols(self):
        model = {"model_title": "inv", "equations": [
            _eq("E1", "Y = C + I + E", ["Y", "C", "I", "E"], []),
            _eq("E2", "C = 0.8*Y", ["C", "Y"], []),
            _eq("E3", "I = 0.1*Y", ["I", "Y"], []),
            _eq("E4", "E = 0.1*Y", ["E", "Y"], []),
        ], "parameters": [], "variables": [
            {"variable_symbol": "Y"}, {"variable_symbol": "C"},
            {"variable_symbol": "I"}, {"variable_symbol": "E"}]}
        gen = generate_module(model)
        assert gen.n_parseable == 4, [r.model_dump() for r in gen.refused]
        assert 'env["E"]' in gen.module_text and 'env["I"]' in gen.module_text

    def test_variable_named_e_does_not_mangle_np_attributes(self):
        # exp(x) prints as np.exp; a variable 'e' must not rewrite it to np.env["e"]xp
        model = {"model_title": "expo", "equations": [
            _eq("E1", "y = exp(e)", ["y", "e"], []),
            _eq("E2", "e = y/2", ["e", "y"], []),
        ], "parameters": [], "variables": [
            {"variable_symbol": "y"}, {"variable_symbol": "e"}]}
        gen = generate_module(model)
        assert gen.n_parseable == 2, [r.model_dump() for r in gen.refused]
        assert "np.env" not in gen.module_text
        compile(gen.module_text, "<generated>", "exec")


class TestCompoundLHS:
    """'P*c + a = ...' atomized into pseudo-variable sym_P___c___a and
    ln(y) became an unknown independent of y — mathematically incoherent modules that the
    Tier-2 judges rightly scored down. A compound LHS is now parsed as an expression."""

    def test_budget_constraint_lhs_keeps_real_unknowns(self):
        model = {"model_title": "budget", "equations": [
            _eq("E1", "P*c + a = R*a + W*l", ["P", "c", "a", "R", "W", "l"], []),
            _eq("E2", "c = 0.8*W*l/P", ["c", "W", "l", "P"], []),
        ], "parameters": [], "variables": [
            {"variable_symbol": s} for s in ("P", "c", "a", "R", "W", "l")]}
        gen = generate_module(model)
        assert gen.n_parseable == 2, [r.model_dump() for r in gen.refused]
        assert not any(v.startswith("sym_") for v in gen.variables), gen.variables
        assert set(gen.variables) == {"P", "c", "a", "R", "W", "l"}

    def test_log_lhs_keeps_y_as_the_unknown(self):
        model = {"model_title": "loglhs", "equations": [
            _eq("E1", "ln(y) = rho*ln(k)", ["y", "k"], ["rho"]),
            _eq("E2", "k = 2*y", ["k", "y"], []),
        ], "parameters": [{"parameter_symbol": "rho"}],
            "variables": [{"variable_symbol": "y"}, {"variable_symbol": "k"}]}
        gen = generate_module(model, {"calibrated_models": [{"calibrated_parameters": [
            {"parameter_symbol": "rho", "calibrated_value": 0.5}]}]})
        assert gen.n_parseable == 2, [r.model_dump() for r in gen.refused]
        assert set(gen.variables) == {"y", "k"}          # no sym_ln_y_ pseudo-unknown
        assert "sym_" not in gen.module_text

    def test_unparseable_lhs_refuses_honestly(self):
        model = {"model_title": "badlhs", "equations": [
            _eq("E1", "max{c} + = y", ["c", "y"], []),
        ], "parameters": [], "variables": [{"variable_symbol": "c"}, {"variable_symbol": "y"}]}
        gen = generate_module(model)
        assert gen.n_parseable == 0


class TestProfessionalEmission:
    def test_module_carries_equation_provenance_and_main(self):
        gen = generate_module(_solow_model(), _solow_calibration())
        assert "# E1" in gen.module_text          # provenance comment per residual
        assert 'if __name__ == "__main__":' in gen.module_text
        compile(gen.module_text, "<generated>", "exec")

    def test_validation_scratch_cleaned(self, tmp_path):
        from CodeTeam.ael.code_harness import cleanup_workdir
        gen = generate_module(_solow_model(), _solow_calibration())
        val = validate_module(gen, str(tmp_path))
        assert val.verdict == "validated"
        cleanup_workdir(str(tmp_path))
        leftovers = [p.name for p in tmp_path.iterdir()]
        assert "_exec_scratch" not in leftovers and "__pycache__" not in leftovers
