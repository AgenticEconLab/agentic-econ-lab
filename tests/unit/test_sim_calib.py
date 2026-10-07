# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Simulation-based (SMM) calibration fallback for models the closed-form harness refuses.

Covers (1) the executability gate — learned/opaque models (neural nets, trained policies) are NOT
auto-simulatable and get an honest 'requires_training' verdict; (2) the SMM engine actually recovers
known parameters of a real stochastic model; (3) the harness routes an uncalibratable model to SMM."""

import pytest

pytest.importorskip("scipy")      # estimate.py -> scipy.least_squares
np = pytest.importorskip("numpy")

from ModelTeam.ael.calib_harness import sim_calib
from ModelTeam.ael.calib_harness.types import TargetRow


# --- (1) executability gate -------------------------------------------------------------

def test_gate_flags_learned_components_as_not_executable():
    rl_eqs = [
        {"equation_latex": r"\pi_\theta(a|s) = \text{Softmax}(\text{MLP}_\theta(\text{Attention}(s)))"},
        {"equation_latex": r"a_t \sim \pi_\theta(\cdot|s_t)"},
    ]
    ok, reason = sim_calib.classify_executability(rl_eqs)
    assert ok is False
    assert "requires_training" in reason
    # names the offending operators so the verdict is honest
    assert any(tok in reason for tok in ("softmax", "mlp", "attention"))


def test_gate_allows_plain_stochastic_equations():
    eqs = [
        {"equation_latex": r"x_t = \rho x_{t-1} + \sigma \varepsilon_t"},
        {"equation_latex": r"y_t = \beta x_t + u_t"},
    ]
    ok, reason = sim_calib.classify_executability(eqs)
    assert ok is True and reason == ""


# --- (2) the SMM engine recovers known parameters of a real stochastic model ------------

def _ar1_simulator():
    """A genuine (seeded) AR(1) simulator: x_t = rho x_{t-1} + sigma eps_t. Returns the sample
    variance and lag-1 autocorrelation — the moments SMM will match."""
    def simulate_moments(params, seed):
        rho, sigma = params["rho"], params["sigma"]
        rng = np.random.default_rng(seed)
        T, burn = 1600, 300
        x = np.zeros(T)
        eps = rng.standard_normal(T)
        for t in range(1, T):
            x[t] = rho * x[t - 1] + sigma * eps[t]
        x = x[burn:]
        return {"x_variance": float(np.var(x)),
                "x_autocorr1": float(np.corrcoef(x[:-1], x[1:])[0, 1])}
    return simulate_moments


def test_smm_recovers_known_ar1_parameters():
    rho_true, sigma_true = 0.6, 0.5
    var_true = sigma_true ** 2 / (1 - rho_true ** 2)   # = 0.390625
    target_book = {
        "x_variance": TargetRow("x_variance", var_true, 0.02, "synthetic"),
        "x_autocorr1": TargetRow("x_autocorr1", rho_true, 0.02, "synthetic"),
    }
    opt, match = sim_calib.smm_estimate(
        _ar1_simulator(), ["x_variance", "x_autocorr1"], target_book,
        free_params=["rho", "sigma"], x0=[0.3, 0.3],
        bounds={"rho": (0.0, 0.95), "sigma": (0.05, 1.5)},
    )
    assert opt is not None
    assert abs(opt["rho"] - rho_true) < 0.12, opt      # recovered, not fabricated
    assert abs(opt["sigma"] - sigma_true) < 0.12, opt
    assert match is not None and match > 0.7           # simulated moments genuinely hit the targets


# --- (3) routing: try_simulation_calibrate + harness.run integration --------------------

def test_try_sim_calibrate_returns_requires_training_for_learned_model():
    model = {"equations": [{"equation_latex": r"a \sim \pi_\theta(s) = \text{MLP}_\theta(s)"}], "parameters": []}
    out = sim_calib.try_simulation_calibrate(model, [])
    assert out.calibration_status == "uncalibratable"
    assert "requires_training" in (out.uncalibratable_reason or "")


def test_try_sim_calibrate_produces_fit_with_injected_simulator():
    """With a runnable simulator supplied, the SMM path yields a genuine simulation_calibrated fit."""
    rho_true, sigma_true = 0.6, 0.5
    var_true = sigma_true ** 2 / (1 - rho_true ** 2)
    tb = {"x_variance": TargetRow("x_variance", var_true, 0.02, "synthetic"),
          "x_autocorr1": TargetRow("x_autocorr1", rho_true, 0.02, "synthetic")}
    sim = _ar1_simulator()

    def builder(formal_model, calibrated_parameters):
        return (sim, ["x_variance", "x_autocorr1"], ["rho", "sigma"], [0.3, 0.3],
                {"rho": (0.0, 0.95), "sigma": (0.05, 1.5)})

    model = {"equations": [{"equation_latex": r"x_t = \rho x_{t-1} + \sigma \varepsilon_t"}], "parameters": []}
    out = sim_calib.try_simulation_calibrate(model, [], simulator_builder=builder, target_book=tb)
    assert out.calibration_status == "simulation_calibrated"
    assert out.fit_score is not None and out.fit_score > 0.7
    assert abs(out.estimated_params["rho"] - rho_true) < 0.12


def test_harness_run_routes_uncalibratable_rl_model_to_honest_reason():
    pytest.importorskip("sympy")   # closed-form path
    from ModelTeam.ael.calib_harness import run as harness_run
    rl_model = {"equations": [
        {"equation_latex": r"\pi_\theta(a|s) = \text{Softmax}(\text{MLP}_\theta(s))", "name": "policy"},
        {"equation_latex": r"a_t \sim \pi_\theta(\cdot|s_t)", "name": "action"},
    ], "parameters": []}
    out = harness_run(rl_model, [])
    assert out.calibration_status == "uncalibratable"
    # the vague closed-form "no_archetype_match" is replaced by the precise SMM-gate reason
    assert "requires_training" in (out.uncalibratable_reason or "")


class TestGateFalsePositives:
    """Standard economics vocabulary must not trigger the opaque gate,
    and an archetype match must not be vetoed by the gate (registry tried FIRST)."""

    def test_econ_vocabulary_is_executable(self):
        eqs = [{"description": "rational inattention: households pay limited attention to news"},
               {"description": "the urban wage gradient declines with distance"}]
        ok, _ = sim_calib.classify_executability(eqs)
        assert ok is True

    def test_ml_usage_still_gated(self):
        for text in ("trained policy network", "policy gradient update", "Attention(s_t)",
                     "self-attention over market features", "stochastic gradient descent"):
            ok, reason = sim_calib.classify_executability([{"description": text}])
            assert ok is False, text
            assert "requires_training" in reason

    def test_archetype_match_beats_the_gate(self):
        """A Taylor-rule model whose prose mentions 'attention'/'gradient' must reach the
        nk_taylor archetype and calibrate — not die at the gate (a routing bug)."""
        model = {
            "model_title": "Monetary policy under rational inattention",
            "model_summary": "central bank Taylor rule with limited attention",
            "parameters": [{"parameter_symbol": "phi_pi",
                            "parameter_name": "Taylor rule inflation response"}],
            "equations": [{"description": "attention to inflation; wage gradient"}],
        }
        out = sim_calib.try_simulation_calibrate(model, [])
        # a registry match is a canonical stand-in, reported as such
        assert out.calibration_status == "archetype_calibrated"
        assert out.simulator == "nk_taylor" and out.simulator_is_archetype


class TestSmmReportsWhatItEstimated:
    """An SMM fit must carry per-moment results and counts that agree
    with its score, and name the simulator, so reports cannot inflate targets or parameters."""

    def _model(self):
        return {
            "model_title": "Heterogeneous-agent New Keynesian model of monetary policy",
            "model_summary": "central bank follows a Taylor rule",
            "parameters": [{"parameter_symbol": "phi_pi", "parameter_name": "Taylor rule inflation response"},
                           {"parameter_symbol": "beta", "parameter_name": "discount factor"}],
            "equations": [],
        }

    def test_archetype_fit_has_moment_results_matching_its_score(self):
        pytest.importorskip("numpy")
        out = sim_calib.try_simulation_calibrate(self._model(), [])
        assert out.calibration_status == "archetype_calibrated"
        assert out.moments, "per-moment results must be populated"
        assert out.coverage["moments_matched"] == len(out.moments)
        assert out.coverage["free_params"] == len(out.estimated_params) == 1
        assert set(out.estimated_params) == {"phi_pi"}   # beta is NOT estimated by the stand-in
        m = out.moments[0]
        assert m.target is not None and m.computed is not None and m.rel_error is not None
        assert out.degrees_of_freedom == 0                # one moment, one parameter: just-identified

    def test_injected_simulator_of_own_model_stays_simulation_calibrated(self):
        tb = {"k": type("T", (), {"value": 1.0, "std_error": 0.1, "source_citation": "x"})()}

        def builder(fm, cp):
            return (lambda p, seed: {"k": p["a"]}, ["k"], ["a"], [0.5], {"a": (0.0, 2.0)})

        out = sim_calib.try_simulation_calibrate({"equations": []}, [], simulator_builder=builder,
                                                 target_book=tb)
        assert out.calibration_status == "simulation_calibrated"
        assert not out.simulator_is_archetype
