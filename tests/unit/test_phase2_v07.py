# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""AEL V0.7 Phase 2 — Economic simulation (ABIDES/LLM-Economist/MALLES)."""

from __future__ import annotations

import pytest

from shared.econ import (
    ABIDESEconomistBackend,
    BACKENDS,
    LLMEconomistBackend,
    MALLESBackend,
    SCENARIOS,
    SimulationResult,
    SimulationSpec,
    run_simulation,
)


# ---------------------------------------------------------------------------
# SimulationSpec validation
# ---------------------------------------------------------------------------

class TestSpec:
    def test_basic(self):
        spec = SimulationSpec(backend="abides_economist", scenario="dsge_monetary")
        assert spec.n_agents == 100
        assert spec.n_periods == 60

    def test_rejects_nonpositive_agents(self):
        with pytest.raises(ValueError):
            SimulationSpec(backend="abides_economist", scenario="x", n_agents=0)

    def test_rejects_nonpositive_periods(self):
        with pytest.raises(ValueError):
            SimulationSpec(backend="abides_economist", scenario="x", n_periods=-5)


# ---------------------------------------------------------------------------
# Registry + scenarios
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_three_backends(self):
        assert set(BACKENDS.keys()) == {"abides_economist", "llm_economist", "malles"}

    def test_backends_instantiable(self):
        for factory in BACKENDS.values():
            obj = factory()
            assert obj.name in {"abides_economist", "llm_economist", "malles"}

    def test_six_builtin_scenarios(self):
        required = {
            "stackelberg_tax", "dsge_monetary", "dsge_fiscal",
            "collusion_pricing", "consumer_choice", "info_asymmetry_credence",
        }
        assert required.issubset(SCENARIOS.keys())
        assert len(SCENARIOS) >= 6

    def test_scenario_default_backend_resolves(self):
        for scen in SCENARIOS.values():
            assert scen.default_backend in BACKENDS

    def test_unknown_backend_rejected(self):
        with pytest.raises(ValueError):
            run_simulation(SimulationSpec(backend="bogus", scenario="x"))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ABIDES-Economist backend
# ---------------------------------------------------------------------------

class TestABIDES:
    def test_monetary_scenario(self):
        spec = SimulationSpec(
            backend="abides_economist", scenario="dsge_monetary",
            n_agents=50, n_periods=80, random_seed=7,
        )
        res = run_simulation(spec, {})
        assert isinstance(res, SimulationResult)
        assert res.backend == "abides_economist"
        assert "inflation" in res.time_series
        assert len(res.time_series["inflation"]) == 80
        assert "mean_inflation" in res.metrics
        assert res.elapsed_s >= 0.0

    def test_fiscal_scenario_adds_multiplier(self):
        spec = SimulationSpec(
            backend="abides_economist", scenario="dsge_fiscal",
            n_agents=40, n_periods=40, random_seed=3,
        )
        res = run_simulation(spec, {})
        assert "fiscal_multiplier" in res.metrics

    def test_equilibrium_found_with_stable_shocks(self):
        spec = SimulationSpec(
            backend="abides_economist", scenario="dsge_monetary",
            n_agents=40, n_periods=200, random_seed=0,
        )
        res = run_simulation(spec, {"shock_std": 0.001})
        assert res.equilibrium_found

    def test_wall_clock_under_budget(self):
        spec = SimulationSpec(
            backend="abides_economist", scenario="dsge_monetary",
            n_agents=100, n_periods=60,
        )
        res = run_simulation(spec, {})
        # Phase NF requirement: 100 agents x 60 periods under 120s (we expect << 1s)
        assert res.elapsed_s < 10.0

    def test_reproducible_under_seed(self):
        kwargs = dict(backend="abides_economist", scenario="dsge_monetary",
                      n_agents=30, n_periods=40, random_seed=1234)
        a = run_simulation(SimulationSpec(**kwargs), {})
        b = run_simulation(SimulationSpec(**kwargs), {})
        assert a.time_series["inflation"] == b.time_series["inflation"]


# ---------------------------------------------------------------------------
# LLM-Economist backend (closed-form mode)
# ---------------------------------------------------------------------------

class TestLLMEconomist:
    def test_stackelberg_converges_to_saez(self):
        spec = SimulationSpec(
            backend="llm_economist", scenario="stackelberg_tax",
            n_agents=200, n_periods=60, random_seed=42,
            heterogeneity={"pareto_alpha": 2.0},
        )
        res = run_simulation(spec, {"etr_elasticity": 0.25})
        # Saez approx: tau* = 1 / (1 + 2.0 * 0.25) = 1/1.5 ≈ 0.6667
        assert abs(res.metrics["tau_saez_optimal"] - 2/3) < 0.01
        # Converges within 5% of Saez-optimal (design target)
        assert res.metrics["tau_gap"] < 0.05
        assert res.equilibrium_found

    def test_collusion_scenario_runs(self):
        # collusion_pricing now runs a Bertrand duopoly,
        # not Saez tax — distinct time-series and metric keys.
        spec = SimulationSpec(
            backend="llm_economist", scenario="collusion_pricing",
            n_agents=50, n_periods=30,
        )
        res = run_simulation(spec, {"marginal_cost": 1.0,
                                     "demand_intercept": 10.0,
                                     "demand_slope": 1.0})
        assert res.scenario == "collusion_pricing"
        assert "price_a" in res.time_series and "price_b" in res.time_series
        assert "mean_price_final" in res.metrics
        assert "supracompetitive_gap" in res.metrics
        # Final mean price should be above marginal cost
        assert res.metrics["mean_price_final"] > 1.0

    def test_info_asymmetry_scenario(self):
        res = run_simulation(
            SimulationSpec(
                backend="llm_economist",
                scenario="info_asymmetry_credence",
                n_agents=20, n_periods=20,
            ),
            {},
        )
        assert res.scenario == "info_asymmetry_credence"

    def test_closed_form_label(self):
        res = run_simulation(
            SimulationSpec(backend="llm_economist", scenario="stackelberg_tax",
                           n_agents=20, n_periods=20),
            {},
        )
        assert "closed-form" in res.notes


# ---------------------------------------------------------------------------
# MALLES backend
# ---------------------------------------------------------------------------

class TestMALLES:
    def test_hit_rate_near_target(self):
        spec = SimulationSpec(
            backend="malles", scenario="consumer_choice",
            n_agents=200, n_periods=50, random_seed=11,
        )
        res = run_simulation(spec, {"n_products": 5})
        # Target hit_rate is 0.775; allow ±5% deviation
        assert 0.73 < res.metrics["hit_rate"] < 0.83
        assert res.equilibrium_found

    def test_period_series_length(self):
        spec = SimulationSpec(
            backend="malles", scenario="consumer_choice",
            n_agents=30, n_periods=25,
        )
        res = run_simulation(spec, {})
        assert len(res.time_series["period_hit_rate"]) == 25


# ---------------------------------------------------------------------------
# Budget guardrail
# ---------------------------------------------------------------------------

class TestBudget:
    def test_budget_overrun_note(self):
        # Extremely tight budget — note should be set even if execution is fast.
        spec = SimulationSpec(
            backend="abides_economist", scenario="dsge_monetary",
            n_agents=10, n_periods=10, max_wall_clock_s=0.0,
        )
        res = run_simulation(spec, {})
        assert "BUDGET_OVERRUN" in res.notes


class TestSimulationStageWrapper:
    """Shared helper used by the 4 ModelTeam AEL modes' 4-SimulationStage.py."""

    def test_skipped_writes_sentinel(self, tmp_path):
        from shared.econ import run_simulation_stage
        out = tmp_path / "sim.json"
        result = run_simulation_stage(
            calibration_output_path=tmp_path / "nonexistent.json",
            output_path=out,
            simulation_enabled=False,
        )
        assert result == {"simulation_skipped": True}
        assert out.exists()

    def test_enabled_produces_payload(self, tmp_path):
        from shared.econ import run_simulation_stage
        calib = tmp_path / "calib.json"
        calib.write_text('{"parameters": {"beta": 0.95}, "recommended_scenario": "dsge_monetary"}')
        out = tmp_path / "sim.json"
        payload = run_simulation_stage(
            calibration_output_path=calib,
            output_path=out,
            simulation_enabled=True,
            n_agents=20,
            n_periods=15,
        )
        assert "simulation_result" in payload
        assert payload["simulation_result"]["backend"] == "abides_economist"
        assert out.exists()

    def test_scenario_override(self, tmp_path):
        from shared.econ import run_simulation_stage
        out = tmp_path / "sim.json"
        calib = tmp_path / "calib.json"
        calib.write_text("{}")
        payload = run_simulation_stage(
            calibration_output_path=calib,
            output_path=out,
            simulation_enabled=True,
            scenario="stackelberg_tax",
            backend="llm_economist",
            n_agents=30, n_periods=20,
        )
        assert payload["simulation_result"]["backend"] == "llm_economist"
        assert payload["simulation_result"]["scenario"] == "stackelberg_tax"

    def test_unknown_scenario_raises(self, tmp_path):
        from shared.econ import run_simulation_stage
        with pytest.raises(ValueError):
            run_simulation_stage(
                calibration_output_path=tmp_path / "x.json",
                output_path=tmp_path / "y.json",
                simulation_enabled=True,
                scenario="completely-bogus",
            )

    def test_all_four_modes_have_stage_file(self):
        from pathlib import Path
        base = Path(__file__).resolve().parent.parent.parent / "ModelTeam" / "ael"
        for mode in ("ModeNoWcNoHITL", "ModeNoWcWithHITL",
                     "ModeWithWcNoHITL", "ModeWithWcWithHITL"):
            p = base / mode / "4-SimulationStage.py"
            assert p.exists(), f"missing stage file: {p}"


class TestResultSerialization:
    def test_to_dict_contains_expected_keys(self):
        res = run_simulation(
            SimulationSpec(backend="abides_economist", scenario="dsge_monetary",
                           n_agents=10, n_periods=10), {},
        )
        d = res.to_dict()
        assert d["backend"] == "abides_economist"
        assert "metrics" in d
        assert "time_series_keys" in d
