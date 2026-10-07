# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Tests for the ModelTeam V0.7 hooks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared.econ import (
    attach_causal_estimate,
    attach_falsifier_reports,
    maybe_run_simulation_stage,
    run_causal_fm_pass,
    run_falsifier_pass,
)
from shared.feature_flags import set_flag, clear_flag


@pytest.fixture(autouse=True)
def _clean_flags(monkeypatch):
    for flag in (
        "causal_fm_enabled",
        "simulation_enabled",
        "falsifier_enabled",
    ):
        monkeypatch.delenv(f"AEL_{flag.upper()}", raising=False)
    yield


# ---------------------------------------------------------------------------
# Falsifier hook
# ---------------------------------------------------------------------------

class TestFalsifierHook:
    def test_disabled_flag_returns_empty(self, monkeypatch):
        monkeypatch.setenv("AEL_FALSIFIER_ENABLED", "0")
        assert run_falsifier_pass({"frameworks": [{"assumptions": ["x"]}]}) == []

    def test_extracts_assumptions(self, monkeypatch):
        monkeypatch.setenv("AEL_FALSIFIER_ENABLED", "1")
        reports = run_falsifier_pass({
            "frameworks": [{
                "assumptions": [
                    "Every agent always converges to a rational equilibrium.",
                    "Monetary easing causes inflation.",
                ],
                "components": [],
            }]
        })
        assert len(reports) == 2
        for r in reports:
            assert "verdict" in r
            assert r["verdict"] in {
                "survives", "corroborated", "needs_revision", "falsified",
                "n/a", "error",
            }

    def test_handles_missing_frameworks(self, monkeypatch):
        monkeypatch.setenv("AEL_FALSIFIER_ENABLED", "1")
        assert run_falsifier_pass({}) == []

    def test_dedupes_repeated_claims(self, monkeypatch):
        monkeypatch.setenv("AEL_FALSIFIER_ENABLED", "1")
        reports = run_falsifier_pass({
            "frameworks": [{
                "assumptions": [
                    "All agents always converge.",
                    "All agents always converge.",
                    "All agents always converge.",
                ]
            }]
        })
        assert len(reports) == 1

    def test_propositions_with_statement_dict(self, monkeypatch):
        monkeypatch.setenv("AEL_FALSIFIER_ENABLED", "1")
        reports = run_falsifier_pass({
            "frameworks": [{
                "propositions": [
                    {"statement": "Inflation always follows money growth proportionally."}
                ],
            }]
        })
        assert any("always" in r["claim"].lower() for r in reports)

    def test_attach_falsifier_reports_writes_in_place(self, tmp_path):
        p = tmp_path / "theory.json"
        p.write_text(json.dumps({"frameworks": []}))
        attach_falsifier_reports(p, [{"claim": "x", "verdict": "survives"}])
        data = json.loads(p.read_text())
        assert data["falsifier_reports"][0]["verdict"] == "survives"

    def test_attach_falsifier_reports_noop_on_empty(self, tmp_path):
        p = tmp_path / "theory.json"
        p.write_text(json.dumps({"frameworks": []}))
        attach_falsifier_reports(p, [])
        data = json.loads(p.read_text())
        assert "falsifier_reports" not in data


# ---------------------------------------------------------------------------
# CausalFM hook
# ---------------------------------------------------------------------------

class TestCausalFMHook:
    def test_disabled_flag_returns_none(self, monkeypatch):
        monkeypatch.setenv("AEL_CAUSAL_FM_ENABLED", "0")
        assert run_causal_fm_pass({}) is None

    def test_missing_models_returns_none(self, monkeypatch):
        monkeypatch.setenv("AEL_CAUSAL_FM_ENABLED", "1")
        assert run_causal_fm_pass({}) is None

    _CALIB = {
        "calibrated_models": [{
            "calibrated_parameters": [
                {"name": "policy"},
                {"name": "capital"},
                {"name": "labour"},
            ],
            "empirical_targets": [{"name": "growth"}],
        }]
    }

    def test_refuses_without_observed_data(self, monkeypatch):
        """No estimate from generated rows: a standalone ModelTeam run has no observations."""
        monkeypatch.setenv("AEL_CAUSAL_FM_ENABLED", "1")
        for rows in (None, []):
            result = run_causal_fm_pass(self._CALIB, data_rows=rows)
            assert result["status"] == "refused"
            assert result["ate"] is None
            assert "sample_size" not in result
            assert (result["treatment"], result["outcome"]) == ("policy", "growth")

    def test_produces_estimate_from_observed_rows(self, monkeypatch):
        monkeypatch.setenv("AEL_CAUSAL_FM_ENABLED", "1")
        import random
        rng = random.Random(0)
        rows = []
        for _ in range(150):
            r = {"policy": rng.gauss(0, 1), "capital": rng.gauss(0, 1), "labour": rng.gauss(0, 1)}
            r["growth"] = 0.5 * r["policy"] + 0.2 * r["capital"] + rng.gauss(0, 0.1)
            rows.append(r)
        result = run_causal_fm_pass(self._CALIB, data_rows=rows)
        assert result.get("method") == "back_door_ols"
        assert result.get("sample_size") == 150
        assert abs(result["ate"] - 0.5) < 0.1

    def test_explicit_adjustment_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("AEL_CAUSAL_FM_ENABLED", "1")
        spec = {
            "method": "back_door",
            "treatment": "T", "outcome": "Y",
            "adjustment_set": ["X1", "X2"],
        }
        import random
        rng = random.Random(42)
        rows = [
            {"T": rng.gauss(0, 1), "Y": rng.gauss(0, 1),
             "X1": rng.gauss(0, 1), "X2": rng.gauss(0, 1)}
            for _ in range(100)
        ]
        result = run_causal_fm_pass(
            {"causal_adjustment": spec},
            data_rows=rows,
        )
        assert isinstance(result, dict)
        assert "ate" in result

    def test_attach_causal_estimate_writes_in_place(self, tmp_path):
        p = tmp_path / "calib.json"
        p.write_text(json.dumps({"x": 1}))
        attach_causal_estimate(p, {"ate": 1.0, "method": "back_door_ols"})
        data = json.loads(p.read_text())
        assert data["causal_estimate"]["ate"] == 1.0


# ---------------------------------------------------------------------------
# Simulation hook
# ---------------------------------------------------------------------------

class TestSimulationHook:
    def test_disabled_flag_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AEL_SIMULATION_ENABLED", "0")
        result = maybe_run_simulation_stage(
            tmp_path / "calib.json", tmp_path, mode="ModeNoWcNoHITL",
        )
        assert result is None

    def test_enabled_produces_payload(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AEL_SIMULATION_ENABLED", "1")
        calib = tmp_path / "calib.json"
        calib.write_text(json.dumps({"recommended_scenario": "dsge_monetary"}))
        result = maybe_run_simulation_stage(calib, tmp_path, mode="ModeNoWcNoHITL")
        assert result is not None
        assert (tmp_path / "simulation_output.json").exists()

    def test_missing_calibration_graceful(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AEL_SIMULATION_ENABLED", "1")
        result = maybe_run_simulation_stage(
            tmp_path / "does-not-exist.json", tmp_path, mode="x",
        )
        # Should either return a payload (with defaults) or write an error file
        assert result is not None
