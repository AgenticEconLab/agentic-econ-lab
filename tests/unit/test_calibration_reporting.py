# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""The calibration summary and metadata must report
what the deterministic harness estimated and matched, separately from the language model's
proposal. A live run reported "12 parameters were calibrated … 12 empirical targets" for a
model whose harness fit estimated one parameter against one cited moment on a stand-in simulator.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("httpx")      # stage -> shared.llm -> httpx
pytest.importorskip("pydantic")   # stage -> ModelTeam.ael.schemas
pytest.importorskip("numpy")      # nk_taylor stand-in simulator

_AGENTS = Path(__file__).resolve().parents[2]
if str(_AGENTS) not in sys.path:
    sys.path.insert(0, str(_AGENTS))
_STAGE = _AGENTS / "ModelTeam" / "ael" / "ModeWithWcWithHITL" / "3-CalibrationStage.py"
_spec = importlib.util.spec_from_file_location("calibration_stage_reporting_under_test", _STAGE)
cs = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = cs
_spec.loader.exec_module(cs)

from ModelTeam.ael.calib_harness import sim_calib  # noqa: E402


def _hank_model():
    return {
        "model_title": "Heterogeneous-agent New Keynesian model of monetary policy",
        "model_summary": "central bank follows a Taylor rule",
        "parameters": [{"parameter_symbol": "phi_pi", "parameter_name": "Taylor rule inflation response"},
                       {"parameter_symbol": "beta", "parameter_name": "discount factor"}],
        "equations": [],
    }


def _synthesize(monkeypatch, fit_metrics, sandbox_validation, n_params=12, n_targets=12):
    captured = {}
    monkeypatch.setattr(cs, "CalibratedModel", lambda **kw: captured.update(kw) or kw)
    c = cs.Calibrator.__new__(cs.Calibrator)
    params = [SimpleNamespace(parameter_symbol=f"p{i}", calibrated_value=0.5, justification="lit")
              for i in range(n_params)]
    c._synthesize_calibrated_model(
        SimpleNamespace(model_title="HANK", based_on_framework="HANK"),
        [object()] * n_targets,
        SimpleNamespace(strategy_type="SMM", description="d", identification_notes="n"),
        params, [], fit_metrics, sandbox_validation)
    return captured


def test_archetype_fit_summary_reports_harness_numbers(monkeypatch):
    outcome = sim_calib.try_simulation_calibrate(_hank_model(), [])
    assert outcome.calibration_status == "archetype_calibrated"
    sv, moments, fm = cs.Calibrator._harness_outcome_to_schema(None, outcome)

    # fit_metrics counts agree with the harness's per-moment results
    assert fm.total_targets == len(moments) == 1
    assert fm.harness_details["estimated_params"].keys() == {"phi_pi"}
    assert fm.harness_details["simulator"] == "nk_taylor"
    assert fm.harness_details["simulator_is_archetype"] is True
    assert sv.validated is False            # a stand-in fit does not validate the model itself

    out = _synthesize(monkeypatch, fm, sv)
    summary = out["calibration_summary"]
    assert "12 proposed parameter values" in summary and "12 candidate empirical targets" in summary
    assert "estimated 1 parameter(s) (phi_pi" in summary
    assert "against 1 cited target(s)" in summary
    assert "canonical stand-in 'nk_taylor'" in summary
    assert "12 parameters were calibrated" not in summary

    md = out["metadata"]
    assert md["calibration_status"] == "archetype_calibrated"
    assert md["num_proposed_params"] == 12 and md["num_candidate_targets"] == 12
    assert md["num_harness_estimated_params"] == 1 and md["num_cited_targets_matched"] == 1
    assert md["simulator_is_archetype"] is True
    assert "num_calibrated_params" not in md


def test_metadata_always_carries_the_verdict(monkeypatch):
    """Reports read metadata.calibration_status; it was written only for uncalibratable models."""
    fm = cs.FitMetrics(total_targets=1, targets_matched=1, mean_absolute_error=0.0,
                       mean_relative_error=0.0, rmse=0.0, fit_score=1.0, fit_assessment="x",
                       harness_details={"calibration_status": "point_calibrated",
                                        "estimated_params": {"psi": 0.21},
                                        "cited_targets_matched": ["inflation_mean"]})
    sv = SimpleNamespace(verdict="point_calibrated", validated=False)
    md = _synthesize(monkeypatch, fm, sv, n_params=15)["metadata"]
    assert md["calibration_status"] == "point_calibrated"
    assert md["harness_estimated_params"] == {"psi": 0.21}
