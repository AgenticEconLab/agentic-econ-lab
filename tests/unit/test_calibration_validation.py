# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Guardrail for the calibration-truthfulness fix.

The bug: ModelTeam's CalibrationStage set `validated = exec_result.success` (the process
exited 0) and computed `fit_metrics` from the LLM's *narrated* moments — so a run whose
sandbox printed `{"validation": "fail"}` still reported validated=True with a high fit.
These tests lock the invariants that `validated` follows the sandbox VERDICT and that the
reported fit is rebuilt from the sandbox's recomputed moments.

The stage imports shared.llm (httpx) + pydantic schemas, so this skips where those are
absent (e.g. a bare login node) and runs in the full test environment.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("httpx")      # stage -> shared.llm -> httpx
pytest.importorskip("pydantic")   # stage -> ModelTeam.ael.schemas

_AGENTS = Path(__file__).resolve().parents[2]
if str(_AGENTS) not in sys.path:
    sys.path.insert(0, str(_AGENTS))
_STAGE = _AGENTS / "ModelTeam" / "ael" / "ModeNoWcNoHITL" / "3-CalibrationStage.py"
_spec = importlib.util.spec_from_file_location("calibration_stage_under_test", _STAGE)
cs = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = cs
_spec.loader.exec_module(cs)

PASS_OUT = '{"validation": "pass", "moments": [{"name": "K/Y", "computed": 4.0, "target": 4.0, "error_pct": 0.0}], "summary": "ok"}'
FAIL_OUT = '{"validation": "fail", "moments": [{"name": "K/Y", "computed": 6.0, "target": 4.0, "error_pct": 50.0}], "summary": "bad"}'


def test_parse_sandbox_json_extracts_verdict():
    assert cs._parse_sandbox_json(FAIL_OUT)["validation"] == "fail"
    assert cs._parse_sandbox_json("noise before\n" + PASS_OUT + "\ntrailing")["validation"] == "pass"
    assert cs._parse_sandbox_json("") is None
    assert cs._parse_sandbox_json("no json at all") is None


def test_validated_requires_pass_not_just_exit_zero():
    """A script that EXECUTED (exit 0) but printed verdict 'fail' must NOT be validated."""
    for out, success, expected in [(PASS_OUT, True, True), (FAIL_OUT, True, False), (PASS_OUT, False, False)]:
        verdict = str((cs._parse_sandbox_json(out) or {}).get("validation", "")).lower()
        validated = bool(success) and verdict == "pass"
        assert validated is expected


def test_sandbox_moments_rebuild_fit_inputs_from_recomputed_values():
    sv = type("SV", (), {"stdout": FAIL_OUT})()
    moms = cs.Calibrator._sandbox_moments_to_model_moments(None, sv)  # self unused
    assert len(moms) == 1
    m = moms[0]
    assert m.moment_name == "K/Y"
    assert m.model_value == 6.0 and m.empirical_value == 4.0   # the SANDBOX's numbers, not narrated
    assert m.relative_error == 50.0 and m.fit_quality == "Poor"


def test_sandbox_moments_empty_when_no_stdout():
    sv = type("SV", (), {"stdout": ""})()
    assert cs.Calibrator._sandbox_moments_to_model_moments(None, sv) == []


# --- targets-only feasibility path -----------

def _bare_calibrator():
    """A Calibrator without __init__ (no LLM client) — enough for the targets-only path,
    which only touches self.agent_name + the two targets-only methods."""
    c = cs.Calibrator.__new__(cs.Calibrator)
    c.agent_name = "Calibrator"
    return c


def test_targets_only_model_is_honest_and_carries_targets():
    from types import SimpleNamespace
    c = _bare_calibrator()
    fm = SimpleNamespace(model_title="Test Model")
    targets = [cs.EmpiricalTarget(target_id="ky", target_name="K/Y", empirical_value=4.0)]
    m = c._targets_only_model(fm, targets)
    # honest, not-calibrated, and clearly flagged for downstream
    assert m.metadata.get("targets_only") is True
    assert m.fit_metrics.fit_score == 0.0
    assert m.calibrated_parameters == []
    assert m.calibration_strategy.strategy_type == "none"
    assert [t.target_id for t in m.empirical_targets] == ["ky"]
    assert "not calibrated" in m.fit_metrics.fit_assessment.lower()


def test_calibrate_model_targets_only_short_circuits_estimation(monkeypatch):
    """targets_only=True must return after target extraction and NEVER run estimation."""
    from types import SimpleNamespace
    c = _bare_calibrator()
    fm = SimpleNamespace(model_title="M")
    targets = [cs.EmpiricalTarget(target_id="ky", target_name="K/Y", empirical_value=4.0)]
    monkeypatch.setattr(c, "_extract_empirical_targets", lambda m: targets)

    def _boom(*a, **k):
        raise AssertionError("estimation ran despite targets_only=True")

    # any of the estimation steps firing would be a regression
    monkeypatch.setattr(c, "_design_calibration_strategy", _boom)
    monkeypatch.setattr(c, "_calibrate_parameters", _boom)

    m = c.calibrate_model(fm, targets_only=True)
    assert m.metadata.get("targets_only") is True
    assert [t.target_id for t in m.empirical_targets] == ["ky"]


def test_calibrate_model_surfaces_uncalibratable_as_na(monkeypatch):
    """An 'uncalibratable' verdict must read as N/A (with the harness's reason), NOT as a
    fitted 0.0. The final CalibratedModel carries that honestly in summary + metadata."""
    from types import SimpleNamespace
    c = _bare_calibrator()
    fm = SimpleNamespace(model_title="RL Model")
    targets = [cs.EmpiricalTarget(target_id="t", target_name="T", empirical_value=1.0)]
    monkeypatch.setattr(c, "_extract_empirical_targets", lambda m: targets)
    monkeypatch.setattr(c, "_design_calibration_strategy", lambda *a, **k: cs.CalibrationStrategy(
        strategy_type="none", description="", parameters_to_calibrate=[], targets_to_match=[],
        calibration_order=[], identification_notes=""))
    monkeypatch.setattr(c, "_calibrate_parameters", lambda *a, **k: [])
    sv = cs.SandboxValidation(validated=False, verdict="uncalibratable",
                              validation_method="deterministic_moment_harness", stdout="",
                              error="requires_training: model has learned/opaque components (mlp, encoder)")
    fmet = cs.FitMetrics(total_targets=0, targets_matched=0, mean_absolute_error=0.0,
                         mean_relative_error=0.0, rmse=0.0, fit_score=0.0, fit_assessment="status=uncalibratable")
    monkeypatch.setattr(c, "_refine_calibration_loop", lambda *a, **k: ([], sv, [], fmet))
    # a fully-valid CalibratedModel to synthesize (reuse the helper, then reset summary/metadata)
    base = c._targets_only_model(fm, targets).model_copy(
        update={"calibration_summary": "original summary", "metadata": {}})
    monkeypatch.setattr(c, "_synthesize_calibrated_model", lambda *a, **k: base)

    m = c.calibrate_model(fm)
    assert m.metadata.get("calibration_status") == "uncalibratable"
    assert m.metadata.get("fit_score_applicable") is False
    assert m.calibration_summary.startswith("N/A")
    assert "requires_training" in m.metadata.get("uncalibratable_reason", "")


# --- non-scalar calibrated_value must DROP, never fabricate a 0.0 -------------------

def test_coerce_scalar_value_repairs_single_numbers():
    f = cs._coerce_scalar_value
    assert f(0.96) == 0.96
    assert f(3) == 3.0
    assert f("0.96") == 0.96
    assert f("0.96 (annual)") == 0.96                  # one number embedded in prose
    assert f("[0.045]") == 0.045
    assert f("1.5e-3") == 0.0015
    assert f([0.5]) == 0.5                              # single-element list is a scalar
    assert f(-2.0) == -2.0


def test_coerce_scalar_value_drops_non_scalars():
    f = cs._coerce_scalar_value
    # the exact failure from the full run: a vector string -> must be None (drop), NOT 0.0
    assert f("Normalized Vector [0.1, 0.2, 0.15, ...]") is None
    assert f([0.1, 0.2, 0.3]) is None                  # genuine vector
    assert f("0.95 to 0.99") is None                   # a range, not a point
    assert f("0.96 (matched to T1)") is None            # ambiguous (2 numbers) -> drop, don't guess
    assert f("high") is None                            # no number at all
    assert f(None) is None
    assert f(True) is None                              # bool rejected (int subclass)
    assert f({}) is None
