# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""CodeTeam stages end-to-end, offline: fake LLM narration, real derivation + execution.

A Solow system flows Generation -> Validation -> Experimentation through the persisted JSON
outputs; the steady state must match the closed form and the ungenerable path must flow
honestly through all three stages."""

import importlib.util
import json
from pathlib import Path

import pytest

_MODES = ["ModeNoWcNoHITL", "ModeNoWcWithHITL"]


def _load(mode, stage_file, alias):
    path = Path(__file__).resolve().parents[2] / "CodeTeam" / "ael" / mode / stage_file
    spec = importlib.util.spec_from_file_location(f"{alias}_{mode}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _eq(eid, plain, variables, parameters):
    return {"equation_id": eid, "equation_plain": plain,
            "variables_used": variables, "parameters_used": parameters}


def _design():
    return {"formal_models": [{
        "model_title": "Solow steady state",
        "equations": [
            _eq("E1", "y = k**alpha", ["y", "k"], ["alpha"]),
            _eq("E2", "k = s*y/delta", ["k", "y"], ["s", "delta"]),
        ],
        "parameters": [{"parameter_symbol": "alpha"}, {"parameter_symbol": "s"},
                       {"parameter_symbol": "delta"}],
        "variables": [{"variable_symbol": "y"}, {"variable_symbol": "k"}],
    }]}


def _calibration():
    return {"calibrated_models": [{
        "model_title": "Solow steady state",
        "calibrated_parameters": [
            {"parameter_symbol": "alpha", "calibrated_value": 0.33},
            {"parameter_symbol": "s", "calibrated_value": 0.2},
            {"parameter_symbol": "delta", "calibrated_value": 0.1},
        ]}]}


def _patch_llm(monkeypatch, text="narration"):
    from shared.llm import LLMClient
    monkeypatch.setattr(LLMClient, "invoke",
                        lambda self, messages, **kw: text, raising=True)


@pytest.mark.parametrize("mode", _MODES)
def test_full_stage_chain_offline(monkeypatch, tmp_path, mode):
    monkeypatch.chdir(tmp_path)
    _patch_llm(monkeypatch)

    s1 = _load(mode, "1-CodeGenerationStage.py", "_code_s1")
    orch1 = s1.CodeGenerationOrchestrator(quiet=True)
    results1 = orch1.run_generation_pipeline(_design(), _calibration(),
                                             modules_dir=str(tmp_path / "generated_models"))
    assert results1[0].verdict == "generated"
    gen_data = json.loads(Path(orch1.save_generation_output()).read_text())
    assert (tmp_path / "generated_models" / "model_1.py").exists()

    s2 = _load(mode, "2-ValidationStage.py", "_code_s2")
    orch2 = s2.CodeValidationOrchestrator(quiet=True)
    results2 = orch2.run_validation_pipeline(gen_data,
                                             workdir=str(tmp_path / "generated_models"))
    assert results2[0].verdict == "validated"
    k_star = (0.2 / 0.1) ** (1.0 / (1.0 - 0.33))
    assert results2[0].steady_state["k"] == pytest.approx(k_star, abs=1e-6)
    val_data = json.loads(Path(orch2.save_validation_output()).read_text())

    s3 = _load(mode, "3-ExperimentationStage.py", "_code_s3")
    orch3 = s3.ExperimentationOrchestrator(quiet=True)
    results3 = orch3.run_experimentation_pipeline(gen_data, val_data,
                                                  workdir=str(tmp_path / "generated_models"))
    assert results3[0].verdict == "completed"
    assert len(results3[0].statics) == 6  # 3 params x (+/-)
    orch3.save_experimentation_output()
    assert (tmp_path / "experimentation_output.json").exists()


@pytest.mark.parametrize("mode", _MODES)
def test_ungenerable_flows_honestly(monkeypatch, tmp_path, mode):
    monkeypatch.chdir(tmp_path)
    _patch_llm(monkeypatch)
    design = {"formal_models": [{"model_title": "opaque",
                                 "equations": [_eq("E1", "no equation here", ["x"], [])],
                                 "parameters": [], "variables": []}]}

    s1 = _load(mode, "1-CodeGenerationStage.py", "_code_s1u")
    orch1 = s1.CodeGenerationOrchestrator(quiet=True)
    results1 = orch1.run_generation_pipeline(design, None,
                                             modules_dir=str(tmp_path / "generated_models"))
    assert results1[0].verdict == "ungenerable"
    gen_data = json.loads(Path(orch1.save_generation_output()).read_text())

    s2 = _load(mode, "2-ValidationStage.py", "_code_s2u")
    orch2 = s2.CodeValidationOrchestrator(quiet=True)
    results2 = orch2.run_validation_pipeline(gen_data,
                                             workdir=str(tmp_path / "generated_models"))
    assert results2[0].verdict == "not_applicable"
    val_data = json.loads(Path(orch2.save_validation_output()).read_text())

    s3 = _load(mode, "3-ExperimentationStage.py", "_code_s3u")
    orch3 = s3.ExperimentationOrchestrator(quiet=True)
    results3 = orch3.run_experimentation_pipeline(gen_data, val_data,
                                                  workdir=str(tmp_path / "generated_models"))
    assert results3[0].verdict == "not_applicable"
