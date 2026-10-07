# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Data requirements must reach retrieval (in the 21 v0.7.1 runs they never did).

The DataTeam runner read model_specification["data_requirements"], a key no team writes; in
21/21 runs the source stage ran with an empty requirement list."""

import importlib
import json
from types import SimpleNamespace

from DataTeam.ael.feasibility.adapters import requirements_for_retrieval
from DataTeam.ael.schemas.stage_outputs import DataRequirement

_DRS = {"research_question": "RQ", "cycle": 1, "source_model": "M", "requirements": [
    {"requirement_id": "var:pi", "variable_name": "Inflation Rate", "description": "CPI inflation",
     "frequency": "unspecified", "time_period": "unspecified", "geographic_coverage": "US",
     "unit_of_measurement": "pct", "priority": "High", "suggested_sources": []},
    {"requirement_id": "var:R", "variable_name": "Regime State",
     "description": "[derived statistic: computed by the pipeline from base series] index",
     "frequency": "unspecified", "time_period": "unspecified", "geographic_coverage": "x",
     "unit_of_measurement": "x", "priority": "Medium", "suggested_sources": ["pipeline-computed"]},
]}

_DESIGN = {"formal_models": [{"model_title": "M", "variables": [
    {"variable_symbol": "g", "variable_name": "Government Spending", "variable_type": "exogenous",
     "description": "federal purchases"},
    {"variable_symbol": "eps", "variable_name": "Demand Shock", "variable_type": "exogenous",
     "description": "iid shock"},
]}]}


def test_drs_artifact_is_used_and_untaggable_rows_dropped():
    reqs = requirements_for_retrieval({"data_requirements_spec": _DRS}, "RQ")
    assert [r.variable_name for r in reqs] == ["Inflation Rate"]
    assert isinstance(reqs[0], DataRequirement)


def test_artifact_envelope_is_unwrapped():
    env = {"name": "data_requirements_spec", "data": _DRS, "producer": "ModelTeam"}
    assert [r.variable_name for r in requirements_for_retrieval({"data_requirements_spec": env})] \
        == ["Inflation Rate"]


def test_falls_back_to_drs_from_model_artifacts():
    reqs = requirements_for_retrieval(
        {"model_design": _DESIGN, "model_specification": {"calibrated_models": []}}, "RQ")
    assert [r.variable_name for r in reqs] == ["Government Spending"]   # the shock is latent


def test_runner_passes_requirements_to_source_stage(monkeypatch, tmp_path):
    from pipeline import team_runners as tr
    captured = {}

    class _O1:
        def __init__(self, collector=None):
            pass

        def run_source_pipeline(self, research_question, data_requirements, available_apis,
                                enable_hitl=True, supplemental_series=None):
            captured["reqs"] = data_requirements

        def save_source_output(self, f):
            open(f, "w").write("{}")

    class _O2:
        def __init__(self, collector=None):
            pass

        def run_cleaning_pipeline(self, **k):
            pass

        def save_cleaning_output(self, f):
            open(f, "w").write("{}")

    class _O3(_O2):
        def run_qa_pipeline(self, **k):
            pass

        def save_qa_output(self, f):
            open(f, "w").write(json.dumps({"ok": True}))

    fakes = {"1-DataSourceStage": SimpleNamespace(DataSourceOrchestrator=_O1),
             "2-DataCleaningStage": SimpleNamespace(DataCleaningOrchestrator=_O2),
             "3-QualityAssuranceStage": SimpleNamespace(QualityAssuranceOrchestrator=_O3)}
    real = importlib.import_module
    monkeypatch.setattr(importlib, "import_module",
                        lambda name, *a, **k: fakes[name] if name in fakes else real(name, *a, **k))
    tr.run_data_team({"research_questions": {"final_questions": [{"question": "RQ"}]},
                      "data_requirements_spec": _DRS}, "open_source_api", str(tmp_path))
    assert [r.variable_name for r in captured["reqs"]] == ["Inflation Rate"]


def test_pipeline_config_feeds_drs_inputs_to_datateam():
    import run_ael_pipeline as rp
    data = next(s for s in rp.build_default_config().stages if s.team == "DataTeam")
    assert {"model_design", "data_requirements_spec"} <= set(data.inputs)
