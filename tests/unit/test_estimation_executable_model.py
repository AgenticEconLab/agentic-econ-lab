# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""The EstimationTeam reads the CodeTeam's
executable model. The lead model's module is shown to the Estimator; the output records
which module variables the specification linked its series to, whether those symbols exist
in the module, and their steady-state values when the module solved one.
"""

import json

import pytest

from EstimationTeam.ael.estim_harness import EstimationSpec, model_link
from pipeline.team_runners import run_estimation_team

from tests.unit.test_estimation_pipeline_runner import _upstream

_LEAD = "Behavioral HANK: Cognitive Discounting"


def _executable_model(solved=True):
    gens = [
        {"model_title": "Formal Model: Another candidate", "verdict": "partial",
         "variables": ["C", "Y"]},
        {"model_title": f"Formal Model: {_LEAD}", "verdict": "partial",
         "variables": ["pi", "y", "R", "sym_c___g"]},
    ]
    vals = [{"verdict": "partial", "steady_state": {}},
            {"verdict": "validated" if solved else "partial",
             "steady_state": {"pi": 1.0, "y": 0.8, "R": 1.01} if solved else {}}]
    return {"generation": {"results": gens}, "validation": {"results": vals},
            "experimentation": {"results": []}}


_MODEL_SPEC = {"calibrated_models": [{"model_title": f"Calibrated Model: {_LEAD}"},
                                     {"model_title": "Calibrated Model: Another candidate"}]}
_MODEL_DESIGN = {"formal_models": [{"model_title": f"Formal Model: {_LEAD}", "variables": [
    {"variable_symbol": "\\pi_t", "variable_name": "Gross inflation"},
    {"variable_symbol": "y_{t}", "variable_name": "Output"}]}]}


def _spec(dep_var="pi", reg_var="y"):
    return EstimationSpec(**{
        "dependent": {"name": "outcome", "series_ref": "YSER", "model_variable": dep_var},
        "regressors": [{"name": "driver", "series_ref": "XSER", "model_variable": reg_var}],
    })


def test_selects_the_lead_models_module_by_title():
    m = model_link.select_module(_executable_model(), _MODEL_SPEC, _MODEL_DESIGN)
    assert m["index"] == 1
    assert m["variable_names"] == {"pi": "Gross inflation", "y": "Output"}
    text = model_link.describe_for_prompt(m)
    assert "pi (Gross inflation), steady state 1" in text
    assert "sym_c___g" not in text


def test_record_links_verified_symbols_and_steady_state():
    m = model_link.select_module(_executable_model(), _MODEL_SPEC, _MODEL_DESIGN)
    rec = model_link.record_use(m, _spec(reg_var="k"), code_team_ran=True)
    assert rec["available"] is True
    assert rec["linked_variables"] == {"outcome": "pi"}
    assert rec["unknown_symbols"] == {"driver": "k"}
    assert rec["steady_state_values"] == {"pi": 1.0}
    assert any("absent from the module" in n for n in rec["notes"])


def test_record_without_steady_state_or_code_team():
    m = model_link.select_module(_executable_model(solved=False), _MODEL_SPEC, None)
    rec = model_link.record_use(m, _spec(), code_team_ran=True)
    assert rec["steady_state_values"] == {}
    assert any("no solved steady state" in n for n in rec["notes"])

    off = model_link.record_use(None, _spec(), code_team_ran=False)
    assert off == {"available": False,
                   "reason": "the CodeTeam did not run (optional node disabled)"}


def test_runner_passes_module_to_estimator_and_records_use(monkeypatch, tmp_path):
    from shared.llm import LLMClient
    prompts = []
    spec_json = json.dumps({
        "dependent": {"name": "outcome", "series_ref": "YSER", "model_variable": "pi"},
        "regressors": [{"name": "driver", "series_ref": "XSER", "model_variable": "y"}],
        "hypotheses": [{"name": "driver_positive", "param": "driver", "restriction": ">0"}],
    })

    def invoke(self, messages, **kw):
        prompts.append(messages[-1]["content"])
        return spec_json

    monkeypatch.setattr(LLMClient, "invoke", invoke, raising=True)
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setenv("AEL_HITL_MODE", "auto")
    up = _upstream()
    up.update(model_specification=_MODEL_SPEC, model_design=_MODEL_DESIGN,
              executable_model=_executable_model())

    results = run_estimation_team(up, "ModeNoWcNoHITL", str(tmp_path))["estimation_results"]

    assert any("Executable model (CodeTeam module" in p and "Gross inflation" in p
               for p in prompts)
    use = results["executable_model_use"]
    assert use["model_title"] == f"Formal Model: {_LEAD}"
    assert use["linked_variables"] == {"outcome": "pi", "driver": "y"}
    assert use["steady_state_values"] == {"pi": 1.0, "y": 0.8}
    assert results["outcome"]["verdict"] in ("estimated", "fragile")


def test_runner_without_code_team_records_absence(monkeypatch, tmp_path):
    from tests.unit.test_estimation_pipeline_runner import _SPEC_JSON
    from shared.llm import LLMClient
    prompts = []
    monkeypatch.setattr(LLMClient, "invoke",
                        lambda self, m, **kw: prompts.append(m[-1]["content"]) or _SPEC_JSON)
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setenv("AEL_HITL_MODE", "auto")
    results = run_estimation_team(_upstream(), "ModeNoWcNoHITL",
                                  str(tmp_path))["estimation_results"]
    assert results["executable_model_use"]["available"] is False
    assert not any("Executable model (CodeTeam module" in p for p in prompts)


def test_report_states_the_linked_module_variables():
    from ReportingTeam.ael.report_harness import assemble_report, check_consistency
    from ReportingTeam.ael.report_harness.interpret import interpret_estimation
    from tests.unit.test_report_harness import _estimation_results
    est = _estimation_results()
    m = model_link.select_module(_executable_model(), _MODEL_SPEC, _MODEL_DESIGN)
    est["executable_model_use"] = model_link.record_use(m, _spec(), code_team_ran=True)
    md = assemble_report({}, {}, {}, {}, est, interpret_estimation(est))
    assert ("series linked to module variables: `outcome` -> `pi` (steady state 1), "
            "`driver` -> `y` (steady state 0.8).") in md
    assert check_consistency(md, [est, interpret_estimation(est).model_dump()]).verdict \
        == "consistent"
