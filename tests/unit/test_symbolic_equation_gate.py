# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""In a live run one model wrote 21/21 equations as English prose — the parsers
refused honestly and the model was ungenerable. The ModelDesign format gate re-prompts
ONCE for symbolic form and keeps whichever set has more '='-bearing equations."""

import importlib.util
import json
from pathlib import Path

import pytest

_STAGE = (Path(__file__).resolve().parent.parent.parent
          / "ModelTeam" / "ael" / "ModeWithWcWithHITL" / "2-ModelDesignStage.py")


@pytest.fixture(scope="module")
def stage():
    spec = importlib.util.spec_from_file_location("model_design_stage_n25", _STAGE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _eq(stage, eid, plain):
    return stage.ModelEquation(
        equation_id=eid, equation_type="behavioral", equation_name=eid,
        equation_latex="", equation_plain=plain, variables_used=[], parameters_used=[],
        interpretation="", derivation="", timing="")


def _designer(stage, llm):
    d = object.__new__(stage.ModelDesigner)
    d.llm = llm
    return d


class _FakeLLM:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def invoke(self, messages, **kw):
        self.calls += 1
        return self.response


def test_mostly_symbolic_set_passes_untouched(stage):
    llm = _FakeLLM("")
    d = _designer(stage, llm)
    eqs = [_eq(stage, "E1", "y = c + i"), _eq(stage, "E2", "c = 0.8*y"),
           _eq(stage, "E3", "Output equals consumption plus investment.")]
    out = d._enforce_symbolic_equations(eqs)
    assert out is eqs and llm.calls == 0     # 2/3 symbolic -> no re-prompt


def test_prose_majority_triggers_one_reprompt_and_adopts_improvement(stage):
    fixed = {"equations": [
        {"equation_id": "E1", "equation_type": "identity", "equation_name": "E1",
         "equation_latex": "", "equation_plain": "y = c + i", "variables_used": [],
         "parameters_used": [], "interpretation": "", "derivation": "", "timing": ""},
        {"equation_id": "E2", "equation_type": "identity", "equation_name": "E2",
         "equation_latex": "", "equation_plain": "c = mpc*y", "variables_used": [],
         "parameters_used": [], "interpretation": "", "derivation": "", "timing": ""},
    ]}
    llm = _FakeLLM(json.dumps(fixed))
    d = _designer(stage, llm)
    eqs = [_eq(stage, "E1", "Output equals consumption plus investment."),
           _eq(stage, "E2", "Consumption equals a fixed fraction of output.")]
    out = d._enforce_symbolic_equations(eqs)
    assert llm.calls == 1
    assert [e.equation_plain for e in out] == ["y = c + i", "c = mpc*y"]


def test_unimproved_reprompt_keeps_original_disclosed(stage):
    llm = _FakeLLM(json.dumps({"equations": []}))   # retry yields nothing usable
    d = _designer(stage, llm)
    eqs = [_eq(stage, "E1", "Marginal utility today equals discounted marginal utility.")]
    out = d._enforce_symbolic_equations(eqs)
    assert llm.calls == 1 and out is eqs             # honest: original kept, parsers judge


def test_reprompt_crash_never_loses_equations(stage):
    class _Boom:
        def invoke(self, *a, **k):
            raise RuntimeError("LLM down")
    d = _designer(stage, _Boom())
    eqs = [_eq(stage, "E1", "Assets tomorrow must equal savings today.")]
    assert d._enforce_symbolic_equations(eqs) is eqs
