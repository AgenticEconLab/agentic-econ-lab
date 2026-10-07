# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""The feasibility loop's
revision requests accumulate across cycles, reach the equation step and the committee rounds,
and the final design is re-graded."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.research_pipeline import ResearchPipelineOrchestrator, merge_revision_requests

_AGENTS = Path(__file__).resolve().parents[2]


class _M:
    def __init__(self, **kw):
        self.kw = kw

    def model_dump(self):
        return dict(self.kw)


def test_merge_keeps_every_cycle():
    m = merge_revision_requests([
        _M(reason="r1", requested_change="drop A", unmet_requirements=["var:A"], cycle=1),
        _M(reason="r2", requested_change="drop N", unmet_requirements=["var:N", "var:A"], cycle=2)])
    assert m["unmet_requirements"] == ["var:A", "var:N"]
    assert "drop A" in m["requested_change"] and "drop N" in m["requested_change"]
    assert m["provenance"]["merged_cycles"] == [1, 2]
    assert merge_revision_requests([]) == {}


def test_equation_step_sees_the_revision_directive(monkeypatch):
    pytest.importorskip("httpx")
    stage = _AGENTS / "ModelTeam" / "ael" / "ModeWithWcWithHITL" / "2-ModelDesignStage.py"
    spec = importlib.util.spec_from_file_location("design_stage_c6", stage)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    prompts = []
    designer = mod.ModelDesigner.__new__(mod.ModelDesigner)
    designer.llm = SimpleNamespace(invoke=lambda msgs, **k: prompts.append(msgs[-1]["content"]) or '{"equations": []}')
    designer.agent_name = "ModelDesigner"
    fw = SimpleNamespace(framework_title="F", theoretical_approach="a", model_structure="s",
                         mathematical_formulations=[])
    designer._formulate_equations(fw, [], [], "REVISION DIRECTIVE: drop A")
    assert prompts and prompts[0].startswith("REVISION DIRECTIVE: drop A")
    assert "Use only the variables listed below" in prompts[0]
    designer._formulate_equations(fw, [], [])
    assert prompts[1].startswith("Formulate a complete equation system")


def test_report_uses_the_regraded_final_design(tmp_path):
    orch = ResearchPipelineOrchestrator.__new__(ResearchPipelineOrchestrator)
    registered = {}
    orch.artifact_store = SimpleNamespace(register=lambda n, d, producer="": registered.__setitem__(n, d))
    orch.output_dir = str(tmp_path)
    loop_dar, final_dar = _M(stage="loop"), _M(stage="final")
    orch._final_regrade = (_M(drs="final"), final_dar)
    lr = SimpleNamespace(status="max_cycles", cycles=2, final_drs=_M(drs="loop"), final_dar=loop_dar,
                         decisions=[], mrrs=[], history=[])
    orch._persist_feasibility(lr)
    assert registered["data_availability_report"] == {"stage": "final"}
    rep = json.loads((tmp_path / "feasibility_report.json").read_text())
    assert rep["final_design_regraded"] is True
    assert rep["loop_data_availability_report"] == {"stage": "loop"}


@pytest.mark.parametrize("unmet,feasible,expected", [
    (["var:N"], False, "unresolved_after_final_design"),
    ([], True, "feasible"),
])
def test_terminal_status_follows_the_final_regrade(tmp_path, unmet, feasible, expected):
    orch = ResearchPipelineOrchestrator.__new__(ResearchPipelineOrchestrator)
    orch.artifact_store = SimpleNamespace(register=lambda *a, **k: None)
    orch.output_dir = str(tmp_path)
    final_dar = _M(stage="final")
    final_dar.unmet_essential, final_dar.overall_feasible = unmet, feasible
    orch._final_regrade = (_M(drs="final"), final_dar)
    lr = SimpleNamespace(status="max_cycles", cycles=2, final_drs=_M(), final_dar=_M(),
                         decisions=[], mrrs=[], history=[])
    orch._persist_feasibility(lr)
    rep = json.loads((tmp_path / "feasibility_report.json").read_text())
    assert rep["status"] == expected and rep["loop_status"] == "max_cycles"
