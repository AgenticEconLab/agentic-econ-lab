# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""IdeationTeam WithHITL: feedback checkpoints forward context to the committee.

Covers BOTH Wc variants (NoWc and WithWc). Additive grounding — in llm_economist mode the
committee sees the material under review; interactive/auto modes ignore context so
standalone behavior is unchanged.
"""

import importlib.util
from pathlib import Path

import pytest

_MODES = ["ModeNoWcWithHITL", "ModeWithWcWithHITL"]


def _load(mode, stage_file, alias):
    path = Path(__file__).resolve().parents[2] / "IdeationTeam" / "ael" / mode / stage_file
    spec = importlib.util.spec_from_file_location(f"{alias}_{mode}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _recorder(seen):
    def fake_auto_input(prompt, default="", input_type="general", timeout=None, context=None):
        seen.append(context)
        return default
    return fake_auto_input


@pytest.mark.parametrize("mode", _MODES)
def test_sourcing_forwards_context(monkeypatch, mode):
    mod = _load(mode, "1-SourcingStage.py", "_ideation_sourcing")
    seen = []
    monkeypatch.setattr(mod, "auto_input", _recorder(seen))
    orch = mod.MultiAgentOrchestrator(openai_api_key="test", quiet=True)
    ctx = {"papers_under_review": [{"rank": 1, "title": "A paper"}]}
    orch.collect_human_feedback(round_number=1, context=ctx)
    assert len(seen) == 5 and all(c is ctx for c in seen)


@pytest.mark.parametrize("mode", _MODES)
def test_sourcing_context_optional(monkeypatch, mode):
    mod = _load(mode, "1-SourcingStage.py", "_ideation_sourcing_opt")
    seen = []
    monkeypatch.setattr(mod, "auto_input", _recorder(seen))
    orch = mod.MultiAgentOrchestrator(openai_api_key="test", quiet=True)
    orch.collect_human_feedback(round_number=1)  # back-compat: no context
    assert seen and all(c is None for c in seen)


@pytest.mark.parametrize("mode", _MODES)
def test_refinement_forwards_context(monkeypatch, tmp_path, mode):
    monkeypatch.chdir(tmp_path)  # method writes a feedback json to cwd
    mod = _load(mode, "2-RefinementStage.py", "_ideation_refine")
    seen = []
    monkeypatch.setattr(mod, "auto_input", _recorder(seen))
    orch = mod.RefinementOrchestrator(quiet=True)
    ctx = {"concepts": [{"rank": 1, "title": "C"}], "questions": [{"rank": 1, "question": "Q"}]}
    orch.collect_human_feedback(round_number=1, context=ctx)
    assert len(seen) == 5 and all(c is ctx for c in seen)


@pytest.mark.parametrize("mode", _MODES)
def test_integration_forwards_context(monkeypatch, tmp_path, mode):
    monkeypatch.chdir(tmp_path)  # method writes a feedback json to cwd
    mod = _load(mode, "3-IntegrationStage.py", "_ideation_integrate")
    seen = []
    monkeypatch.setattr(mod, "auto_input", _recorder(seen))
    orch = mod.IntegrationOrchestrator(quiet=True)
    ctx = {"prioritized_questions": [{"rank": 1, "question": "Q", "score": 0.9}]}
    orch.collect_integration_feedback(round_number=1, num_questions=3, context=ctx)
    assert len(seen) == 4 and all(c is ctx for c in seen)
