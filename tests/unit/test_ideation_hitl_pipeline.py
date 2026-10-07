# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Control-flow tests for the WithHITL IdeationTeam pipeline driver (_run_ideation_hitl).

Uses fake stage orchestrators (real LLM flow needs a cluster smoke run). Verifies the
develop-until-approved loop: auto mode -> the team's 2-round design; llm_economist mode
-> the committee gates the rounds.
"""

import importlib
import json
import types
from pathlib import Path

import pytest

from pipeline import team_runners


def _fakes(counts):
    class _S1:
        def __init__(self, quiet=True, collector=None):
            pass

        def run_search_round(self, research_topic, round_number, feedback, max_results_per_agent):
            counts["s1"] += 1
            return [types.SimpleNamespace(title="P", source="arxiv")]

        def collect_human_feedback(self, round_number, context=None):
            counts["fb"] += 1
            return {"round": round_number}

        def save_results(self, file):
            open(file, "w").write("csv")

    class _S2:
        def __init__(self, quiet=True, collector=None):
            pass

        def load_literature(self, file):
            return "df"

        def run_refinement_round(self, literature_df, round_number, feedback, num_concepts, num_questions):
            counts["s2"] += 1
            return ([types.SimpleNamespace(concept_title="C")], [types.SimpleNamespace(question="Q")])

        def collect_human_feedback(self, round_number, context=None):
            counts["fb"] += 1
            return {"round": round_number}

        def save_results(self, file):
            open(file, "w").write("{}")

    class _S3:
        def __init__(self, quiet=True, collector=None):
            pass

        def load_refinement_results(self, file):
            return ["q"]

        def run_integration_round(self, questions, round_number, feedback, max_final_questions):
            counts["s3"] += 1
            return ([], [types.SimpleNamespace(question="Final Q?")])

        def collect_integration_feedback(self, round_number, num_questions, context=None):
            counts["fb"] += 1
            return {"round": round_number}

        def convert_prioritized_to_research_questions(self, prioritized):
            return ["q"]

        def save_final_questions(self, file):
            json.dump({"final_questions": [{"question": "Final Q?"}]}, open(file, "w"))

    return {
        "1-SourcingStage": types.SimpleNamespace(MultiAgentOrchestrator=_S1),
        "2-RefinementStage": types.SimpleNamespace(RefinementOrchestrator=_S2),
        "3-IntegrationStage": types.SimpleNamespace(IntegrationOrchestrator=_S3),
    }


def _patch_imports(monkeypatch, mapping):
    real = importlib.import_module

    def fake(name, *a, **k):
        return mapping[name] if name in mapping else real(name, *a, **k)

    monkeypatch.setattr(importlib, "import_module", fake)


def _run(monkeypatch, tmp_path, counts):
    _patch_imports(monkeypatch, _fakes(counts))
    return team_runners.run_ideation_team(
        upstream_artifacts={"research_topic": {"research_topic": "AI in macro"}},
        mode="ModeNoWcWithHITL", output_dir=str(tmp_path),
    )


def test_team_context_registers_search_tools(tmp_path):
    # Regression: the pipeline imports stage modules directly (not the standalone
    # orchestrators), so it must register the ToolRegistry itself — else searches return 0.
    from shared.tools.tool_registry import ToolRegistry

    with team_runners._team_context(tmp_path):
        assert ToolRegistry.get_tool("openalex_search") is not None
        assert ToolRegistry.get_tool("arxiv_search") is not None


class TestIdeationHitlDriver:
    def test_auto_mode_runs_two_rounds_per_stage(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AEL_HITL_MODE", "auto")
        counts = {"s1": 0, "s2": 0, "s3": 0, "fb": 0}
        out = _run(monkeypatch, tmp_path, counts)
        assert out["research_questions"]["final_questions"][0]["question"] == "Final Q?"
        # team's standard 2-round design in auto mode
        assert counts["s1"] == 2 and counts["s2"] == 2 and counts["s3"] == 2
        assert counts["fb"] == 3  # one feedback collection per stage

    def test_committee_approval_stops_after_one_round(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AEL_HITL_MODE", "llm_economist")

        from shared.llm_economist import PropositionResult

        class _FakeCommittee:
            def __init__(self, *a, **k):
                pass

            def vote_proposition(self, question, context=None):
                return PropositionResult(question=question, passed=True)  # approve immediately

        monkeypatch.setattr("shared.llm_economist.LLMEconomistCommittee", _FakeCommittee)
        counts = {"s1": 0, "s2": 0, "s3": 0, "fb": 0}
        out = _run(monkeypatch, tmp_path, counts)
        assert out["research_questions"]["final_questions"]
        # committee approves round 1 -> no refinement round, no feedback collected
        assert counts["s1"] == 1 and counts["s2"] == 1 and counts["s3"] == 1
        assert counts["fb"] == 0


def test_diversity_floor_runs_extra_integration_round(monkeypatch, tmp_path):
    """When the committee's gates thin the finals below 3 while the refined pool is larger,
    ONE extra integration round runs over the full pool (seeded with committee critique) and its
    broader result is what gets saved."""
    monkeypatch.setenv("AEL_HITL_MODE", "auto")
    monkeypatch.setenv("AEL_HITL_MAX_ROUNDS", "2")          # normal rounds: 1 + 1 objection
    counts = {"s1": 0, "s2": 0, "s3": 0, "fb": 0}
    mapping = _fakes(counts)

    class _ThinS3:
        def __init__(self, quiet=True, collector=None):
            pass

        def load_refinement_results(self, file):
            return [f"q{i}" for i in range(6)]              # a 6-question refined pool

        def run_integration_round(self, questions, round_number, feedback, max_final_questions):
            counts["s3"] += 1
            counts["last_round"] = round_number
            # REAL contract (a bare string once crashed a live run): feedback must be None or a structured
            # IntegrationFeedback with additional_guidance — never a bare string.
            assert feedback is None or hasattr(feedback, "additional_guidance"), \
                f"feedback must be IntegrationFeedback-shaped, got {type(feedback).__name__}"
            if feedback is not None:
                counts["guidance"] = str(getattr(feedback, "additional_guidance", ""))
            n = 4 if round_number > 2 else 2                # thin in normal rounds; the extra round broadens
            self._last = [types.SimpleNamespace(question=f"Q{i}?") for i in range(n)]
            return ([], self._last)

        def collect_integration_feedback(self, round_number, num_questions, context=None):
            # mirror the REAL return type (a structured feedback, not a dict/str)
            return types.SimpleNamespace(questions_to_prioritize=[], additional_guidance="")

        def convert_prioritized_to_research_questions(self, prioritized):
            return ["q"]

        def save_final_questions(self, file):
            json.dump({"final_questions": [{"question": q.question} for q in self._last]},
                      open(file, "w"))

    mapping["3-IntegrationStage"] = types.SimpleNamespace(IntegrationOrchestrator=_ThinS3)
    _patch_imports(monkeypatch, mapping)
    out = team_runners.run_ideation_team(
        upstream_artifacts={"research_topic": {"research_topic": "AI in macro"}},
        mode="ModeNoWcWithHITL", output_dir=str(tmp_path))
    # normal rounds (1 + objection round 2, both thin) + the diversity-floor extra round (broad)
    assert counts["s3"] == 3 and counts["last_round"] == 3
    assert len(out["research_questions"]["final_questions"]) == 4   # broadened set saved
    assert "diverse" in counts.get("guidance", "").lower()  # critique rode in additional_guidance
