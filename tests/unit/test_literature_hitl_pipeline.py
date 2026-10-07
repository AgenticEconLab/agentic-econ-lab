# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Control-flow tests for the WithHITL LiteratureTeam pipeline driver (_run_literature_hitl).

Fake stage orchestrators (real flow needs a cluster run). Literature runs each stage once
in auto mode; in llm_economist mode the committee gates gathering (re-gather on reject).
"""

import importlib
import json

import pytest

from pipeline import team_runners


def _fakes(counts):
    class _RQ:
        def __init__(self, **kw):
            pass

    class _S1:
        def __init__(self, collector=None):
            pass

        def run_gathering_pipeline(self, research_questions, max_papers_per_question):
            counts["s1"] += 1
            counts["max_papers"] = max_papers_per_question
            return {"literature_items": [{"title": "P"}] * 3}

        def save_literature_batch(self, file):
            json.dump({"literature_items": [{"title": "P"}] * 3}, open(file, "w"))

    class _S2:
        def __init__(self, collector=None):
            pass

        def run_gap_detection_pipeline(self, batch):
            counts["s2"] += 1
            return {"gaps": [{"gap": "g1"}]}

        def save_gap_analysis(self, file):
            json.dump({"gaps": [{"gap": "g1"}]}, open(file, "w"))

        def save_graph_json(self, file):
            json.dump({"nodes": []}, open(file, "w"))

    class _S3:
        def __init__(self, collector=None):
            pass

        def load_literature_metadata(self, filepath):
            counts["metadata_from"] = filepath
            return {}

        def run_synthesis_pipeline(self, gap, literature_metadata=None):
            counts["s3"] += 1
            return {"review": "ok"}

        def save_literature_review(self, file):
            open(file, "w").write("review")

        def save_synthesis_result(self, file):
            json.dump({"review": "ok"}, open(file, "w"))

    return {
        "1-LiteratureGatheringStage": type("M", (), {"ResearchQuestion": _RQ,
                                                     "LiteratureGatheringOrchestrator": _S1}),
        "2-GapDetectionStage": type("M", (), {"GapDetectionOrchestrator": _S2}),
        "3-SynthesisStage": type("M", (), {"SynthesisOrchestrator": _S3}),
    }


def _patch_imports(monkeypatch, mapping):
    real = importlib.import_module
    monkeypatch.setattr(importlib, "import_module",
                        lambda name, *a, **k: mapping[name] if name in mapping else real(name, *a, **k))


def _run(monkeypatch, tmp_path, counts):
    _patch_imports(monkeypatch, _fakes(counts))
    return team_runners.run_literature_team(
        upstream_artifacts={"research_questions": {"final_questions": [{"question": "Q1"}]}},
        mode="ModeNoWcWithHITL", output_dir=str(tmp_path),
    )


class TestLiteratureHitlDriver:
    def test_auto_mode_runs_each_stage_once(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AEL_HITL_MODE", "auto")
        monkeypatch.delenv("AEL_HITL_LOG", raising=False)
        counts = {"s1": 0, "s2": 0, "s3": 0}
        out = _run(monkeypatch, tmp_path, counts)
        # synthesis metadata is loaded from THIS run's batch file
        assert counts.pop("metadata_from") == str(tmp_path / "literature_batch.json")
        assert counts == {"s1": 1, "s2": 1, "s3": 1, "max_papers": 15}
        # ballots of this team go to the run directory, and the env is restored after
        import os
        assert "AEL_HITL_LOG" not in os.environ
        assert set(out) == {"literature_review", "gap_analysis", "knowledge_graph"}
        assert out["gap_analysis"]["gaps"] and out["literature_review"]["review"] == "ok"

    def test_committee_rejects_then_regathers_more(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AEL_HITL_MODE", "llm_economist")
        from shared.llm_economist import PropositionResult

        class _FakeCommittee:
            def __init__(self, *a, **k):
                pass

            def vote_proposition(self, question, context=None):
                # approve gathering only once it has escalated past the initial 15 papers
                passed = not ("adequate to proceed" in question and (context or {}).get("max_papers", 0) <= 15)
                return PropositionResult(question=question, passed=passed)

        monkeypatch.setattr("shared.llm_economist.LLMEconomistCommittee", _FakeCommittee)
        counts = {"s1": 0, "s2": 0, "s3": 0}
        out = _run(monkeypatch, tmp_path, counts)
        assert counts["s1"] == 2 and counts["max_papers"] > 15  # re-gathered with more papers
        assert counts["s2"] == 1 and counts["s3"] == 1
        assert set(out) == {"literature_review", "gap_analysis", "knowledge_graph"}
