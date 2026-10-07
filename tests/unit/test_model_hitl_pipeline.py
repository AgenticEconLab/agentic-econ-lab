# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Control-flow tests for the WithHITL ModelTeam pipeline driver (_run_model_hitl).

Fake stage orchestrators. Model gates each stage; ModelDesign re-runs with the committee's
feedback as a revision (feasibility-loop mechanism) until approved.
"""

import importlib
import json

import pytest

from pipeline import team_runners


def _fakes(counts):
    class _S1:
        def __init__(self, collector=None):
            pass

        def run_theory_pipeline(self, research_questions, literature_batch):
            counts["s1"] += 1

        def save_theory_output(self, file):
            json.dump({"theoretical_frameworks": [{"framework_title": "F"}]}, open(file, "w"))

    class _S2:
        def __init__(self, collector=None):
            pass

        def run_design_pipeline(self, theory_output, revision_request=None):
            counts["s2"] += 1
            counts["revised"] = counts.get("revised", 0) + (1 if revision_request else 0)
            nvars = 4 if revision_request else 2  # revision produces a richer design
            counts["_design"] = {"formal_models": [{"variables": [{}] * nvars, "equations": [{}] * 3}]}

        def save_design_output(self, file):
            json.dump(counts["_design"], open(file, "w"))

    class _S3:
        def __init__(self, collector=None):
            pass

        def run_calibration_pipeline(self, model_design_output, targets_only=False):
            counts["s3"] += 1
            counts["targets_only"] = targets_only

        def save_calibration_output(self, file):
            json.dump({"calibrated_models": [{"empirical_targets": []}]}, open(file, "w"))

    return {
        "1-TheoryStage": type("M", (), {"TheoryStageOrchestrator": _S1}),
        "2-ModelDesignStage": type("M", (), {"ModelDesignOrchestrator": _S2}),
        "3-CalibrationStage": type("M", (), {"CalibrationOrchestrator": _S3}),
    }


def _patch_imports(monkeypatch, mapping):
    real = importlib.import_module
    monkeypatch.setattr(importlib, "import_module",
                        lambda name, *a, **k: mapping[name] if name in mapping else real(name, *a, **k))


def _run(monkeypatch, tmp_path, counts, **kwargs):
    _patch_imports(monkeypatch, _fakes(counts))
    return team_runners.run_model_team(
        upstream_artifacts={"research_questions": {"final_questions": [{"question": "Q1"}]},
                            "literature_review": {"synthesis": "x"}},
        mode="ModeNoWcWithHITL", output_dir=str(tmp_path), **kwargs,
    )


class TestModelHitlDriver:
    def test_auto_mode_runs_each_stage_once(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AEL_HITL_MODE", "auto")
        counts = {"s1": 0, "s2": 0, "s3": 0}
        out = _run(monkeypatch, tmp_path, counts)
        assert counts["s1"] == 1 and counts["s2"] == 1 and counts["s3"] == 1
        assert counts.get("revised", 0) == 0
        assert set(out) == {"model_specification", "model_design"}
        assert out["model_design"]["formal_models"]

    def test_committee_rejects_design_then_revises(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AEL_HITL_MODE", "llm_economist")
        from shared.llm_economist import PropositionResult, FeedbackResult

        class _FakeCommittee:
            def __init__(self, *a, **k):
                pass

            def vote_proposition(self, question, context=None):
                # approve the design only once it has >= 3 variables; other gates always pass
                if "model design" in question.lower():
                    passed = (context or {}).get("n_variables", 0) >= 3
                else:
                    passed = True
                return PropositionResult(question=question, passed=passed)

            def synthesize_feedback(self, question, context=None):
                return FeedbackResult(feedback="add structural variables to identify the mechanism", drafts=[])

        monkeypatch.setattr("shared.llm_economist.LLMEconomistCommittee", _FakeCommittee)
        counts = {"s1": 0, "s2": 0, "s3": 0}
        out = _run(monkeypatch, tmp_path, counts)
        # design ran twice: initial (2 vars, rejected) -> revised (4 vars, approved)
        assert counts["s2"] == 2 and counts["revised"] == 1
        assert counts["s1"] == 1 and counts["s3"] == 1
        assert out["model_design"]["formal_models"][0]["variables"]  # revised design kept

    def test_drs_only_threads_targets_only_and_returns_early(self, monkeypatch, tmp_path):
        """drs_only=True runs Calibration in targets-only mode and skips the calibration
        review gate (the DRS cycle only needs variables + targets)."""
        monkeypatch.setenv("AEL_HITL_MODE", "auto")
        counts = {"s1": 0, "s2": 0, "s3": 0}
        out = _run(monkeypatch, tmp_path, counts, drs_only=True)
        assert counts["targets_only"] is True          # estimation skipped
        assert set(out) == {"model_specification", "model_design"}

    def test_drs_only_reuses_theory_and_single_design_round(self, monkeypatch, tmp_path):
        """A feasibility cycle (drs_only + MRR + existing theory) must NOT regenerate theory and
        must apply the MRR in ONE design round with no committee re-refinement — even if the committee
        would reject everything."""
        monkeypatch.setenv("AEL_HITL_MODE", "llm_economist")
        from shared.llm_economist import PropositionResult, FeedbackResult

        class _RejectAll:
            def __init__(self, *a, **k):
                pass

            def vote_proposition(self, question, context=None):
                return PropositionResult(question=question, passed=False)   # reject everything

            def synthesize_feedback(self, question, context=None):
                return FeedbackResult(feedback="change it", drafts=[])

        monkeypatch.setattr("shared.llm_economist.LLMEconomistCommittee", _RejectAll)
        # a prior theory exists (from the linear pass) -> the drs_only cycle should reuse it
        (tmp_path / "theory_output.json").write_text(
            json.dumps({"theoretical_frameworks": [{"framework_title": "F"}]}))

        counts = {"s1": 0, "s2": 0, "s3": 0}
        _patch_imports(monkeypatch, _fakes(counts))
        out = team_runners.run_model_team(
            upstream_artifacts={"research_questions": {"final_questions": [{"question": "Q"}]},
                                "literature_review": {"synthesis": "x"},
                                "model_revision_request": {"reason": "drop X", "requested_change": "drop X"}},
            mode="ModeNoWcWithHITL", output_dir=str(tmp_path), drs_only=True)

        assert counts["s1"] == 0                        # theory REUSED, not regenerated
        assert counts["s2"] == 1                        # design ran ONCE (no committee refinement)
        assert counts.get("targets_only") is True
        assert set(out) == {"model_specification", "model_design"}


class TestCrossModeDispatch:
    """run_model_team must work in ALL FOUR modes — WithHITL modes route through the
    committee driver, NoHITL modes through the automated branch; every mode dir must exist."""

    @pytest.mark.parametrize("mode", ["ModeNoWcNoHITL", "ModeNoWcWithHITL",
                                      "ModeWithWcNoHITL", "ModeWithWcWithHITL"])
    def test_all_four_modes_dispatch_and_produce_artifacts(self, monkeypatch, tmp_path, mode):
        monkeypatch.setenv("AEL_HITL_MODE", "auto")
        counts = {"s1": 0, "s2": 0, "s3": 0}
        _patch_imports(monkeypatch, _fakes(counts))
        out = team_runners.run_model_team(
            upstream_artifacts={"research_questions": {"final_questions": [{"question": "Q"}]},
                                "literature_review": {"synthesis": "x"}},
            mode=mode, output_dir=str(tmp_path))
        assert counts["s1"] == 1 and counts["s2"] == 1 and counts["s3"] == 1
        assert set(out) == {"model_specification", "model_design"}
