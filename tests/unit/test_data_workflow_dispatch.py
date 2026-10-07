# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""run_data_team must drive ALL THREE DataTeam workflows, not just open_source_api.

The three workflows expose the same orchestrator names but workflow-specific stage-1
signatures — premium takes budget/credentials, user_uploaded REQUIRES file_path. Blindly
passing the open-source kwargs raised TypeError the moment premium/user_uploaded ran
(caught statically by test_mode_copy_consistency; fixed via signature-aware kwarg
filtering). These tests drive the REAL run_data_team with fake stage modules whose
signatures mirror each workflow, and assert each receives exactly what it can accept."""

import importlib
import json

import pytest

from pipeline import team_runners


def _mk_fakes(captured, source_sig):
    """Fake stage modules; the stage-1 run_source_pipeline signature is workflow-specific."""

    class _S1:
        def __init__(self, collector=None):
            pass

        run_source_pipeline = source_sig(captured)

        def save_source_output(self, file):
            json.dump({"retrieved_data": []}, open(file, "w"))

    class _S2:
        def __init__(self, collector=None):
            pass

        def run_cleaning_pipeline(self, data_source_output, research_question, enable_hitl=True):
            captured["cleaning_ran"] = True

        def save_cleaning_output(self, file):
            json.dump({"cleaned": True}, open(file, "w"))

    class _S3:
        def __init__(self, collector=None):
            pass

        def run_qa_pipeline(self, data_cleaning_output, research_question, enable_hitl=True):
            captured["qa_ran"] = True

        def save_qa_output(self, file):
            json.dump({"documented_datasets": [{"name": "d"}]}, open(file, "w"))

    return {
        "1-DataSourceStage": type("M", (), {"DataSourceOrchestrator": _S1}),
        "2-DataCleaningStage": type("M", (), {"DataCleaningOrchestrator": _S2}),
        "3-QualityAssuranceStage": type("M", (), {"QualityAssuranceOrchestrator": _S3}),
    }


def _run(monkeypatch, tmp_path, mode, source_sig):
    captured = {}
    mapping = _mk_fakes(captured, source_sig)
    real = importlib.import_module
    monkeypatch.setattr(importlib, "import_module",
                        lambda name, *a, **k: mapping[name] if name in mapping else real(name, *a, **k))
    monkeypatch.setenv("AEL_HITL_MODE", "auto")
    out = team_runners.run_data_team(
        upstream_artifacts={"research_questions": {"final_questions": [{"question": "Q"}]}},
        mode=mode, output_dir=str(tmp_path),
    )
    return captured, out


def test_open_source_api_receives_full_kwargs(monkeypatch, tmp_path):
    def sig(captured):
        def run_source_pipeline(self, research_question, data_requirements=None,
                                available_apis=None, enable_hitl=True):
            captured["kwargs"] = dict(research_question=research_question,
                                      data_requirements=data_requirements,
                                      available_apis=available_apis)
        return run_source_pipeline

    captured, out = _run(monkeypatch, tmp_path, "open_source_api", sig)
    assert captured["kwargs"]["available_apis"]            # api list delivered
    assert captured["qa_ran"] and "validated_dataset" in out


def test_premium_subscribed_gets_only_accepted_kwargs(monkeypatch, tmp_path):
    """Premium's signature (research_question, budget_limit, credentials_status, enable_hitl):
    the runner must NOT pass data_requirements/available_apis (was an instant TypeError)."""
    def sig(captured):
        def run_source_pipeline(self, research_question, budget_limit=None,
                                credentials_status=None, enable_hitl=True):
            captured["research_question"] = research_question
        return run_source_pipeline

    captured, out = _run(monkeypatch, tmp_path, "premium_subscribed", sig)
    assert captured["research_question"] == "Q"
    assert captured["qa_ran"] and "validated_dataset" in out


def test_user_uploaded_receives_file_path(monkeypatch, tmp_path):
    """user_uploaded REQUIRES file_path — the runner must supply one (env override or the
    bundled sample) instead of crashing on a missing positional argument."""
    def sig(captured):
        def run_source_pipeline(self, file_path, research_question="", enable_hitl=True):
            captured["file_path"] = file_path
        return run_source_pipeline

    monkeypatch.delenv("AEL_USER_DATA_PATH", raising=False)
    captured, out = _run(monkeypatch, tmp_path, "user_uploaded", sig)
    assert captured["file_path"].endswith("example_data.csv")   # bundled sample fallback
    assert captured["qa_ran"] and "validated_dataset" in out


def test_user_uploaded_env_override_wins(monkeypatch, tmp_path):
    def sig(captured):
        def run_source_pipeline(self, file_path, research_question="", enable_hitl=True):
            captured["file_path"] = file_path
        return run_source_pipeline

    monkeypatch.setenv("AEL_USER_DATA_PATH", "/data/my_panel.csv")
    captured, _ = _run(monkeypatch, tmp_path, "user_uploaded", sig)
    assert captured["file_path"] == "/data/my_panel.csv"

def test_supplemental_series_delivered_as_list(monkeypatch, tmp_path):
    """The runner passes supplemental_series through as a plain list — whether the
    artifact arrived as the registered {"series": [...]} envelope or the bare list the
    orchestrator hands over via extra_artifacts."""
    def sig(captured):
        def run_source_pipeline(self, research_question, data_requirements=None,
                                available_apis=None, enable_hitl=True,
                                supplemental_series=None):
            captured["supp"] = supplemental_series
        return run_source_pipeline

    supp = [{"variable_name": "Government Spending", "source": "FRED", "series_id": "GCE"}]
    for form in (supp, {"series": supp}):
        captured = {}
        mapping = _mk_fakes(captured, sig)
        real = importlib.import_module
        monkeypatch.setattr(importlib, "import_module",
                            lambda name, *a, **k: mapping[name] if name in mapping else real(name, *a, **k))
        monkeypatch.setenv("AEL_HITL_MODE", "auto")
        team_runners.run_data_team(
            upstream_artifacts={"research_questions": {"final_questions": [{"question": "Q"}]},
                                "supplemental_series": form},
            mode="open_source_api", output_dir=str(tmp_path),
        )
        assert captured["supp"] == supp, f"failed for artifact form: {type(form).__name__}"
