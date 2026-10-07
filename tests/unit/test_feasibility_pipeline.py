# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Integration tests for the feasibility loop wired inside ResearchPipelineOrchestrator (b1)."""

import json
import os

import pytest

from pipeline.research_pipeline import ResearchPipelineOrchestrator, _TEAM_RUNNERS
from pipeline.pipeline_config import PipelineConfig, StageConfig, BudgetConfig


@pytest.fixture(autouse=True)
def clean_runners():
    _TEAM_RUNNERS.clear()
    yield
    _TEAM_RUNNERS.clear()


def _config(tmp_path, feasibility_loop=True):
    return PipelineConfig(
        name="feas",
        stages=[
            StageConfig(team="ModelTeam", mode="ModeNoWcNoHITL",
                        inputs=["research_topic"], outputs=["model_specification", "model_design"]),
            StageConfig(team="DataTeam", mode="ModeOpenSourceAPI",
                        inputs=["model_specification"], outputs=["validated_dataset", "data_source"],
                        depends_on=["ModelTeam"]),
        ],
        budget=BudgetConfig(),
        output_dir=str(tmp_path / "out"),
        feasibility_loop=feasibility_loop,
    )


def _model_runner(calls=None):
    """ModelTeam: cycle 1 needs 'capital stock' (unavailable); after an MRR it drops it.
    If ``calls`` is given, each invocation appends its ``drs_only`` flag (fast-path check)."""
    def runner(upstream_artifacts, mode, output_dir, collector=None, drs_only=False):
        if calls is not None:
            calls.append(drs_only)
        revised = "model_revision_request" in upstream_artifacts
        variables = [{"variable_symbol": "g", "variable_name": "government spending",
                      "variable_type": "exogenous", "description": "g"}]
        if not revised:
            variables.insert(0, {"variable_symbol": "K", "variable_name": "capital stock",
                                 "variable_type": "exogenous", "description": "K"})
        return {
            "model_specification": {"calibrated_models": [{"empirical_targets": []}]},
            "model_design": {"formal_models": [{"model_title": "M", "variables": variables}]},
        }
    return runner


def _data_runner():
    """DataTeam supplies only 'government spending'."""
    def runner(upstream_artifacts, mode, output_dir, collector=None):
        return {
            "validated_dataset": {"ok": True},
            "data_source": {"selected_series": [
                {"series_id": "FYONGDA188S", "series_name": "government spending",
                 "source_name": "FRED", "frequency": "annual", "units": "pct"},
            ], "retrieved_data": [   # FEASIBLE needs a retrieval record + title match
                {"series_id": "FYONGDA188S", "series_name": "government spending",
                 "source_name": "FRED", "retrieval_status": "Success", "num_observations": 60,
                 "data_simulated": False,
                 "source_title": "Government Consumption Expenditures and Gross Investment"},
            ]},
        }
    return runner


def _orchestrator(tmp_path, model_calls=None, **cfg):
    orch = ResearchPipelineOrchestrator(_config(tmp_path, **cfg))
    orch.register_team_runner("ModelTeam", _model_runner(model_calls))
    orch.register_team_runner("DataTeam", _data_runner())
    return orch


def _report(orch):
    path = os.path.join(orch.output_dir, "feasibility_report.json")
    with open(path) as f:
        return json.load(f)


class TestFeasibilityPipeline:
    def test_loop_runs_and_persists(self, tmp_path, monkeypatch):
        # auto mode -> committee approves nothing -> unmet resolves to a documented limitation
        monkeypatch.setenv("AEL_HITL_MODE", "auto")
        orch = _orchestrator(tmp_path)
        res = orch.run(research_topic="fiscal policy")
        assert res.success
        rep = _report(orch)
        assert rep["status"] == "resolved_without_revision" and rep["cycles"] == 1
        # capital stock was unavailable and accepted as a documented limitation
        actions = [d["action"] for d in rep["decisions"]]
        assert actions == ["accept_as_is"]
        assert orch.artifact_store.get("data_availability_report") is not None
        assert orch.artifact_store.get("data_requirements_spec") is not None

    def test_revise_model_cycle_closes(self, tmp_path, monkeypatch):
        # committee approves "revise the model" -> ModelTeam re-runs, drops capital stock -> feasible
        monkeypatch.setenv("AEL_HITL_MODE", "llm_economist")

        from shared.llm_economist import PropositionResult

        class _FakeCommittee:
            def __init__(self, *a, **k):
                pass

            def vote_proposition(self, question, context=None):
                return PropositionResult(question=question, passed="revise the model" in question.lower())

        monkeypatch.setattr("shared.llm_economist.LLMEconomistCommittee", _FakeCommittee)

        orch = _orchestrator(tmp_path)
        res = orch.run(research_topic="fiscal policy")
        assert res.success
        rep = _report(orch)
        assert rep["status"] == "feasible" and rep["cycles"] == 2
        assert len(rep["model_revision_requests"]) == 1
        assert "capital stock" in rep["model_revision_requests"][0]["requested_change"]
        # the MRR was registered as a first-class artifact
        assert orch.artifact_store.get("model_revision_request") is not None

    def test_drs_cycle_uses_fast_path_full_calibration_at_convergence(self, tmp_path, monkeypatch):
        """A DRS re-run cycle calls ModelTeam with drs_only=True (skip estimation); a
        single FULL calibration (drs_only=False) runs last, at convergence."""
        monkeypatch.setenv("AEL_HITL_MODE", "llm_economist")

        from shared.llm_economist import PropositionResult

        class _FakeCommittee:
            def __init__(self, *a, **k):
                pass

            def vote_proposition(self, question, context=None):
                return PropositionResult(question=question, passed="revise the model" in question.lower())

        monkeypatch.setattr("shared.llm_economist.LLMEconomistCommittee", _FakeCommittee)

        calls = []
        orch = _orchestrator(tmp_path, model_calls=calls)
        res = orch.run(research_topic="fiscal policy")
        assert res.success
        # the DRS re-run cycle used the fast targets-only path...
        assert any(calls), f"expected a drs_only=True cycle, got {calls}"
        # ...and the LAST ModelTeam invocation is a FULL calibration at convergence.
        assert calls[-1] is False, f"final ModelTeam call must be full calibration, got {calls}"

    def test_disabled_by_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AEL_HITL_MODE", "auto")
        orch = _orchestrator(tmp_path, feasibility_loop=False)
        res = orch.run(research_topic="fiscal policy")
        assert res.success
        assert not os.path.exists(os.path.join(orch.output_dir, "feasibility_report.json"))
        assert orch.artifact_store.get("data_availability_report") is None

    def test_fetchable_requirement_triggers_supplemental_refresh(self, tmp_path, monkeypatch):
        """A requirement missing from the retrieved pool but covered by
        the curated table grades FEASIBLE (fetchable) — no MRR — and the post-loop refresh
        re-invokes DataTeam with the supplemental fetch list."""
        monkeypatch.setenv("AEL_HITL_MODE", "auto")
        # a curated hit is FEASIBLE only when its official title passes the concept
        # check — serve the World Bank title offline (search rungs stubbed)
        import shared.tools.econ_connectors as ec
        ec._RESOLVE_CACHE.clear()
        monkeypatch.setenv("AEL_PROVIDER_SEARCH", "1")
        monkeypatch.setattr(ec, "search_fred_series", lambda *a, **k: None)
        monkeypatch.setattr(ec, "search_wdi_series", lambda *a, **k: None)
        monkeypatch.setattr(ec, "fetch_series_title",
                            lambda src, sid, *a, **k: {"GC.TAX.TOTL.GD.ZS": "Tax revenue (% of GDP)"}
                            .get(sid))

        def model_runner(upstream_artifacts, mode, output_dir, collector=None, drs_only=False):
            return {
                "model_specification": {"calibrated_models": [{"empirical_targets": []}]},
                "model_design": {"formal_models": [{"model_title": "M", "variables": [
                    {"variable_symbol": "g", "variable_name": "government spending",
                     "variable_type": "exogenous", "description": "g"},
                    # a ratio requirement: the curated id is the ratio 'Tax revenue (% of
                    # GDP)', which a level request ('tax revenue') does not match
                    {"variable_symbol": "tau", "variable_name": "tax revenue (% of GDP)",
                     "variable_type": "exogenous", "description": "tau"},
                ]}]},
            }

        data_calls = []
        def data_runner(upstream_artifacts, mode, output_dir, collector=None):
            data_calls.append(upstream_artifacts.get("supplemental_series"))
            return {
                "validated_dataset": {"ok": True},
                "data_source": {"selected_series": [
                    {"series_id": "FYONGDA188S", "series_name": "government spending",
                     "source_name": "FRED", "frequency": "annual", "units": "pct"},
                ], "retrieved_data": [   # FEASIBLE needs a retrieval record + title match
                    {"series_id": "FYONGDA188S", "series_name": "government spending",
                     "source_name": "FRED", "retrieval_status": "Success", "num_observations": 60,
                     "data_simulated": False,
                     "source_title": "Government Consumption Expenditures and Gross Investment"},
                ]},
            }

        orch = ResearchPipelineOrchestrator(_config(tmp_path))
        orch.register_team_runner("ModelTeam", model_runner)
        orch.register_team_runner("DataTeam", data_runner)
        res = orch.run(research_topic="fiscal policy")
        assert res.success
        rep = _report(orch)
        # tax revenue: absent from the pool, curated (GC.TAX.TOTL.GD.ZS) -> feasible, no MRR
        assert rep["status"] == "feasible" and not rep["model_revision_requests"]
        tax = next(f for f in rep["final_data_availability_report"]["findings"]
                   if f["variable_name"] == "tax revenue (% of GDP)")
        assert tax["fetchable"] is True and tax["matched_series"] == "GC.TAX.TOTL.GD.ZS"
        # the post-loop refresh re-invoked DataTeam WITH the supplemental list
        assert len(data_calls) == 2, "expected linear pass + supplemental refresh"
        supp = data_calls[-1]
        supp_list = supp.get("series") if isinstance(supp, dict) else supp
        assert supp_list and supp_list[0]["series_id"] == "GC.TAX.TOTL.GD.ZS"
        # and registered the list as a first-class artifact
        assert orch.artifact_store.get("supplemental_series") is not None
