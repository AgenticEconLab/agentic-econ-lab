# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""EstimationTeam pipeline integration: runner contract + orchestrator deferral.

run_estimation_team is exercised with the REAL stage modules and harness — only
LLMClient.invoke is faked (returns a canned EstimationSpec JSON), so the test proves the
whole runner->stages->harness chain offline. The deferral test proves EstimationTeam runs
AFTER the other teams even when topological order would schedule it first.
"""

import json

import numpy as np
import pytest

from pipeline.pipeline_config import BudgetConfig, PipelineConfig, StageConfig
from pipeline.research_pipeline import ResearchPipelineOrchestrator
from pipeline.team_runners import run_estimation_team


def _preview(series_id, values):
    dates = [f"{1990 + i // 12}-{i % 12 + 1:02d}-01" for i in range(len(values))]
    return [{"date": d, series_id: float(v)} for d, v in zip(dates, values)]


def _upstream(n=80, seed=5):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    y = 1.0 + 2.0 * x + rng.normal(0, 1, n)
    return {
        "research_questions": {"final_questions": [
            {"question": "Does the driver series move the outcome series?"}]},
        "model_specification": {"calibrated_models": [{"model_title": "toy"}]},
        "data_source": {"retrieved_data": [
            {"series_id": "XSER", "series_name": "Driver Series", "source_name": "CustomSource",
             "num_observations": n, "data_preview": _preview("XSER", x), "data_simulated": False},
            {"series_id": "YSER", "series_name": "Outcome Series", "source_name": "CustomSource",
             "num_observations": n, "data_preview": _preview("YSER", y), "data_simulated": False},
        ]},
    }


_SPEC_JSON = json.dumps({
    "dependent": {"name": "outcome", "series_ref": "YSER", "transform": "level", "lag": 0},
    "regressors": [{"name": "driver", "series_ref": "XSER", "transform": "level", "lag": 0}],
    "method": "ols", "cov_type": "HAC", "add_constant": True,
    "hypotheses": [{"name": "driver_positive", "param": "driver", "restriction": ">0",
                    "rationale": "toy DGP has positive slope"}],
    "rationale": "test spec",
})


@pytest.fixture()
def fake_llm(monkeypatch):
    """Every LLM call (spec proposal AND narration) returns the canned spec JSON —
    narration consumers only str() it, the Estimator parses it."""
    from shared.llm import LLMClient
    monkeypatch.setattr(LLMClient, "invoke",
                        lambda self, messages, **kw: _SPEC_JSON, raising=True)


@pytest.mark.parametrize("mode", ["ModeNoWcNoHITL", "ModeWithWcWithHITL"])
def test_run_estimation_team_end_to_end(fake_llm, monkeypatch, tmp_path, mode):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setenv("AEL_HITL_MODE", "auto")  # no committee in unit tests
    out = run_estimation_team(
        upstream_artifacts=_upstream(), mode=mode, output_dir=str(tmp_path))

    results = out["estimation_results"]
    assert results["outcome"]["verdict"] in ("estimated", "fragile")
    hyp = results["inference"]["hypotheses"][0]
    assert hyp["name"] == "driver_positive" and hyp["supported"] is True
    for fname in ("estimation_output.json", "validation_output.json", "inference_output.json"):
        assert (tmp_path / fname).exists(), f"{fname} not written to pipeline output dir"
    # provenance: analysis panel stored for offline reproducibility
    est = json.loads((tmp_path / "estimation_output.json").read_text())
    assert est["outcome"]["analysis_data"], "aligned analysis panel must be persisted"


def test_run_estimation_team_declines_honestly_without_data(fake_llm, monkeypatch, tmp_path):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setenv("AEL_HITL_MODE", "auto")
    upstream = _upstream()
    upstream["data_source"] = {"retrieved_data": []}
    out = run_estimation_team(
        upstream_artifacts=upstream, mode="ModeNoWcNoHITL", output_dir=str(tmp_path))
    assert out["estimation_results"]["outcome"]["verdict"] == "inestimable"
    assert out["estimation_results"]["outcome"]["reason"] == "no_data"


def test_estimation_team_is_deferred_after_other_teams(tmp_path):
    """Topological order would run EstimationTeam FIRST (alphabetical, no deps) — the
    orchestrator must defer it past the linear pass so it sees final artifacts."""
    order = []

    def z_runner(upstream_artifacts, mode, output_dir, collector=None):
        order.append("ZTeam")
        return {"z_artifact": {"payload": 1}}

    seen_upstream = {}

    def estimation_runner(upstream_artifacts, mode, output_dir, collector=None):
        order.append("EstimationTeam")
        seen_upstream.update(upstream_artifacts)
        return {"estimation_results": {"ok": True}}

    config = PipelineConfig(
        name="deferral-test", mode="ModeNoWcNoHITL",
        stages=[
            StageConfig(team="EstimationTeam", mode="ModeNoWcNoHITL",
                        inputs=["z_artifact"], outputs=["estimation_results"], depends_on=[]),
            StageConfig(team="ZTeam", mode="ModeNoWcNoHITL",
                        inputs=["research_topic"], outputs=["z_artifact"], depends_on=[]),
        ],
        budget=BudgetConfig(), output_dir=str(tmp_path),
        feasibility_loop=False,
    )
    assert config.get_execution_order()[0] == "EstimationTeam"  # the trap this guards

    pipeline = ResearchPipelineOrchestrator(config, team_runners={
        "ZTeam": z_runner, "EstimationTeam": estimation_runner})
    result = pipeline.run(research_topic="t")

    assert result.success
    assert order == ["ZTeam", "EstimationTeam"]
    assert seen_upstream.get("z_artifact") == {"payload": 1}, \
        "deferred EstimationTeam must see artifacts produced by the linear pass"
    assert "estimation_results" in result.artifact_names


def test_code_team_enabled_via_env_runs_deferred(monkeypatch, tmp_path):
    """AEL_CODE_TEAM_ENABLED=1 adds the optional CodeTeam node, deferred with the
    other post-feasibility teams and ordered before Estimation/Reporting."""
    monkeypatch.setenv("AEL_CODE_TEAM_ENABLED", "1")
    from run_ael_pipeline import build_default_config
    config = build_default_config()
    code = config.get_stage("CodeTeam")
    assert code is not None and code.enabled is True
    config.output_dir = str(tmp_path)
    config.feasibility_loop = False

    order = []

    def mk(team, outputs):
        def runner(upstream_artifacts, mode, output_dir, collector=None):
            order.append(team)
            return outputs
        return runner

    pipeline_runners = {
        "IdeationTeam": mk("IdeationTeam", {"research_questions": {"final_questions": []}}),
        "LiteratureTeam": mk("LiteratureTeam", {"literature_review": {}}),
        "ModelTeam": mk("ModelTeam", {"model_specification": {}, "model_design": {}}),
        "DataTeam": mk("DataTeam", {"validated_dataset": {}, "data_source": {}}),
        "CodeTeam": mk("CodeTeam", {"executable_model": {}}),
        "EstimationTeam": mk("EstimationTeam", {"estimation_results": {}}),
        "ReportingTeam": mk("ReportingTeam", {"research_report": {}}),
    }
    from pipeline.research_pipeline import ResearchPipelineOrchestrator
    result = ResearchPipelineOrchestrator(config, team_runners=pipeline_runners).run("t")
    assert result.success and len(result.teams_completed) == 7
    assert order.index("CodeTeam") > order.index("DataTeam")        # deferred past linear pass
    assert order.index("CodeTeam") < order.index("EstimationTeam")  # dependency order kept
    assert order[-1] == "ReportingTeam"


def test_loader_prefers_first_class_fetch_id(monkeypatch):
    """fetch_id/fetch_source fields win over the legacy quality_notes regex."""
    import pandas as pd
    from EstimationTeam.ael.estim_harness import data_loader as dl

    calls = []

    def fake_connector(source, sid, collector=None):
        calls.append((source, sid))
        if sid == "WB/WDI/A-GOOD-USA":
            dates = pd.date_range("2000-01-01", periods=30, freq="YS")
            return pd.Series(range(30), index=dates, dtype=float)
        return None
    monkeypatch.setattr(dl, "_fetch_connector", fake_connector)
    artifact = {"retrieved_data": [{
        "series_id": "BROKEN.ID", "series_name": "Some Series", "source_name": "World Bank",
        "num_observations": 30, "data_preview": [], "data_simulated": False,
        "fetch_id": "WB/WDI/A-GOOD-USA", "fetch_source": "DBnomics",
        "quality_notes": "[id repaired via WDI search: 'BROKEN.ID' -> WB/WDI/A-GOOD-USA]",
    }]}
    series_map, _aliases, notes = dl.load_panel(artifact)
    assert series_map is not None and "BROKEN.ID" in series_map
    assert calls[0] == ("DBnomics", "WB/WDI/A-GOOD-USA")   # first-class field used FIRST
    assert any("canonical fetch_id" in n for n in notes)


def test_resume_skip_teams_does_not_swallow_never_run_optional_teams(tmp_path):
    """Live-run lesson: the order-prefix resume heuristic marked the never-run optional
    CodeTeam as completed. skip_teams (the checkpointed set) must be authoritative."""
    order = []

    def mk(team, outputs):
        def runner(upstream_artifacts, mode, output_dir, collector=None):
            order.append(team)
            return outputs
        return runner

    config = PipelineConfig(
        name="resume-test", mode="ModeNoWcNoHITL",
        stages=[
            StageConfig(team="AlphaTeam", mode="m", inputs=[], outputs=["a"], depends_on=[]),
            StageConfig(team="MidOptionalTeam", mode="m", inputs=[], outputs=["opt"],
                        depends_on=["AlphaTeam"]),
            StageConfig(team="ZetaTeam", mode="m", inputs=[], outputs=["z"],
                        depends_on=["MidOptionalTeam"]),
        ],
        budget=BudgetConfig(), output_dir=str(tmp_path), feasibility_loop=False,
    )
    pipeline = ResearchPipelineOrchestrator(config, team_runners={
        "AlphaTeam": mk("AlphaTeam", {"a": {}}),
        "MidOptionalTeam": mk("MidOptionalTeam", {"opt": {}}),
        "ZetaTeam": mk("ZetaTeam", {"z": {}}),
    })
    # Prior run completed Alpha only; MidOptional never ran. Resume at Zeta would have
    # skipped MidOptional under the old prefix heuristic.
    result = pipeline.run("t", resume_from="ZetaTeam", skip_teams={"AlphaTeam"})
    assert order == ["MidOptionalTeam", "ZetaTeam"], order
    assert result.teams_completed == ["AlphaTeam", "MidOptionalTeam", "ZetaTeam"]
