# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""ReportingTeam pipeline integration: the terminal node end-to-end, offline.

Real stages + real harness; only LLMClient.invoke is faked. The clean-narrative case must
come out `consistent`; a narrative with a hallucinated number must be flagged on the report."""

import json

import numpy as np
import pytest

from pipeline.team_runners import run_reporting_team


def _estimation_results(n=60, b=2.0, seed=4):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1.5, n)
    y = 1.0 + b * x + rng.normal(0, 0.5, n)
    dates = [f"{1990 + i // 12}-{i % 12 + 1:02d}-01" for i in range(n)]
    analysis = [{"date": d, "outcome": float(yy), "driver": float(xx)}
                for d, yy, xx in zip(dates, y, x)]
    return {
        "outcome": {
            "verdict": "estimated", "dependent_name": "outcome", "n_obs": n,
            "r_squared": 0.9, "cov_type": "HAC",
            "sample_start": "1990-01-01", "sample_end": "1994-12-01", "frequency": "monthly",
            "coefficients": [
                {"name": "const", "estimate": 1.0, "std_error": 0.1, "t_stat": 10.0,
                 "p_value": 0.0, "ci_low": 0.8, "ci_high": 1.2},
                {"name": "driver", "estimate": b, "std_error": 0.05, "t_stat": 40.0,
                 "p_value": 0.0, "ci_low": b - 0.1, "ci_high": b + 0.1}],
            "analysis_data": analysis, "notes": [],
            "spec": {"dependent": {"name": "outcome", "series_ref": "Y", "transform": "level",
                                   "lag": 0},
                     "regressors": [{"name": "driver", "series_ref": "X", "transform": "level",
                                     "lag": 0}],
                     "method": "ols", "cov_type": "HAC", "add_constant": True,
                     "hypotheses": [], "rationale": "", "fallback_spec": False,
                     "sample_start": None, "sample_end": None},
        },
        "inference": {"hypotheses": [], "robustness": [], "stability_score": None,
                      "interpretation": ""},
    }


def _upstream():
    return {
        "research_questions": {"final_questions": [{"question": "Does X drive Y?"}]},
        "literature_review": {"review_text": "Prior evidence is mixed."},
        "model_specification": {"calibrated_models": []},
        "data_source": {"retrieved_data": [
            {"series_id": "XSER", "series_name": "Driver", "source_name": "FRED",
             "num_observations": 60, "start_date": "1990-01-01", "end_date": "1994-12-01",
             "data_simulated": False, "quality_notes": "ok"}]},
        "estimation_results": _estimation_results(),
    }


def _patch_llm(monkeypatch, text):
    from shared.llm import LLMClient
    monkeypatch.setattr(LLMClient, "invoke",
                        lambda self, messages, **kw: text, raising=True)


@pytest.mark.parametrize("mode", ["ModeNoWcNoHITL", "ModeWithWcWithHITL"])
def test_run_reporting_team_end_to_end_consistent(monkeypatch, tmp_path, mode):
    monkeypatch.setenv("AEL_HITL_MODE", "auto")
    _patch_llm(monkeypatch, "The driver clearly matters for the outcome.")  # no numbers
    out = run_reporting_team(upstream_artifacts=_upstream(), mode=mode,
                             output_dir=str(tmp_path))
    report = out["research_report"]
    assert (tmp_path / "research_report.md").exists()
    assert report["consistency"]["verdict"] == "consistent"
    md = report["report_markdown"]
    assert "## 7. Limitations" in md and "Number-consistency check" in md
    assert (tmp_path / "figures").exists() and report["figures"]


def test_hallucinated_narrative_number_is_flagged_on_the_report(monkeypatch, tmp_path):
    monkeypatch.setenv("AEL_HITL_MODE", "auto")
    _patch_llm(monkeypatch, "The effect equals 7.7734 which is enormous.")  # invented number
    out = run_reporting_team(upstream_artifacts=_upstream(), mode="ModeNoWcNoHITL",
                             output_dir=str(tmp_path))
    consistency = out["research_report"]["consistency"]
    assert consistency["verdict"] == "has_unverified"
    assert any(abs(u["value"] - 7.7734) < 1e-9 for u in consistency["unverified"])
    assert "7.7734" in out["research_report"]["report_markdown"].split("Number-consistency")[1]


def test_inestimable_upstream_still_produces_an_honest_report(monkeypatch, tmp_path):
    monkeypatch.setenv("AEL_HITL_MODE", "auto")
    _patch_llm(monkeypatch, "No estimation was possible.")
    upstream = _upstream()
    upstream["estimation_results"] = {"outcome": {"verdict": "inestimable",
                                                  "reason": "no_data", "analysis_data": [],
                                                  "coefficients": [], "notes": []},
                                      "inference": {"hypotheses": [], "robustness": []}}
    out = run_reporting_team(upstream_artifacts=upstream, mode="ModeNoWcNoHITL",
                             output_dir=str(tmp_path))
    md = out["research_report"]["report_markdown"]
    assert "`inestimable`" in md
    assert "Estimation honestly declined" in md
