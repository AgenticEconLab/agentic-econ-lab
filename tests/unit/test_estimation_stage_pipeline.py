# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""EstimationTeam stages end-to-end, offline: fake LLM proposal, real harness numerics.

Stage 1 (proposal faked, estimation real) -> Stage 2 (diagnostics) -> Stage 3 (inference),
chained through the same JSON dicts the orchestrators persist. No network: the synthetic
data artifact uses a source with no connector, so the loader takes the disclosed preview
fallback — the previews here carry 80 observations so the harness gate passes.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from EstimationTeam.ael.estim_harness import EstimationSpec, HypothesisSpec, VariableSpec

_MODES = ["ModeNoWcNoHITL", "ModeNoWcWithHITL"]


def _load(mode, stage_file, alias):
    path = Path(__file__).resolve().parents[2] / "EstimationTeam" / "ael" / mode / stage_file
    spec = importlib.util.spec_from_file_location(f"{alias}_{mode}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _preview(series_id, values):
    dates = [f"{1990 + i // 12}-{i % 12 + 1:02d}-01" for i in range(len(values))]
    return [{"date": d, series_id: float(v)} for d, v in zip(dates, values)]


def _data_artifact(n=80, seed=11):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    y = 2.0 + 3.0 * x + rng.normal(0, 1, n)
    return {"retrieved_data": [
        {"series_id": "XSER", "series_name": "Driver Series", "source_name": "CustomSource",
         "num_observations": n, "data_preview": _preview("XSER", x), "data_simulated": False},
        {"series_id": "YSER", "series_name": "Outcome Series", "source_name": "CustomSource",
         "num_observations": n, "data_preview": _preview("YSER", y), "data_simulated": False},
    ]}


def _fake_spec():
    return EstimationSpec(
        dependent=VariableSpec(name="outcome", series_ref="YSER"),
        regressors=[VariableSpec(name="driver", series_ref="XSER")],
        hypotheses=[HypothesisSpec(name="driver_positive", param="driver", restriction=">0")],
        rationale="synthetic test spec")


def _patch_llm_free(monkeypatch, mod, agent_attr, method, value):
    monkeypatch.setattr(getattr(mod, agent_attr), method,
                        lambda self, *a, **k: value, raising=True)


@pytest.mark.parametrize("mode", _MODES)
def test_full_stage_chain_offline(monkeypatch, tmp_path, mode):
    monkeypatch.chdir(tmp_path)  # stages write output/feedback json to cwd
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    # ---- Stage 1: proposal faked, estimation real
    s1 = _load(mode, "1-EstimationStage.py", "_estim_s1")
    _patch_llm_free(monkeypatch, s1, "Estimator", "propose_spec", _fake_spec())
    orch1 = s1.EstimationOrchestrator(quiet=True)
    out1 = orch1.run_estimation_pipeline({"model": "test"}, _data_artifact(),
                                         research_question="does X drive Y?")
    assert out1.outcome.verdict == "estimated"
    slope = next(c for c in out1.outcome.coefficients if c.name == "driver")
    assert slope.ci_low < 3.0 < slope.ci_high
    path1 = orch1.save_estimation_output("estimation_output.json")
    stage1_data = json.loads(Path(path1).read_text())

    # ---- Stage 2: deterministic battery, narration faked
    s2 = _load(mode, "2-ValidationDiagnosticsStage.py", "_estim_s2")
    _patch_llm_free(monkeypatch, s2, "Validator", "interpret", "canned narration")
    orch2 = s2.ValidationOrchestrator(quiet=True)
    out2 = orch2.run_validation_pipeline(stage1_data)
    assert len(out2.diagnostics.results) >= 6
    assert out2.outcome.verdict in ("estimated", "fragile")
    path2 = orch2.save_validation_output("validation_output.json")
    stage2_data = json.loads(Path(path2).read_text())

    # ---- Stage 3: hypothesis tests + robustness, narration faked
    s3 = _load(mode, "3-InferenceRobustnessStage.py", "_estim_s3")
    _patch_llm_free(monkeypatch, s3, "HypothesisTester", "interpret", "canned narration")
    orch3 = s3.InferenceOrchestrator(quiet=True)
    out3 = orch3.run_inference_pipeline(stage2_data, research_question="does X drive Y?")
    assert out3.inference.hypotheses[0].supported is True
    assert out3.inference.robustness, "expected split-sample/cov-swap robustness checks"
    orch3.save_inference_output("inference_output.json")
    assert (tmp_path / "inference_output.json").exists()


@pytest.mark.parametrize("mode", _MODES)
def test_inestimable_flows_honestly_through_all_stages(monkeypatch, tmp_path, mode):
    """An LLM that never yields a valid spec must produce inestimable, not a substitute fit —
    and stages 2/3 must pass the refusal through instead of crashing."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    s1 = _load(mode, "1-EstimationStage.py", "_estim_s1i")

    def _always_fails(self, *a, **k):
        raise ValueError("no valid spec")
    monkeypatch.setattr(s1.Estimator, "propose_spec", _always_fails, raising=True)
    orch1 = s1.EstimationOrchestrator(quiet=True)
    out1 = orch1.run_estimation_pipeline({}, _data_artifact(), research_question="q")
    assert out1.outcome.verdict == "inestimable" and out1.outcome.reason == "invalid_spec"
    stage1_data = json.loads(Path(orch1.save_estimation_output()).read_text())

    s2 = _load(mode, "2-ValidationDiagnosticsStage.py", "_estim_s2i")
    orch2 = s2.ValidationOrchestrator(quiet=True)
    out2 = orch2.run_validation_pipeline(stage1_data)
    assert out2.outcome.verdict == "inestimable"
    stage2_data = json.loads(Path(orch2.save_validation_output()).read_text())

    s3 = _load(mode, "3-InferenceRobustnessStage.py", "_estim_s3i")
    orch3 = s3.InferenceOrchestrator(quiet=True)
    out3 = orch3.run_inference_pipeline(stage2_data)
    assert out3.outcome.verdict == "inestimable"
    assert not out3.inference.hypotheses and not out3.inference.robustness


@pytest.mark.parametrize("mode", _MODES)
def test_feedback_checkpoint_forwards_context_and_writes_to_cwd(monkeypatch, tmp_path, mode):
    monkeypatch.chdir(tmp_path)  # method writes a feedback json to cwd
    s1 = _load(mode, "1-EstimationStage.py", "_estim_fb")
    seen = []

    def fake_auto_input(prompt, default="", input_type="general", timeout=None, context=None):
        seen.append(context)
        return "concern about identification"
    monkeypatch.setattr(s1, "auto_input", fake_auto_input)
    orch = s1.EstimationOrchestrator(quiet=True)
    ctx = {"specification": {"dependent": "y"}, "verdict": "estimated"}
    feedback = orch.collect_human_feedback(round_number=1, context=ctx)
    assert seen and all(c is ctx for c in seen)
    assert "identification" in feedback
    assert (tmp_path / "round1_estimation_feedback.json").exists()
