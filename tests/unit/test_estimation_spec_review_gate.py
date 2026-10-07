# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Estimation specification review gate: specification first, estimation second.

The committee saw coefficients/p/R^2 of the first fit, rejected it in 22/22 runs (mostly for
low R^2 or insignificance), one LLM revision followed without a re-vote and became the
reported result; the first fit was never disclosed; in 4 runs an estimable spec was replaced
by an inestimable revision. These tests pin the new order:

(a) the review sees the PROPOSED spec, the series and the budget only — no fit statistics;
(b) a rejection triggers one revision that is re-voted; a rejected revision leaves the
    original proposal in place with the objections recorded;
(c) after estimation a refusal-triggered repair (re-voted before fitting, see
    test_estimation_repair_gate.py) may replace a refused spec, never an estimated one;
(d) every fitted specification and its verdict is recorded.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from EstimationTeam.ael.estim_harness import EstimationSpec, VariableSpec

_FIT_KEYS = {"coefficients", "r_squared", "p_value", "p_values", "n_obs", "verdict",
             "adj_r_squared", "harness_notes"}


def _load(mode, alias):
    path = (Path(__file__).resolve().parents[2] / "EstimationTeam" / "ael" / mode
            / "1-EstimationStage.py")
    spec = importlib.util.spec_from_file_location(f"{alias}_{mode}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _preview(series_id, values):
    dates = [f"{1990 + i // 12}-{i % 12 + 1:02d}-01" for i in range(len(values))]
    return [{"date": d, series_id: float(v)} for d, v in zip(dates, values)]


def _artifact(n=80, seed=11):
    rng = np.random.default_rng(seed)
    x, z = rng.normal(0, 1, n), rng.normal(0, 1, n)
    y = 2.0 + 0.1 * x + rng.normal(0, 1, n)
    items = []
    for sid, v in (("XSER", x), ("YSER", y), ("ZSER", z)):
        items.append({"series_id": sid, "series_name": sid, "source_name": "CustomSource",
                      "num_observations": n, "data_preview": _preview(sid, v),
                      "data_simulated": False})
    return {"retrieved_data": items}


SPEC_A = EstimationSpec(dependent=VariableSpec(name="y", series_ref="YSER"),
                        regressors=[VariableSpec(name="x", series_ref="XSER")])
SPEC_B = EstimationSpec(dependent=VariableSpec(name="y", series_ref="YSER"),
                        regressors=[VariableSpec(name="z", series_ref="ZSER")])
# 12 distinct regressors: 13 x 8 = 104 > 80 obs, so the harness refuses it
SPEC_BIG = EstimationSpec(dependent=VariableSpec(name="y", series_ref="YSER"),
                          regressors=[VariableSpec(name=f"r{i}", series_ref=("XSER", "ZSER")[i % 2], lag=i // 2)
                                      for i in range(12)])
SPEC_MISSING = EstimationSpec(dependent=VariableSpec(name="y", series_ref="YSER"),
                              regressors=[VariableSpec(name="w", series_ref="NOPE")])


def _orch(monkeypatch, tmp_path, mode, proposals):
    s1 = _load(mode, "_gate")
    queue = list(proposals)
    calls = []

    def fake_propose(self, research_question, model_context, variables_desc,
                     feedback=None, prior_error=None, executable_model_desc=""):
        calls.append({"feedback": feedback, "prior_error": prior_error})
        return queue.pop(0)
    monkeypatch.setattr(s1.Estimator, "propose_spec", fake_propose, raising=True)
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    return s1.EstimationOrchestrator(quiet=True, output_dir=str(tmp_path)), calls


def _review_script(*answers):
    seen = []
    answers = list(answers)

    def review(ctx, round_number):
        seen.append((round_number, ctx))
        return answers.pop(0)
    return review, seen


@pytest.mark.parametrize("mode", ["ModeNoWcNoHITL", "ModeNoWcWithHITL"])
def test_review_sees_spec_and_budget_but_no_fit(monkeypatch, tmp_path, mode):
    orch, _ = _orch(monkeypatch, tmp_path, mode, [SPEC_A])
    review, seen = _review_script((True, "approve"))
    out = orch.run_reviewed_pipeline({}, _artifact(), review=review, research_question="q")
    (rnd, ctx), = seen
    assert rnd == 1 and ctx["specification"]["regressors"][0]["series_ref"] == "XSER"
    assert not (_FIT_KEYS & set(ctx)), "no fit statistics before the vote"
    assert "requires" in ctx["observation_budget"] and "overlap on 80" in ctx["observation_budget"]
    assert "XSER" in ctx["available_series"]
    fitted = out.metadata["fitted_specifications"]
    assert len(fitted) == 1 and fitted[0]["reported"] and fitted[0]["committee"] == "approved"
    assert out.outcome.verdict == "estimated"


def test_rejected_revision_is_revoted_and_original_kept(monkeypatch, tmp_path):
    orch, calls = _orch(monkeypatch, tmp_path, "ModeNoWcWithHITL", [SPEC_A, SPEC_B])
    review, seen = _review_script((False, "use z instead"), (False, "still weak"))
    out = orch.run_reviewed_pipeline({}, _artifact(), review=review)
    assert [r for r, _ in seen] == [1, 2]                 # the revision was re-voted
    assert seen[1][1]["specification"]["regressors"][0]["series_ref"] == "ZSER"
    assert calls[1]["feedback"] == "use z instead"
    # the original proposal is estimated and reported; the revision is never fitted
    assert out.outcome.spec.regressors[0].series_ref == "XSER"
    fitted = out.metadata["fitted_specifications"]
    assert len(fitted) == 1 and fitted[0]["label"] == "proposal"
    assert "objections" in fitted[0]["selection"]
    objections = out.metadata["committee_objections"]
    assert any("use z instead" in o for o in objections)
    assert any("still weak" in o for o in objections)
    assert [r["approved"] for r in out.metadata["spec_review"]] == [False, False]


def test_approved_revision_is_estimated(monkeypatch, tmp_path):
    orch, _ = _orch(monkeypatch, tmp_path, "ModeNoWcWithHITL", [SPEC_A, SPEC_B])
    review, _ = _review_script((False, "use z"), (True, "approve"))
    out = orch.run_reviewed_pipeline({}, _artifact(), review=review)
    assert out.outcome.spec.regressors[0].series_ref == "ZSER"
    assert out.spec_source == "llm_revised"
    assert "committee_objections" not in out.metadata


def test_inestimable_repair_never_replaces_an_estimated_spec(monkeypatch, tmp_path):
    """v0.7.1: in 4 runs an estimable spec was replaced by an inestimable revision."""
    orch, calls = _orch(monkeypatch, tmp_path, "ModeNoWcWithHITL", [SPEC_A])
    review, _ = _review_script((True, ""))
    out = orch.run_reviewed_pipeline({}, _artifact(), review=review)
    assert out.outcome.verdict == "estimated" and len(calls) == 1   # no repair attempted


def test_refused_spec_is_repaired_once_and_both_fits_recorded(monkeypatch, tmp_path):
    orch, calls = _orch(monkeypatch, tmp_path, "ModeNoWcWithHITL", [SPEC_BIG, SPEC_A])
    review, _ = _review_script((True, ""), (True, ""))   # proposal, then the repair
    out = orch.run_reviewed_pipeline({}, _artifact(), review=review)
    assert "insufficient_observations" in (calls[1]["prior_error"] or "")
    fitted = out.metadata["fitted_specifications"]
    assert [f["verdict"] for f in fitted] == ["inestimable", "estimated"]
    assert [f["reported"] for f in fitted] == [False, True]
    assert out.outcome.verdict == "estimated" and out.spec_source == "llm_repaired"
    assert "repair" in fitted[1]["selection"]


def test_refused_repair_keeps_the_original_refusal(monkeypatch, tmp_path):
    orch, _ = _orch(monkeypatch, tmp_path, "ModeNoWcWithHITL", [SPEC_BIG, SPEC_MISSING])
    review, _ = _review_script((True, ""), (True, ""))
    out = orch.run_reviewed_pipeline({}, _artifact(), review=review)
    fitted = out.metadata["fitted_specifications"]
    assert [f["reported"] for f in fitted] == [True, False]
    assert out.outcome.reason == "insufficient_observations"


def test_unreviewed_pipeline_records_its_single_fit(monkeypatch, tmp_path):
    orch, _ = _orch(monkeypatch, tmp_path, "ModeNoWcNoHITL", [SPEC_A])
    out = orch.run_estimation_pipeline({}, _artifact())
    fitted = out.metadata["fitted_specifications"]
    assert len(fitted) == 1 and fitted[0]["reported"] and fitted[0]["verdict"] == "estimated"


def test_feedback_file_goes_to_output_dir(monkeypatch, tmp_path):
    s1 = _load("ModeNoWcWithHITL", "_gate_fb")
    monkeypatch.setattr(s1, "auto_input", lambda *a, **k: "concern")
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    cwd = tmp_path / "code_tree"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    s1.EstimationOrchestrator(quiet=True, output_dir=str(out_dir)).collect_human_feedback(
        round_number=1, context={})
    assert (out_dir / "round1_estimation_feedback.json").exists()
    assert not (cwd / "round1_estimation_feedback.json").exists()


def test_pipeline_runner_votes_before_estimation(monkeypatch, tmp_path):
    """run_estimation_team in committee mode: the vote context carries no fit statistics,
    a rejected revision is re-voted, and the fitted-spec record reaches estimation_results."""
    from shared.llm import LLMClient
    import pipeline.team_runners as tr

    proposals = [SPEC_A, SPEC_B]
    monkeypatch.setattr(LLMClient, "invoke", lambda self, messages, **kw: "narration")
    votes = []

    def fake_approved(round_number, context, question, auto_rounds=2):
        votes.append((round_number, dict(context)))
        return False

    monkeypatch.setattr(tr, "_stage_approved", fake_approved)
    monkeypatch.setattr(tr, "_collect_feedback", lambda method, ctx, **kw: "objection")
    monkeypatch.setenv("AEL_HITL_MODE", "llm_economist")
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    def fake_propose(self, *a, **k):
        return proposals.pop(0)

    import sys
    real_import = __import__("importlib").import_module

    def patched_import(name, *args):
        mod = real_import(name, *args)
        if name == "1-EstimationStage":
            monkeypatch.setattr(mod.Estimator, "propose_spec", fake_propose)
        return mod
    monkeypatch.setattr("importlib.import_module", patched_import)
    sys.modules.pop("1-EstimationStage", None)

    upstream = {"research_questions": {"final_questions": [{"question": "q?"}]},
                "model_specification": {}, "data_source": _artifact()}
    out = tr.run_estimation_team(upstream, "ModeWithWcWithHITL", str(tmp_path))
    res = out["estimation_results"]
    assert [r for r, _ in votes] == [1, 2]
    assert all(not (_FIT_KEYS & set(ctx)) for _, ctx in votes)
    assert res["fitted_specifications"][0]["spec"]["regressors"][0]["series_ref"] == "XSER"
    assert len(res["committee_objections"]) == 2
    saved = json.loads((tmp_path / "inference_output.json").read_text())
    assert saved["metadata"]["fitted_specifications"]
