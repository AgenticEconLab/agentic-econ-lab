# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Repair gate: after a harness refusal, a second LLM proposal was fitted and
published as 'not_reviewed', with nothing restricting it to repairing the approved
specification - it could change the dependent, the sample and the hypotheses.

Now (a) the repair must keep the approved dependent variable and the approved hypotheses'
parameters (deterministic check, every mode); (b) with a reviewer the repair is voted on
BEFORE it is fitted, with the same specification-only context as the first gate; a
rejected repair leaves the estimation inestimable with the objections recorded; (c) every
repair proposal, fitted or not, is listed in fitted_specifications."""

import pytest

from EstimationTeam.ael.estim_harness import (EstimationSpec, HypothesisSpec, VariableSpec,
                                              repair_departures)

from tests.unit.test_estimation_spec_review_gate import (_FIT_KEYS, _artifact, _orch,
                                                          _review_script)

HYP = [HypothesisSpec(name="h", param="x", restriction=">0")]
# 12 regressors on 80 obs: refused (insufficient_observations); tests 'x' > 0 on XSER
SPEC_BIG_H = EstimationSpec(
    dependent=VariableSpec(name="y", series_ref="YSER"),
    regressors=[VariableSpec(name="x", series_ref="XSER")]
    + [VariableSpec(name=f"r{i}", series_ref=("XSER", "ZSER")[i % 2], lag=1 + i // 2)
       for i in range(11)],
    hypotheses=HYP)
REPAIR_OK = EstimationSpec(dependent=VariableSpec(name="y", series_ref="YSER"),
                           regressors=[VariableSpec(name="x", series_ref="XSER")],
                           hypotheses=HYP)
REPAIR_NEW_DEP = EstimationSpec(dependent=VariableSpec(name="z", series_ref="ZSER"),
                                regressors=[VariableSpec(name="x", series_ref="XSER")],
                                hypotheses=HYP)
REPAIR_NO_HYP = EstimationSpec(dependent=VariableSpec(name="y", series_ref="YSER"),
                               regressors=[VariableSpec(name="z", series_ref="ZSER")])


@pytest.mark.parametrize("mode", ["ModeNoWcNoHITL", "ModeNoWcWithHITL"])
def test_repair_changing_the_dependent_is_not_fitted(monkeypatch, tmp_path, mode):
    orch, _ = _orch(monkeypatch, tmp_path, mode, [SPEC_BIG_H, REPAIR_NEW_DEP])
    review, seen = _review_script((True, ""))
    out = orch.run_reviewed_pipeline({}, _artifact(), review=review)
    assert out.outcome.verdict == "inestimable"
    assert out.outcome.reason == "insufficient_observations"
    assert len(seen) == 1                          # a drifting repair is not even voted on
    fitted = out.metadata["fitted_specifications"]
    assert [f["label"] for f in fitted] == ["proposal", "refusal_repair"]
    assert fitted[1]["verdict"] == "not_fitted" and fitted[1]["fitted"] is False
    assert "dependent" in fitted[1]["reason"]
    assert [f["reported"] for f in fitted] == [True, False]


def test_repair_dropping_a_hypothesis_is_not_fitted(monkeypatch, tmp_path):
    orch, _ = _orch(monkeypatch, tmp_path, "ModeNoWcWithHITL", [SPEC_BIG_H, REPAIR_NO_HYP])
    review, _ = _review_script((True, ""))
    out = orch.run_reviewed_pipeline({}, _artifact(), review=review)
    assert out.outcome.verdict == "inestimable"
    fitted = out.metadata["fitted_specifications"]
    assert fitted[1]["verdict"] == "not_fitted" and "hypothesis 'h'" in fitted[1]["reason"]


def test_repair_is_voted_before_fitting_and_rejection_keeps_inestimable(monkeypatch, tmp_path):
    orch, _ = _orch(monkeypatch, tmp_path, "ModeNoWcWithHITL", [SPEC_BIG_H, REPAIR_OK])
    review, seen = _review_script((True, ""), (False, "repair omits the controls"))
    out = orch.run_reviewed_pipeline({}, _artifact(), review=review)
    assert [r for r, _ in seen] == [1, 2]
    ctx = seen[1][1]
    assert ctx["candidate"] == "refusal_repair"
    assert not (_FIT_KEYS & set(ctx)), "no fit statistics before the repair vote"
    assert "requires" in ctx["observation_budget"]
    assert out.outcome.verdict == "inestimable"
    fitted = out.metadata["fitted_specifications"]
    assert fitted[1]["verdict"] == "not_fitted" and fitted[1]["committee"] == "rejected"
    assert any("repair omits the controls" in o for o in out.metadata["committee_objections"])
    assert out.metadata["spec_review"][-1]["candidate"] == "refusal_repair"


def test_approved_repair_is_fitted_and_reported(monkeypatch, tmp_path):
    orch, _ = _orch(monkeypatch, tmp_path, "ModeNoWcWithHITL", [SPEC_BIG_H, REPAIR_OK])
    review, seen = _review_script((True, ""), (True, "approve"))
    out = orch.run_reviewed_pipeline({}, _artifact(), review=review)
    assert out.outcome.verdict == "estimated" and out.spec_source == "llm_repaired"
    fitted = out.metadata["fitted_specifications"]
    assert fitted[1]["committee"] == "approved" and fitted[1]["reported"]


def test_repair_departures_allows_lag_change_of_tested_regressor():
    approved = EstimationSpec(
        dependent=VariableSpec(name="y", series_ref="YSER", subtract_ref="XSER"),
        regressors=[VariableSpec(name="x", series_ref="XSER")], hypotheses=HYP)
    lagged = approved.model_copy(update={"regressors": [
        VariableSpec(name="x", series_ref="xser", lag=1)]})
    assert repair_departures(approved, lagged, ["YSER", "XSER"], {}) == []
    retransformed = approved.model_copy(update={"regressors": [
        VariableSpec(name="x", series_ref="XSER", transform="log")]})
    assert repair_departures(approved, retransformed, ["YSER", "XSER"], {})
    flipped = approved.model_copy(update={"hypotheses": [
        HypothesisSpec(name="h", param="x", restriction="<0")]})
    assert repair_departures(approved, flipped, ["YSER", "XSER"], {})
