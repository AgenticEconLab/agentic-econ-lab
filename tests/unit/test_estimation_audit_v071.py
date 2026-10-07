# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Estimation-harness fixes found by auditing the v0.7.1 runs.

- interaction with a subtracted series is (X - W) * Z, not X * Z - W
- spurious-regression guard for every transform; severe residual autocorrelation -> fragile;
  HAC bandwidth from residual persistence
- pct_change on a sign-changing base is refused per observation; robustness counts only
  estimable variants
"""

import numpy as np
import pandas as pd
import pytest

from EstimationTeam.ael.estim_harness import EstimationSpec, run_estimation
from EstimationTeam.ael.estim_harness.harness import build_design


def _panel(n=80, seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1950-01-01", periods=n, freq="YS")
    x = pd.Series(5 + rng.normal(0, 2, n), index=idx)
    w = pd.Series(2 + rng.normal(0, 1, n), index=idx)
    z = pd.Series(40 + rng.normal(0, 3, n), index=idx)
    y = pd.Series(rng.normal(0, 1, n), index=idx)
    return {"X": x, "W": w, "Z": z, "Y": y}


# ---------------------------------------------------------------- interaction with a subtracted series
def test_interaction_with_subtraction_is_difference_times_partner():
    panel = _panel()
    spec = EstimationSpec(**{
        "dependent": {"name": "y", "series_ref": "Y"},
        "regressors": [{"name": "(X-W)xZ", "series_ref": "X", "subtract_ref": "W",
                        "interact_with": "Z"}]})
    design, _freq, _notes = build_design(spec, panel, {})
    expected = (panel["X"] - panel["W"]) * panel["Z"]
    np.testing.assert_allclose(design["(X-W)xZ"].values, expected.loc[design.index].values)
    wrong = panel["X"] * panel["Z"] - panel["W"]
    assert not np.allclose(design["(X-W)xZ"].values, wrong.loc[design.index].values)


def test_interaction_partner_takes_term_transform_and_subtrahend_its_own():
    panel = _panel()
    panel["P"] = pd.Series(100 * np.cumprod(1 + np.full(80, 0.02)), index=panel["X"].index)
    spec = EstimationSpec(**{
        "dependent": {"name": "y", "series_ref": "Y"},
        "regressors": [{"name": "rr_x_z", "series_ref": "X", "transform": "level",
                        "subtract_ref": "P", "subtract_transform": "pct_change",
                        "interact_with": "Z"}]})
    design, _f, _n = build_design(spec, panel, {})
    expected = (panel["X"] - panel["P"].pct_change() * 100.0) * panel["Z"]
    np.testing.assert_allclose(design["rr_x_z"].values, expected.loc[design.index].values)


# ---------------------------------------------------------------- spurious-regression guard, HAC bandwidth
from EstimationTeam.ael.estim_harness import run_diagnostics, apply_diagnostics_verdict  # noqa: E402
from EstimationTeam.ael.estim_harness.harness import hac_maxlags  # noqa: E402


def _ar1(n, rho, rng):
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = rho * e[t - 1] + rng.normal()
    return e


def test_growth_rate_regression_with_random_walk_residuals_is_fragile():
    """A pct_change dependent was exempt from every residual guard (a v0.7.1 run with DW = 0.02 reported
    'estimated'). A growth-rate regression whose residual is a random walk is downgraded."""
    rng = np.random.default_rng(7)
    n = 80
    idx = pd.date_range("1940-01-01", periods=n, freq="YS")
    x = pd.Series(rng.normal(0, 1, n), index=idx)
    growth = 0.5 * x.values + np.cumsum(rng.normal(0, 1, n))     # I(1) error in the growth rate
    level = pd.Series(100 * np.cumprod(1 + growth / 100.0), index=idx)
    spec = EstimationSpec(**{
        "dependent": {"name": "g", "series_ref": "L", "transform": "pct_change"},
        "regressors": [{"name": "x", "series_ref": "X"}]})
    out = run_estimation(spec, {"L": level, "X": x})
    assert out.verdict == "estimated"
    report = run_diagnostics(out)
    by = {r.name: r for r in report.results}
    assert by["residual_adf"].verdict != "not_applicable"
    assert by["residual_ar1"].statistic > 0.9 and by["residual_ar1"].verdict == "fail"
    assert apply_diagnostics_verdict(out, report).verdict == "fragile"


def test_severe_durbin_watson_fails_even_with_hac():
    rng = np.random.default_rng(3)
    n = 120
    idx = pd.date_range("1990-01-01", periods=n, freq="MS")
    x = pd.Series(rng.normal(0, 1, n), index=idx)
    y = pd.Series(1 + x.values + _ar1(n, 0.97, rng), index=idx)
    spec = EstimationSpec(**{"dependent": {"name": "y", "series_ref": "Y"},
                             "regressors": [{"name": "x", "series_ref": "X"}]})
    report = run_diagnostics(run_estimation(spec, {"Y": y, "X": x}))
    dw = {r.name: r for r in report.results}["durbin_watson"]
    assert dw.statistic < 0.5 and dw.verdict == "fail"
    assert report.overall == "fragile"


def test_white_noise_residuals_pass_the_autocorrelation_guards():
    panel = _panel()
    spec = EstimationSpec(**{"dependent": {"name": "y", "series_ref": "Y", "transform": "diff"},
                             "regressors": [{"name": "x", "series_ref": "X"}]})
    report = run_diagnostics(run_estimation(spec, panel))
    by = {r.name: r for r in report.results}
    assert by["residual_ar1"].verdict != "fail" and by["durbin_watson"].verdict != "fail"


def test_hac_bandwidth_grows_with_residual_persistence():
    rng = np.random.default_rng(1)
    white, _ = hac_maxlags(rng.normal(size=200))
    persistent, rho = hac_maxlags(_ar1(200, 0.8, rng))
    assert rho > 0.6
    assert persistent > white
    assert persistent <= 200 // 4


def test_hac_note_records_bandwidth_basis():
    spec = EstimationSpec(**{"dependent": {"name": "y", "series_ref": "Y"},
                             "regressors": [{"name": "x", "series_ref": "X"}]})
    out = run_estimation(spec, _panel())
    assert any("residual persistence" in n for n in out.notes)


# ---------------------------------------------------------------- pct_change on a sign-changing base
from EstimationTeam.ael.estim_harness.transforms import apply_transform  # noqa: E402


def test_pct_change_refused_where_series_changes_sign():
    idx = pd.date_range("2000-01-01", periods=6, freq="YS")
    s = pd.Series([2.0, 1.0, -1.0, -2.0, 0.0, 4.0], index=idx)
    notes = []
    out = apply_transform(s, "pct_change", "annual", notes)
    assert out.iloc[1] == pytest.approx(-50.0)       # 2 -> 1: defined
    assert np.isnan(out.iloc[2])                       # 1 -> -1: sign change
    assert out.iloc[3] == pytest.approx(100.0)        # -1 -> -2: same sign, defined
    assert np.isnan(out.iloc[4])                       # -2 -> 0: sign change
    assert np.isnan(out.iloc[5])                       # base 0
    assert notes and "3 observation(s)" in notes[0]


def test_sign_change_note_reaches_the_outcome():
    rng = np.random.default_rng(2)
    n = 120
    idx = pd.date_range("1900-01-01", periods=n, freq="YS")
    bal = pd.Series(rng.normal(0.5, 1, n), index=idx)       # a balance that crosses zero
    x = pd.Series(rng.normal(0, 1, n), index=idx)
    spec = EstimationSpec(**{"dependent": {"name": "y", "series_ref": "X"},
                             "regressors": [{"name": "bal_growth", "series_ref": "B",
                                             "transform": "pct_change"}]})
    out = run_estimation(spec, {"B": bal, "X": x})
    assert out.verdict == "estimated"
    assert any("changes sign" in n for n in out.notes)
    assert out.n_obs < n - 1


def test_inference_summary_counts_only_estimable_variants(monkeypatch):
    import importlib
    import sys
    from pathlib import Path
    from shared.llm import LLMClient
    from EstimationTeam.ael.schemas.stage_outputs import ValidationStageOutput
    from EstimationTeam.ael.estim_harness.types import DiagnosticsReport

    monkeypatch.setattr(LLMClient, "invoke", lambda self, messages, **kw: "narration")
    stage_dir = Path(__file__).resolve().parents[2] / "EstimationTeam" / "ael" / "ModeNoWcNoHITL"
    monkeypatch.syspath_prepend(str(stage_dir))
    sys.modules.pop("3-InferenceRobustnessStage", None)
    stage3 = importlib.import_module("3-InferenceRobustnessStage")

    rng = np.random.default_rng(4)
    n = 50                       # halves of 25 obs < 32 needed for 4 parameters
    idx = pd.date_range("1960-01-01", periods=n, freq="YS")
    panel = {k: pd.Series(rng.normal(0, 1, n), index=idx) for k in ("Y", "A", "B", "C")}
    spec = EstimationSpec(**{"dependent": {"name": "y", "series_ref": "Y"},
                             "regressors": [{"name": "a", "series_ref": "A"},
                                            {"name": "b", "series_ref": "B"},
                                            {"name": "c", "series_ref": "C"}]})
    outcome = run_estimation(spec, panel)
    val = ValidationStageOutput(diagnostics=DiagnosticsReport(), outcome=outcome)
    out = stage3.InferenceOrchestrator(quiet=True).run_inference_pipeline(val.model_dump())
    n_not = sum(1 for c in out.inference.robustness if c.variant_estimate is None)
    n_est = len(out.inference.robustness) - n_not
    assert n_not == 2
    assert f"{n_est} estimable robustness variant(s) ({n_not} not estimable)" in out.summary


def test_hac_lags_capped_at_sqrt_n_for_persistent_residuals():
    import math
    rng = np.random.default_rng(1)
    lags, rho = hac_maxlags(_ar1(5000, 0.97, rng))
    assert rho > 0.9 and lags <= math.ceil(math.sqrt(5000))


def test_main_effect_must_match_the_interaction_construction():
    from EstimationTeam.ael.estim_harness.harness import missing_main_effects
    spec = EstimationSpec(**{"dependent": {"name": "y", "series_ref": "Y"},
                             "regressors": [
                                 {"name": "log x", "series_ref": "X", "transform": "log"},
                                 {"name": "z", "series_ref": "Z"},
                                 {"name": "x z", "series_ref": "X", "interact_with": "Z"}]})
    notes = missing_main_effects(spec)
    assert notes and "'X'" in notes[0]
