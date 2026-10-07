# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""A v0.7.1 demo run regressed annual CPI inflation on
a 'real rate' built as FEDFUNDS minus year-on-year CPI inflation, i.e. on itself (R^2 = 0.996,
coefficient about -1). A regressor or instrument that uses the dependent's series at the
dependent's own lag is refused; a lagged use stays allowed."""

import numpy as np
import pandas as pd
import pytest

from EstimationTeam.ael.estim_harness import EstimationSpec, dependent_in_regressors, run_estimation


def _panel(n=70, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1955-01-01", periods=n, freq="YS")
    cpi = pd.Series(100 * np.cumprod(1 + rng.normal(0.03, 0.02, n)), index=idx)
    ff = pd.Series(4 + rng.normal(0, 2, n), index=idx)
    gini = pd.Series(40 + rng.normal(0, 1, n), index=idx)
    return {"CPIAUCSL": cpi, "FEDFUNDS": ff, "SI.POV.GINI": gini}


def _spec(lag=0):
    return EstimationSpec(**{
        "dependent": {"name": "Inflation", "series_ref": "CPIAUCSL", "transform": "pct_change"},
        "regressors": [
            {"name": "Real Interest Rate", "series_ref": "FEDFUNDS", "transform": "level",
             "subtract_ref": "CPIAUCSL", "subtract_transform": "yoy_pct_change", "lag": lag},
            {"name": "Gini", "series_ref": "SI.POV.GINI", "transform": "level"}],
    })


def test_demo_specification_is_refused():
    out = run_estimation(_spec(lag=0), _panel())
    assert out.verdict == "inestimable"
    assert out.reason == "dependent_in_regressor"
    assert "Real Interest Rate" in out.notes[-1]


def test_lagged_use_is_estimated():
    out = run_estimation(_spec(lag=1), _panel())
    assert out.verdict in ("estimated", "fragile")


def test_alias_and_interaction_are_caught():
    spec = EstimationSpec(**{
        "dependent": {"name": "Inflation", "series_ref": "cpi", "transform": "pct_change"},
        "regressors": [{"name": "Rate x CPI", "series_ref": "FEDFUNDS", "transform": "level",
                        "interact_with": "CPIAUCSL"}]})
    hits = dependent_in_regressors(spec, _panel(), {"cpi": "CPIAUCSL"})
    assert hits == ["'Rate x CPI' (interact_with)"]


def test_lagged_dependent_regressor_allowed():
    spec = EstimationSpec(**{
        "dependent": {"name": "Gini", "series_ref": "SI.POV.GINI", "transform": "level"},
        "regressors": [{"name": "Lagged Gini", "series_ref": "SI.POV.GINI", "transform": "level",
                        "lag": 1}]})
    assert dependent_in_regressors(spec, _panel(), {}) == []


# The dependent's own construction (subtract_ref, interact_with)
# is checked too - (A - B) regressed on contemporaneous B gave R^2 0.9999.

def _ab_panel(n=70, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1955-01-01", periods=n, freq="YS")
    return {"A": pd.Series(rng.normal(5, 0.01, n), index=idx),
            "B": pd.Series(rng.normal(3, 2, n), index=idx),
            "C": pd.Series(rng.normal(0, 1, n), index=idx)}


def _dep(**extra):
    return {"name": "Gap", "series_ref": "A", "transform": "level", **extra}


def test_dependent_subtrahend_as_regressor_is_refused():
    spec = EstimationSpec(**{
        "dependent": _dep(subtract_ref="B"),
        "regressors": [{"name": "B level", "series_ref": "B", "transform": "level"}],
        "hypotheses": [{"name": "h", "param": "B level", "restriction": "<0"}]})
    out = run_estimation(spec, _ab_panel())
    assert out.verdict == "inestimable"
    assert out.reason == "dependent_in_regressor"


def test_dependent_interaction_partner_in_regressor_is_refused():
    spec = EstimationSpec(**{
        "dependent": _dep(interact_with="C"),
        "regressors": [{"name": "B minus C", "series_ref": "B", "transform": "level",
                        "subtract_ref": "C"}]})
    assert dependent_in_regressors(spec, _ab_panel(), {}) == ["'B minus C' (subtract_ref)"]


def test_dependent_subtrahend_as_instrument_is_refused():
    spec = EstimationSpec(**{
        "dependent": _dep(subtract_ref="B"),
        "regressors": [{"name": "C", "series_ref": "C", "transform": "level"}],
        "method": "iv2sls", "endogenous": ["C"],
        "instruments": [{"name": "B iv", "series_ref": "b", "transform": "level"}]})
    assert dependent_in_regressors(spec, _ab_panel(), {}) == ["'B iv' (series_ref)"]


def test_lagged_dependent_constituent_allowed():
    spec = EstimationSpec(**{
        "dependent": _dep(subtract_ref="B"),
        "regressors": [{"name": "B lag", "series_ref": "B", "transform": "level", "lag": 1}]})
    assert dependent_in_regressors(spec, _ab_panel(), {}) == []
    assert run_estimation(spec, _ab_panel()).verdict in ("estimated", "fragile")


def test_unrelated_regressor_allowed_with_constructed_dependent():
    spec = EstimationSpec(**{
        "dependent": _dep(subtract_ref="B"),
        "regressors": [{"name": "C", "series_ref": "C", "transform": "level"}]})
    assert dependent_in_regressors(spec, _ab_panel(), {}) == []
