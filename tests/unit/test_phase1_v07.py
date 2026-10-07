# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""AEL V0.7 Phase 1 tests — CausalFM wrapper (back-door, IV, front-door, fallback)."""

from __future__ import annotations

import random
import pytest

from shared.econ import CausalAdjustment, CausalEstimate, CausalFMClient
from shared.econ.causal_fm import _ols_ate, _iv_ate, _front_door_ate, _solve_normal_equations


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def gen_back_door_data(n=500, true_ate=2.0, seed=42):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        x1 = rng.gauss(0, 1)
        x2 = rng.gauss(0, 1)
        # T depends on confounders
        t = 0.5 * x1 - 0.3 * x2 + rng.gauss(0, 0.5)
        # Y = ATE * T + confounders
        y = true_ate * t + 0.7 * x1 + 0.4 * x2 + rng.gauss(0, 0.3)
        rows.append({"T": t, "Y": y, "X1": x1, "X2": x2})
    return rows


def gen_iv_data(n=500, true_ate=1.5, seed=7):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        z = rng.gauss(0, 1)
        u = rng.gauss(0, 1)        # unobserved confounder
        t = 0.8 * z + 0.6 * u + rng.gauss(0, 0.3)
        y = true_ate * t + 0.9 * u + rng.gauss(0, 0.3)
        rows.append({"T": t, "Y": y, "Z": z})
    return rows


def gen_front_door_data(n=500, true_ate=1.2, seed=3):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        u = rng.gauss(0, 1)
        t = 0.6 * u + rng.gauss(0, 0.5)
        # Mediator depends only on T
        m = 0.9 * t + rng.gauss(0, 0.3)
        # Y depends on M and U (not T directly)
        y = (true_ate / 0.9) * m + 0.5 * u + rng.gauss(0, 0.3)
        rows.append({"T": t, "Y": y, "M": m})
    return rows


# ---------------------------------------------------------------------------
# Core estimator functions
# ---------------------------------------------------------------------------

class TestLinearAlgebra:
    def test_solves_small_system(self):
        # 2x + 3y = 8; x + 2y = 5 → x=1, y=2
        # Phrased as OLS: X=[[1,2],[1,1]] regressing y=[8,5] on (intercept??)
        # Simpler: X = [[2,3],[1,2]], beta = solve normal eqns won't be identity.
        # Use a canonical OLS test instead:
        X = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]
        y = [2.1, 3.9, 6.0, 8.1]
        beta = _solve_normal_equations(X, y)
        # slope ~= 2, intercept ~= 0.05
        assert abs(beta[1] - 2.0) < 0.1
        assert abs(beta[0]) < 0.3

    def test_singular_returns_zeros(self):
        # X has duplicate columns -> singular X'X
        X = [[1.0, 1.0], [1.0, 1.0]]
        y = [1.0, 2.0]
        beta = _solve_normal_equations(X, y)
        assert beta == [0.0, 0.0]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _solve_normal_equations([], [])


class TestOLSATE:
    def test_recovers_back_door_ate(self):
        rows = gen_back_door_data(n=800, true_ate=2.0)
        ate = _ols_ate(rows, "T", "Y", ["X1", "X2"])
        assert 1.7 < ate < 2.3

    def test_bias_without_adjustment(self):
        rows = gen_back_door_data(n=800, true_ate=2.0)
        naive = _ols_ate(rows, "T", "Y", [])
        adjusted = _ols_ate(rows, "T", "Y", ["X1", "X2"])
        assert abs(adjusted - 2.0) < abs(naive - 2.0)


class Test2SLS:
    def test_recovers_iv_ate(self):
        rows = gen_iv_data(n=800, true_ate=1.5)
        ate = _iv_ate(rows, "T", "Y", "Z")
        assert 1.2 < ate < 1.8

    def test_ols_is_biased(self):
        rows = gen_iv_data(n=800, true_ate=1.5)
        ols = _ols_ate(rows, "T", "Y", [])
        iv = _iv_ate(rows, "T", "Y", "Z")
        assert abs(iv - 1.5) < abs(ols - 1.5)


class TestFrontDoor:
    def test_recovers_front_door_ate(self):
        rows = gen_front_door_data(n=800, true_ate=1.2)
        ate = _front_door_ate(rows, "T", "Y", ["M"])
        assert 0.9 < ate < 1.5

    def test_requires_mediator(self):
        rows = gen_front_door_data(n=200)
        with pytest.raises(ValueError):
            _front_door_ate(rows, "T", "Y", [])


# ---------------------------------------------------------------------------
# CausalAdjustment schema
# ---------------------------------------------------------------------------

class TestCausalAdjustment:
    def test_back_door_basic(self):
        spec = CausalAdjustment(method="back_door", treatment="T", outcome="Y",
                                 adjustment_set=["X1"])
        assert spec.method == "back_door"

    def test_iv_requires_instrument(self):
        with pytest.raises(ValueError):
            CausalAdjustment(method="iv", treatment="T", outcome="Y")

    def test_iv_with_instrument_ok(self):
        spec = CausalAdjustment(method="iv", treatment="T", outcome="Y", instrument="Z")
        assert spec.instrument == "Z"


# ---------------------------------------------------------------------------
# CausalFMClient end-to-end
# ---------------------------------------------------------------------------

class TestCausalFMClient:
    def test_back_door_estimate(self):
        rows = gen_back_door_data(n=400, true_ate=2.0)
        client = CausalFMClient(n_bootstrap=80, seed=0)
        est = client.estimate(rows, CausalAdjustment(
            method="back_door", treatment="T", outcome="Y",
            adjustment_set=["X1", "X2"],
        ))
        assert isinstance(est, CausalEstimate)
        assert 1.6 < est.ate < 2.4
        assert est.sample_size == 400
        assert est.method == "back_door_ols"
        assert est.backend == "fallback_ols"
        assert est.ci_low < est.ate < est.ci_high
        assert len(est.posterior_samples) > 0

    def test_iv_estimate(self):
        rows = gen_iv_data(n=400, true_ate=1.5)
        client = CausalFMClient(n_bootstrap=80, seed=1)
        est = client.estimate(rows, CausalAdjustment(
            method="iv", treatment="T", outcome="Y", instrument="Z",
        ))
        assert 1.1 < est.ate < 1.9
        assert est.method == "iv_2sls"

    def test_front_door_estimate(self):
        rows = gen_front_door_data(n=400, true_ate=1.2)
        client = CausalFMClient(n_bootstrap=80, seed=2)
        est = client.estimate(rows, CausalAdjustment(
            method="front_door", treatment="T", outcome="Y",
            adjustment_set=["M"],
        ))
        assert 0.8 < est.ate < 1.6

    def test_empty_data_raises(self):
        client = CausalFMClient()
        with pytest.raises(ValueError):
            client.estimate([], CausalAdjustment(
                method="back_door", treatment="T", outcome="Y",
            ))

    def test_missing_column_raises(self):
        rows = [{"T": 1, "Y": 2}]
        client = CausalFMClient()
        with pytest.raises(ValueError):
            client.estimate(rows, CausalAdjustment(
                method="back_door", treatment="T", outcome="Y",
                adjustment_set=["Missing"],
            ))

    def test_accepts_dict_of_columns(self):
        rows = gen_back_door_data(n=200, true_ate=2.0)
        cols = {k: [r[k] for r in rows] for k in rows[0]}
        client = CausalFMClient(n_bootstrap=40)
        est = client.estimate(cols, CausalAdjustment(
            method="back_door", treatment="T", outcome="Y",
            adjustment_set=["X1", "X2"],
        ))
        assert 1.6 < est.ate < 2.4

    def test_accepts_pandas_like_via_to_dict(self):
        class FakeDF:
            def __init__(self, rows): self._rows = rows
            def to_dict(self, orient): return self._rows
        rows = gen_back_door_data(n=200, true_ate=2.0)
        est = CausalFMClient(n_bootstrap=30).estimate(FakeDF(rows), CausalAdjustment(
            method="back_door", treatment="T", outcome="Y",
            adjustment_set=["X1", "X2"],
        ))
        assert 1.5 < est.ate < 2.5

    def test_estimate_reproducible_under_seed(self):
        rows = gen_back_door_data(n=200, true_ate=2.0)
        a = CausalFMClient(n_bootstrap=50, seed=1234).estimate(
            rows, CausalAdjustment("back_door", "T", "Y", ["X1", "X2"]),
        )
        b = CausalFMClient(n_bootstrap=50, seed=1234).estimate(
            rows, CausalAdjustment("back_door", "T", "Y", ["X1", "X2"]),
        )
        assert a.ate == pytest.approx(b.ate)
        assert a.ci_low == pytest.approx(b.ci_low)

    def test_posterior_ci_coverage(self):
        # 95% CI should contain true ATE in well-identified back-door case
        rows = gen_back_door_data(n=600, true_ate=2.0)
        client = CausalFMClient(n_bootstrap=150)
        est = client.estimate(rows, CausalAdjustment(
            method="back_door", treatment="T", outcome="Y",
            adjustment_set=["X1", "X2"],
        ))
        assert est.ci_low < 2.0 < est.ci_high

    def test_to_dict_serializable(self):
        rows = gen_back_door_data(n=200)
        est = CausalFMClient(n_bootstrap=30).estimate(
            rows, CausalAdjustment("back_door", "T", "Y", ["X1", "X2"]),
        )
        d = est.to_dict()
        assert d["method"] == "back_door_ols"
        assert "posterior_samples_n" in d
        assert d["posterior_samples_n"] > 0

    def test_collector_argument_backward_compat(self):
        # Constructor must accept collector=None for back-compat with V0.3+
        client = CausalFMClient(collector=None)
        rows = gen_back_door_data(n=100)
        est = client.estimate(rows, CausalAdjustment(
            "back_door", "T", "Y", ["X1", "X2"],
        ))
        assert est.backend == "fallback_ols"

    def test_ate_std_non_negative(self):
        rows = gen_back_door_data(n=300)
        est = CausalFMClient(n_bootstrap=50).estimate(
            rows, CausalAdjustment("back_door", "T", "Y", ["X1", "X2"]),
        )
        assert est.ate_std >= 0.0
