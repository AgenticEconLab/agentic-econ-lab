# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""EstimationTeam deterministic harness: recovery, honest refusals, diagnostics, inference.

The non-negotiable test is coefficient recovery on synthetic data with known parameters —
if the harness cannot recover b=3 from y = 2 + 3x + eps, nothing downstream is trustworthy.
"""

import numpy as np
import pandas as pd
import pytest

from EstimationTeam.ael.estim_harness import (
    VERDICT_ESTIMATED,
    VERDICT_FRAGILE,
    VERDICT_INESTIMABLE,
    EstimationSpec,
    HypothesisSpec,
    VariableSpec,
    apply_diagnostics_verdict,
    find_retrieved_data,
    load_panel,
    run_diagnostics,
    run_estimation,
    run_inference,
)
from EstimationTeam.ael.estim_harness.transforms import align_panel, apply_transform, infer_frequency


def _dates(n, freq="MS", start="1990-01-01"):
    return pd.date_range(start, periods=n, freq=freq)


def _panel_xy(n=200, b0=2.0, b1=3.0, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    y = b0 + b1 * x + rng.normal(0, noise, n)
    return pd.DataFrame({"X1": x, "Y": y}, index=_dates(n))


def _spec(**kw):
    base = dict(
        dependent=VariableSpec(name="y", series_ref="Y"),
        regressors=[VariableSpec(name="x1", series_ref="X1")],
    )
    base.update(kw)
    return EstimationSpec(**base)


class TestRecovery:
    def test_ols_recovers_known_coefficients(self):
        out = run_estimation(_spec(), _panel_xy())
        assert out.verdict == VERDICT_ESTIMATED
        slope = next(c for c in out.coefficients if c.name == "x1")
        const = next(c for c in out.coefficients if c.name == "const")
        assert slope.ci_low < 3.0 < slope.ci_high
        assert const.ci_low < 2.0 < const.ci_high
        assert out.r_squared > 0.8
        assert out.analysis_data and out.spec is not None

    def test_alias_and_case_insensitive_refs(self):
        panel = _panel_xy()
        spec = _spec(dependent=VariableSpec(name="y", series_ref="output gap"),
                     regressors=[VariableSpec(name="x1", series_ref="x1")])
        out = run_estimation(spec, panel, alias_map={"output gap": "Y", "x1": "X1"})
        assert out.verdict == VERDICT_ESTIMATED


class TestHonestRefusals:
    def test_missing_variable(self):
        out = run_estimation(_spec(regressors=[VariableSpec(name="z", series_ref="NOPE")]),
                             _panel_xy())
        assert out.verdict == VERDICT_INESTIMABLE and out.reason == "missing_variable"

    def test_insufficient_observations(self):
        out = run_estimation(_spec(), _panel_xy(n=12))
        assert out.verdict == VERDICT_INESTIMABLE and out.reason == "insufficient_observations"

    def test_singular_design(self):
        panel = _panel_xy()
        panel["X2"] = 2.0 * panel["X1"]
        spec = _spec(regressors=[VariableSpec(name="x1", series_ref="X1"),
                                 VariableSpec(name="x2", series_ref="X2")])
        out = run_estimation(spec, panel)
        assert out.verdict == VERDICT_INESTIMABLE and out.reason == "singular_design"

    def test_degenerate_dependent(self):
        panel = _panel_xy()
        panel["Y"] = 1.23
        out = run_estimation(_spec(), panel)
        assert out.verdict == VERDICT_INESTIMABLE and out.reason == "degenerate_dependent"

    def test_panel_fe_is_declined_not_substituted(self):
        out = run_estimation(_spec(method="panel_fe"), _panel_xy())
        assert out.verdict == VERDICT_INESTIMABLE and out.reason == "method_unavailable"
        assert not out.coefficients

    def test_iv_without_instruments_is_invalid_spec(self):
        out = run_estimation(_spec(method="iv2sls"), _panel_xy())
        assert out.verdict == VERDICT_INESTIMABLE and out.reason == "invalid_spec"

    def test_iv_under_identified_declines(self):
        spec = _spec(method="iv2sls",
                     regressors=[VariableSpec(name="x1", series_ref="X1"),
                                 VariableSpec(name="x2", series_ref="X2")])
        spec.endogenous = ["x1", "x2"]
        spec.instruments = [VariableSpec(name="z1", series_ref="Z1")]
        panel = _panel_xy()
        panel["X2"] = panel["X1"] * 0.5 + 1.0
        panel["Z1"] = panel["X1"] * 0.3
        out = run_estimation(spec, panel)
        assert out.verdict == VERDICT_INESTIMABLE and out.reason == "under_identified"

    def test_log_of_negative_series_refused(self):
        panel = _panel_xy()  # X1 is standard normal -> negative values
        spec = _spec(regressors=[VariableSpec(name="x1", series_ref="X1", transform="log")])
        out = run_estimation(spec, panel)
        assert out.verdict == VERDICT_INESTIMABLE and out.reason == "invalid_spec"


class TestTransformsAndAlignment:
    def test_infer_frequency(self):
        assert infer_frequency(_dates(24, "MS")) == "monthly"
        assert infer_frequency(_dates(24, "QS")) == "quarterly"
        assert infer_frequency(_dates(24, "YS")) == "annual"

    def test_align_mixed_frequencies_to_coarsest(self):
        monthly = pd.Series(np.arange(120.0), index=_dates(120, "MS"))
        annual = pd.Series(np.arange(10.0), index=_dates(10, "YS"))
        panel, freq = align_panel({"m": monthly, "a": annual})
        assert freq == "annual"
        assert 8 <= len(panel) <= 10

    def test_yoy_pct_change(self):
        s = pd.Series([100.0] * 12 + [110.0] * 12, index=_dates(24, "MS"))
        out = apply_transform(s, "yoy_pct_change", "monthly").dropna()
        assert out.iloc[0] == pytest.approx(10.0)

    def test_lag_shifts_sample(self):
        spec = _spec(regressors=[VariableSpec(name="x1", series_ref="X1", lag=2)])
        out = run_estimation(spec, _panel_xy(n=100))
        assert out.verdict == VERDICT_ESTIMATED and out.n_obs == 98


class TestDiagnostics:
    def test_clean_dgp_not_fragile(self):
        out = run_estimation(_spec(), _panel_xy())
        report = run_diagnostics(out)
        assert report.overall in ("clean", "caveats")
        assert not report.severe_failures

    def test_spurious_levels_regression_flagged_and_downgraded(self):
        rng = np.random.default_rng(0)  # canonical draw: residual ADF p=0.83 (I(1) residuals)
        n = 300
        walk1 = np.cumsum(rng.normal(0, 1, n)) + 100
        walk2 = np.cumsum(rng.normal(0, 1, n)) + 100
        panel = pd.DataFrame({"Y": walk1, "X1": walk2}, index=_dates(n))
        out = run_estimation(_spec(), panel)
        assert out.verdict == VERDICT_ESTIMATED
        report = run_diagnostics(out)
        adf = next(r for r in report.results if r.name == "residual_adf")
        assert adf.verdict == "fail"
        downgraded = apply_diagnostics_verdict(out, report)
        assert downgraded.verdict == VERDICT_FRAGILE
        assert any("downgraded" in note for note in downgraded.notes)

    def test_differenced_regression_runs_adf_guard(self):
        # the spurious-regression guard runs for every transform; stationary residuals pass.
        out = run_estimation(
            _spec(dependent=VariableSpec(name="y", series_ref="Y", transform="diff"),
                  regressors=[VariableSpec(name="x1", series_ref="X1", transform="diff")]),
            _panel_xy())
        report = run_diagnostics(out)
        adf = next(r for r in report.results if r.name == "residual_adf")
        assert adf.verdict == "pass"


class TestInference:
    def test_true_positive_slope_hypotheses(self):
        spec = _spec(hypotheses=[
            HypothesisSpec(name="slope_positive", param="x1", restriction=">0"),
            HypothesisSpec(name="slope_negative", param="x1", restriction="<0"),
            HypothesisSpec(name="slope_zero", param="x1", restriction="=0"),
        ])
        report = run_inference(run_estimation(spec, _panel_xy()))
        by_name = {h.name: h for h in report.hypotheses}
        assert by_name["slope_positive"].supported is True
        assert by_name["slope_negative"].supported is False
        assert by_name["slope_zero"].supported is False  # b=3 rejects b=0

    def test_outcome_distinguishes_contradicted_from_inconclusive(self):
        """A wrong-sign prediction is 'contradicted' only when the
        estimate is significantly on the other side; an imprecise estimate is 'inconclusive'
        (a live run reported b = 0.0001, se = 0.042, p = 0.998 as a rejected hypothesis)."""
        strong = _spec(hypotheses=[
            HypothesisSpec(name="pos", param="x1", restriction=">0"),
            HypothesisSpec(name="neg", param="x1", restriction="<0"),
            HypothesisSpec(name="zero", param="x1", restriction="=0"),
        ])
        by = {h.name: h for h in run_inference(run_estimation(strong, _panel_xy())).hypotheses}
        assert by["pos"].outcome == "supported"
        assert by["neg"].outcome == "contradicted"          # b = 3, precisely estimated
        assert by["zero"].outcome == "rejected"

        weak = _spec(hypotheses=[
            HypothesisSpec(name="neg", param="x1", restriction="<0"),
            HypothesisSpec(name="zero", param="x1", restriction="=0"),
        ])
        by = {h.name: h for h in run_inference(run_estimation(weak, _panel_xy(b1=0.0))).hypotheses}
        assert by["neg"].outcome == "inconclusive"          # true slope 0: not evidence against
        assert by["neg"].supported is False
        assert by["zero"].outcome == "consistent"

    def test_missing_param_hypothesis_is_reported_not_dropped(self):
        spec = _spec(hypotheses=[HypothesisSpec(name="ghost", param="zz", restriction=">0")])
        report = run_inference(run_estimation(spec, _panel_xy()))
        assert report.hypotheses[0].supported is None

    def test_stable_dgp_survives_robustness(self):
        rng = np.random.default_rng(3)
        n = 240
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        y = 1.0 + 2.5 * x1 + 0.5 * x2 + rng.normal(0, 1, n)
        panel = pd.DataFrame({"Y": y, "X1": x1, "X2": x2}, index=_dates(n))
        spec = _spec(regressors=[VariableSpec(name="x1", series_ref="X1"),
                                 VariableSpec(name="x2", series_ref="X2")],
                     hypotheses=[HypothesisSpec(name="h", param="x1", restriction=">0")])
        report = run_inference(run_estimation(spec, panel))
        assert report.robustness, "expected drop-one/split-sample/cov-swap checks"
        assert report.stability_score == pytest.approx(1.0)


class TestDataLoader:
    def _artifact(self):
        preview = [{"date": f"20{10 + i}-01-01", "SER1": float(100 + i)} for i in range(5)]
        return {"name": "validated_dataset", "data": {"metadata": {}, "retrieved_data": [
            {"series_id": "SER1", "series_name": "Some Series", "source_name": "CustomSource",
             "num_observations": 5, "data_preview": preview, "data_simulated": False},
            {"series_id": "SIM1", "series_name": "Simulated", "source_name": "CustomSource",
             "num_observations": 40, "data_preview": preview, "data_simulated": True},
        ]}}

    def test_find_retrieved_data_in_nested_envelope(self):
        assert len(find_retrieved_data(self._artifact())) == 2

    def test_simulated_and_short_series_excluded_disclosed(self, monkeypatch):
        """Simulated series never become empirics; a 5-point preview fallback is excluded
        rather than allowed to poison every other series' overlap in the inner join."""
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        panel, _alias_map, notes = load_panel(self._artifact())
        assert panel is None
        assert any("data_simulated" in n for n in notes)
        assert any("poison the panel overlap" in n for n in notes)

    def test_preview_fallback_fails_min_obs_gate_honestly(self, monkeypatch):
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        panel, alias_map, notes = load_panel(self._artifact(), min_series_obs=3)
        assert panel is not None and list(panel.keys()) == ["SER1"]
        assert alias_map["some series"] == "SER1"
        assert any("preview" in n for n in notes)
        spec = EstimationSpec(
            dependent=VariableSpec(name="y", series_ref="SER1"),
            regressors=[VariableSpec(name="x", series_ref="SER1", lag=1)])
        out = run_estimation(spec, panel, alias_map)
        assert out.verdict == VERDICT_INESTIMABLE
        assert out.reason == "insufficient_observations"

    def test_repaired_canonical_id_recovered_from_quality_notes(self):
        from EstimationTeam.ael.estim_harness.data_loader import repaired_fetch_id
        item = {"quality_notes": "Successfully retrieved 8 observations from BLS (open API) "
                                 "[id repaired via DBnomics search: 'LAU20000000000003' (BLS) "
                                 "-> OECD/DSD_SDBSBSC_ISIC4@DF_SDBS_ISIC4/A.USA.EMPE.A._T.PS]"}
        assert repaired_fetch_id(item) == "OECD/DSD_SDBSBSC_ISIC4@DF_SDBS_ISIC4/A.USA.EMPE.A._T.PS"
        assert repaired_fetch_id({"quality_notes": "Successfully retrieved"}) is None


class TestIVEstimation:
    """2SLS on a textbook endogenous DGP — OLS is biased, IV recovers the truth."""

    @staticmethod
    def _endogenous_panel(n=400, b=2.0, seed=12):
        rng = np.random.default_rng(seed)
        z = rng.normal(0, 1, n)                       # instrument
        u = rng.normal(0, 1, n)                       # confounder
        x = 0.8 * z + u                               # endogenous regressor
        y = b * x + 0.9 * u + rng.normal(0, 0.3, n)   # corr(x, error) != 0
        return pd.DataFrame({"Y": y, "X": x, "Z": z}, index=_dates(n))

    def _iv_spec(self):
        spec = EstimationSpec(
            dependent=VariableSpec(name="y", series_ref="Y"),
            regressors=[VariableSpec(name="x", series_ref="X")],
            method="iv2sls",
            endogenous=["x"],
            instruments=[VariableSpec(name="z", series_ref="Z")],
            hypotheses=[HypothesisSpec(name="pos", param="x", restriction=">0")])
        return spec

    def test_iv_recovers_truth_where_ols_is_biased(self):
        panel = self._endogenous_panel(b=2.0)
        ols = run_estimation(_spec(dependent=VariableSpec(name="y", series_ref="Y"),
                                   regressors=[VariableSpec(name="x", series_ref="X")]), panel)
        ols_slope = next(c for c in ols.coefficients if c.name == "x")
        assert ols_slope.estimate > 2.2               # attenuated toward the confounder

        iv = run_estimation(self._iv_spec(), panel)
        assert iv.verdict == VERDICT_ESTIMATED and iv.method == "iv2sls"
        iv_slope = next(c for c in iv.coefficients if c.name == "x")
        assert iv_slope.ci_low < 2.0 < iv_slope.ci_high
        assert abs(iv_slope.estimate - 2.0) < abs(ols_slope.estimate - 2.0)

    def test_iv_diagnostics_and_inference_run(self):
        panel = self._endogenous_panel()
        out = run_estimation(self._iv_spec(), panel)
        report = run_diagnostics(out)
        names = {r.name for r in report.results}
        assert "breusch_pagan" in names and "durbin_watson" in names
        reset = next(r for r in report.results if r.name == "ramsey_reset")
        assert reset.verdict == "not_applicable"      # statsmodels-only test degrades honestly
        inf = run_inference(out)
        assert inf.hypotheses[0].supported is True
        assert inf.robustness                          # split-sample + cov swap still run


class TestTrendOption:
    def test_include_trend_adds_deterministic_trend_regressor(self):
        spec = _spec(include_trend=True)
        out = run_estimation(spec, _panel_xy())
        names = [c.name for c in out.coefficients]
        assert "trend" in names and out.verdict == VERDICT_ESTIMATED
        assert any("common-trend guard" in n for n in out.notes)
        # trend column persisted for deterministic stage-2/3 refits
        assert "trend" in out.analysis_data[0]

    def test_trend_survives_refit_and_robustness(self):
        spec = _spec(include_trend=True,
                     hypotheses=[HypothesisSpec(name="h", param="x1", restriction=">0")])
        out = run_estimation(spec, _panel_xy())
        report = run_diagnostics(out)
        assert report.results, "refit with trend must run the battery"
        inf = run_inference(out)
        assert inf.hypotheses[0].supported is True


class TestInteractionsAndDedup:
    """A live spec encoded 'Fed Funds x Debt Service' as a THIRD
    regressor identical to regressor 1 -> rank-deficient -> total refusal. The harness now
    (a) drops duplicate columns with a disclosed note and estimates the rest, and (b) makes
    interactions genuinely expressible via `interact_with`."""

    @staticmethod
    def _panel_interaction(n=300, seed=0):
        rng = np.random.default_rng(seed)
        a = rng.normal(0, 1, n)
        b = rng.normal(0, 1, n)
        y = 1.0 + 2.0 * a + 3.0 * b + 4.0 * a * b + rng.normal(0, 0.5, n)
        return pd.DataFrame({"A": a, "B": b, "Y": y}, index=_dates(n))

    def test_duplicate_interaction_column_estimates_remainder_with_note(self):
        # the live-run shape: regressor 3 duplicates regressor 1's column
        panel = self._panel_interaction()
        spec = _spec(regressors=[
            VariableSpec(name="Fed Funds Rate", series_ref="A"),
            VariableSpec(name="Debt Service", series_ref="B"),
            VariableSpec(name="Interaction: Fed Funds x Debt Service", series_ref="A"),
        ])
        out = run_estimation(spec, panel)
        assert out.verdict == VERDICT_ESTIMATED, (out.reason, out.notes)
        assert len(out.spec.regressors) == 2                      # duplicate dropped
        assert any("dropped duplicate regressor" in n for n in out.notes)
        assert any("interact_with" in n for n in out.notes)       # remedy disclosed

    def test_distinct_transforms_of_same_series_are_not_deduped(self):
        panel = self._panel_interaction()
        spec = _spec(regressors=[
            VariableSpec(name="a_level", series_ref="A"),
            VariableSpec(name="a_lag", series_ref="A", lag=1),
        ])
        out = run_estimation(spec, panel)
        assert len(out.spec.regressors) == 2
        assert not any("dropped duplicate" in n for n in out.notes)

    def test_interact_with_recovers_product_coefficient(self):
        panel = self._panel_interaction()
        spec = _spec(regressors=[
            VariableSpec(name="a", series_ref="A"),
            VariableSpec(name="b", series_ref="B"),
            VariableSpec(name="a_x_b", series_ref="A", interact_with="B"),
        ])
        out = run_estimation(spec, panel)
        assert out.verdict == VERDICT_ESTIMATED, (out.reason, out.notes)
        inter = next(c for c in out.coefficients if c.name == "a_x_b")
        assert inter.ci_low < 4.0 < inter.ci_high
        assert out.r_squared > 0.9

    def test_interact_with_distinguishes_otherwise_identical_regressors(self):
        panel = self._panel_interaction()
        spec = _spec(regressors=[
            VariableSpec(name="a", series_ref="A"),
            VariableSpec(name="a_x_b", series_ref="A", interact_with="B"),
        ])
        out = run_estimation(spec, panel)
        assert out.verdict == VERDICT_ESTIMATED
        assert len(out.spec.regressors) == 2      # NOT deduped: interaction is distinct

    def test_interact_with_missing_series_is_honest_refusal(self):
        panel = self._panel_interaction()
        spec = _spec(regressors=[
            VariableSpec(name="a_x_z", series_ref="A", interact_with="NO_SUCH"),
        ])
        out = run_estimation(spec, panel)
        assert out.verdict == "inestimable" and out.reason == "missing_variable"


class TestRobustnessOnInteractions:
    """Every robustness variant returned None stability because _rerun
    kept `interact_with` live while re-estimating on the already-transformed panel."""

    def test_robustness_produces_verdicts_for_interaction_spec(self):
        from EstimationTeam.ael.estim_harness.inference import run_inference
        rng = np.random.default_rng(0)
        n = 200
        a, b = rng.normal(0, 1, n), rng.normal(0, 1, n)
        y = 1 + 2*a + 3*b + 4*a*b + rng.normal(0, 0.5, n)
        panel = pd.DataFrame({"A": a, "B": b, "Y": y}, index=_dates(n))
        spec = _spec(regressors=[
            VariableSpec(name="a", series_ref="A"),
            VariableSpec(name="b", series_ref="B"),
            VariableSpec(name="a_x_b", series_ref="A", interact_with="B"),
        ], hypotheses=[HypothesisSpec(name="h", param="a_x_b", restriction=">0")])
        out = run_estimation(spec, panel)
        assert out.verdict == VERDICT_ESTIMATED
        rep = run_inference(out)
        judged = [c for c in rep.robustness if c.sign_stable is not None]
        assert judged, [c.detail for c in rep.robustness]   # variants must be estimable
        assert all(c.sign_stable for c in judged)           # x4 interaction is robust
        assert rep.stability_score is not None


class TestDerivedDifferences:
    """'Real interest rate' was proxied by the NOMINAL rate
    because r = i - pi had no spec form. subtract_ref makes the derivation expressible."""

    @staticmethod
    def _panel_real_rate(n=250, seed=0):
        rng = np.random.default_rng(seed)
        i_nom = rng.normal(4, 1, n)
        pi = rng.normal(2, 1, n)
        y = 1.0 - 0.8 * (i_nom - pi) + rng.normal(0, 0.4, n)
        return pd.DataFrame({"NOM": i_nom, "INF": pi, "Y": y}, index=_dates(n))

    def test_subtract_ref_recovers_real_rate_coefficient(self):
        panel = self._panel_real_rate()
        spec = _spec(regressors=[
            VariableSpec(name="real_rate", series_ref="NOM", subtract_ref="INF"),
        ])
        out = run_estimation(spec, panel)
        assert out.verdict == VERDICT_ESTIMATED, (out.reason, out.notes)
        rr = next(c for c in out.coefficients if c.name == "real_rate")
        assert rr.ci_low < -0.8 < rr.ci_high
        assert out.r_squared > 0.7

    def test_subtract_ref_distinguishes_dedup_key(self):
        panel = self._panel_real_rate()
        spec = _spec(regressors=[
            VariableSpec(name="nominal", series_ref="NOM"),
            VariableSpec(name="real_rate", series_ref="NOM", subtract_ref="INF"),
        ])
        out = run_estimation(spec, panel)
        assert out.verdict == VERDICT_ESTIMATED
        assert len(out.spec.regressors) == 2      # NOT deduped: derivation is distinct

    def test_subtract_ref_missing_series_honest_refusal(self):
        panel = self._panel_real_rate()
        spec = _spec(regressors=[
            VariableSpec(name="real_rate", series_ref="NOM", subtract_ref="NO_SUCH"),
        ])
        out = run_estimation(spec, panel)
        assert out.verdict == "inestimable" and out.reason == "missing_variable"

    def test_robustness_variants_work_on_derived_spec(self):
        from EstimationTeam.ael.estim_harness.inference import run_inference
        panel = self._panel_real_rate()
        spec = _spec(regressors=[
            VariableSpec(name="real_rate", series_ref="NOM", subtract_ref="INF"),
            VariableSpec(name="infl", series_ref="INF"),
        ], hypotheses=[HypothesisSpec(name="h", param="real_rate", restriction="<0")])
        out = run_estimation(spec, panel)
        rep = run_inference(out)
        judged = [c for c in rep.robustness if c.sign_stable is not None]
        assert judged and rep.stability_score is not None   # interaction neutralization covers subtract_ref


class TestMixedTransformDifferences:
    """A live 'real rate' subtracted the price LEVEL from the
    nominal rate because subtract_ref forced a common transform — a degenerate regressor
    the harness could only report as an exact zero. subtract_transform fixes the class."""

    def test_real_rate_from_rate_minus_index_inflation(self):
        rng = np.random.default_rng(0)
        n = 300
        infl = rng.normal(2, 0.8, n)                       # true yoy inflation (%)
        price = 100.0 * np.cumprod(1 + infl / 100 / 12)    # a price INDEX level
        nom = 2.0 + infl + rng.normal(0, 0.3, n)           # nominal rate tracks inflation
        panel = pd.DataFrame({"NOM": nom, "PIDX": price}, index=_dates(n))
        real_true = nom - pd.Series(price, index=panel.index).pct_change(12).mul(100.0)
        y = 1.0 - 0.5 * real_true + rng.normal(0, 0.2, n)
        panel["Y"] = y
        spec = _spec(regressors=[
            VariableSpec(name="real_rate", series_ref="NOM", transform="level",
                         subtract_ref="PIDX", subtract_transform="yoy_pct_change"),
        ])
        out = run_estimation(spec, panel)
        assert out.verdict == VERDICT_ESTIMATED, (out.reason, out.notes)
        rr = next(c for c in out.coefficients if c.name == "real_rate")
        assert rr.ci_low < -0.5 < rr.ci_high      # recovers the true coefficient
        assert out.r_squared > 0.6

    def test_subtract_transform_distinguishes_dedup_key(self):
        rng = np.random.default_rng(1)
        n = 200
        panel = pd.DataFrame({"A": rng.normal(4, 1, n) + 10,
                              "B": rng.normal(2, 1, n) + 10,
                              "Y": rng.normal(0, 1, n)}, index=_dates(n))
        panel["Y"] = panel["A"] - panel["B"] + rng.normal(0, 0.3, n)
        spec = _spec(regressors=[
            VariableSpec(name="d_level", series_ref="A", subtract_ref="B"),
            VariableSpec(name="d_mixed", series_ref="A", subtract_ref="B",
                         subtract_transform="diff"),
        ])
        out = run_estimation(spec, panel)
        assert len(out.spec.regressors) == 2      # NOT deduped: different subtrahend transforms


class TestIdentificationStatement:
    """The identification statement is part of the artifact —
    method-derived, discipline-free; proxy disclosure keys on the spec's own naming."""

    def test_ols_carries_association_statement(self):
        out = run_estimation(_spec(), _panel_xy())
        assert any(n.startswith("identification:") and "no causal" in n for n in out.notes)

    def test_proxy_named_regressor_is_disclosed(self):
        panel = _panel_xy()
        spec = _spec(regressors=[VariableSpec(name="Expectations Proxy", series_ref="X1")])
        out = run_estimation(spec, panel)
        assert any(n.startswith("proxy disclosure:") and "Expectations Proxy" in n
                   for n in out.notes)

    def test_no_proxy_note_without_proxy_naming(self):
        out = run_estimation(_spec(), _panel_xy())
        assert not any(n.startswith("proxy disclosure:") for n in out.notes)
