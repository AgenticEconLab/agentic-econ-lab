# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""ReportingTeam deterministic harness: magnitudes, assembly, and the consistency checker.

The non-negotiable tests: (a) effect sizes computed correctly from a known panel, and
(b) the consistency checker flags a hallucinated number but accepts correct roundings."""

import numpy as np
import pytest

from ReportingTeam.ael.report_harness import (
    assemble_report,
    check_consistency,
    collect_limitations,
    interpret_estimation,
)


def _estimation_results(n=60, b=2.0, verdict="estimated", seed=4):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1.5, n)
    y = 1.0 + b * x + rng.normal(0, 0.5, n)
    dates = [f"{1990 + i // 12}-{i % 12 + 1:02d}-01" for i in range(n)]
    analysis = [{"date": d, "outcome": float(yy), "driver": float(xx)}
                for d, yy, xx in zip(dates, y, x)]
    return {
        "outcome": {
            "verdict": verdict, "dependent_name": "outcome", "n_obs": n,
            "r_squared": 0.9, "cov_type": "HAC",
            "sample_start": "1990-01-01", "sample_end": "1994-12-01", "frequency": "monthly",
            "coefficients": [
                {"name": "const", "estimate": 1.0, "std_error": 0.1, "t_stat": 10.0,
                 "p_value": 0.0, "ci_low": 0.8, "ci_high": 1.2},
                {"name": "driver", "estimate": b, "std_error": 0.05, "t_stat": 40.0,
                 "p_value": 0.0, "ci_low": b - 0.1, "ci_high": b + 0.1},
            ],
            "analysis_data": analysis,
            "notes": [],
            "spec": {"dependent": {"name": "outcome", "series_ref": "Y", "transform": "level",
                                   "lag": 0},
                     "regressors": [{"name": "driver", "series_ref": "X", "transform": "level",
                                     "lag": 0}],
                     "method": "ols", "cov_type": "HAC", "add_constant": True,
                     "hypotheses": [], "rationale": "", "fallback_spec": False,
                     "sample_start": None, "sample_end": None},
        },
        "inference": {"hypotheses": [
            {"name": "h1", "param": "driver", "restriction": ">0", "p_value": 0.0,
             "supported": True, "estimate": b, "detail": ""}],
            "robustness": [], "stability_score": 1.0, "interpretation": ""},
    }


class TestInterpret:
    def test_effect_sizes_from_known_panel(self):
        res = interpret_estimation(_estimation_results(b=2.0))
        assert res.verdict == "estimated" and len(res.effects) == 1
        e = res.effects[0]
        assert e.param == "driver" and e.interpretation_kind == "marginal_effect"
        # one-sd effect = b * sd_x, standardized beta = b*sd_x/sd_y
        sd_x = res.descriptives["driver"]["sd"]
        sd_y = res.descriptives["outcome"]["sd"]
        assert e.one_sd_effect == pytest.approx(2.0 * sd_x)
        assert e.standardized_beta == pytest.approx(2.0 * sd_x / sd_y)

    def test_loglog_labeled_elasticity(self):
        data = _estimation_results()
        data["outcome"]["spec"]["dependent"]["transform"] = "log"
        data["outcome"]["spec"]["regressors"][0]["transform"] = "log"
        res = interpret_estimation(data)
        assert res.effects[0].interpretation_kind == "elasticity"
        assert res.effects[0].elasticity_at_means is None  # b IS the elasticity

    def test_inestimable_passthrough(self):
        res = interpret_estimation({"outcome": {"verdict": "inestimable",
                                                "reason": "no_data", "analysis_data": []}})
        assert res.effects == [] and any("no magnitudes" in n for n in res.notes)


class TestConsistency:
    def test_hallucinated_number_flagged_correct_rounding_accepted(self):
        artifacts = [{"outcome": {"estimate": 0.128473, "n_obs": 928, "r2": 0.7312}}]
        good = "The coefficient is 0.13 across 928 observations (R2 = 0.73)."
        rep = check_consistency(good, artifacts)
        assert rep.verdict == "consistent" and rep.verified == 3

        bad = "The coefficient is 0.31 across 928 observations."
        rep = check_consistency(bad, artifacts)
        assert rep.verdict == "has_unverified"
        assert rep.unverified[0].value == pytest.approx(0.31)

    def test_percentage_rescaling_accepted(self):
        artifacts = [{"share": 0.128}]
        rep = check_consistency("about 12.8 percent", artifacts)
        assert rep.verdict == "consistent"

    def test_small_int_enumerations_not_flagged(self):
        rep = check_consistency("We ran 3 stages across 4 teams.", [{}])
        assert rep.verdict == "consistent" and rep.skipped_small_ints == 2

    def test_llm_narration_cannot_self_verify(self):
        """A hallucinated number stored in the artifact's own LLM-narrative field must not
        back-verify the draft that repeats it."""
        artifacts = [{"interpretation": {"narrative": "The effect equals 7.7734.",
                                         "n_obs": 60}}]
        rep = check_consistency("The effect equals 7.7734.", artifacts)
        assert rep.verdict == "has_unverified"
        # but genuine numeric leaves under the same parent still verify
        rep2 = check_consistency("We used 60 observations.", artifacts)
        assert rep2.verdict == "consistent"


class TestAssembly:
    def test_report_numbers_are_all_verifiable_against_artifacts(self):
        """The assembled skeleton must never introduce a number the artifacts can't back."""
        est = _estimation_results()
        rq = {"final_questions": [{"question": "Does the driver move the outcome?"}]}
        lit = {"review_text": "Prior work is inconclusive."}
        model = {"calibrated_models": [{
            "model_title": "Toy Model",
            "metadata": {"calibration_status": "point_calibrated"},
            "calibrated_parameters": [{"parameter_symbol": "beta", "calibrated_value": 0.99,
                                       "calibration_basis": "cited"}]}]}
        data = {"retrieved_data": [
            {"series_id": "XSER", "series_name": "Driver", "source_name": "FRED",
             "num_observations": 60, "start_date": "1990-01-01", "end_date": "1994-12-01",
             "data_simulated": False, "quality_notes": "ok"},
            {"series_id": "SIM", "series_name": "Simulated", "source_name": "FRED",
             "num_observations": 100, "start_date": "1990", "end_date": "1999",
             "data_simulated": True, "quality_notes": "Simulated data"}]}
        interp = interpret_estimation(est)
        md = assemble_report(rq, lit, model, data, est, interp,
                             narratives={"discussion": "The driver matters."},
                             pipeline_run_id="pipeline-test")
        assert "## 7. Limitations" in md
        assert "DISCLOSED SIMULATION" in md
        rep = check_consistency(md, [rq, lit, model, data, est, interp.model_dump()])
        assert rep.verdict == "consistent", rep.unverified

    def test_limitations_collect_all_disclosure_classes(self):
        est = {"outcome": {"verdict": "fragile", "notes": ["downgraded to fragile by ..."]}}
        model = {"calibrated_models": [{"model_title": "M", "metadata": {
            "calibration_status": "uncalibratable", "uncalibratable_reason": "requires_training"}}]}
        data = {"retrieved_data": [
            {"series_id": "A", "data_simulated": True, "quality_notes": ""},
            {"series_id": "B", "data_simulated": False,
             "quality_notes": "[id repaired via curated table: 'X' -> FRED/CURRCIR]"}]}
        feas = {"status": "not_converging", "cycles": 3}
        lims = collect_limitations(data, model, est, feas)
        text = " ".join(lims)
        for token in ("DISCLOSED SIMULATION", "id repair", "uncalibratable",
                      "fragile", "not_converging"):
            assert token in text, token


class TestReferences:
    def _batch(self):
        return {"literature_items": [
            {"title": "HANK and the Missing Middle", "authors": ["Smith, A.", "Wu, B."],
             "year": 2024, "venue": "Journal of Monetary Economics",
             "url": "https://doi.org/10.1000/jme.2024.1"},
            {"title": "Deep Surrogates for DSGE", "authors": ["Ali, C."], "year": 2025,
             "source": "arXiv", "url": "https://arxiv.org/abs/2501.00001"},
        ]}

    def test_references_section_from_literature_batch(self):
        est = _estimation_results()
        interp = interpret_estimation(est)
        md = assemble_report({"final_questions": [{"question": "Q?"}]}, {}, {}, {},
                             est, interp, literature_batch=self._batch())
        assert "## 8. References" in md and "## 9. Provenance" in md
        assert "Smith, A., Wu, B. (2024). HANK and the Missing Middle." in md
        assert "https://arxiv.org/abs/2501.00001" in md
        # reference years/URLs verify when the batch is in the checker's artifact pool
        rep = check_consistency(md, [est, interp.model_dump(), self._batch()])
        assert rep.verdict == "consistent", rep.unverified

    def test_no_batch_keeps_provenance_at_section_8(self):
        est = _estimation_results()
        md = assemble_report({}, {}, {}, {}, est, interpret_estimation(est))
        assert "## 8. Provenance" in md and "## 8. References" not in md


class TestLiteratureRendering:
    def test_synthesis_dict_renders_parts_not_repr(self):
        est = _estimation_results()
        lit = {"literature_review": {
            "title": "Bridging Theory and Data in HANK",
            "abstract": "This review synthesizes recent advances.",
            "synthesis": "The landscape is shifting toward heterogeneity.",
            "sections": [{"section_title": "GE vs PE"}]}}
        md = assemble_report({}, lit, {}, {}, est, interpret_estimation(est))
        assert "**Bridging Theory and Data in HANK**" in md
        assert "This review synthesizes recent advances." in md
        assert "{'section_title'" not in md and "{'title'" not in md


class TestScientificNotation:
    """'2.2e-05' in the narrative vs stored 2.223e-05 was flagged
    unverified — decimal-place rounding is meaningless for exponent-formatted floats."""

    def test_sci_notation_rounding_matches(self):
        from ReportingTeam.ael.report_harness.consistency import _matches
        assert _matches(2.2e-05, {2.223e-05})
        assert _matches(2e-05, {2.223e-05})       # 1 sig fig
        assert _matches(-3.1e-04, {-3.089e-04})   # negative side

    def test_sci_notation_mismatch_still_flagged(self):
        from ReportingTeam.ael.report_harness.consistency import _matches
        assert not _matches(2.9e-05, {2.223e-05})  # not a rounding of the stored value
        assert not _matches(2.2e-04, {2.223e-05})  # wrong order of magnitude

    def test_normal_magnitudes_unaffected(self):
        from ReportingTeam.ael.report_harness.consistency import _matches
        assert _matches(0.48, {0.4818096})
        assert not _matches(0.6, {0.4818096, 3.0})  # not a rounding of anything stored


class TestCodeSection:
    """The generated CodeTeam modules must appear in the research report."""

    def _base_args(self):
        from ReportingTeam.ael.report_harness.interpret import InterpretationResult
        return dict(
            research_questions={"final_questions": [{"question": "Q?"}]},
            literature_review={}, model_specification={}, data_source={},
            estimation_results={"outcome": {"verdict": "inestimable", "reason": "no_data"}},
            interpretation=InterpretationResult(),
        )

    def test_code_section_renders_inventory(self):
        from ReportingTeam.ael.report_harness.assemble import assemble_report
        gen = {"results": [
            {"model_title": "HANK A", "verdict": "partial", "n_parseable": 7,
             "n_equations": 12, "refused": [{"equation_id": "E1"}]},
        ]}
        val = {"results": [
            {"verdict": "partial", "checks": [{"name": "module_compiles", "passed": True},
                                              {"name": "steady_state_solves", "passed": False}]},
        ]}
        md = assemble_report(**self._base_args(), code_generation=gen, code_validation=val)
        assert "## 5b. Code Implementation" in md
        assert "| HANK A | `partial` | 7/12 | `partial` | 1/2 |" in md
        assert "refused 1 equation" in md
        assert "5_code/generated_models" in md          # storage pointer

    def test_section_omitted_without_code_artifacts(self):
        from ReportingTeam.ael.report_harness.assemble import assemble_report
        md = assemble_report(**self._base_args())
        assert "Code Implementation" not in md


class TestReportRendering:
    def test_identification_notes_travel_into_the_report(self):
        from ReportingTeam.ael.report_harness.assemble import assemble_report
        from ReportingTeam.ael.report_harness.interpret import InterpretationResult
        md = assemble_report(
            research_questions={"final_questions": [{"question": "Q?"}]},
            literature_review={}, model_specification={}, data_source={},
            estimation_results={"outcome": {
                "verdict": "estimated", "method": "ols", "cov_type": "HAC",
                "dependent_name": "y", "n_obs": 50, "r_squared": 0.5,
                "sample_start": "1990", "sample_end": "2020", "frequency": "annual",
                "coefficients": [{"name": "x", "estimate": 1.0, "std_error": 0.1,
                                  "t_stat": 10.0, "p_value": 0.001, "ci_low": 0.8,
                                  "ci_high": 1.2}],
                "notes": ["identification: least squares on observational time series — "
                          "coefficients are associations conditional on the included "
                          "regressors; no causal identification strategy is claimed",
                          "proxy disclosure: Expectations Proxy — proxy measurement(s)"]}},
            interpretation=InterpretationResult(),
        )
        assert "*identification: least squares" in md
        assert "*proxy disclosure: Expectations Proxy" in md


class TestReportD4:
    """The report distinguishes proposed from
    harness-estimated parameters, lists unsupported hypotheses and stand-in calibrations among
    the limitations, and the consistency check records where each value occurs."""

    def _model_spec(self):
        return {"calibrated_models": [{
            "model_title": "HANK with Taylor rule",
            "calibrated_parameters": [
                {"parameter_symbol": "phi_pi", "calibrated_value": 1.5, "justification": "lit"},
                {"parameter_symbol": "beta", "calibrated_value": 0.99, "justification": "lit"}],
            "metadata": {"calibration_status": "archetype_calibrated",
                         "harness_estimated_params": {"phi_pi": 2.3609},
                         "simulator": "nk_taylor", "simulator_is_archetype": True}}]}

    def _est(self):
        return {"outcome": {"verdict": "estimated"},
                "inference": {"hypotheses": [
                    {"name": "real_rate", "param": "r", "restriction": "<0", "estimate": 0.0001,
                     "p_value": 0.5011, "supported": False, "outcome": "inconclusive"},
                    {"name": "gini", "param": "g", "restriction": ">0", "estimate": -1.628,
                     "p_value": 1.0, "supported": False, "outcome": "contradicted"},
                    {"name": "ok", "param": "y", "restriction": ">0", "estimate": 0.4,
                     "p_value": 0.01, "supported": True, "outcome": "supported"}]}}

    def test_limitations_list_hypotheses_and_stand_in_scope(self):
        lims = collect_limitations({}, self._model_spec(), self._est())
        text = "\n".join(lims)
        assert "`inconclusive`" in text and "`contradicted`" in text
        assert "'ok'" not in text                          # supported predictions are not limitations
        assert "canonical stand-in 'nk_taylor'" in text and "estimated only phi_pi" in text

    def test_parameter_table_separates_proposal_from_estimate(self):
        md = assemble_report({}, {}, self._model_spec(), {}, self._est(),
                             interpret_estimation(self._est()))
        assert "| Parameter | Proposed value (LLM) | Harness estimate | Basis |" in md
        assert "| phi_pi | 1.5 | 2.361" in md
        assert "| beta | 0.99 | — |" in md
        assert "## 7. Limitations" in md and "Honest Limitations" not in md

    def test_consistency_records_sources_and_match_kind(self):
        arts = {"estimation_results": {"coef": 0.128473}, "model_specification": {"phi": 2.3609}}
        rep = check_consistency("The coefficient is 0.13 and phi is 2.3609.", arts)
        assert rep.verdict == "consistent" and rep.verified == 2
        by_val = {v.value: v for v in rep.verified_numbers}
        assert by_val[0.13].match == "rounding" and by_val[0.13].sources == ["estimation_results"]
        assert by_val[2.3609].match == "exact" and by_val[2.3609].sources == ["model_specification"]
        assert rep.match_kinds == {"exact": 1, "rounding": 1}


class TestProxySeriesD5:
    """A series whose source title differs materially from the requested name is shown
    under the source title and disclosed as a proxy (a live run: World Bank SI.POV.GINI,
    'Gini index', presented as 'Gini Coefficient of Consumption')."""

    def _data(self):
        return {"retrieved_data": [{
            "series_id": "SI.POV.GINI", "series_name": "Gini Coefficient of Consumption",
            "source_name": "World Bank", "source_title": "Gini index",
            "proxy_for": "Gini Coefficient of Consumption", "num_observations": 62,
            "start_date": "1963", "end_date": "2024", "quality_notes": "", "data_simulated": False}]}

    def test_report_names_the_source_series_and_discloses_proxy(self):
        est = {"outcome": {"verdict": "estimated"}}
        md = assemble_report({}, {}, {}, self._data(), est, interpret_estimation(est))
        assert "`SI.POV.GINI` (Gini index)" in md
        assert "proxy for 'Gini Coefficient of Consumption'" in md
        lims = "\n".join(collect_limitations(self._data(), {}, est))
        assert "is a proxy: the source's own series is 'Gini index'" in lims


def test_title_proxy_mismatch_rule():
    from shared.tools.econ_connectors import title_proxy_mismatch as f
    assert f("Gini Coefficient of Consumption", "Gini index")
    assert not f("Real GDP", "Real Gross Domestic Product")          # acronym
    assert not f("Labor share", "Share of Labour Compensation in GDP")  # British spelling
    assert not f("Unemployment Rate", "Unemployment Rate")
    assert not f("Anything", None)                                   # no title, no claim
