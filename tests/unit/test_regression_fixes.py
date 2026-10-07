# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Fixes found by auditing a validated full run: ideation diversity floor, series-id repair,
LaTeX-free DRS ids, pipeline-constructible requirements, VAR-growth runnable archetype."""

from unittest.mock import patch, MagicMock

import pytest

from DataTeam.ael.feasibility.adapters import drs_from_model_artifacts
from shared.tools import econ_connectors as ec


def _resp(json_data=None, text=""):
    r = MagicMock()
    r.json.return_value = json_data
    r.text = text
    return r


# --- id repair ----------------------------------------------------------------------------

class TestIdRepair:
    def test_country_alias_repair(self):
        wb_ok = [{"page": 1}, [{"date": "2023", "value": 3.7}]]

        def fake_get(url, params=None, collector=None, agent="", retries=0):
            if "/country/GER/" in url:
                return _resp([{"message": [{"key": "Invalid format"}]}])   # WB rejects GER
            return _resp(wb_ok)

        with patch("shared.observability.tracked_get", side_effect=fake_get):
            out, note = ec.fetch_series_with_repair("World Bank", "GER:SL.UEM.TOTL.ZS",
                                                    series_name="Unemployment Germany")
            assert out and out["n_obs"] == 1
            assert "GER -> DEU" in note                       # repair DISCLOSED

    def test_dbnomics_search_repair(self):
        def fake_get(url, params=None, collector=None, agent="", retries=0):
            if url.endswith("/search"):
                return _resp({"results": {"docs": [{"provider_code": "WB", "code": "WDI"}]}})
            if "/series/WB/WDI" in url and (params or {}).get("observations") == "0":
                return _resp({"series": {"docs": [{"series_code": "A-TM.TAX.WM-USA"}]}})
            if "/series/WB/WDI/A-TM.TAX.WM-USA" in url:
                return _resp({"series": {"docs": [{"period": ["2020", "2021"], "value": [2.1, 2.0],
                    "series_name": "Tariff rate, applied, weighted mean, all products (%) - United States"}]}})
            return _resp({})                                   # original id fails

        with patch("shared.observability.tracked_get", side_effect=fake_get):
            out, note = ec.fetch_series_with_repair("OECD", "TARIF/TARIFF_APPLIED",
                                                    series_name="tariff rate applied weighted mean")
            assert out and out["n_obs"] == 2
            assert "repaired via DBnomics search" in note and "WB/WDI" in note

    def test_unrepairable_returns_none_with_empty_note(self):
        with patch("shared.observability.tracked_get", return_value=_resp({})):
            out, note = ec.fetch_series_with_repair("IMF", "WEO/NOPE", series_name="zzz")
            assert out is None and note == ""


# --- DRS hygiene + constructed inputs -------------------------------------------------------

class TestDrsHygiene:
    def test_latex_stripped_and_constructed_downgraded(self):
        md = {"formal_models": [{"variables": [
            {"variable_symbol": r"\mathbf{z}_t^{LLM}", "variable_name": "LLM-Derived Features",
             "variable_type": "exogenous", "description": "extracted via LLM from text corpus"},
            {"variable_symbol": "S_t", "variable_name": "Consumer Sentiment Index",
             "variable_type": "exogenous", "description": "University of Michigan survey index"},
        ]}]}
        drs = drs_from_model_artifacts("RQ", md, None)
        by_id = {r.requirement_id: r for r in drs.requirements}
        assert "var:z_t^LLM" in by_id                          # LaTeX stripped
        constructed = by_id["var:z_t^LLM"]
        assert constructed.priority == "Medium"                # not essential
        assert constructed.suggested_sources == ["pipeline-constructed"]
        assert "pipeline-constructible" in constructed.description
        # a genuine retrievable survey series stays High-essential
        assert by_id["var:S_t"].priority == "High"
        assert [r.variable_name for r in drs.essential()] == ["Consumer Sentiment Index"]


# --- VAR-growth archetype -------------------------------------------------------------------

class TestVarGrowthArchetype:
    def setup_method(self):
        pytest.importorskip("scipy")
        pytest.importorskip("numpy")

    def _bvar(self):
        return {"model_title": "LLM-Augmented Regime-Switching Bayesian VAR",
                "parameters": [
                    {"parameter_symbol": "rho_g", "parameter_name": "growth persistence (AR(1)) coefficient"},
                    {"parameter_symbol": "sigma_g", "parameter_name": "growth shock volatility"}],
                "equations": []}

    def test_moments_respond_to_parameters(self):
        from ModelTeam.ael.calib_harness.sim_backends import var_growth as vg
        lo = vg.simulate_moments({"rho": 0.1, "sigma": 0.01}, 1)
        hi = vg.simulate_moments({"rho": 0.7, "sigma": 0.01}, 1)
        assert hi["output_growth_persistence"] > lo["output_growth_persistence"] + 0.3
        big = vg.simulate_moments({"rho": 0.3, "sigma": 0.03}, 1)
        assert big["output_growth_volatility"] > 2 * vg.simulate_moments({"rho": 0.3, "sigma": 0.01}, 1)["output_growth_volatility"]

    def test_build_matches_bvar_and_declines_others(self):
        from ModelTeam.ael.calib_harness.sim_backends import var_growth as vg
        spec = vg.build(self._bvar(), [])
        assert spec is not None and spec[2] == ["rho_g", "sigma_g"]
        # no VAR wording -> decline
        assert vg.build({"model_title": "Static CGE model", "parameters": []}, []) is None
        # VAR wording but missing the shock-scale param -> decline (can't identify 2 moments)
        assert vg.build({"model_title": "A VAR framework", "parameters": [
            {"parameter_symbol": "rho", "parameter_name": "persistence"}]}, []) is None

    def test_harness_simulation_calibrates_bvar(self):
        pytest.importorskip("sympy")
        from ModelTeam.ael.calib_harness import run as harness_run
        out = harness_run(self._bvar(), [])
        assert out.calibration_status == "archetype_calibrated"
        assert out.simulator == "var_growth"
        assert set(out.estimated_params) == {"rho_g", "sigma_g"}
        # estimates should land near the cited targets (rho ~0.30, sigma s.t. std ~0.018)
        assert -0.2 < out.estimated_params["rho_g"] < 0.7
        assert 0.005 < out.estimated_params["sigma_g"] < 0.03


class TestGeoGuardedRepair:
    """The search repair must never substitute a wrong-country series (a live run
    repaired 'CO2 emissions' to an ARUBA series that QA then scored 84/100)."""

    def _fake(self, sdocs):
        calls = {}
        def fake_get(url, params=None, collector=None, agent="", retries=0):
            if url.endswith("/search"):
                return _resp({"results": {"docs": [{"provider_code": "WB", "code": "WDI"}]}})
            if "/series/WB/WDI" in url and (params or {}).get("observations") == "0":
                return _resp({"series": {"docs": sdocs}})
            # the mirror rung fires FIRST on the same URL the post-search fetch uses;
            # these tests exercise the geo-guarded SEARCH rung, so the first hit (the
            # mirror attempt) is an outage and later hits succeed.
            if "/series/WB/WDI/A-EN.ATM.CO2E.KT-USA" in url:
                calls.setdefault("mirror", 0)
                calls["mirror"] += 1
                if calls["mirror"] == 1:
                    return _resp({})
            if "/series/WB/WDI/" in url:
                return _resp({"series": {"docs": [{"period": ["2020"], "value": [1.0],
                                                   "series_name": "CO2 emissions (kt) - United States"}]}})
            return _resp({})
        return fake_get

    def test_wrong_geo_candidates_skipped_for_matching_one(self):
        sdocs = [{"series_code": "A-EN.ATM.CO2E.KT-ABW", "series_name": "CO2 emissions (kt) - Aruba"},
                 {"series_code": "A-EN.ATM.CO2E.KT-USA", "series_name": "CO2 emissions (kt) - United States"}]
        with patch("shared.observability.tracked_get", side_effect=self._fake(sdocs)):
            out, note = ec.fetch_series_with_repair("World Bank", "EN.ATM.CO2E.KT",
                                                    series_name="CO2 emissions kt")
            assert out is not None
            assert "USA" in note and "ABW" not in note      # Aruba skipped, USA chosen

    def test_all_wrong_geo_declines(self):
        sdocs = [{"series_code": "A-EN.ATM.CO2E.KT-ABW", "series_name": "CO2 emissions - Aruba"},
                 {"series_code": "A-EN.ATM.CO2E.KT-DEU", "series_name": "CO2 emissions - Germany"}]
        with patch("shared.observability.tracked_get", side_effect=self._fake(sdocs)):
            out, note = ec.fetch_series_with_repair("World Bank", "EN.ATM.CO2E.KT",
                                                    series_name="CO2 emissions kt")
            assert out is None and note == ""               # honest decline -> disclosed simulation


class TestBookSatisfiedTargets:
    """Calibration targets the harness matched against the CITED target book are
    satisfied — in a live run the MRRs literally asked the model to remove K/Y and the capital
    share. They stay in the DRS as non-essential, attributed to the book."""

    def test_book_matched_target_downgraded_unmatched_stays_high(self):
        md = {"formal_models": [{"variables": []}]}
        cal = {"calibrated_models": [{
            "empirical_targets": [
                {"target_id": "T1", "target_name": "Steady-State Capital-Output Ratio", "importance": "High"},
                {"target_id": "T2", "target_name": "Capital Share of Income", "importance": "High"},
                {"target_id": "T3", "target_name": "Negotiation Duration (Rounds)", "importance": "High"},
            ],
            "model_moments": [
                {"moment_name": "capital_output_ratio", "model_value": 3.28, "empirical_value": 3.0},
                {"moment_name": "capital_share", "model_value": 0.36, "empirical_value": 0.36},
            ],
        }]}
        drs = drs_from_model_artifacts("RQ", md, cal)
        by = {r.variable_name: r for r in drs.requirements}
        assert by["Steady-State Capital-Output Ratio"].priority == "Medium"
        assert "target book" in by["Steady-State Capital-Output Ratio"].description
        assert by["Capital Share of Income"].priority == "Medium"
        assert by["Negotiation Duration (Rounds)"].priority == "High"     # no book moment -> stays
        assert [r.variable_name for r in drs.essential()] == ["Negotiation Duration (Rounds)"]


class TestConceptSimilarity:
    """Canonical series must match their economic concepts — token-Jaccard alone
    scored 'Annualized Inflation Rate' vs the CPI series name at 0.0."""

    def test_concept_pairs_reach_proxy_grade(self):
        from DataTeam.ael.feasibility.scout import similarity
        assert similarity("Annualized Inflation Rate",
                          "Consumer Price Index for All Urban Consumers: All Items") >= 0.5
        assert similarity("Policy Rate", "Federal Funds Effective Rate") >= 0.5
        assert similarity("Unemployment Rate", "UNRATE jobless share") >= 0.5

    def test_floor_is_proxy_not_direct_match(self):
        from DataTeam.ael.feasibility.scout import similarity
        # concept floor (0.5) sits BELOW the direct-match threshold (0.6): a derivation is a
        # close proxy, not the exact variable
        assert 0.5 <= similarity("Annualized Inflation Rate",
                                 "Consumer Price Index for All Urban Consumers") < 0.6

    def test_unrelated_names_unaffected(self):
        from DataTeam.ael.feasibility.scout import similarity
        assert similarity("Annualized Inflation Rate", "Agricultural land share") < 0.3
        assert similarity("Temperature Anomaly", "Federal Funds Effective Rate") < 0.3

    def test_exact_token_matches_keep_their_score(self):
        from DataTeam.ael.feasibility.scout import similarity
        assert similarity("government spending", "government spending") == 1.0


class TestRuleOfThumbRoleContext:
    """A backward-looking PRICE-SETTER share described as 'rule-of-thumb' must not be mapped to
    the household hand-to-mouth share (a live run calibrated psi, a pricing parameter, to the aggregate MPC target)."""

    def _role(self, symbol, name, desc, rng="[0, 0.5]"):
        from ModelTeam.ael.calib_harness import archetypes
        roles = archetypes.map_param_roles([{"parameter_symbol": symbol, "parameter_name": name,
                                             "description": desc, "typical_range": rng}])
        return roles[0].role if roles else None

    def test_firm_rule_of_thumb_is_not_hand_to_mouth(self):
        assert self._role("psi", "Backward-looking weight",
                          "The fraction of firms that set prices based on past inflation "
                          "(rule-of-thumb) rather than forward-looking optimization.") is None

    def test_household_rule_of_thumb_is_hand_to_mouth(self):
        assert self._role("lambda", "Rule-of-thumb share",
                          "Share of rule-of-thumb households that consume their current income.",
                          "[0, 1]") == "hand_to_mouth_share"

    def test_explicit_hand_to_mouth_unaffected(self):
        assert self._role("chi", "Hand-to-mouth share",
                          "Fraction of hand-to-mouth agents.", "[0, 1]") == "hand_to_mouth_share"


def test_coverage_accounts_for_every_equation():
    """Parse failures were missing from coverage, so parseable + refused < total."""
    pytest.importorskip("sympy")
    from ModelTeam.ael.calib_harness import harness
    model = {"model_title": "t", "parameters": [], "equations": [
        {"equation_plain": "y_t = a * k_t"},
        {"equation_plain": "u'(c_t) = c_t^(-sigma)"},          # parse failure (derivative LHS)
        {"equation_plain": "Y_t = integral(c_{i,t} di)"},      # refused (non-algebraic)
    ]}
    out = harness._run_closed_form(model, [])
    cov = out.coverage
    assert cov["total_eqs"] == 3
    assert cov["parseable_eqs"] + cov["refused_eqs"] + cov["parse_fail_eqs"] == 3
