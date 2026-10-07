# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""DataTeam concept checks (fixes found by auditing the v0.7.1 runs).

- the requested-name vs source-title check (token Jaccard) passed different concepts.
- the curated-table repair stored a different concept/country under the requested id.
- the feasibility scout graded any name hit FEASIBLE.
- IMF WEO projections (to 2030) were labelled as observed data."""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from DataTeam.ael.feasibility import (
    AvailabilityScout, DataRequirementsSpec, RequirementVerdict, available_series_from_source,
)
from DataTeam.ael.schemas.stage_outputs import DataRequirement
from shared.tools import econ_connectors as ec


# --------------------------------------------------------------------------------------------
# concept check
# --------------------------------------------------------------------------------------------

AUDIT_MISSES = [
    ("Consumption-to-GDP Ratio", "GDP (current US$)"),
    ("FDI, net inflows (% of GDP)", "Foreign direct investment, net outflows (% of GDP)"),
    ("US Corporate HY OAS",
     "ICE BofA High Yield Emerging Markets Corporate Plus Index Option-Adjusted Spread"),
    ("Interest Rate Volatility",
     "Equity Market Volatility Tracker: Macroeconomic News and Outlook: Interest Rates"),
    ("Unemployment Rate Volatility", "Unemployment Rate"),
    ("Money supply M3 in the euro area", "M2"),
    ("Average Hourly Earnings", "All Employees, Total Nonfarm"),
    ("Domestic credit to private sector (% of GDP)", "GDP (current US$)"),
    ("Wealth Gini", "Gini index"),
    ("Steady State Inflation Rate",
     "Consumer Price Index for All Urban Consumers: All Items in U.S. City Average"),
    ("Real Government Purchases", "Government Consumption Expenditures and Gross Investment"),
    ("Gini Coefficient of Consumption", "Gini index"),
]

TRUE_MATCHES = [
    ("Real GDP", "Real Gross Domestic Product"),
    ("Labor share", "Share of Labour Compensation in GDP at Current National Prices for United States"),
    ("Unemployment Rate", "Unemployment Rate"),
    ("Unemployment Rate", "Unemployment, total (% of total labor force) (modeled ILO estimate)"),
    ("CPI", "Consumer Price Index for All Urban Consumers: All Items in U.S. City Average"),
    ("Federal Funds Rate", "Federal Funds Effective Rate"),
    ("Government Spending", "Government Consumption Expenditures and Gross Investment"),
    ("Household Debt Service Ratio",
     "Household Debt Service Payments as a Percent of Disposable Income"),
    ("Housing Starts", "New Privately-Owned Housing Units Started: Total Units"),
    ("Tax revenue (% of GDP)", "Tax revenue (% of GDP)"),
    ("GDP growth", "GDP growth (annual %)"),
    ("CBOE VIX Index (Close)", "CBOE Volatility Index: VIX"),
    ("Germany unemployment rate", "Unemployment, total (% of total labor force) (national estimate)"),
    # replay of the 21 run artifacts: bare-code titles, price fields, 'excluding' = 'less'
    ("M2 Money Stock", "M2"),
    ("CBOE Volatility Index (VIX) Close", "CBOE Volatility Index: VIX"),
    ("PCE Price Index Less Food and Energy",
     "Personal Consumption Expenditures Excluding Food and Energy (Chain-Type Price Index)"),
]


@pytest.mark.parametrize("requested,title", AUDIT_MISSES)
def test_audit_misses_are_flagged(requested, title):
    status, reason = ec.title_concept_check(requested, title)
    assert status == "mismatch", (requested, title)
    assert reason
    assert ec.title_proxy_mismatch(requested, title)


@pytest.mark.parametrize("requested,title", TRUE_MATCHES)
def test_true_matches_still_match(requested, title):
    assert ec.title_concept_check(requested, title) == ("match", "")


def test_empty_or_uninformative_title_is_unverified_not_match():
    assert ec.title_concept_check("Exchange rate", None)[0] == "unverified"
    assert ec.title_concept_check("Exchange rate", "")[0] == "unverified"
    assert ec.title_concept_check("Money stock", "M2SL", series_id="M2SL")[0] == "unverified"
    # the bool wrapper does not CLAIM a mismatch without a title (status is separate)
    assert not ec.title_proxy_mismatch("Exchange rate", None)


# --------------------------------------------------------------------------------------------
# repair acceptance
# --------------------------------------------------------------------------------------------

def _obs(title, sid="X"):
    return {"observations": [{"date": "2020-01-01", "value": 1.0}], "n_obs": 1,
            "start_date": "2020-01-01", "end_date": "2020-01-01", "source_name": "x",
            "series_title": title}


@pytest.fixture()
def offline(monkeypatch):
    monkeypatch.setenv("AEL_PROVIDER_SEARCH", "0")
    monkeypatch.setattr(ec, "fetch_dbnomics", lambda *a, **k: None)
    monkeypatch.setattr(ec, "search_dbnomics_series", lambda *a, **k: None)
    monkeypatch.setattr(ec, "search_wdi_series", lambda *a, **k: None)


def test_credit_to_gdp_is_not_repaired_to_gdp_level(monkeypatch, offline):
    # v0.7.1 runs: 'gdp' inside '(% of GDP)' hit the curated NY.GDP.MKTP.CD entry
    served = {"NY.GDP.MKTP.CD": _obs("GDP (current US$)")}
    monkeypatch.setattr(ec, "fetch_series", lambda src, sid, **k: served.get(sid))
    out, note = ec.fetch_series_with_repair(
        "World Bank", "FS.AST.PRVT.GDP", series_name="Domestic credit to private sector (% of GDP)")
    assert out is None
    assert "declined" in note and "NY.GDP.MKTP.CD" in note


def test_euro_area_m3_is_not_repaired_to_us_m2(monkeypatch, offline):
    # a v0.7.1 run: ECB euro M3 -> FRED M2SL (US), no proxy flag
    monkeypatch.setattr(ec, "fetch_series", lambda *a, **k: None)
    monkeypatch.setattr(ec, "_fetch_fred_native", lambda sid: _obs("M2") if sid == "M2SL" else None)
    out, note = ec.fetch_series_with_repair(
        "ECB", "BSI/M.U2.Y.V.M30.X.1.U2.2300.Z01.E",
        series_name="Money supply M3 in the euro area")
    assert out is None and "M2SL" in note and "declined" in note


def test_hourly_earnings_is_not_repaired_to_payroll_employment(monkeypatch, offline):
    # a v0.7.1 run: BLS average hourly earnings -> PAYEMS (employment level), no flag
    monkeypatch.setattr(ec, "fetch_series", lambda *a, **k: None)
    monkeypatch.setattr(ec, "_fetch_fred_native",
                        lambda sid: _obs("All Employees, Total Nonfarm") if sid == "PAYEMS" else None)
    out, note = ec.fetch_series_with_repair(
        "BLS", "CES0500000003X",
        series_name="Average Hourly Earnings of All Employees, Total Nonfarm")
    assert out is None and "PAYEMS" in note


def test_geography_mismatch_declines_even_when_title_matches(monkeypatch, offline):
    # an ECB (euro-area) unemployment request must not become the US UNRATE
    monkeypatch.setattr(ec, "fetch_series", lambda *a, **k: None)
    monkeypatch.setattr(ec, "_fetch_fred_native",
                        lambda sid: _obs("Unemployment Rate") if sid == "UNRATE" else None)
    out, note = ec.fetch_series_with_repair("ECB", "LFSI/M.I9.S.UNEHRT.TOTAL0.15_74.T",
                                            series_name="Unemployment rate")
    assert out is None and "geography" in note


def test_matching_repair_still_accepted(monkeypatch, offline):
    monkeypatch.setattr(ec, "fetch_series", lambda *a, **k: None)
    monkeypatch.setattr(ec, "_fetch_fred_native",
                        lambda sid: _obs("Unemployment Rate") if sid == "UNRATE" else None)
    out, note = ec.fetch_series_with_repair("FRED", "UNRATEX", series_name="Unemployment rate")
    assert out is not None and out["resolved_id"] == "UNRATE" and "curated table" in note


# --------------------------------------------------------------------------------------------
# feasibility scout
# --------------------------------------------------------------------------------------------

def _req(rid, var):
    return DataRequirement(requirement_id=rid, variable_name=var, description=var,
                           frequency="annual", time_period="1990-2023",
                           geographic_coverage="US", unit_of_measurement="x", priority="High")


@pytest.fixture()
def resolver(monkeypatch):
    ec._RESOLVE_CACHE.clear()
    monkeypatch.setenv("AEL_PROVIDER_SEARCH", "1")
    monkeypatch.setattr(ec, "search_fred_series", lambda *a, **k: None)
    monkeypatch.setattr(ec, "search_wdi_series", lambda *a, **k: None)
    titles = {"NY.GDP.MKTP.CD": "GDP (current US$)", "SI.POV.GINI": "Gini index",
              "CPIAUCSL": "Consumer Price Index for All Urban Consumers: All Items in U.S. City "
                          "Average",
              "GCE": "Government Consumption Expenditures and Gross Investment"}
    monkeypatch.setattr(ec, "fetch_series_title", lambda src, sid, *a, **k: titles.get(sid))
    yield ec.resolve_fetchable_series
    ec._RESOLVE_CACHE.clear()


@pytest.mark.parametrize("var", ["Tax-to-GDP Ratio", "Wealth Gini", "Inflation Belief Dispersion"])
def test_scout_name_hit_with_wrong_concept_is_proxy_only(resolver, var):
    drs = DataRequirementsSpec(research_question="RQ", requirements=[_req("R1", var)])
    f = AvailabilityScout(fetchable_lookup=resolver).grade(drs, [])[0]
    assert f.verdict == RequirementVerdict.PROXY_ONLY
    assert not f.fetchable
    assert f.proxy_candidates and "different construct" in f.proxy_candidates[0].bias_caveat


def test_scout_verified_hit_stays_feasible(resolver):
    drs = DataRequirementsSpec(research_question="RQ", requirements=[_req("R1", "Government Spending")])
    f = AvailabilityScout(fetchable_lookup=resolver).grade(drs, [])[0]
    assert f.verdict == RequirementVerdict.FEASIBLE and f.fetchable


def test_scout_unverifiable_curated_hit_is_proxy_only(monkeypatch):
    ec._RESOLVE_CACHE.clear()
    monkeypatch.setenv("AEL_PROVIDER_SEARCH", "0")          # offline: no title available
    drs = DataRequirementsSpec(research_question="RQ", requirements=[_req("R1", "Government Spending")])
    f = AvailabilityScout(fetchable_lookup=ec.resolve_fetchable_series).grade(drs, [])[0]
    assert f.verdict == RequirementVerdict.PROXY_ONLY
    assert "could not be verified" in f.proxy_candidates[0].bias_caveat
    ec._RESOLVE_CACHE.clear()


def test_scout_pool_match_with_mismatched_source_title_is_proxy():
    # selected as "Gini Coefficient of Wealth", retrieved series is the income Gini
    source = {"selected_series": [{"series_id": "SI.POV.GINI", "series_name": "Wealth Gini",
                                   "source_name": "World Bank"}],
              "retrieved_data": [{"series_id": "SI.POV.GINI", "source_title": "Gini index"}]}
    pool = available_series_from_source(source)
    drs = DataRequirementsSpec(research_question="RQ", requirements=[_req("R1", "Wealth Gini")])
    f = AvailabilityScout().grade(drs, pool)[0]
    assert f.verdict == RequirementVerdict.PROXY_ONLY
    assert "Gini index" in f.proxy_candidates[0].bias_caveat


def test_scout_pool_simulated_series_never_feasible():
    source = {"selected_series": [{"series_id": "UNRATE", "series_name": "Unemployment Rate",
                                   "source_name": "FRED"}],
              "retrieved_data": [{"series_id": "UNRATE", "data_simulated": True}]}
    drs = DataRequirementsSpec(research_question="RQ", requirements=[_req("R1", "Unemployment Rate")])
    f = AvailabilityScout().grade(drs, available_series_from_source(source))[0]
    assert f.verdict == RequirementVerdict.PROXY_ONLY


# --------------------------------------------------------------------------------------------
# projections
# --------------------------------------------------------------------------------------------

def test_period_end_parsing():
    assert ec.period_end("2024") == date(2024, 12, 31)
    assert ec.period_end("2024-Q2") == date(2024, 6, 30)
    assert ec.period_end("2024-02") == date(2024, 2, 29)
    assert ec.period_end("2024-S1") == date(2024, 6, 30)
    assert ec.period_end("2024-05-17") == date(2024, 5, 17)
    assert ec.period_end("garbage") is None


def test_weo_vintage_projections_are_excluded(monkeypatch):
    monkeypatch.setattr(ec, "_today", lambda: date(2026, 10, 1))
    doc = {"series": {"docs": [{
        "series_name": "United States – Gross domestic product, constant prices – Percent change",
        "indexed_at": "2025-05-15T11:13:27.379Z",
        "period": [str(y) for y in range(2019, 2031)],
        "value": [2.0] * 12}]}}
    resp = MagicMock()
    resp.json.return_value = doc
    with patch("shared.observability.tracked_get", return_value=resp):
        out = ec.fetch_series("IMF", "WEO:2025-04/USA.NGDP_RPCH")
    assert out["end_date"] == "2024"                 # 2025..2030 are vintage projections
    assert out["n_projection_excluded"] == 6
    assert out["projection_cutoff"] == "2025-05-15"


def test_observations_after_retrieval_date_are_projections(monkeypatch):
    monkeypatch.setattr(ec, "_today", lambda: date(2026, 10, 1))
    out = ec._norm([{"date": "2026-07-01", "value": 1}, {"date": "2035-01-01", "value": 2}], "FRED")
    assert out["n_obs"] == 1 and out["n_projection_excluded"] == 1


def test_rate_vocabulary_and_interest_rate_titles():
    from shared.tools.econ_connectors import title_concept_check as t
    assert t("10-Year Treasury Yield Minus 2-Year Treasury Yield",
             "10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity")[0] == "match"
    assert t("TED Rate", "TED Spread (DISCONTINUED)")[0] == "match"
    assert t("Average Loan-to-Value Ratio",
             "Large Bank Consumer Mortgage Originations: Average Interest Rate at Origination by "
             "LTV: 30-Year Fixed Rate Mortgage: 66-79 Loan-to-Value")[0] == "mismatch"
    assert t("Federal Funds Rate", "Federal Funds Effective Rate")[0] == "match"
