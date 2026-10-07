# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""DataTeam connector review fixes.

- the World Bank -> DBnomics WDI mirror mapped every unrecognized two-letter country code
  to USA and kept the response under the requested id without the repair acceptance check.
- the title concept check dropped parenthetical identifiers (M3 vs M2), ignored rate vs
  level, and checked the ratio construct in one direction only."""

import pytest

from shared.tools import econ_connectors as ec
from shared.tools.iso_countries import ISO2_TO_ISO3, to_iso3


def _obs(title):
    return {"observations": [{"date": "2020", "value": 1.0}], "n_obs": 1,
            "start_date": "2020", "end_date": "2020", "source_name": "DBnomics",
            "series_title": title}


@pytest.fixture()
def wb_down(monkeypatch):
    monkeypatch.setenv("AEL_PROVIDER_SEARCH", "0")
    monkeypatch.setattr(ec, "fetch_series", lambda *a, **k: None)
    monkeypatch.setattr(ec, "_fetch_fred_native", lambda *a, **k: None)
    monkeypatch.setattr(ec, "search_dbnomics_series", lambda *a, **k: None)
    monkeypatch.setattr(ec, "search_wdi_series", lambda *a, **k: None)


# --------------------------------------------------------------------------------------------
# #3 — country normalization + mirror acceptance
# --------------------------------------------------------------------------------------------

def test_iso_table_is_complete_and_refuses_unknown_codes():
    assert len(ISO2_TO_ISO3) >= 249
    assert to_iso3("FR") == "FRA" and to_iso3("ke") == "KEN" and to_iso3("SK") == "SVK"
    assert to_iso3("FRA") == "FRA" and to_iso3("XC") == "EMU"
    assert to_iso3("QQ") is None and to_iso3("ZZZ") is None and to_iso3("") is None


@pytest.mark.parametrize("country,iso3", [("FR", "FRA"), ("KE", "KEN"), ("BR", "BRA"),
                                          ("VN", "VNM"), ("SK", "SVK")])
def test_mirror_requests_the_requested_country(monkeypatch, wb_down, country, iso3):
    seen = []
    monkeypatch.setattr(ec, "fetch_dbnomics", lambda sid, **k: seen.append(sid))
    ec.fetch_series_with_repair("World Bank", f"{country}:NY.GDP.MKTP.CD",
                                series_name="GDP (current US$)")
    assert seen == [f"WB/WDI/A-NY.GDP.MKTP.CD-{iso3}"]


def test_unknown_country_code_is_refused_not_defaulted_to_usa(monkeypatch, wb_down):
    seen = []
    monkeypatch.setattr(ec, "fetch_dbnomics", lambda sid, **k: seen.append(sid) or _obs("x"))
    out, note = ec.fetch_series_with_repair("World Bank", "QQ:NY.GDP.MKTP.CD",
                                            series_name="GDP (current US$)")
    assert out is None
    assert not any(s.endswith("-USA") for s in seen)
    assert "unknown country code 'QQ'" in note


def test_wrong_country_mirror_response_is_declined(monkeypatch, wb_down):
    # the host answers with a US series for a French request: geography check declines it
    monkeypatch.setattr(ec, "fetch_dbnomics",
                        lambda sid, **k: _obs("GDP (current US$) – United States"))
    out, note = ec.fetch_series_with_repair("World Bank", "FR:NY.GDP.MKTP.CD",
                                            series_name="GDP (current US$)")
    assert out is None
    assert "geography" in note


def test_wrong_country_mirror_declined_without_series_name(monkeypatch, wb_down):
    monkeypatch.setattr(ec, "fetch_dbnomics",
                        lambda sid, **k: _obs("GDP (current US$) – United States"))
    out, note = ec.fetch_series_with_repair("World Bank", "FR:NY.GDP.MKTP.CD")
    assert out is None and "geography" in note


def test_mirror_response_with_different_concept_is_declined(monkeypatch, wb_down):
    monkeypatch.setattr(ec, "fetch_dbnomics",
                        lambda sid, **k: _obs("Population, total – France"))
    out, note = ec.fetch_series_with_repair("World Bank", "FR:NY.GDP.MKTP.CD",
                                            series_name="GDP (current US$)")
    assert out is None and "concept check" in note


def test_matching_mirror_response_is_kept(monkeypatch, wb_down):
    monkeypatch.setattr(ec, "fetch_dbnomics",
                        lambda sid, **k: _obs("GDP (current US$) – France")
                        if sid == "WB/WDI/A-NY.GDP.MKTP.CD-FRA" else None)
    out, note = ec.fetch_series_with_repair("World Bank", "FR:NY.GDP.MKTP.CD",
                                            series_name="GDP (current US$)")
    assert out and out["resolved_id"] == "WB/WDI/A-NY.GDP.MKTP.CD-FRA"
    assert "provider redundancy" in note


def test_country_without_name_pattern_uses_id_geography(monkeypatch, wb_down):
    # Kenya has no name pattern: the mirror id's ISO3 must agree with the requested code
    assert ec.implied_geography("World Bank", "KE:NY.GDP.MKTP.CD") == \
        ec.implied_geography("DBnomics", "WB/WDI/A-NY.GDP.MKTP.CD-KEN")
    assert not ec._geo_compatible(ec.implied_geography("World Bank", "KE:X.Y"),
                                  ec.implied_geography("DBnomics", "WB/WDI/A-X.Y-USA"))


# --------------------------------------------------------------------------------------------
# #4 — concept check: identifiers, measurement type, two-sided ratio
# --------------------------------------------------------------------------------------------

NEW_MISMATCHES = [
    ("Money stock (M3)", "M2 Money Stock"),
    ("GDP per capita (PPP)", "GDP per capita (constant 2015 US$)"),
    ("Current account (BoP, current US$)", "Current account balance (% of GDP)"),
    ("Unemployment rate", "Unemployment Level"),
    ("Number of unemployed persons", "Unemployment rate (percent)"),
    ("Price level", "Inflation rate"),
    ("FDI net inflows", "Foreign direct investment, net inflows (% of GDP)"),
    ("Household debt", "Household debt to GDP ratio"),
]

STILL_MATCH = [
    ("Federal Funds Rate", "Federal Funds Effective Rate"),
    ("Real GDP", "Real Gross Domestic Product"),
    ("CPI", "Consumer Price Index for All Urban Consumers: All Items in U.S. City Average"),
    ("Consumer Price Index (CPI)", "Consumer Price Index for All Urban Consumers: All Items"),
    ("Gross domestic product (GDP)", "Gross Domestic Product"),
    ("Money stock (M3)", "M3 for the Euro Area"),
    ("CBOE Volatility Index (VIX)", "CBOE Volatility Index: VIX"),
    ("GDP per capita, PPP (constant 2017 international $)",
     "GDP per capita, PPP (constant 2017 international $)"),
    ("Foreign direct investment, net inflows (BoP, current US$)",
     "Foreign direct investment, net inflows (BoP, current US$)"),
    ("Unemployment rate", "Unemployment, total (% of total labor force) (modeled ILO estimate)"),
    ("Unemployment Rate Level", "Unemployment Rate"),
    ("Labor force participation rate",
     "Labor force participation rate, total (% of total population ages 15+)"),
    ("Industrial Production Index", "Industrial Production: Total Index"),
    ("Tax revenue (% of GDP)", "Tax revenue (% of GDP)"),
    ("Top 1% Wealth Share", "Share of Net Worth Held by the Top 1% (99th to 100th Wealth Percentiles)"),
]


@pytest.mark.parametrize("requested,title", NEW_MISMATCHES)
def test_review_cases_are_mismatches(requested, title):
    status, reason = ec.title_concept_check(requested, title)
    assert status == "mismatch", (requested, title, reason)


@pytest.mark.parametrize("requested,title", STILL_MATCH)
def test_obvious_equivalents_still_match(requested, title):
    assert ec.title_concept_check(requested, title) == ("match", ""), (requested, title)
