# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Real open connectors for economic data (World Bank / DBnomics / OECD / IMF / BIS / ECB /
Eurostat / BLS — all keyless). FRED was structurally the ONLY real source (the dispatch had 2
connectors and simulated everything else); these tests lock each connector's parsing (mocked HTTP,
payload shapes captured from the LIVE APIs on 2026-07-09) and the registry dispatch rules."""

from unittest.mock import patch, MagicMock

import pytest

from shared.tools import econ_connectors as ec


def _resp(json_data=None, text=""):
    r = MagicMock()
    r.json.return_value = json_data
    r.text = text
    return r


def test_world_bank_parse_and_country_prefix():
    payload = [{"page": 1}, [
        {"date": "2023", "value": 27811000000000.0},
        {"date": "2022", "value": 26006000000000.0},
        {"date": "2021", "value": None},                      # nulls dropped
    ]]
    with patch("shared.observability.tracked_get", return_value=_resp(payload)) as g:
        out = ec.fetch_series("World Bank", "NY.GDP.MKTP.CD")
        assert out["n_obs"] == 2
        assert out["observations"][0]["date"] == "2022"       # chronological
        assert "country/US/" in g.call_args.args[0]           # default country
        ec.fetch_series("World Bank", "DE:SL.UEM.TOTL.ZS")
        assert "country/DE/" in g.call_args.args[0]           # COUNTRY:INDICATOR honored


def test_dbnomics_parse_and_oecd_imf_aliases():
    payload = {"series": {"docs": [{"period": ["2023-01", "2023-02"], "value": [1.0, 2.0]}]}}
    with patch("shared.observability.tracked_get", return_value=_resp(payload)) as g:
        out = ec.fetch_series("DBnomics", "OECD/KEI/PRINTO01.USA.GY.M")
        assert out["n_obs"] == 2 and out["source_name"] == "DBnomics"
        # OECD/IMF ride DBnomics with the provider prepended and keep their display name
        out = ec.fetch_series("OECD", "KEI/PRINTO01.USA.GY.M")
        assert out["source_name"] == "OECD"
        assert "/series/OECD/KEI/PRINTO01.USA.GY.M" in g.call_args.args[0]
        out = ec.fetch_series("IMF", "WEO:2025-04/USA.NGDP_RPCH")
        assert out["source_name"] == "IMF"
    # malformed id (not PROVIDER/DATASET/SERIES) -> None, not a bad request
    assert ec.fetch_dbnomics("just-one-part") is None


def test_ecb_csv_parse():
    csv = ("KEY,FREQ,TIME_PERIOD,OBS_VALUE,OBS_STATUS\n"
           "EXR.D.USD.EUR.SP00.A,D,2026-07-08,1.1401,A\n"
           "EXR.D.USD.EUR.SP00.A,D,2026-07-09,1.1435,A\n")
    with patch("shared.observability.tracked_get", return_value=_resp(text=csv)):
        out = ec.fetch_series("ECB", "EXR/D.USD.EUR.SP00.A")
        assert out["n_obs"] == 2 and out["observations"][-1]["value"] == 1.1435


def test_eurostat_jsonstat_parse():
    payload = {"value": {"0": 3.9, "1": 3.8},
               "dimension": {"time": {"category": {"index": {"2026-04": 0, "2026-05": 1}}}}}
    with patch("shared.observability.tracked_get", return_value=_resp(payload)):
        out = ec.fetch_series("Eurostat", "une_rt_m?geo=DE&s_adj=SA&sex=T&age=TOTAL&unit=PC_ACT")
        assert out["n_obs"] == 2
        assert out["observations"][-1] == {"date": "2026-05", "value": 3.8}


def test_bls_parse_builds_monthly_dates():
    payload = {"status": "REQUEST_SUCCEEDED", "Results": {"series": [{"data": [
        {"year": "2026", "period": "M05", "value": "335.123"},
        {"year": "2026", "period": "M04", "value": "334.201"},
    ]}]}}
    with patch("shared.observability.tracked_get", return_value=_resp(payload)):
        out = ec.fetch_series("BLS", "CUUR0000SA0")
        assert out["n_obs"] == 2
        assert out["observations"][-1]["date"] == "2026-05"


def test_unconnected_source_returns_none_and_never_raises():
    # sources with no connector -> None (caller falls back to DISCLOSED simulation)
    assert ec.fetch_series("Google Trends API", "anything") is None
    assert ec.fetch_series("LinkedIn Jobs API", "x") is None
    assert ec.fetch_series("", "") is None
    # a connector whose HTTP call explodes -> None, never an exception
    with patch("shared.observability.tracked_get", side_effect=RuntimeError("net down")):
        assert ec.fetch_series("World Bank", "NY.GDP.MKTP.CD") is None


def test_connected_sources_registry_is_truthful():
    names = set(ec.connected_sources())
    # every advertised name (beyond the stage-native FRED/Yahoo) must resolve in the dispatch
    for name in names - {"FRED", "Yahoo Finance"}:
        assert name.lower() in ec._CONNECTORS, f"{name} advertised but not dispatchable"
    # and the dispatch has no hidden sources missing from the advertisement
    assert {n.lower() for n in names} >= set(ec._CONNECTORS)

class TestReviewHardening:
    """Connector fixes: quote-aware ECB CSV, Eurostat unpinned-dimension decline,
    BLS M13 exclusion."""

    def test_ecb_quoted_commas_do_not_shift_columns(self):
        csv = ('KEY,FREQ,TIME_PERIOD,OBS_VALUE,TITLE\n'
               'EXR.D.USD.EUR.SP00.A,D,2026-07-09,1.1435,"US dollar/Euro, spot rate"\n')
        with patch("shared.observability.tracked_get", return_value=_resp(text=csv)):
            out = ec.fetch_series("ECB", "EXR/D.USD.EUR.SP00.A")
            assert out["observations"][0]["value"] == 1.1435   # not shifted by the quoted comma

    def test_eurostat_unpinned_dimension_declines(self):
        # 'sex' has 2 categories -> flat index no longer aligns with time -> honest None
        payload = {"value": {"0": 3.9, "1": 3.8},
                   "dimension": {
                       "sex": {"category": {"index": {"M": 0, "F": 1}}},
                       "time": {"category": {"index": {"2026-05": 0}}}}}
        with patch("shared.observability.tracked_get", return_value=_resp(payload)):
            assert ec.fetch_series(
                "Eurostat", "une_rt_m?geo=DE&s_adj=SA&age=TOTAL&unit=PC_ACT") is None

    def test_bls_m13_annual_average_excluded(self):
        payload = {"status": "REQUEST_SUCCEEDED", "Results": {"series": [{"data": [
            {"year": "2025", "period": "M13", "value": "310.0"},   # annual avg pseudo-month
            {"year": "2025", "period": "M12", "value": "312.1"},
        ]}]}}
        with patch("shared.observability.tracked_get", return_value=_resp(payload)):
            out = ec.fetch_series("BLS", "CUUR0000SA0")
            assert out["n_obs"] == 1
            assert out["observations"][0]["date"] == "2025-12"


class TestRepairUpgrades:
    """Curated exact-id table + WDI-pinned search + FRED-capable repair ladder —
    a live run lost 3/6 series to LLM id near-misses the old ladder couldn't fix."""

    def test_curated_lookup_fixes_live_near_misses(self):
        assert ec.lookup_known_series("Currency in Circulation") == ("FRED", "CURRCIR")
        assert ec.lookup_known_series("Tax revenue (% of GDP)") == ("World Bank", "GC.TAX.TOTL.GD.ZS")
        assert ec.lookup_known_series(
            "People with accounts at a financial institution or mobile-money-service "
            "provider (% of population ages 15+)") == ("World Bank", "FX.OWN.TOTL.ZS")

    def test_curated_lookup_declines_unknown_and_respects_word_boundaries(self):
        assert ec.lookup_known_series("Quantum Flux Capacitance Index") is None
        assert ec.lookup_known_series("") is None
        # 'm2' must not fire inside 'km2'
        assert ec.lookup_known_series("land area km2") is None

    def test_curated_lookup_specificity_order(self):
        # 'real gdp' must win over the generic 'gdp' entry
        assert ec.lookup_known_series("Real GDP") == ("FRED", "GDPC1")
        assert ec.lookup_known_series("GDP (current US$)") == ("World Bank", "NY.GDP.MKTP.CD")


class TestComponentQualifierGuard:
    """'CPI-U Shelter' was repaired to CPIAUCSL (ALL-ITEMS CPI) because
    the matcher saw 'cpi' and ignored the component qualifier. A component-naming query must
    match a component-covering entry or decline — never the aggregate id."""

    def test_shelter_maps_to_the_shelter_id_not_all_items(self):
        assert ec.lookup_known_series("CPI-U Shelter") == ("FRED", "CUSR0000SAH1")
        assert ec.lookup_known_series("Shelter (CPI-U)") == ("FRED", "CUSR0000SAH1")

    def test_uncovered_component_declines_instead_of_aggregate(self):
        # no curated core-CPI id -> honest None, NOT CPIAUCSL
        assert ec.lookup_known_series("Core CPI (ex food and energy)") is None
        assert ec.lookup_known_series("CPI Food at Home") is None

    def test_aggregate_queries_unaffected(self):
        assert ec.lookup_known_series("Consumer Price Index for All Urban Consumers") == \
            ("FRED", "CPIAUCSL")
        assert ec.lookup_known_series("Inflation") == ("FRED", "CPIAUCSL")

    def test_curated_additions_debt_service_and_government(self):
        # a live run force-mapped debt service to HOUST and lacked any G series
        assert ec.lookup_known_series("Household Debt Service Ratio") == ("FRED", "TDSP")
        assert ec.lookup_known_series("Government Spending") == ("FRED", "GCE")
        assert ec.lookup_known_series("Real Government Purchases") == ("FRED", "GCE")

    def test_repair_uses_curated_table_for_fred_near_miss(self, monkeypatch):
        fake = {"observations": [{"date": "2026-01-01", "value": 2.3}], "n_obs": 1,
                "start_date": "2026-01-01", "end_date": "2026-01-01", "source_name": "FRED",
                "series_title": "Currency in Circulation"}   # repairs are title-checked
        fetched = []

        def fake_fred(series_id):
            fetched.append(series_id)
            return fake if series_id == "CURRCIR" else None
        monkeypatch.setattr(ec, "_fetch_fred_native", fake_fred)
        out, note = ec.fetch_series_with_repair("FRED", "CURRALL",
                                                series_name="Currency in Circulation")
        assert out is fake
        assert "curated table" in note and "CURRCIR" in note
        assert fetched == ["CURRALL", "CURRCIR"]   # direct attempt first, then the repair

    def test_repair_uses_curated_table_for_world_bank_near_miss(self, monkeypatch):
        fake = {"observations": [{"date": "2024-01-01", "value": 11.2}], "n_obs": 1,
                "start_date": "2024-01-01", "end_date": "2024-01-01", "source_name": "World Bank",
                "series_title": "Tax revenue (% of GDP)"}

        def fake_fetch(source, sid, collector=None, agent=""):
            return fake if sid == "GC.TAX.TOTL.GD.ZS" else None
        monkeypatch.setattr(ec, "fetch_series", fake_fetch)
        out, note = ec.fetch_series_with_repair("World Bank", "GC.TAX.TOTL.ZS",
                                                series_name="Tax revenue (% of GDP)")
        assert out is fake
        assert "curated table" in note and "GC.TAX.TOTL.GD.ZS" in note

    def test_repair_falls_through_to_wdi_search_when_curated_misses(self, monkeypatch):
        fake = {"observations": [{"date": "2024-01-01", "value": 5.0}], "n_obs": 1,
                "start_date": "2024-01-01", "end_date": "2024-01-01", "source_name": "DBnomics",
                "series_title": "Obscure WDI measure - United States"}  # title-checked
        monkeypatch.setattr(ec, "fetch_series", lambda *a, **k: None)
        monkeypatch.setattr(ec, "search_wdi_series",
                            lambda q, **k: "WB/WDI/A-XX.YY.ZZ-USA")
        # the mirror rung tries WB/WDI/A-XX.YY.WRONG-USA first — a WRONG id's mirror
        # fails in reality, so the mock only serves the SEARCHED id (shape-aware)
        monkeypatch.setattr(ec, "fetch_dbnomics",
                            lambda sid, **k: fake if sid == "WB/WDI/A-XX.YY.ZZ-USA" else None)
        out, note = ec.fetch_series_with_repair("World Bank", "XX.YY.WRONG",
                                                series_name="Obscure WDI measure")
        assert out is fake
        assert "WDI search" in note

    def test_wdi_search_is_geo_guarded(self):
        payload = {"series": {"docs": [
            {"series_code": "A-XX.YY.ZZ-ABW", "series_name": "Something - Aruba"},
            {"series_code": "A-XX.YY.ZZ-USA", "series_name": "Something - United States"},
        ]}}
        with patch("shared.observability.tracked_get", return_value=_resp(payload)):
            assert ec.search_wdi_series("something") == "WB/WDI/A-XX.YY.ZZ-USA"


class TestMechanismFirstResolver:
    """The curated table is FROZEN; coverage grows via runtime
    provider-index search (field-agnostic), guarded for relevance like the geography and component guards."""

    def _fred_payload(self, *pairs):
        return {"seriess": [{"id": i, "title": t} for i, t in pairs]}

    def test_fred_search_accepts_full_token_coverage(self, monkeypatch):
        monkeypatch.setenv("FRED_API_KEY", "k")
        payload = self._fred_payload(
            ("TDSP", "Household Debt Service Payments as a Percent of Disposable Income"))
        monkeypatch.setattr(ec, "_get_json", lambda *a, **k: payload)
        assert ec.search_fred_series("household debt service ratio") == "TDSP"

    def test_fred_search_declines_drifted_hits(self, monkeypatch):
        # the live case: 'infant mortality' returns life-expectancy series -> decline
        monkeypatch.setenv("FRED_API_KEY", "k")
        payload = self._fred_payload(
            ("SPDYNLE00INUSA", "Life Expectancy at Birth, Total for the United States"))
        monkeypatch.setattr(ec, "_get_json", lambda *a, **k: payload)
        assert ec.search_fred_series("infant mortality rate") is None

    def test_fred_search_rejects_foreign_geo_titles(self, monkeypatch):
        # live case: 'secondary school enrollment' top-popularity hit was Lao PDR data
        monkeypatch.setenv("FRED_API_KEY", "k")
        payload = self._fred_payload(
            ("SEENRSECOFMZSLAO", "School Enrollment, Secondary (% net) for Lao PDR"),
            ("SEENRSECUSA", "School Enrollment, Secondary (% net) for the United States"))
        monkeypatch.setattr(ec, "_get_json", lambda *a, **k: payload)
        assert ec.search_fred_series("secondary school enrollment") == "SEENRSECUSA"

    def test_fred_search_negation_blind_spot_closed(self, monkeypatch):
        # live case: 'CPI: Food' matched core CPI 'All Items LESS FOOD and
        # Energy' — component words inside an exclusion clause must not certify a hit.
        monkeypatch.setenv("FRED_API_KEY", "k")
        payload = self._fred_payload(
            ("CPILFESL", "Consumer Price Index for All Urban Consumers: "
                         "All Items Less Food and Energy in U.S. City Average"))
        monkeypatch.setattr(ec, "_get_json", lambda *a, **k: payload)
        assert ec.search_fred_series(
            "Consumer Price Index for All Urban Consumers: Food in U.S. City Average") is None
        # ...while a GENUINE core-CPI query still matches (negation stripped on both sides)
        assert ec.search_fred_series(
            "Consumer Price Index less food and energy") == "CPILFESL"

    def test_fred_search_without_key_is_none(self, monkeypatch):
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        monkeypatch.setattr(ec, "_get_json",
                            lambda *a, **k: (_ for _ in ()).throw(AssertionError("no HTTP")))
        assert ec.search_fred_series("anything at all") is None

    def test_resolver_curated_first_then_search_then_wdi(self, monkeypatch):
        ec._RESOLVE_CACHE.clear()
        calls = []
        monkeypatch.setenv("AEL_PROVIDER_SEARCH", "1")
        monkeypatch.setattr(ec, "search_fred_series",
                            lambda q, **k: calls.append(("fred", q)) or None)
        monkeypatch.setattr(ec, "search_wdi_series",
                            lambda q, **k: calls.append(("wdi", q)) or "WB/WDI/A-SP.DYN.IMRT.IN-USA")
        # every candidate's official title is concept-checked
        titles = {"GCE": "Government Consumption Expenditures and Gross Investment",
                  "WB/WDI/A-SP.DYN.IMRT.IN-USA":
                      "Mortality rate, infant (per 1,000 live births) - United States"}
        monkeypatch.setattr(ec, "fetch_series_title", lambda src, sid, *a, **k: titles.get(sid))
        # curated hit short-circuits: no search calls
        assert ec.resolve_fetchable_series("Government Spending") == ("FRED", "GCE")
        assert calls == []
        # non-curated falls through FRED -> WDI
        out = ec.resolve_fetchable_series("infant mortality rate")
        assert out == ("DBnomics", "WB/WDI/A-SP.DYN.IMRT.IN-USA")
        assert [c[0] for c in calls] == ["fred", "wdi"]
        # cached: no further search calls
        ec.resolve_fetchable_series("infant mortality rate")
        assert len(calls) == 2
        ec._RESOLVE_CACHE.clear()

    def test_resolver_offline_env_skips_network_rungs(self, monkeypatch):
        ec._RESOLVE_CACHE.clear()
        monkeypatch.setenv("AEL_PROVIDER_SEARCH", "0")
        monkeypatch.setattr(ec, "search_fred_series",
                            lambda *a, **k: (_ for _ in ()).throw(AssertionError("network rung fired")))
        assert ec.resolve_fetchable_series("some exotic requirement") is None
        assert ec.resolve_fetchable_series("Government Spending") == ("FRED", "GCE")
        ec._RESOLVE_CACHE.clear()


class TestProviderRedundancy:
    """A transient World Bank outage turned three known-good ids into
    disclosed simulations. For an exact WDI-shaped id, the DBnomics WDI mirror is tried
    deterministically before any search — same indicator, different host."""

    def test_wb_outage_served_from_dbnomics_mirror(self, monkeypatch):
        # the mirror response passes the repair acceptance check (title concept + geography)
        mirror = {"observations": [{"date": "2020", "value": 1.0}], "n_obs": 1,
                  "start_date": "2020", "end_date": "2020", "source_name": "DBnomics",
                  "series_title": "Poverty headcount ratio at $2.15 a day (2017 PPP) "
                                  "(% of population) – United States"}
        monkeypatch.setattr(ec, "fetch_series", lambda *a, **k: None)      # WB down
        monkeypatch.setattr(ec, "_fetch_fred_native", lambda *a, **k: None)
        monkeypatch.setattr(ec, "fetch_dbnomics",
                            lambda sid, **k: mirror if sid == "WB/WDI/A-SI.POV.DDAY-USA" else None)
        out, note = ec.fetch_series_with_repair("World Bank", "SI.POV.DDAY",
                                                series_name="Poverty headcount ratio (% of population)")
        assert out and out["resolved_source"] == "DBnomics"
        assert "provider redundancy" in note and "WDI mirror" in note

    def test_country_prefixed_id_maps_geo(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(ec, "fetch_series", lambda *a, **k: None)
        monkeypatch.setattr(ec, "_fetch_fred_native", lambda *a, **k: None)
        def fake_dbn(sid, **k):
            seen["sid"] = sid
            return None
        monkeypatch.setattr(ec, "fetch_dbnomics", fake_dbn)
        monkeypatch.setenv("AEL_PROVIDER_SEARCH", "0")
        ec.fetch_series_with_repair("World Bank", "DEU:SL.UEM.TOTL.ZS", series_name="x")
        assert seen["sid"] == "WB/WDI/A-SL.UEM.TOTL.ZS-DEU"

    def test_non_wdi_shaped_id_skips_mirror(self, monkeypatch):
        calls = []
        monkeypatch.setattr(ec, "fetch_series", lambda *a, **k: None)
        monkeypatch.setattr(ec, "_fetch_fred_native", lambda *a, **k: None)
        monkeypatch.setattr(ec, "fetch_dbnomics", lambda sid, **k: calls.append(sid))
        monkeypatch.setenv("AEL_PROVIDER_SEARCH", "0")
        ec.fetch_series_with_repair("World Bank", "not-an-indicator", series_name="")
        assert not [c for c in calls if c and c.startswith("WB/WDI/A-not")]
