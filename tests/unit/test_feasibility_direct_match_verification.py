# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""A direct FEASIBLE needs a successful retrieval AND a title match.

The scout graded a direct pool match FEASIBLE when its title was merely "not a mismatch"
(missing/unverified titles passed), and the adapter gave quality 1.0 to selected ids with no
retrieval record: a selected 'Wealth Gini' with empty retrieved_data was FEASIBLE with
fetchable=False, so no supplemental retrieval was requested."""

from DataTeam.ael.feasibility import (
    AvailabilityScout, DataRequirementsSpec, RequirementVerdict, available_series_from_source,
)
from DataTeam.ael.schemas.stage_outputs import DataRequirement


def _drs(name):
    return DataRequirementsSpec(
        research_question="rq",
        requirements=[DataRequirement(requirement_id="R1", variable_name=name,
                                      description=name, frequency="annual",
                                      time_period="1990-2023", geographic_coverage="US",
                                      unit_of_measurement="x", priority="High")])


def _source(selected, retrieved):
    return {"selected_series": selected, "retrieved_data": retrieved}


def _sel(sid, name, source="World Bank"):
    return {"series_id": sid, "series_name": name, "source_name": source}


def _ret(sid, name, title, **kw):
    d = {"series_id": sid, "series_name": name, "source_name": "World Bank",
         "source_title": title, "data_simulated": False, "retrieval_status": "Success",
         "num_observations": 30}
    d.update(kw)
    return d


def test_selected_but_unretrieved_series_is_not_feasible():
    avail = available_series_from_source(_source([_sel("WG.X", "Wealth Gini")], []))
    assert avail[0].retrieved is False and avail[0].quality < 1.0
    f = AvailabilityScout().grade(_drs("Wealth Gini"), avail)[0]
    assert f.verdict != RequirementVerdict.FEASIBLE
    assert any("no successful retrieval" in p.bias_caveat for p in f.proxy_candidates)


def test_unretrieved_series_resolved_by_verified_lookup_schedules_fetch():
    avail = available_series_from_source(_source([_sel("WG.X", "Wealth Gini")], []))
    scout = AvailabilityScout(
        fetchable_lookup=lambda name: ("DBnomics", "WID/X/WEALTH-GINI", "Wealth Gini"))
    f = scout.grade(_drs("Wealth Gini"), avail)[0]
    assert f.verdict == RequirementVerdict.FEASIBLE and f.fetchable is True


def test_retrieved_series_without_title_is_not_feasible():
    avail = available_series_from_source(
        _source([_sel("SL.UEM.TOTL.ZS", "Unemployment rate")],
                [_ret("SL.UEM.TOTL.ZS", "Unemployment rate", "")]))
    f = AvailabilityScout().grade(_drs("Unemployment rate"), avail)[0]
    assert f.verdict == RequirementVerdict.PROXY_ONLY
    assert any("could not be verified" in p.bias_caveat for p in f.proxy_candidates)


def test_failed_retrieval_record_is_not_feasible():
    avail = available_series_from_source(
        _source([_sel("A.B", "Unemployment rate")],
                [_ret("A.B", "Unemployment rate", "Unemployment rate",
                      retrieval_status="Failed", num_observations=0)]))
    assert avail[0].retrieved is False
    f = AvailabilityScout().grade(_drs("Unemployment rate"), avail)[0]
    assert f.verdict != RequirementVerdict.FEASIBLE


def test_retrieved_and_title_matched_series_is_feasible():
    avail = available_series_from_source(
        _source([_sel("SL.UEM.TOTL.ZS", "Unemployment rate")],
                [_ret("SL.UEM.TOTL.ZS", "Unemployment rate",
                      "Unemployment, total (% of total labor force) (modeled ILO estimate)",
                      num_observations="34")]))
    assert avail[0].retrieved is True and avail[0].quality == 1.0
    f = AvailabilityScout().grade(_drs("Unemployment rate"), avail)[0]
    assert f.verdict == RequirementVerdict.FEASIBLE


def test_simulated_string_flag_is_not_retrieved():
    avail = available_series_from_source(
        _source([_sel("A.B", "Unemployment rate")],
                [_ret("A.B", "Unemployment rate", "Unemployment rate", data_simulated="True")]))
    assert avail[0].retrieved is False
