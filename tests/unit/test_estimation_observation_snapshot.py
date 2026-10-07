# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Estimation loads the DataTeam's archived observations.

The loader re-fetched every series from the live APIs, ignoring the full-observations
sidecar the DataTeam source stage writes (metadata.observations_file). An API outage at
estimation time then fell back to 5-row previews, and a series the DataTeam's deterministic
checks rejected (constant, all-NaN, duplicate dates) re-entered estimation."""

import json

import pytest

from EstimationTeam.ael.estim_harness import data_loader as dl


def _obs(n, start=2000, const=None, dup=False):
    rows = [{"date": f"{start + i}-01-01", "value": (const if const is not None else float(i))}
            for i in range(n)]
    if dup:
        rows.append({"date": f"{start}-01-01", "value": 99.0})
    return rows


def _item(sid, source="World Bank", **kw):
    d = {"series_id": sid, "series_name": sid.lower(), "source_name": source,
         "data_simulated": False, "retrieval_date": "2026-09-30",
         "data_preview": [{"date": "2019-01-01", sid: 1.0}, {"date": "2020-01-01", sid: 2.0}]}
    d.update(kw)
    return d


@pytest.fixture()
def outage(monkeypatch):
    calls = []
    monkeypatch.setattr(dl, "_fetch_connector", lambda *a, **k: calls.append(a) or None)
    monkeypatch.setattr(dl, "_fetch_fred", lambda *a, **k: calls.append(a) or None)
    monkeypatch.setattr(dl, "_fetch_yahoo", lambda *a, **k: calls.append(a) or None)
    return calls


def _artifact(tmp_path, items, snapshot):
    path = tmp_path / "api_source_output_observations.json"
    path.write_text(json.dumps(snapshot))
    return {"retrieved_data": items, "metadata": {"observations_file": str(path)}}


def test_snapshot_used_instead_of_refetch_during_outage(tmp_path, outage):
    art = _artifact(tmp_path, [_item("A.B.C"), _item("GDPC1", source="FRED")],
                    {"A.B.C": _obs(30), "GDPC1": _obs(40)})
    series_map, _alias, notes = dl.load_panel(art)
    assert set(series_map) == {"A.B.C", "GDPC1"}
    assert len(series_map["A.B.C"]) == 30 and len(series_map["GDPC1"]) == 40
    assert outage == []                                   # no live API call at all
    assert any("archived observations" in n and "A.B.C" in n for n in notes)
    assert not any("preview" in n for n in notes)


@pytest.mark.parametrize("obs,word", [(_obs(30, const=5.0), "constant"),
                                      (_obs(30, dup=True), "duplicate"),
                                      ([{"date": f"{2000 + i}", "value": None} for i in range(30)],
                                       "no numeric")])
def test_series_failing_datateam_checks_is_excluded(tmp_path, outage, obs, word):
    art = _artifact(tmp_path, [_item("BAD"), _item("GOOD")], {"BAD": obs, "GOOD": _obs(30)})
    series_map, _alias, notes = dl.load_panel(art)
    assert series_map is not None and "BAD" not in series_map and "GOOD" in series_map
    assert any(n.startswith("excluded BAD") and word in n for n in notes)
    assert outage == []                                   # a rejected series is not re-fetched


def test_series_absent_from_snapshot_is_refetched_and_disclosed(tmp_path, monkeypatch):
    import pandas as pd
    fresh = pd.Series(range(25), index=pd.date_range("2000-01-01", periods=25, freq="YS"),
                      dtype=float)
    monkeypatch.setattr(dl, "_fetch_connector", lambda src, sid, **k: fresh if sid == "NEW" else None)
    art = _artifact(tmp_path, [_item("OLD"), _item("NEW")], {"OLD": _obs(30)})
    series_map, _alias, notes = dl.load_panel(art)
    assert set(series_map) == {"OLD", "NEW"}
    assert any("NEW" in n and "new vintage" in n for n in notes)


def test_simulated_series_still_excluded_with_snapshot(tmp_path, outage):
    art = _artifact(tmp_path, [_item("SIM", data_simulated=True), _item("OK")],
                    {"SIM": _obs(30), "OK": _obs(30)})
    series_map, _alias, notes = dl.load_panel(art)
    assert "SIM" not in series_map
    assert any(n.startswith("excluded SIM") for n in notes)


def test_standalone_artifact_without_snapshot_still_refetches(monkeypatch):
    import pandas as pd
    fresh = pd.Series(range(25), index=pd.date_range("2000-01-01", periods=25, freq="YS"),
                      dtype=float)
    monkeypatch.setattr(dl, "_fetch_connector", lambda *a, **k: fresh)
    series_map, _alias, notes = dl.load_panel({"retrieved_data": [_item("X.Y")]})
    assert series_map and len(series_map["X.Y"]) == 25


def test_snapshot_found_inside_pipeline_envelope(tmp_path, outage):
    inner = _artifact(tmp_path, [_item("A.B.C")], {"A.B.C": _obs(30)})
    envelope = {"name": "data_source", "data": inner, "producer": "DataTeam"}
    series_map, _alias, _notes = dl.load_panel(envelope)
    assert series_map and len(series_map["A.B.C"]) == 30


def test_bad_date_in_snapshot_does_not_abort():
    from EstimationTeam.ael.estim_harness.transforms import to_series
    obs = [{"date": f"{1980 + i}-01-01", "value": float(i)} for i in range(40)]
    obs.append({"date": "bad-date", "value": 7.0})
    s = to_series(obs)
    assert s is not None and len(s) == 40
    q = to_series([{"date": "2020-Q1", "value": 1.0}, {"date": "2020-Q2", "value": 2.0}])
    assert q is not None and len(q) == 2
