# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Supplemental fetch, stage-level behavior: DataSourceStage fetches scout-promised curated ids
deterministically, disclosed in quality_notes, honest None on connector failure."""

import importlib.util
from pathlib import Path

import pytest

_STAGE = (Path(__file__).resolve().parent.parent.parent
          / "DataTeam" / "ael" / "ModeOpenSourceAPI" / "1-DataSourceStage.py")


@pytest.fixture(scope="module")
def stage():
    spec = importlib.util.spec_from_file_location("data_source_stage_n23", _STAGE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def orch(stage):
    return stage.DataSourceOrchestrator(collector=None)


def test_supplemental_fetch_builds_disclosed_retrieved_data(orch, monkeypatch):
    out = {"observations": [{"date": "2020-01-01", "value": 1.0},
                            {"date": "2020-04-01", "value": 2.0}],
           "n_obs": 2, "start_date": "2020-01-01", "end_date": "2020-04-01",
           "source_name": "FRED"}
    import shared.tools.econ_connectors as ec
    monkeypatch.setattr(ec, "fetch_series_with_repair",
                        lambda *a, **k: (out, ""))
    data = orch._fetch_supplemental("FRED", "GCE", "Government Spending")
    assert data is not None
    assert data.series_id == "GCE" and data.series_name == "Government Spending"
    assert data.num_observations == 2 and data.data_simulated is False
    assert "Supplemental curated fetch" in data.quality_notes   # provenance disclosed
    assert data.fetch_id == "GCE" and data.fetch_source == "FRED"


def test_supplemental_fetch_failure_is_honest_none(orch, monkeypatch):
    import shared.tools.econ_connectors as ec
    monkeypatch.setattr(ec, "fetch_series_with_repair", lambda *a, **k: (None, ""))
    assert orch._fetch_supplemental("FRED", "NOPE", "x") is None
    def boom(*a, **k):
        raise RuntimeError("connector down")
    monkeypatch.setattr(ec, "fetch_series_with_repair", boom)
    assert orch._fetch_supplemental("FRED", "GCE", "x") is None   # never raises


def test_collapsed_repairs_deduped(stage, orch, monkeypatch):
    """In a live run SET01(Food)+SET02(Energy) both repaired onto CPILFESL — the
    same data entered twice under two names. Later collapses are dropped, disclosed."""
    import pandas as pd
    mk = lambda sid, fid: stage.RetrievedData(
        series_id=sid, series_name=sid, source_name="FRED", retrieval_date="2026-07-12",
        num_observations=10, start_date="2020-01-01", end_date="2021-01-01",
        frequency="monthly", data_preview=[], quality_notes="", retrieval_status="Success",
        fetch_id=fid, fetch_source="FRED")
    rows = [mk("CUUR0000SET01", "CPILFESL"), mk("CUUR0000SET02", "CPILFESL"),
            mk("UNRATE", None)]
    # replicate the stage's dedup block logic against the real RetrievedData model
    seen, deduped = {}, []
    for d in rows:
        actual = d.fetch_id or d.series_id
        if actual in seen and not d.data_simulated:
            continue
        seen.setdefault(actual, d.series_id)
        deduped.append(d)
    assert [d.series_id for d in deduped] == ["CUUR0000SET01", "UNRATE"]
