# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""The DataTeam's main FRED path must carry FRED's own series title
and flag a proxy when it differs from the requested variable, as the open-API connectors do."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("pydantic")

_AGENTS = Path(__file__).resolve().parents[2]
_STAGE = _AGENTS / "DataTeam" / "ael" / "ModeOpenSourceAPI" / "1-DataSourceStage.py"
_spec = importlib.util.spec_from_file_location("data_source_stage_fred_title", _STAGE)
ds = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ds
_spec.loader.exec_module(ds)


class _FakeFred:
    def __init__(self, api_key=None):
        pass

    def get_series_info(self, series_id):
        return {"GDPC1": {"title": "Real Gross Domestic Product"},
                "FEDFUNDS": {"title": "Federal Funds Effective Rate"}}[series_id]


def _agent(monkeypatch):
    monkeypatch.setattr(ds, "FRED_AVAILABLE", True, raising=False)
    monkeypatch.setattr(ds, "fredapi", SimpleNamespace(Fred=_FakeFred), raising=False)
    obs = {"observations": [{"date": f"2020-0{i}-01", "value": str(i)} for i in range(1, 7)]}
    monkeypatch.setattr(ds.ToolRegistry, "invoke",
                        lambda *a, **k: SimpleNamespace(success=True, data=obs, error=None))
    agent = ds.DataRetrievalAgent.__new__(ds.DataRetrievalAgent)
    agent.fred_api_key, agent.collector, agent.agent_name = "k", None, "DataRetrievalAgent"
    return agent


def _series(sid, name):
    return SimpleNamespace(series_id=sid, series_name=name, frequency="Quarterly")


def test_fred_series_carries_source_title(monkeypatch):
    out = _agent(monkeypatch)._retrieve_fred_data(_series("GDPC1", "Real GDP"))
    assert out.source_title == "Real Gross Domestic Product"
    assert out.proxy_for is None


def test_fred_series_flags_proxy(monkeypatch):
    out = _agent(monkeypatch)._retrieve_fred_data(
        _series("FEDFUNDS", "Real Policy Rate (Treatment Proxy)"))
    assert out.source_title == "Federal Funds Effective Rate"
    assert out.proxy_for == "Real Policy Rate (Treatment Proxy)"
    assert "PROXY" in out.quality_notes
