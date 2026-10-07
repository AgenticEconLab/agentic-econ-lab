# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""DataSourceStage fixes found by auditing the v0.7.1 runs.

- every Yahoo series was simulated (corrupt shared yfinance cache, error dropped, no repair).
- FRED key printed in error URLs; frequency taken from the LLM label; discontinued series
  not flagged.
- committee votes taken on a blank "\\n> " prompt; objections printed and ignored."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_STAGE = (Path(__file__).resolve().parent.parent.parent
          / "DataTeam" / "ael" / "ModeOpenSourceAPI" / "1-DataSourceStage.py")


@pytest.fixture(scope="module")
def ds():
    spec = importlib.util.spec_from_file_location("data_source_stage_v071", _STAGE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _spec(ds, sid, name, source="FRED", freq="monthly", end="present"):
    return ds.DataSeries(series_id=sid, series_name=name, source_name=source, description="",
                         frequency=freq, units="", seasonal_adjustment="", start_date="",
                         end_date=end, last_updated="", variable_role="outcome")


def _agent(ds):
    return ds.DataRetrievalAgent("sk-test")


def _monthly(start_year, n):
    return [{"date": f"{start_year + i // 12}-{i % 12 + 1:02d}-01", "value": float(i % 7)}
            for i in range(n)]


# --------------------------------------------------------------------------------------------
# Yahoo Finance
# --------------------------------------------------------------------------------------------

def test_yahoo_failure_is_logged_and_routed_through_repair(ds, monkeypatch, capsys):
    import shared.tools.econ_connectors as ec
    monkeypatch.setattr(ds, "YFINANCE_AVAILABLE", True)
    monkeypatch.setattr(ds.ToolRegistry, "invoke", lambda *a, **k: SimpleNamespace(
        success=False, data=None, error="database disk image is malformed"))
    sp500 = {"observations": _monthly(2000, 60), "n_obs": 60, "start_date": "2000-01-01",
             "end_date": "2004-12-01", "source_name": "FRED", "series_title": "S&P 500",
             "resolved_id": "SP500", "resolved_source": "FRED"}
    monkeypatch.setattr(ec, "fetch_series_with_repair",
                        lambda *a, **k: (sp500, "id repaired via FRED index search: '^GSPC' "
                                                "(Yahoo Finance) -> FRED/SP500"))
    out = _agent(ds).retrieve_data([_spec(ds, "^GSPC", "S&P 500", source="Yahoo Finance")])
    assert len(out) == 1 and not out[0].data_simulated
    assert out[0].fetch_id == "SP500" and out[0].title_check == "match"
    assert "database disk image is malformed" in capsys.readouterr().out   # error logged


def test_yfinance_cache_is_per_process(monkeypatch, tmp_path):
    import shared.tools.yfinance_tool as yt
    calls = {}
    fake = SimpleNamespace(set_tz_cache_location=lambda d: calls.setdefault("tz", d),
                           set_cache_location=lambda d: calls.setdefault("cookie", d))
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    monkeypatch.setattr(yt, "_YF_CACHE_DIR", None)
    monkeypatch.setenv("AEL_YF_CACHE_DIR", str(tmp_path / "yf"))
    d = yt.ensure_yfinance_cache()
    assert d == str(tmp_path / "yf") and calls == {"tz": d, "cookie": d}
    assert ".cache/py-yfinance" not in d
    monkeypatch.setattr(yt, "_YF_CACHE_DIR", None)


# --------------------------------------------------------------------------------------------
# key redaction, frequency from data, discontinued series
# --------------------------------------------------------------------------------------------

def test_tool_errors_redact_api_keys():
    from shared.tools.redact import redact_secrets
    from shared.tools.tool_registry import ToolRegistry
    from pydantic import BaseModel

    url = ("400 Client Error: Bad Request for url: https://api.stlouisfed.org/fred/series/"
           "observations?series_id=DGS10Y&api_key=abcdef0123456789&file_type=json")
    assert "abcdef0123456789" not in redact_secrets(url)
    assert "api_key=REDACTED" in redact_secrets(url) and "series_id=DGS10Y" in redact_secrets(url)

    class _In(BaseModel):
        x: int = 0

    def boom(x=0, collector=None, agent=""):
        raise RuntimeError(url)
    ToolRegistry.register(name="_v071_boom", description="t", input_schema=_In, handler=boom)
    try:
        res = ToolRegistry.invoke("_v071_boom", {})
        assert not res.success and "abcdef0123456789" not in res.error
    finally:
        ToolRegistry._tools.pop("_v071_boom", None)


def test_frequency_is_inferred_from_dates_not_label(ds, monkeypatch):
    monkeypatch.setattr(ds, "FRED_AVAILABLE", False)
    obs = {"observations": [{"date": o["date"], "value": str(o["value"])} for o in _monthly(2010, 48)]}
    monkeypatch.setattr(ds.ToolRegistry, "invoke",
                        lambda *a, **k: SimpleNamespace(success=True, data=obs, error=None))
    agent = _agent(ds)
    agent.fred_api_key = "k"
    spec = _spec(ds, "FEDFUNDS", "Federal Funds Rate", freq="daily", end="2013-12-31")
    out = agent._retrieve_fred_data(spec)
    assert out.frequency == "monthly" and out.declared_frequency == "daily"
    notes = agent.validate_retrieved_data([spec], [out])
    assert any("Frequency mismatch" in n and "monthly" in n for n in notes)


def test_discontinued_series_flagged(ds, monkeypatch):
    monkeypatch.setattr(ds, "FRED_AVAILABLE", False)
    obs = {"observations": [{"date": o["date"], "value": str(o["value"])} for o in _monthly(2015, 85)]}
    monkeypatch.setattr(ds.ToolRegistry, "invoke",
                        lambda *a, **k: SimpleNamespace(success=True, data=obs, error=None))
    agent = _agent(ds)
    agent.fred_api_key = "k"
    out = agent._retrieve_fred_data(_spec(ds, "TEDRATE", "TED Spread", end="present"))
    assert out.end_date == "2022-01-01"
    assert out.discontinued and "DISCONTINUED" in out.quality_notes


# --------------------------------------------------------------------------------------------
# checkpoints vote on the real question; objections act / are recorded
# --------------------------------------------------------------------------------------------

def _orchestrator(ds, tmp_path):
    orch = ds.DataSourceOrchestrator.__new__(ds.DataSourceOrchestrator)
    src = ds.APISource(source_id="A", source_name="FRED", source_type="central_bank", url="",
                       description="FRED", coverage="US", data_quality="High",
                       requires_api_key=True, api_key_available=True, update_frequency="d",
                       series_available=["UNRATE"])
    picks = [[_spec(ds, "PAYEMS", "Unemployment Rate")], [_spec(ds, "UNRATE", "Unemployment Rate")]]
    selections = []

    def identify(rq, reqs, sources):
        selections.append(rq)
        return picks[len(selections) - 1]
    orch.discovery_agent = SimpleNamespace(discover_sources=lambda *a: [src])
    orch.connection_agent = SimpleNamespace(verify_connections=lambda s: s)
    orch.series_agent = SimpleNamespace(identify_series=identify)
    agent = ds.DataRetrievalAgent("sk-test")

    def fake_fred(series):
        return agent._build_retrieved(series, _monthly(2000, 300), "FRED", "Unemployment Rate",
                                      "ok.")
    agent._retrieve_fred_data = fake_fred
    orch.retrieval_agent = agent
    orch.source_output = None
    return orch, selections


def test_checkpoints_ask_the_printed_question_and_act_on_objection(ds, tmp_path, monkeypatch):
    import DataTeam.ael.hitl as hitl
    asked = []

    def fake_input(prompt, default="", context=None, **k):
        asked.append((prompt, context))
        # object to the FIRST series selection only
        if "Series Selection" in prompt and sum("Series Selection" in p for p, _ in asked) == 1:
            return "no"
        if "Data Quality" in prompt:
            return "no — the sample is too short for the question"
        return default
    monkeypatch.setattr(hitl, "auto_input", fake_input)
    orch, selections = _orchestrator(ds, tmp_path)
    out = orch.run_source_pipeline("Does unemployment respond to policy?",
                                   [], ["FRED"], enable_hitl=True)
    prompts = [p for p, _ in asked]
    assert all(p.strip() != ">" and len(p) > 40 for p in prompts)        # not a blank prompt
    assert "Does unemployment respond to policy?" in prompts[0]
    assert any("PAYEMS" in p for p in prompts) and all(c for _, c in asked)  # context passed
    # objection at series review -> one bounded re-selection carrying the objection
    assert len(selections) == 2 and "REVIEWER OBJECTION" in selections[1]
    assert [d.series_id for d in out.retrieved_data] == ["UNRATE"]
    # the data-quality objection is recorded and disclosed
    objections = out.metadata["hitl_objections"]
    assert [o["checkpoint"] for o in objections] == ["Data Quality Review"]
    assert any("Data Quality Review" in lim and "did not approve" in lim for lim in out.limitations)
    rec = {c["checkpoint"]: c for c in out.metadata["hitl_checkpoints"]}
    assert rec["Series Selection Review"]["revised"] and rec["Series Selection Review"]["approved"]


def test_full_observations_written_to_sidecar(ds, tmp_path, monkeypatch):
    import DataTeam.ael.hitl as hitl
    monkeypatch.setattr(hitl, "auto_input", lambda p, default="", context=None, **k: default)
    orch, _ = _orchestrator(ds, tmp_path)
    orch.run_source_pipeline("RQ", [], ["FRED"], enable_hitl=False)
    target = tmp_path / "api_source_output.json"
    orch.save_source_output(str(target))
    saved = json.loads(target.read_text())
    obs_file = saved["metadata"]["observations_file"]
    full = json.loads(Path(obs_file).read_text())
    assert len(full["PAYEMS"]) == 300                     # full series, not the 5-row preview
    assert len(saved["retrieved_data"][0]["data_preview"]) == 5


def test_yahoo_series_carries_the_instrument_name(ds, monkeypatch):
    import pandas as pd
    monkeypatch.setattr(ds, "YFINANCE_AVAILABLE", True)
    idx = pd.date_range("2000-01-03", periods=40, freq="B")
    frame = pd.DataFrame({"Close": [100.0 + i for i in range(40)]}, index=idx)
    monkeypatch.setattr(ds.ToolRegistry, "invoke",
                        lambda *a, **k: SimpleNamespace(success=True, data=frame, error=None))
    fake_yf = SimpleNamespace(Ticker=lambda t: SimpleNamespace(
        get_info=lambda: {"longName": "S&P 500", "shortName": "S&P 500"}))
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)
    out = _agent(ds)._retrieve_yahoo_data(_spec(ds, "^GSPC", "S&P 500 Index",
                                                source="Yahoo Finance", freq="daily"))
    assert out is not None and out.source_title == "S&P 500"
