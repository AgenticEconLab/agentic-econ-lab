# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Fair direction discovery: field-balanced slate + committee choice with provenance.

The properties under test: no hardcoded seed anywhere in the path; fields without signals
are dropped (never invented); the ungrounded fallback is disclosed; the committee ballot
trail (model, lens, ranking, rationale) is returned; failure refuses honestly instead of
falling back to a macro default."""

import json
from unittest.mock import patch

import pytest

import shared.tools.direction_scout as ds


def _fake_hits(field_name):
    return [{"title": f"New {field_name} study", "url": f"https://x.org/{field_name[:4]}",
             "content": f"Recent developments in {field_name} research."}]


class TestDiscover:
    def test_slate_is_field_balanced_and_grounded(self, monkeypatch):
        searched = []

        def fake_search(query, max_results=4):
            searched.append(query)
            return _fake_hits(query)
        monkeypatch.setattr("shared.tools.news_search.search_news", fake_search)

        extracted = json.dumps([
            {"field_code": c, "direction": f"A concrete {n} question?", "evidence": "url"}
            for c, n in ds.JEL_FIELDS])
        from shared.llm import LLMClient
        monkeypatch.setattr(LLMClient, "invoke", lambda self, messages, **kw: extracted)

        slate = ds.discover_directions()
        assert slate["grounded"] is True
        assert len(searched) == len(ds.JEL_FIELDS)          # one search per field, no favorites
        codes = {c["field_code"] for c in slate["candidates"]}
        assert codes == {c for c, _ in ds.JEL_FIELDS}        # every field represented

    def test_fields_without_signals_dropped_not_invented(self, monkeypatch):
        def fake_search(query, max_results=4):
            return _fake_hits(query) if "labor" in query else []
        monkeypatch.setattr("shared.tools.news_search.search_news", fake_search)
        from shared.llm import LLMClient
        monkeypatch.setattr(LLMClient, "invoke", lambda self, messages, **kw: json.dumps(
            [{"field_code": "J", "direction": "A labor question?", "evidence": "url"}]))
        slate = ds.discover_directions()
        assert [c["field_code"] for c in slate["candidates"]] == ["J"]
        assert sum("dropped, not invented" in n for n in slate["notes"]) == len(ds.JEL_FIELDS) - 1

    def test_no_signals_anywhere_falls_back_ungrounded_and_disclosed(self, monkeypatch):
        monkeypatch.setattr("shared.tools.news_search.search_news",
                            lambda q, max_results=4: [])
        from shared.llm import LLMClient
        monkeypatch.setattr(LLMClient, "invoke", lambda self, messages, **kw: json.dumps(
            [{"field_code": c, "direction": f"{n}?", "evidence": "none (ungrounded)"}
             for c, n in ds.JEL_FIELDS[:3]]))
        slate = ds.discover_directions()
        assert slate["grounded"] is False
        assert any("UNGROUNDED" in n for n in slate["notes"])


class TestChoose:
    def _slate(self):
        return {"candidates": [
            {"field_code": "J", "field": "labor", "direction": "Labor Q?", "evidence": "u"},
            {"field_code": "E", "field": "macro", "direction": "Macro Q?", "evidence": "u"},
            {"field_code": "I", "field": "health", "direction": "Health Q?", "evidence": "u"},
        ], "notes": [], "grounded": True}

    def test_committee_borda_with_full_ballot_provenance(self, monkeypatch):
        monkeypatch.setenv("AEL_HITL_MODE", "llm_economist")
        monkeypatch.setattr("shared.llm_economist.get_committee_models",
                            lambda: ["m-theory", "m-data", "m-policy"])
        rankings = iter([
            {"ranking": [0, 2, 1], "rationale": "labor best"},
            {"ranking": [0, 1, 2], "rationale": "labor again"},
            {"ranking": [2, 0, 1], "rationale": "health first"},
        ])
        from shared.llm import LLMClient
        monkeypatch.setattr(LLMClient, "invoke",
                            lambda self, messages, **kw: json.dumps(next(rankings)))
        out = ds.choose_direction(self._slate())
        assert out["method"] == "committee_borda"
        assert out["chosen"]["field_code"] == "J"   # Borda: J=3+3+2=8, I=2+1+3=6, E=1+2+1=4
        assert len(out["ballots"]) == 3
        assert all(b.get("lens") for b in out["ballots"])
        assert all(b.get("rationale") for b in out["ballots"])

    def test_empty_slate_refuses_instead_of_seeding(self):
        out = ds.choose_direction({"candidates": []})
        assert out["chosen"] is None and "explicitly" in out["error"]

    def test_all_ballots_failing_refuses(self, monkeypatch):
        monkeypatch.setenv("AEL_HITL_MODE", "auto")
        from shared.llm import LLMClient
        monkeypatch.setattr(LLMClient, "invoke",
                            lambda self, messages, **kw: (_ for _ in ()).throw(OSError("down")))
        out = ds.choose_direction(self._slate())
        assert out["chosen"] is None and out["method"] == "committee_failed"


def test_pipeline_dev_default_and_auto_release_path():
    """The development default topic stays macro (the evaluating expert is a macro ABM/DSGE
    economist — expert judgment is the development-stage validation instrument); the fair 'auto' discovery path must exist and be wired for release."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2] / "run_ael_pipeline.py").read_text()
    assert "_DEV_STAGE_DEFAULT_TOPIC" in src and "macro" in src
    assert 'default=_DEV_STAGE_DEFAULT_TOPIC' in src
    # the release path: --topic auto runs the scout with provenance
    assert "discover_directions" in src and "direction_provenance" in src
