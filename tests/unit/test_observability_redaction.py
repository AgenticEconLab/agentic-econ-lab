# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""HTTP errors re-raised by the tracked tool calls carry no credentials."""

import pytest
import requests

from shared import observability as ob


def test_fred_error_is_redacted_when_reraised(monkeypatch):
    class _Resp:
        status_code, ok = 400, False

        def raise_for_status(self):
            raise requests.HTTPError("400 Client Error: Bad Request for url: "
                                     "https://api.stlouisfed.org/fred/series/observations"
                                     "?series_id=X&api_key=abcdef0123456789&file_type=json")

    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp())
    with pytest.raises(requests.HTTPError) as ei:
        ob.tracked_fred_get_series("X", api_key="abcdef0123456789")
    assert "abcdef0123456789" not in str(ei.value)
    assert "api_key=REDACTED" in str(ei.value)


def test_llm_error_is_redacted(monkeypatch):
    import httpx
    from shared.llm import LLMClient
    from shared.observability import MetricsCollector

    def boom(*a, **k):
        raise httpx.HTTPError("429 for url https://x.example/v1?key=SECRETKEY123456&alt=json")

    c = MetricsCollector()
    llm = LLMClient(model="gpt-4o-mini", api_key="dummy", collector=c, agent_name="T")
    monkeypatch.setattr(llm, "_invoke_openai", boom)
    with pytest.raises(httpx.HTTPError) as ei:
        llm.invoke([{"role": "user", "content": "hi"}])
    assert "SECRETKEY123456" not in str(ei.value)
    assert all("SECRETKEY123456" not in str(getattr(r, "error", "")) for r in c.llm_calls)
