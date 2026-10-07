# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Tests for the WebCrawl abstraction (2026-04-20 Firecrawl → WebCrawl refactor).

Covers:
- base types (WebCrawlResult, UnsupportedCapability)
- four providers via mocked HTTP / SDK (Firecrawl, Tavily, Jina, Crawl4AI)
- WebCrawlClient fallback-chain dispatch
- provider_map.WEBCRAWL_PROVIDERS registry entries
- ToolRegistry registration of webcrawl_* tools
- DeprecationWarning from the _firecrawl_compat shim
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from shared.webcrawl.base import UnsupportedCapability, WebCrawlResult
from shared.webcrawl.providers import (
    Crawl4AIProvider,
    FirecrawlProvider,
    JinaProvider,
    TavilyProvider,
    get_provider,
)
from shared.tools.webcrawl_tool import WebCrawlClient


# ============================================================================
# Base types
# ============================================================================


class TestBaseTypes:
    def test_result_defaults(self):
        r = WebCrawlResult(success=True)
        assert r.success is True
        assert r.data is None
        assert r.markdown is None
        assert r.provider == ""
        assert r.attempts == []

    def test_unsupported_capability_message(self):
        e = UnsupportedCapability("jina", "crawl")
        assert "jina" in str(e) and "crawl" in str(e)
        assert e.provider == "jina"
        assert e.capability == "crawl"


# ============================================================================
# Provider registry
# ============================================================================


class TestProviderLookup:
    def test_get_known_providers(self):
        assert get_provider("firecrawl") is FirecrawlProvider
        assert get_provider("tavily") is TavilyProvider
        assert get_provider("jina") is JinaProvider
        assert get_provider("crawl4ai") is Crawl4AIProvider

    def test_get_unknown_provider_raises(self):
        with pytest.raises(KeyError):
            get_provider("scrapy")


# ============================================================================
# WEBCRAWL_PROVIDERS in provider_map
# ============================================================================


class TestProviderMap:
    def test_webcrawl_providers_dict_exists(self):
        from shared import provider_map

        assert hasattr(provider_map, "WEBCRAWL_PROVIDERS")
        assert set(provider_map.WEBCRAWL_PROVIDERS.keys()) == {
            "firecrawl",
            "tavily",
            "jina",
            "crawl4ai",
        }

    def test_firecrawl_capabilities_complete(self):
        from shared.provider_map import WEBCRAWL_PROVIDERS

        assert WEBCRAWL_PROVIDERS["firecrawl"]["capabilities"] == {
            "scrape",
            "crawl",
            "extract",
            "map",
        }

    def test_jina_has_no_crawl_or_map(self):
        from shared.provider_map import WEBCRAWL_PROVIDERS

        caps = WEBCRAWL_PROVIDERS["jina"]["capabilities"]
        assert "crawl" not in caps
        assert "map" not in caps


# ============================================================================
# FirecrawlProvider
# ============================================================================


class TestFirecrawlProvider:
    def test_is_available_without_key(self, monkeypatch):
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        p = FirecrawlProvider()
        assert p.is_available() is False

    def test_is_available_with_key(self, monkeypatch):
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fk_test")
        p = FirecrawlProvider()
        assert p.is_available() is True

    def test_scrape_uses_sdk(self, monkeypatch):
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fk_test")
        p = FirecrawlProvider()
        mock_app = MagicMock()
        mock_app.scrape_url.return_value = SimpleNamespace(markdown="# hi")
        p._app = mock_app

        result = p.scrape("https://example.com")

        assert result.success is True
        assert result.markdown == "# hi"
        assert result.provider == "firecrawl"
        mock_app.scrape_url.assert_called_once_with(
            "https://example.com", formats=["markdown"]
        )

    def test_scrape_captures_sdk_exceptions(self, monkeypatch):
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fk_test")
        p = FirecrawlProvider()
        mock_app = MagicMock()
        mock_app.scrape_url.side_effect = ValueError("rate limited")
        p._app = mock_app

        result = p.scrape("https://example.com")
        assert result.success is False
        assert "rate limited" in result.error
        assert result.provider == "firecrawl"


# ============================================================================
# TavilyProvider
# ============================================================================


def _mock_response(json_body, status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.ok = status == 200
    resp.json.return_value = json_body
    return resp


class TestTavilyProvider:
    def test_scrape_without_key_returns_error(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        p = TavilyProvider()
        result = p.scrape("https://example.com")
        assert result.success is False
        assert "TAVILY_API_KEY" in result.error

    def test_scrape_posts_and_returns_markdown(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "tvly_test")
        p = TavilyProvider()
        with patch("shared.observability.tracked_post") as mock_post:
            mock_post.return_value = _mock_response(
                {"results": [{"url": "https://example.com", "raw_content": "# hi"}]}
            )
            result = p.scrape("https://example.com")

        assert result.success is True
        assert result.markdown == "# hi"
        assert result.provider == "tavily"
        mock_post.assert_called_once()

    def test_map_raises_unsupported(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "tvly_test")
        p = TavilyProvider()
        with pytest.raises(UnsupportedCapability):
            p.map("https://example.com")


# ============================================================================
# JinaProvider
# ============================================================================


class TestJinaProvider:
    def test_is_available_without_key(self, monkeypatch):
        monkeypatch.delenv("JINA_API_KEY", raising=False)
        assert JinaProvider().is_available() is True

    def test_scrape_gets_reader_url(self, monkeypatch):
        p = JinaProvider()
        with patch("shared.observability.tracked_get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.text = "# from jina"
            mock_get.return_value = mock_resp
            result = p.scrape("https://example.com")

        assert result.success is True
        assert result.markdown == "# from jina"
        assert result.provider == "jina"
        called_url = mock_get.call_args.kwargs["url"]
        assert called_url.startswith("https://r.jina.ai/")
        assert called_url.endswith("https://example.com")

    def test_crawl_raises_unsupported(self):
        with pytest.raises(UnsupportedCapability):
            JinaProvider().crawl("https://example.com")

    def test_map_raises_unsupported(self):
        with pytest.raises(UnsupportedCapability):
            JinaProvider().map("https://example.com")


# ============================================================================
# Crawl4AIProvider
# ============================================================================


class TestCrawl4AIProvider:
    def test_is_available_requires_base_url_or_library(self, monkeypatch):
        # No base_url AND library not importable -> not available.
        monkeypatch.delenv("CRAWL4AI_BASE_URL", raising=False)
        monkeypatch.setattr("shared.webcrawl.providers.crawl4ai._lib_importable", lambda: False)
        assert Crawl4AIProvider().is_available() is False

    def test_is_available_true_in_library_mode(self, monkeypatch):
        # crawl4ai library present (no base_url) -> available via library mode.
        monkeypatch.delenv("CRAWL4AI_BASE_URL", raising=False)
        monkeypatch.setattr("shared.webcrawl.providers.crawl4ai._lib_importable", lambda: True)
        assert Crawl4AIProvider().is_available() is True

    def test_scrape_posts_to_crawl_endpoint(self, monkeypatch):
        monkeypatch.setenv("CRAWL4AI_BASE_URL", "http://localhost:11235")
        p = Crawl4AIProvider()
        with patch("shared.observability.tracked_post") as mock_post:
            mock_post.return_value = _mock_response(
                {"results": [{"url": "https://example.com", "markdown": "# hi"}]}
            )
            result = p.scrape("https://example.com")

        assert result.success is True
        assert result.markdown == "# hi"
        assert result.provider == "crawl4ai"
        called_url = mock_post.call_args.kwargs["url"]
        assert called_url == "http://localhost:11235/crawl"

    def test_map_raises_unsupported(self, monkeypatch):
        monkeypatch.setenv("CRAWL4AI_BASE_URL", "http://localhost:11235")
        with pytest.raises(UnsupportedCapability):
            Crawl4AIProvider().map("https://example.com")


# ============================================================================
# WebCrawlClient fallback chain
# ============================================================================


class TestWebCrawlClient:
    def test_default_primary_is_trafilatura(self, monkeypatch):
        # open-first: with no env override the default primary must be the keyless
        # trafilatura, NOT firecrawl (which only burns credits as a last fallback).
        monkeypatch.delenv("WEBCRAWL_PROVIDER", raising=False)
        monkeypatch.delenv("WEBCRAWL_FALLBACK_CHAIN", raising=False)
        c = WebCrawlClient()
        assert c.chain[0] == "trafilatura"

    def test_env_selects_primary(self, monkeypatch):
        monkeypatch.setenv("WEBCRAWL_PROVIDER", "tavily")
        monkeypatch.setenv("WEBCRAWL_FALLBACK_CHAIN", "firecrawl,jina")
        c = WebCrawlClient()
        assert c.chain == ["tavily", "firecrawl", "jina"]

    def test_explicit_args_override_env(self, monkeypatch):
        monkeypatch.setenv("WEBCRAWL_PROVIDER", "tavily")
        c = WebCrawlClient(primary="jina", fallback_chain=["firecrawl"])
        assert c.chain == ["jina", "firecrawl"]

    def test_is_available_false_when_nothing_configured(self, monkeypatch):
        for v in (
            "FIRECRAWL_API_KEY",
            "TAVILY_API_KEY",
            "CRAWL4AI_BASE_URL",
        ):
            monkeypatch.delenv(v, raising=False)
        # Jina is always "available" — pin the chain to exclude it. trafilatura/crawl4ai
        # are open (no key), so disable crawl4ai's library mode for this "nothing
        # configured" scenario.
        monkeypatch.setattr("shared.webcrawl.providers.crawl4ai._lib_importable", lambda: False)
        c = WebCrawlClient(primary="firecrawl", fallback_chain=["tavily", "crawl4ai"])
        assert c.is_available() is False

    def test_falls_back_when_primary_unconfigured(self, monkeypatch):
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        monkeypatch.setenv("TAVILY_API_KEY", "tvly_test")

        c = WebCrawlClient(primary="firecrawl", fallback_chain=["tavily"])
        with patch("shared.observability.tracked_post") as mock_post:
            mock_post.return_value = _mock_response(
                {"results": [{"url": "u", "raw_content": "# fallback"}]}
            )
            result = c.scrape("https://example.com")

        assert result.success is True
        assert result.provider == "tavily"
        # The first attempt was the firecrawl "not configured" skip.
        providers_tried = [a["provider"] for a in result.attempts]
        assert providers_tried == ["firecrawl", "tavily"]
        assert result.attempts[0]["success"] is False
        assert result.attempts[1]["success"] is True

    def test_falls_back_on_unsupported_capability(self, monkeypatch):
        # Jina doesn't support crawl; chain falls through to configured firecrawl.
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fk_test")

        c = WebCrawlClient(primary="jina", fallback_chain=["firecrawl"])
        # Stub the firecrawl SDK layer
        fc = c._resolve("firecrawl")
        fc._app = MagicMock()
        fc._app.crawl_url.return_value = {"pages": []}

        result = c.crawl("https://example.com", limit=5)

        assert result.success is True
        assert result.provider == "firecrawl"
        # First attempt was the jina UnsupportedCapability skip.
        assert result.attempts[0]["provider"] == "jina"
        assert result.attempts[0]["success"] is False
        assert "map" not in (result.attempts[0]["error"] or "")  # it was crawl, not map

    def test_all_providers_fail_returns_last_error(self, monkeypatch):
        # Nothing configured anywhere (+ crawl4ai library mode off so it can't scrape).
        for v in ("FIRECRAWL_API_KEY", "TAVILY_API_KEY", "CRAWL4AI_BASE_URL"):
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setattr("shared.webcrawl.providers.crawl4ai._lib_importable", lambda: False)
        c = WebCrawlClient(primary="firecrawl", fallback_chain=["tavily", "crawl4ai"])
        result = c.scrape("https://example.com")
        assert result.success is False
        assert [a["provider"] for a in result.attempts] == [
            "firecrawl",
            "tavily",
            "crawl4ai",
        ]

    def test_scrape_url_shim_returns_markdown_attribute(self, monkeypatch):
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fk_test")
        c = WebCrawlClient(primary="firecrawl", fallback_chain=[])
        fc = c._resolve("firecrawl")
        fc._app = MagicMock()
        fc._app.scrape_url.return_value = SimpleNamespace(markdown="# shim ok")

        result = c.scrape_url("https://example.com")
        assert result.markdown == "# shim ok"

    def test_active_provider_prefers_first_configured(self, monkeypatch):
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        monkeypatch.setenv("TAVILY_API_KEY", "tvly_test")
        c = WebCrawlClient(primary="firecrawl", fallback_chain=["tavily"])
        assert c.active_provider() == "tavily"


# ============================================================================
# ToolRegistry integration
# ============================================================================


class TestToolRegistryIntegration:
    """Registry is class-level state; other tests may clear it. These helpers
    force a fresh registration before assertion."""

    @staticmethod
    def _fresh_register():
        from shared.tools.tool_registry import ToolRegistry
        import shared.tools.register_all as reg_mod

        ToolRegistry.clear()
        reg_mod._registered = False
        reg_mod.register_all_tools()

    def test_four_webcrawl_tools_registered(self):
        self._fresh_register()
        from shared.tools.tool_registry import ToolRegistry

        names = ToolRegistry.tool_names()
        for t in ("webcrawl_scrape", "webcrawl_crawl", "webcrawl_extract", "webcrawl_map"):
            assert t in names, f"missing tool {t!r}; have {names!r}"

    def test_tools_categorized_as_web(self):
        self._fresh_register()
        from shared.tools.tool_registry import ToolRegistry

        spec = ToolRegistry.get_tool("webcrawl_scrape")
        assert spec is not None
        assert spec.category == "web"


# ============================================================================
# Deprecation shim
# ============================================================================


class TestModeNameDeprecationAlias:
    """Phase E: old Mode*Fc* names accepted with a DeprecationWarning."""

    def setup_method(self):
        # Clear the one-shot latch so each test sees a fresh warning.
        from pipeline import pipeline_config as pc

        pc._DEPRECATION_WARNED.clear()

    def test_stage_config_accepts_legacy_mode(self):
        from pipeline.pipeline_config import StageConfig

        with pytest.warns(DeprecationWarning, match="ModeNoFcNoHITL"):
            s = StageConfig(team="IdeationTeam", mode="ModeNoFcNoHITL")
        assert s.mode == "ModeNoWcNoHITL"

    def test_pipeline_config_accepts_legacy_mode(self):
        from pipeline.pipeline_config import PipelineConfig, StageConfig

        with pytest.warns(DeprecationWarning):
            cfg = PipelineConfig(
                name="t",
                mode="ModeWithFcNoHITL",
                stages=[StageConfig(team="IdeationTeam", mode="ModeNoWcNoHITL")],
            )
        assert cfg.mode == "ModeWithWcNoHITL"

    def test_new_names_do_not_warn(self):
        import warnings as _w
        from pipeline.pipeline_config import StageConfig

        with _w.catch_warnings(record=True) as captured:
            _w.simplefilter("always")
            _ = StageConfig(team="IdeationTeam", mode="ModeNoWcNoHITL")
        assert not any(
            issubclass(w.category, DeprecationWarning) for w in captured
        )

    def test_warning_is_one_shot_per_value(self):
        import warnings as _w
        from pipeline.pipeline_config import StageConfig

        # First call warns.
        with pytest.warns(DeprecationWarning):
            StageConfig(team="A", mode="ModeNoFcNoHITL")
        # Second call with the same old name does NOT warn again.
        with _w.catch_warnings(record=True) as captured:
            _w.simplefilter("always")
            StageConfig(team="B", mode="ModeNoFcNoHITL")
        assert not any(
            issubclass(w.category, DeprecationWarning) for w in captured
        )


class TestFirecrawlCompatShim:
    def test_importing_shim_class_warns_once(self, monkeypatch):
        import shared.webcrawl._firecrawl_compat as compat
        compat._WARNED = False  # reset the one-shot latch

        monkeypatch.setenv("FIRECRAWL_API_KEY", "fk_test")
        with pytest.warns(DeprecationWarning):
            _ = compat.FirecrawlApp(api_key="explicit-key")

        # Second construction must not re-warn.
        import warnings as _w

        with _w.catch_warnings(record=True) as captured:
            _w.simplefilter("always")
            _ = compat.FirecrawlApp(api_key="explicit-key")
        assert not any(
            issubclass(w.category, DeprecationWarning) for w in captured
        )


# ---------------------------------------------------------------------------
# Hard per-provider crawl timeout (a hung crawl4ai/playwright page must not
# stall the pipeline — observed a ~3h hang on one DOI page).
# ---------------------------------------------------------------------------

class TestCrawlTimeout:
    def test_returns_result_within_timeout(self):
        from shared.tools.webcrawl_tool import _call_with_timeout
        assert _call_with_timeout(lambda: 42, 5.0) == 42

    def test_raises_timeout_on_hang(self):
        import time as _t
        import pytest
        from shared.tools.webcrawl_tool import _call_with_timeout
        with pytest.raises(TimeoutError):
            _call_with_timeout(lambda: _t.sleep(10), 0.3)

    def test_propagates_callee_error(self):
        import pytest
        from shared.tools.webcrawl_tool import _call_with_timeout

        def boom():
            raise ValueError("provider failed")

        with pytest.raises(ValueError):
            _call_with_timeout(boom, 5.0)

    def test_dispatch_falls_through_on_timeout(self, monkeypatch):
        """A hanging provider times out and the chain continues to the next."""
        import time as _t
        from shared.tools import webcrawl_tool as wct
        from shared.webcrawl.base import WebCrawlResult

        monkeypatch.setattr(wct, "_CRAWL_TIMEOUT", 0.3)

        class _Hang:
            def is_available(self):
                return True

            def scrape(self, *a, **k):
                _t.sleep(10)  # never returns

        class _Good:
            def is_available(self):
                return True

            def scrape(self, *a, **k):
                return WebCrawlResult(success=True, provider="good", data={"markdown": "ok"})

        client = wct.WebCrawlClient(primary="hang", fallback_chain=["good"])
        monkeypatch.setattr(client, "_resolve",
                            lambda name: _Hang() if name == "hang" else _Good())
        result = client.scrape("https://example.com")
        assert result.success and result.provider == "good"
        assert any("exceeded" in (a.get("error") or "") for a in result.attempts)
