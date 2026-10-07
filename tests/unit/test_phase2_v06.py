# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
AEL V0.6 Phase 2 — Domain Data Expansion Tests

Tests for MCP data source registry, Alpha Vantage, OpenBB, Financial Datasets,
Elicit, Semantic Scholar, FRED V2, and IMF clients.
All tests are offline (no network calls).
"""

import os
from unittest.mock import MagicMock, patch

import pytest


# ── MCP Data Source Registry ─────────────────────────────────────────────


class TestMCPDataSourceRegistry:
    def test_module_imports(self):
        from shared.data.mcp_data_sources import (
            MCPDataSourceConfig,
            MCPDataSourceRegistry,
            MCP_DATA_SOURCES,
        )

    def test_default_sources_exist(self):
        from shared.data.mcp_data_sources import MCP_DATA_SOURCES
        assert "alpha_vantage" in MCP_DATA_SOURCES
        assert "openbb" in MCP_DATA_SOURCES
        assert "financial_datasets" in MCP_DATA_SOURCES
        assert "semantic_scholar" in MCP_DATA_SOURCES

    def test_list_all_sources(self):
        from shared.data.mcp_data_sources import MCPDataSourceRegistry
        reg = MCPDataSourceRegistry()
        sources = reg.list_sources()
        assert len(sources) >= 4

    def test_list_sources_by_workflow(self):
        from shared.data.mcp_data_sources import MCPDataSourceRegistry
        reg = MCPDataSourceRegistry()
        osa = reg.list_sources(workflow="open_source_api")
        assert any(s.name == "alpha_vantage" for s in osa)
        # "all" workflow sources should also appear
        assert any(s.name == "openbb" for s in osa)

    def test_list_sources_premium(self):
        from shared.data.mcp_data_sources import MCPDataSourceRegistry
        reg = MCPDataSourceRegistry()
        premium = reg.list_sources(workflow="premium_subscribed")
        assert any(s.name == "financial_datasets" for s in premium)

    def test_get_config(self):
        from shared.data.mcp_data_sources import MCPDataSourceRegistry
        reg = MCPDataSourceRegistry()
        config = reg.get_config("alpha_vantage")
        assert config is not None
        assert config.env_key == "ALPHA_VANTAGE_API_KEY"
        assert "macro_indicators" in config.capabilities

    def test_get_config_nonexistent(self):
        from shared.data.mcp_data_sources import MCPDataSourceRegistry
        reg = MCPDataSourceRegistry()
        assert reg.get_config("nonexistent") is None

    def test_is_enabled_default_false(self):
        from shared.data.mcp_data_sources import MCPDataSourceRegistry
        reg = MCPDataSourceRegistry()
        assert reg.is_enabled("alpha_vantage") is False

    def test_is_enabled_with_env(self):
        from shared.data.mcp_data_sources import MCPDataSourceRegistry
        reg = MCPDataSourceRegistry()
        with patch.dict(os.environ, {"ALPHA_VANTAGE_MCP_ENABLED": "true"}):
            assert reg.is_enabled("alpha_vantage") is True

    def test_has_api_key_false(self):
        from shared.data.mcp_data_sources import MCPDataSourceRegistry
        reg = MCPDataSourceRegistry()
        with patch.dict(os.environ, {}, clear=True):
            assert reg.has_api_key("alpha_vantage") is False

    def test_has_api_key_true(self):
        from shared.data.mcp_data_sources import MCPDataSourceRegistry
        reg = MCPDataSourceRegistry()
        with patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": "test-key"}):
            assert reg.has_api_key("alpha_vantage") is True

    def test_get_client_disabled_returns_none(self):
        from shared.data.mcp_data_sources import MCPDataSourceRegistry
        reg = MCPDataSourceRegistry()
        assert reg.get_client("alpha_vantage") is None

    def test_get_status(self):
        from shared.data.mcp_data_sources import MCPDataSourceRegistry
        reg = MCPDataSourceRegistry()
        status = reg.get_status()
        assert "alpha_vantage" in status
        assert "enabled" in status["alpha_vantage"]
        assert "has_api_key" in status["alpha_vantage"]
        assert "fallback" in status["alpha_vantage"]

    def test_register_custom_source(self):
        from shared.data.mcp_data_sources import MCPDataSourceRegistry, MCPDataSourceConfig
        reg = MCPDataSourceRegistry()
        custom = MCPDataSourceConfig(
            name="custom_source",
            server_url="http://localhost:9999/mcp",
            workflow="open_source_api",
        )
        reg.register(custom)
        assert reg.get_config("custom_source") is not None

    def test_source_tools_defined(self):
        from shared.data.mcp_data_sources import MCP_DATA_SOURCES
        av = MCP_DATA_SOURCES["alpha_vantage"]
        assert len(av.tools) >= 2
        assert av.tools[0]["name"] == "av_get_macro_indicator"


# ── Alpha Vantage ────────────────────────────────────────────────────────


class TestAlphaVantageClient:
    def test_module_imports(self):
        from shared.data.alpha_vantage import (
            AlphaVantageClient,
            MACRO_INDICATORS,
            TECHNICAL_INDICATORS,
        )

    def test_supported_macro_indicators(self):
        from shared.data.alpha_vantage import AlphaVantageClient
        indicators = AlphaVantageClient.supported_macro_indicators()
        assert "REAL_GDP" in indicators
        assert "CPI" in indicators
        assert "UNEMPLOYMENT" in indicators
        assert len(indicators) >= 10

    def test_supported_technical_indicators(self):
        from shared.data.alpha_vantage import AlphaVantageClient
        indicators = AlphaVantageClient.supported_technical_indicators()
        assert "SMA" in indicators
        assert "RSI" in indicators
        assert len(indicators) >= 20

    def test_no_api_key_returns_error(self, monkeypatch):
        monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
        from shared.data.alpha_vantage import AlphaVantageClient
        client = AlphaVantageClient(api_key="")
        result = client.get_macro_indicator("REAL_GDP")
        assert "error" in result

    def test_rest_request_with_mock(self):
        from shared.data.alpha_vantage import AlphaVantageClient
        client = AlphaVantageClient(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": [{"date": "2024-01-01", "value": "3.1"}]}
        mock_resp.raise_for_status = MagicMock()
        with patch("shared.data.alpha_vantage.httpx.get", return_value=mock_resp):
            result = client.get_macro_indicator("REAL_GDP")
            assert "data" in result

    def test_search_symbols_with_mock(self):
        from shared.data.alpha_vantage import AlphaVantageClient
        client = AlphaVantageClient(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"bestMatches": []}
        mock_resp.raise_for_status = MagicMock()
        with patch("shared.data.alpha_vantage.httpx.get", return_value=mock_resp):
            result = client.search_symbols("Apple")
            assert "bestMatches" in result


# ── OpenBB ───────────────────────────────────────────────────────────────


class TestOpenBBClient:
    def test_module_imports(self):
        from shared.data.openbb_client import OpenBBClient

    def test_supported_categories(self):
        from shared.data.openbb_client import OpenBBClient
        cats = OpenBBClient.supported_categories()
        assert "equity" in cats
        assert "economy" in cats
        assert "fixedincome" in cats

    def test_sdk_not_installed_returns_error(self):
        from shared.data.openbb_client import OpenBBClient
        client = OpenBBClient()
        result = client.get_economy_indicators("GDP")
        assert "error" in result

    def test_equity_price_no_sdk(self):
        from shared.data.openbb_client import OpenBBClient
        client = OpenBBClient()
        result = client.get_equity_price("AAPL")
        assert "error" in result

    def test_fixed_income_no_sdk(self):
        from shared.data.openbb_client import OpenBBClient
        client = OpenBBClient()
        result = client.get_fixed_income("10y")
        assert "error" in result


# ── Financial Datasets ───────────────────────────────────────────────────


class TestFinancialDatasetsClient:
    def test_module_imports(self):
        from shared.data.financial_datasets import (
            FinancialDatasetsClient,
            STATEMENT_TYPES,
            DATA_TYPES,
        )

    def test_supported_statements(self):
        from shared.data.financial_datasets import FinancialDatasetsClient
        stmts = FinancialDatasetsClient.supported_statements()
        assert "income" in stmts
        assert "balance" in stmts
        assert "cashflow" in stmts

    def test_no_api_key_returns_error(self, monkeypatch):
        monkeypatch.delenv("FINANCIAL_DATASETS_API_KEY", raising=False)
        from shared.data.financial_datasets import FinancialDatasetsClient
        client = FinancialDatasetsClient(api_key="")
        result = client.get_financials("AAPL")
        assert "error" in result

    def test_get_prices_no_key(self, monkeypatch):
        monkeypatch.delenv("FINANCIAL_DATASETS_API_KEY", raising=False)
        from shared.data.financial_datasets import FinancialDatasetsClient
        client = FinancialDatasetsClient(api_key="")
        result = client.get_prices("AAPL")
        assert "error" in result

    def test_get_financials_with_mock(self):
        from shared.data.financial_datasets import FinancialDatasetsClient
        client = FinancialDatasetsClient(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"income_statements": [{"revenue": 100000}]}
        mock_resp.raise_for_status = MagicMock()
        with patch("shared.data.financial_datasets.httpx.get", return_value=mock_resp):
            result = client.get_financials("AAPL", statement="income")
            assert "income_statements" in result


# ── Elicit ───────────────────────────────────────────────────────────────


class TestElicitClient:
    def test_module_imports(self):
        from shared.data.elicit_client import ElicitClient

    def test_supported_capabilities(self):
        from shared.data.elicit_client import ElicitClient
        caps = ElicitClient.supported_capabilities()
        assert "search" in caps
        assert "extract" in caps
        assert "report" in caps

    def test_no_api_key_returns_error(self, monkeypatch):
        monkeypatch.delenv("ELICIT_API_KEY", raising=False)
        from shared.data.elicit_client import ElicitClient
        client = ElicitClient(api_key="")
        result = client.search_papers("AI economics")
        assert "error" in result

    def test_search_papers_with_mock(self):
        from shared.data.elicit_client import ElicitClient
        client = ElicitClient(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"papers": [{"title": "AI and GDP"}]}
        mock_resp.raise_for_status = MagicMock()
        with patch("shared.data.elicit_client.httpx.post", return_value=mock_resp):
            result = client.search_papers("AI economics", max_papers=5)
            assert "papers" in result

    def test_extract_with_mock(self):
        from shared.data.elicit_client import ElicitClient
        client = ElicitClient(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"rows": [{"sample_size": 1000}]}
        mock_resp.raise_for_status = MagicMock()
        with patch("shared.data.elicit_client.httpx.post", return_value=mock_resp):
            result = client.extract(["doi:123"], ["sample_size"])
            assert "rows" in result

    def test_get_report_with_mock(self):
        from shared.data.elicit_client import ElicitClient
        client = ElicitClient(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"report": "Summary of findings..."}
        mock_resp.raise_for_status = MagicMock()
        with patch("shared.data.elicit_client.httpx.post", return_value=mock_resp):
            result = client.get_report("What is the effect of AI on GDP?")
            assert "report" in result


# ── Semantic Scholar ─────────────────────────────────────────────────────


class TestSemanticScholarClient:
    def test_module_imports(self):
        from shared.data.s2_client import SemanticScholarClient

    def test_supported_capabilities(self):
        from shared.data.s2_client import SemanticScholarClient
        caps = SemanticScholarClient.supported_capabilities()
        assert "search" in caps
        assert "recommend" in caps
        assert "citations" in caps

    def test_search_with_mock(self):
        from shared.data.s2_client import SemanticScholarClient
        client = SemanticScholarClient()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": [{"paperId": "abc", "title": "Test"}]}
        mock_resp.raise_for_status = MagicMock()
        with patch("shared.data.s2_client.httpx.get", return_value=mock_resp):
            result = client.search("AI economics")
            assert "data" in result

    def test_recommend_with_mock(self):
        from shared.data.s2_client import SemanticScholarClient
        client = SemanticScholarClient()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"recommendedPapers": []}
        mock_resp.raise_for_status = MagicMock()
        with patch("shared.data.s2_client.httpx.get", return_value=mock_resp):
            result = client.recommend("abc123")
            assert "recommendedPapers" in result

    def test_get_paper_with_mock(self):
        from shared.data.s2_client import SemanticScholarClient
        client = SemanticScholarClient()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"paperId": "abc", "title": "Test Paper"}
        mock_resp.raise_for_status = MagicMock()
        with patch("shared.data.s2_client.httpx.get", return_value=mock_resp):
            result = client.get_paper("abc")
            assert result["title"] == "Test Paper"

    def test_get_citations_with_mock(self):
        from shared.data.s2_client import SemanticScholarClient
        client = SemanticScholarClient()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": []}
        mock_resp.raise_for_status = MagicMock()
        with patch("shared.data.s2_client.httpx.get", return_value=mock_resp):
            result = client.get_citations("abc")
            assert "data" in result


# ── FRED V2 ─────────────────────────────────────────────────────────────


class TestFREDv2Client:
    def test_module_imports(self):
        from shared.data.fred_v2 import FREDv2Client, NEW_V2_SERIES, COMMON_SERIES

    def test_new_v2_series(self):
        from shared.data.fred_v2 import FREDv2Client
        series = FREDv2Client.new_v2_series()
        assert "NASDAQCOM" in series
        assert "SOFR" in series

    def test_common_series(self):
        from shared.data.fred_v2 import FREDv2Client
        series = FREDv2Client.common_series()
        assert "GDP" in series
        assert "CPIAUCSL" in series
        assert "UNRATE" in series

    def test_no_api_key_returns_error(self, monkeypatch):
        # Client falls back to FRED_API_KEY env var when api_key is falsy;
        # clear it so the test exercises the no-key path regardless of .env.
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        from shared.data.fred_v2 import FREDv2Client
        client = FREDv2Client(api_key="")
        result = client.get_series("GDP")
        assert "error" in result

    def test_get_series_with_mock(self):
        from shared.data.fred_v2 import FREDv2Client
        client = FREDv2Client(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"observations": [{"date": "2024-01-01", "value": "28000"}]}
        mock_resp.raise_for_status = MagicMock()
        with patch("shared.data.fred_v2.httpx.get", return_value=mock_resp):
            result = client.get_series("GDP")
            assert "observations" in result

    def test_get_series_bulk(self):
        from shared.data.fred_v2 import FREDv2Client
        client = FREDv2Client(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"observations": []}
        mock_resp.raise_for_status = MagicMock()
        with patch("shared.data.fred_v2.httpx.get", return_value=mock_resp):
            results = client.get_series_bulk(["GDP", "CPI"])
            assert "GDP" in results
            assert "CPI" in results

    def test_search_series_with_mock(self):
        from shared.data.fred_v2 import FREDv2Client
        client = FREDv2Client(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"seriess": [{"id": "GDP"}]}
        mock_resp.raise_for_status = MagicMock()
        with patch("shared.data.fred_v2.httpx.get", return_value=mock_resp):
            result = client.search_series("gross domestic product")
            assert "seriess" in result

    def test_get_releases_with_mock(self):
        from shared.data.fred_v2 import FREDv2Client
        client = FREDv2Client(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"releases": []}
        mock_resp.raise_for_status = MagicMock()
        with patch("shared.data.fred_v2.httpx.get", return_value=mock_resp):
            result = client.get_releases()
            assert "releases" in result


# ── IMF ──────────────────────────────────────────────────────────────────


class TestIMFClient:
    def test_module_imports(self):
        from shared.data.imf_client import IMFClient, IMF_DATASETS, WEO_INDICATORS

    def test_supported_datasets(self):
        from shared.data.imf_client import IMFClient
        datasets = IMFClient.supported_datasets()
        assert "WEO" in datasets
        assert "IFS" in datasets
        assert "BOP" in datasets

    def test_common_weo_indicators(self):
        from shared.data.imf_client import IMFClient
        indicators = IMFClient.common_weo_indicators()
        assert "NGDP_RPCH" in indicators
        assert "PCPIPCH" in indicators

    def test_get_indicator_with_mock(self):
        from shared.data.imf_client import IMFClient
        client = IMFClient()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"values": {"US": {"2024": 2.8}}}
        mock_resp.raise_for_status = MagicMock()
        with patch("shared.data.imf_client.httpx.get", return_value=mock_resp):
            result = client.get_indicator("NGDP_RPCH", countries=["US"])
            assert "values" in result

    def test_get_datasets_with_mock(self):
        from shared.data.imf_client import IMFClient
        client = IMFClient()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"datasets": []}
        mock_resp.raise_for_status = MagicMock()
        with patch("shared.data.imf_client.httpx.get", return_value=mock_resp):
            result = client.get_datasets()
            assert "datasets" in result


# ── Cross-client fallback ────────────────────────────────────────────────


class TestMCPFallbackBehavior:
    def test_alpha_vantage_mcp_fallback(self):
        """When MCP is unavailable, client falls back to REST."""
        from shared.data.alpha_vantage import AlphaVantageClient
        client = AlphaVantageClient(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": "from_rest"}
        mock_resp.raise_for_status = MagicMock()
        with patch("shared.data.alpha_vantage.httpx.get", return_value=mock_resp):
            result = client.get_macro_indicator("CPI")
            assert result.get("data") == "from_rest"

    def test_s2_mcp_fallback(self):
        """When S2 MCP is unavailable, client falls back to REST."""
        from shared.data.s2_client import SemanticScholarClient
        client = SemanticScholarClient()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": []}
        mock_resp.raise_for_status = MagicMock()
        with patch("shared.data.s2_client.httpx.get", return_value=mock_resp):
            result = client.search("test query")
            assert "data" in result

    def test_registry_status_all_disabled(self):
        """Default status shows all sources disabled."""
        from shared.data.mcp_data_sources import MCPDataSourceRegistry
        reg = MCPDataSourceRegistry()
        status = reg.get_status()
        for name, info in status.items():
            assert info["enabled"] is False
