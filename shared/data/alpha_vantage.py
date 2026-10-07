# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Alpha Vantage Client — MCP + direct API fallback for macro indicators (V0.6 Phase 2).

Provides access to GDP, CPI, Treasury yields, unemployment, non-farm payrolls,
and 50+ technical indicators via Alpha Vantage MCP server or direct REST API.

Usage:
    from shared.data.alpha_vantage import AlphaVantageClient

    client = AlphaVantageClient()
    data = client.get_macro_indicator("REAL_GDP")
    tech = client.get_technical("SMA", "AAPL", interval="daily")
"""

import os
from typing import Any, Dict, List, Optional

import httpx

from shared.data.mcp_data_sources import MCPDataSourceRegistry


# Supported macro indicator functions
MACRO_INDICATORS = [
    "REAL_GDP", "REAL_GDP_PER_CAPITA", "TREASURY_YIELD", "FEDERAL_FUNDS_RATE",
    "CPI", "INFLATION", "RETAIL_SALES", "DURABLES", "UNEMPLOYMENT", "NONFARM_PAYROLL",
]

# Common technical indicator functions
TECHNICAL_INDICATORS = [
    "SMA", "EMA", "WMA", "DEMA", "TEMA", "TRIMA", "KAMA", "MAMA", "VWAP",
    "RSI", "STOCH", "STOCHF", "WILLR", "ADX", "CCI", "AROON", "BBANDS",
    "AD", "OBV", "HT_TRENDLINE", "HT_SINE", "HT_TRENDMODE",
    "MACD", "MACDEXT", "APO", "PPO", "MOM", "ROC", "ROCR",
]


class AlphaVantageClient:
    """Alpha Vantage data client with MCP-first, REST-fallback strategy.

    Args:
        api_key: Alpha Vantage API key. Defaults to ALPHA_VANTAGE_API_KEY env var.
        registry: Optional MCPDataSourceRegistry.
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(
        self,
        api_key: Optional[str] = None,
        registry: Optional[MCPDataSourceRegistry] = None,
    ):
        self.api_key = api_key or os.environ.get("ALPHA_VANTAGE_API_KEY", "")
        self._registry = registry or MCPDataSourceRegistry()

    def get_macro_indicator(
        self,
        function: str,
        interval: str = "annual",
    ) -> Dict[str, Any]:
        """Fetch a macro economic indicator.

        Args:
            function: Indicator name (e.g., "REAL_GDP", "CPI", "UNEMPLOYMENT").
            interval: "annual", "quarterly", or "monthly".

        Returns:
            Dict with indicator data or error.
        """
        # Try MCP first
        mcp_client = self._registry.get_client("alpha_vantage")
        if mcp_client:
            try:
                return mcp_client.call_tool(
                    "av_get_macro_indicator",
                    {"function": function, "interval": interval},
                )
            except Exception:
                pass  # Fall through to REST

        # REST fallback
        return self._rest_request({
            "function": function,
            "interval": interval,
        })

    def get_technical(
        self,
        function: str,
        symbol: str,
        interval: str = "daily",
        time_period: int = 20,
    ) -> Dict[str, Any]:
        """Fetch a technical indicator.

        Args:
            function: Indicator (e.g., "SMA", "RSI", "BBANDS").
            symbol: Ticker symbol.
            interval: Time interval.
            time_period: Number of data points for calculation.

        Returns:
            Dict with technical indicator data or error.
        """
        mcp_client = self._registry.get_client("alpha_vantage")
        if mcp_client:
            try:
                return mcp_client.call_tool(
                    "av_get_technical",
                    {"function": function, "symbol": symbol, "interval": interval},
                )
            except Exception:
                pass

        return self._rest_request({
            "function": function,
            "symbol": symbol,
            "interval": interval,
            "time_period": str(time_period),
        })

    def search_symbols(self, keywords: str) -> Dict[str, Any]:
        """Search for symbols by keywords.

        Args:
            keywords: Search keywords.

        Returns:
            Dict with matching symbols.
        """
        return self._rest_request({
            "function": "SYMBOL_SEARCH",
            "keywords": keywords,
        })

    def _rest_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make a direct REST API request to Alpha Vantage.

        Args:
            params: Query parameters.

        Returns:
            Response JSON or error dict.
        """
        if not self.api_key:
            return {"error": "ALPHA_VANTAGE_API_KEY not set"}

        params["apikey"] = self.api_key
        try:
            resp = httpx.get(self.BASE_URL, params=params, timeout=30.0)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def supported_macro_indicators() -> List[str]:
        """Return list of supported macro indicator functions."""
        return list(MACRO_INDICATORS)

    @staticmethod
    def supported_technical_indicators() -> List[str]:
        """Return list of supported technical indicator functions."""
        return list(TECHNICAL_INDICATORS)
