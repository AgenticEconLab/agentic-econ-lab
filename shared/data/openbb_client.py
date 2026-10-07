# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
OpenBB Client — MCP + SDK integration for 350+ financial datasets (V0.6 Phase 2).

Provides access to equity, economy, fixed income, crypto, and ETF data
via OpenBB Platform MCP server or direct SDK.

Usage:
    from shared.data.openbb_client import OpenBBClient

    client = OpenBBClient()
    data = client.get_economy_indicators("GDP", provider="fred")
"""

import os
from typing import Any, Dict, List, Optional

from shared.data.mcp_data_sources import MCPDataSourceRegistry


class OpenBBClient:
    """OpenBB data client with MCP-first strategy.

    Args:
        token: OpenBB PAT token. Defaults to OPENBB_TOKEN env var.
        registry: Optional MCPDataSourceRegistry.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        registry: Optional[MCPDataSourceRegistry] = None,
    ):
        self.token = token or os.environ.get("OPENBB_TOKEN", "")
        self._registry = registry or MCPDataSourceRegistry()

    def get_economy_indicators(
        self,
        symbol: str,
        provider: str = "fred",
    ) -> Dict[str, Any]:
        """Fetch economic indicators via OpenBB.

        Args:
            symbol: Indicator symbol (e.g., "GDP", "CPIAUCSL").
            provider: Data provider backend (e.g., "fred", "econdb").

        Returns:
            Dict with indicator data or error.
        """
        mcp_client = self._registry.get_client("openbb")
        if mcp_client:
            try:
                return mcp_client.call_tool(
                    "openbb_economy_indicators",
                    {"symbol": symbol, "provider": provider},
                )
            except Exception:
                pass

        return self._sdk_fallback("economy.indicators", symbol=symbol, provider=provider)

    def get_equity_price(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        provider: str = "yfinance",
    ) -> Dict[str, Any]:
        """Fetch equity price history.

        Args:
            symbol: Ticker symbol.
            start_date: Start date (YYYY-MM-DD).
            provider: Data provider.

        Returns:
            Dict with price data or error.
        """
        return self._sdk_fallback(
            "equity.price.historical",
            symbol=symbol,
            start_date=start_date,
            provider=provider,
        )

    def get_fixed_income(
        self,
        maturity: str = "10y",
        provider: str = "fred",
    ) -> Dict[str, Any]:
        """Fetch fixed income/Treasury yield data.

        Args:
            maturity: Bond maturity (e.g., "3m", "2y", "10y", "30y").
            provider: Data provider.

        Returns:
            Dict with yield data or error.
        """
        return self._sdk_fallback(
            "fixedincome.rate.treasury",
            maturity=maturity,
            provider=provider,
        )

    def _sdk_fallback(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Attempt OpenBB SDK call (if installed).

        Args:
            endpoint: OpenBB endpoint path (e.g., "economy.indicators").
            **kwargs: Endpoint parameters.

        Returns:
            Dict with data or error.
        """
        try:
            from openbb import obb
            parts = endpoint.split(".")
            obj = obb
            for part in parts:
                obj = getattr(obj, part)
            result = obj(**kwargs)
            if hasattr(result, "to_dict"):
                return result.to_dict()
            return {"data": str(result)}
        except ImportError:
            return {"error": "openbb SDK not installed and MCP unavailable"}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def supported_categories() -> List[str]:
        """List supported OpenBB data categories."""
        return ["equity", "economy", "fixedincome", "crypto", "etf", "forex"]
