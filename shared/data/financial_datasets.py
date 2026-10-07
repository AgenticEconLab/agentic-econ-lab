# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Financial Datasets Client — MCP + REST for corporate financial data (V0.6 Phase 2).

Provides access to income statements, balance sheets, cash flow statements,
stock prices, and market news via Financial Datasets MCP or REST API.

Usage:
    from shared.data.financial_datasets import FinancialDatasetsClient

    client = FinancialDatasetsClient()
    data = client.get_financials("AAPL", statement="income")
"""

import os
from typing import Any, Dict, List, Optional

import httpx

from shared.data.mcp_data_sources import MCPDataSourceRegistry


STATEMENT_TYPES = ["income", "balance", "cashflow"]
DATA_TYPES = ["financials", "prices", "news", "insider_trades"]


class FinancialDatasetsClient:
    """Financial Datasets data client with MCP-first, REST-fallback strategy.

    Args:
        api_key: API key. Defaults to FINANCIAL_DATASETS_API_KEY env var.
        registry: Optional MCPDataSourceRegistry.
    """

    BASE_URL = "https://api.financialdatasets.ai"

    def __init__(
        self,
        api_key: Optional[str] = None,
        registry: Optional[MCPDataSourceRegistry] = None,
    ):
        self.api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY", "")
        self._registry = registry or MCPDataSourceRegistry()

    def get_financials(
        self,
        ticker: str,
        statement: str = "income",
        period: str = "annual",
        limit: int = 5,
    ) -> Dict[str, Any]:
        """Fetch financial statements for a ticker.

        Args:
            ticker: Company ticker symbol.
            statement: Statement type ("income", "balance", "cashflow").
            period: "annual" or "quarterly".
            limit: Number of periods to return.

        Returns:
            Dict with financial data or error.
        """
        mcp_client = self._registry.get_client("financial_datasets")
        if mcp_client:
            try:
                return mcp_client.call_tool(
                    "fd_get_financials",
                    {"ticker": ticker, "statement": statement, "period": period},
                )
            except Exception:
                pass

        endpoint_map = {
            "income": "income-statements",
            "balance": "balance-sheets",
            "cashflow": "cash-flow-statements",
        }
        ep = endpoint_map.get(statement, "income-statements")
        return self._rest_request(
            f"/financials/{ep}",
            params={"ticker": ticker, "period": period, "limit": str(limit)},
        )

    def get_prices(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch stock price history.

        Args:
            ticker: Ticker symbol.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).

        Returns:
            Dict with price data or error.
        """
        params: Dict[str, str] = {"ticker": ticker}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._rest_request("/prices", params=params)

    def _rest_request(
        self, path: str, params: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Make REST API request.

        Args:
            path: API endpoint path.
            params: Query parameters.

        Returns:
            Response JSON or error dict.
        """
        if not self.api_key:
            return {"error": "FINANCIAL_DATASETS_API_KEY not set"}

        headers = {"X-API-KEY": self.api_key}
        try:
            resp = httpx.get(
                f"{self.BASE_URL}{path}",
                params=params or {},
                headers=headers,
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def supported_statements() -> List[str]:
        """List supported financial statement types."""
        return list(STATEMENT_TYPES)

    @staticmethod
    def supported_data_types() -> List[str]:
        """List supported data types."""
        return list(DATA_TYPES)
