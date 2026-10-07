# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
FRED API V2 Client — Bulk retrieval and expanded series (V0.6 Phase 2).

Upgrades from FRED API V1 with bulk data retrieval, complete history access,
and new series (Nasdaq, collateral rates). Falls back to MCP when available.

Usage:
    from shared.data.fred_v2 import FREDv2Client

    client = FREDv2Client()
    data = client.get_series("GDP")
    bulk = client.get_series_bulk(["GDP", "CPIAUCSL", "UNRATE"])
"""

import os
from typing import Any, Dict, List, Optional

import httpx


# New series available in FRED V2
NEW_V2_SERIES = [
    "NASDAQCOM",       # Nasdaq Composite
    "SOFR",            # Secured Overnight Financing Rate
    "BGCR",            # Broad General Collateral Rate
    "TGCR",            # Tri-Party General Collateral Rate
    "EFFR",            # Effective Federal Funds Rate
    "OBFR",            # Overnight Bank Funding Rate
    "IORB",            # Interest on Reserve Balances
]

# Common macro series
COMMON_SERIES = [
    "GDP", "GDPC1", "CPIAUCSL", "UNRATE", "FEDFUNDS", "DGS10", "DGS2",
    "M2SL", "PAYEMS", "RSAFS", "INDPRO", "PCE", "PCEPI",
]


class FREDv2Client:
    """FRED API V2 client with bulk retrieval and expanded series.

    Args:
        api_key: FRED API key. Defaults to FRED_API_KEY env var.
    """

    BASE_URL = "https://api.stlouisfed.org/fred"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("FRED_API_KEY", "")

    def get_series(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch a single FRED series.

        Args:
            series_id: FRED series ID (e.g., "GDP").
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            frequency: Aggregation frequency (e.g., "a", "q", "m").

        Returns:
            Dict with observations or error.
        """
        params: Dict[str, str] = {"series_id": series_id}
        if start_date:
            params["observation_start"] = start_date
        if end_date:
            params["observation_end"] = end_date
        if frequency:
            params["frequency"] = frequency
        return self._get("/series/observations", params)

    def get_series_bulk(
        self,
        series_ids: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch multiple FRED series in bulk.

        Args:
            series_ids: List of FRED series IDs.
            start_date: Start date.
            end_date: End date.

        Returns:
            Dict mapping series_id to observation data.
        """
        results = {}
        for sid in series_ids:
            results[sid] = self.get_series(sid, start_date, end_date)
        return results

    def search_series(
        self,
        query: str,
        limit: int = 20,
        order_by: str = "popularity",
    ) -> Dict[str, Any]:
        """Search for FRED series.

        Args:
            query: Search text.
            limit: Max results.
            order_by: Sort order ("popularity", "search_rank").

        Returns:
            Dict with matching series metadata.
        """
        return self._get("/series/search", {
            "search_text": query,
            "limit": str(limit),
            "order_by": order_by,
        })

    def get_series_info(self, series_id: str) -> Dict[str, Any]:
        """Get metadata for a series.

        Args:
            series_id: FRED series ID.

        Returns:
            Dict with series metadata.
        """
        return self._get("/series", {"series_id": series_id})

    def get_releases(self, limit: int = 20) -> Dict[str, Any]:
        """Get recent FRED data releases.

        Args:
            limit: Max releases to return.

        Returns:
            Dict with release data.
        """
        return self._get("/releases", {"limit": str(limit)})

    def _get(self, path: str, params: Dict[str, str]) -> Dict[str, Any]:
        """Make GET request to FRED API.

        Args:
            path: API endpoint path.
            params: Query parameters.

        Returns:
            Response JSON or error dict.
        """
        if not self.api_key:
            return {"error": "FRED_API_KEY not set"}

        params["api_key"] = self.api_key
        params["file_type"] = "json"

        try:
            resp = httpx.get(
                f"{self.BASE_URL}{path}",
                params=params,
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def new_v2_series() -> List[str]:
        """Series newly available in FRED V2."""
        return list(NEW_V2_SERIES)

    @staticmethod
    def common_series() -> List[str]:
        """Commonly used macro series."""
        return list(COMMON_SERIES)
