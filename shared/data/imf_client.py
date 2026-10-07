# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
IMF Data Portal Client — Updated API for international economic data (V0.6 Phase 2).

Provides access to IMF datasets (WEO, IFS, BOP, DOTS) via the new data.imf.org API.

Usage:
    from shared.data.imf_client import IMFClient

    client = IMFClient()
    data = client.get_indicator("NGDP_RPCH", countries=["US", "GB", "JP"])
"""

from typing import Any, Dict, List, Optional

import httpx


# Common IMF datasets
IMF_DATASETS = {
    "WEO": "World Economic Outlook",
    "IFS": "International Financial Statistics",
    "BOP": "Balance of Payments",
    "DOTS": "Direction of Trade Statistics",
    "GFS": "Government Finance Statistics",
    "PCPS": "Primary Commodity Price System",
}

# Common WEO indicator codes
WEO_INDICATORS = {
    "NGDP_RPCH": "Real GDP growth (%)",
    "PCPIPCH": "Inflation (CPI, %)",
    "LUR": "Unemployment rate (%)",
    "GGXWDG_NGDP": "Government gross debt (% GDP)",
    "BCA_NGDPD": "Current account balance (% GDP)",
    "NGDPDPC": "GDP per capita (USD)",
}


class IMFClient:
    """IMF Data Portal client (new API at data.imf.org).

    Args:
        base_url: API base URL.
    """

    DEFAULT_BASE_URL = "https://www.imf.org/external/datamapper/api/v1"

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")

    def get_indicator(
        self,
        indicator: str,
        countries: Optional[List[str]] = None,
        dataset: str = "WEO",
    ) -> Dict[str, Any]:
        """Fetch an IMF indicator for specified countries.

        Args:
            indicator: Indicator code (e.g., "NGDP_RPCH").
            countries: ISO country codes (e.g., ["US", "GB"]). None for all.
            dataset: Dataset name (e.g., "WEO", "IFS").

        Returns:
            Dict with indicator data or error.
        """
        path = f"/{dataset}/{indicator}"
        if countries:
            path += "/" + "+".join(countries)
        return self._get(path)

    def get_datasets(self) -> Dict[str, Any]:
        """List available IMF datasets.

        Returns:
            Dict with dataset metadata.
        """
        return self._get("/datasets")

    def get_indicators(self, dataset: str = "WEO") -> Dict[str, Any]:
        """List indicators available in a dataset.

        Args:
            dataset: Dataset name.

        Returns:
            Dict with indicator metadata.
        """
        return self._get(f"/{dataset}/indicators")

    def get_countries(self) -> Dict[str, Any]:
        """List available countries/regions.

        Returns:
            Dict with country metadata.
        """
        return self._get("/countries")

    def _get(self, path: str) -> Dict[str, Any]:
        """Make GET request to IMF API.

        Args:
            path: API endpoint path.

        Returns:
            Response JSON or error dict.
        """
        try:
            resp = httpx.get(
                f"{self.base_url}{path}",
                timeout=30.0,
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def supported_datasets() -> Dict[str, str]:
        """List supported IMF datasets."""
        return dict(IMF_DATASETS)

    @staticmethod
    def common_weo_indicators() -> Dict[str, str]:
        """Common WEO indicator codes and descriptions."""
        return dict(WEO_INDICATORS)
