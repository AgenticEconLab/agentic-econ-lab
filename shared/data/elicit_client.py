# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Elicit API Client — Programmatic access to 138M+ papers (V0.6 Phase 2).

Provides paper search, structured extraction, and research report generation
via Elicit's REST API for LiteratureTeam integration.

Usage:
    from shared.data.elicit_client import ElicitClient

    client = ElicitClient()
    results = client.search_papers("AI impact on monetary policy", max_papers=20)
"""

import os
from typing import Any, Dict, List, Optional

import httpx


class ElicitClient:
    """Elicit API client for academic paper search and extraction.

    Args:
        api_key: Elicit API key. Defaults to ELICIT_API_KEY env var.
        base_url: API base URL.
    """

    DEFAULT_BASE_URL = "https://elicit.com/api/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("ELICIT_API_KEY", "")
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")

    def search_papers(
        self,
        query: str,
        max_papers: int = 20,
        fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Search for papers matching a query.

        Args:
            query: Natural language research query.
            max_papers: Maximum papers to return.
            fields: Optional specific fields to extract.

        Returns:
            Dict with papers list or error.
        """
        body: Dict[str, Any] = {
            "query": query,
            "max_papers": max_papers,
        }
        if fields:
            body["fields"] = fields

        return self._post("/search", body)

    def extract(
        self,
        paper_ids: List[str],
        columns: List[str],
    ) -> Dict[str, Any]:
        """Extract structured data from papers.

        Args:
            paper_ids: List of Elicit paper IDs or DOIs.
            columns: Column names to extract (e.g., "sample_size", "methodology").

        Returns:
            Dict with extracted data table or error.
        """
        return self._post("/extract", {
            "paper_ids": paper_ids,
            "columns": columns,
        })

    def get_report(
        self,
        query: str,
        max_papers: int = 40,
    ) -> Dict[str, Any]:
        """Generate a structured research report.

        Args:
            query: Research question.
            max_papers: Papers to include.

        Returns:
            Dict with report content or error.
        """
        return self._post("/report", {
            "query": query,
            "max_papers": max_papers,
        })

    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Make POST request to Elicit API.

        Args:
            path: API endpoint path.
            body: Request body.

        Returns:
            Response JSON or error dict.
        """
        if not self.api_key:
            return {"error": "ELICIT_API_KEY not set"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = httpx.post(
                f"{self.base_url}{path}",
                json=body,
                headers=headers,
                timeout=60.0,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def supported_capabilities() -> List[str]:
        """List supported Elicit API capabilities."""
        return ["search", "extract", "report"]
