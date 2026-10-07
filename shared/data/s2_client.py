# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Semantic Scholar Enhanced Client — MCP + REST for paper discovery (V0.6 Phase 2).

Extends existing S2 integration with Recommendations API and SPECTER2 embeddings
for LiteratureTeam GapDetectionStage.

Usage:
    from shared.data.s2_client import SemanticScholarClient

    client = SemanticScholarClient()
    papers = client.search("AI monetary policy", limit=20)
    recs = client.recommend("649def34f8be52c8b66281af98ae884c09aef38b")
"""

import os
from typing import Any, Dict, List, Optional

import httpx

from shared.data.mcp_data_sources import MCPDataSourceRegistry

# Module-level throttle state shared by every client instance (the unauthenticated
# Semantic Scholar tier allows ~1 request/second).
_S2_MIN_INTERVAL = float(os.environ.get("S2_MIN_INTERVAL", "1.1"))
_S2_LAST_CALL = [0.0]


DEFAULT_FIELDS = "paperId,title,abstract,year,citationCount,influentialCitationCount,authors"


class SemanticScholarClient:
    """Enhanced Semantic Scholar client with MCP-first, REST-fallback strategy.

    Args:
        api_key: S2 API key. Defaults to S2_API_KEY env var.
        registry: Optional MCPDataSourceRegistry.
    """

    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    RECOMMENDATIONS_URL = "https://api.semanticscholar.org/recommendations/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        registry: Optional[MCPDataSourceRegistry] = None,
    ):
        self.api_key = api_key or os.environ.get("S2_API_KEY", "")
        self._registry = registry or MCPDataSourceRegistry()

    def search(
        self,
        query: str,
        limit: int = 10,
        fields: str = DEFAULT_FIELDS,
        year: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search for papers.

        Args:
            query: Search query string.
            limit: Maximum papers to return.
            fields: Comma-separated list of fields.
            year: Year filter (e.g., "2020-2026", "2023-").

        Returns:
            Dict with papers or error.
        """
        mcp_client = self._registry.get_client("semantic_scholar")
        if mcp_client:
            try:
                return mcp_client.call_tool(
                    "s2_search_papers",
                    {"query": query, "limit": limit, "fields": fields},
                )
            except Exception:
                pass

        params: Dict[str, Any] = {
            "query": query,
            "limit": limit,
            "fields": fields,
        }
        if year:
            params["year"] = year
        return self._get("/paper/search", params)

    def recommend(
        self,
        paper_id: str,
        limit: int = 10,
        fields: str = DEFAULT_FIELDS,
    ) -> Dict[str, Any]:
        """Get paper recommendations based on a seed paper.

        Args:
            paper_id: Semantic Scholar paper ID or DOI.
            limit: Number of recommendations.
            fields: Fields to include.

        Returns:
            Dict with recommended papers or error.
        """
        mcp_client = self._registry.get_client("semantic_scholar")
        if mcp_client:
            try:
                return mcp_client.call_tool(
                    "s2_recommend",
                    {"paper_id": paper_id, "limit": limit},
                )
            except Exception:
                pass

        return self._get(
            f"/paper/{paper_id}/recommendations",
            {"limit": limit, "fields": fields},
            base_url=self.RECOMMENDATIONS_URL,
        )

    def get_paper(
        self,
        paper_id: str,
        fields: str = DEFAULT_FIELDS,
    ) -> Dict[str, Any]:
        """Get details for a specific paper.

        Args:
            paper_id: Paper ID, DOI, or ArXiv ID.
            fields: Fields to include.

        Returns:
            Dict with paper data or error.
        """
        return self._get(f"/paper/{paper_id}", {"fields": fields})

    def get_citations(
        self,
        paper_id: str,
        limit: int = 100,
        fields: str = "paperId,title,year,citationCount",
    ) -> Dict[str, Any]:
        """Get papers that cite this paper.

        Args:
            paper_id: Paper ID.
            limit: Max citations to return.
            fields: Fields per citation.

        Returns:
            Dict with citation data or error.
        """
        return self._get(
            f"/paper/{paper_id}/citations",
            {"limit": limit, "fields": f"citingPaper.{fields}"},
        )

    def _get(
        self,
        path: str,
        params: Dict[str, Any],
        base_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Make GET request to Semantic Scholar API.

        Args:
            path: API endpoint path.
            params: Query parameters.
            base_url: Override base URL.

        Returns:
            Response JSON or error dict.
        """
        url = (base_url or self.BASE_URL) + path
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        # The unauthenticated S2 tier is ~1 req/s — unthrottled burst calls hit 429
        # several times per sourcing pass and fell to Crossref (losing S2's better abstracts).
        # Throttle to the documented rate and retry a 429 once, honoring Retry-After.
        import time as _time
        for attempt in (1, 2):
            wait = _S2_MIN_INTERVAL - (_time.monotonic() - _S2_LAST_CALL[0])
            if wait > 0:
                _time.sleep(wait)
            _S2_LAST_CALL[0] = _time.monotonic()
            try:
                resp = httpx.get(url, params=params, headers=headers, timeout=30.0)
                if resp.status_code == 429 and attempt == 1:
                    try:
                        retry_after = float(resp.headers.get("retry-after") or 2.0)
                    except (TypeError, ValueError):
                        retry_after = 2.0
                    _time.sleep(min(max(retry_after, 1.0), 10.0))
                    continue
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                if attempt == 1 and "429" in str(e):
                    _time.sleep(2.0)
                    continue
                return {"error": str(e)}
        return {"error": "unreachable"}  # pragma: no cover

    @staticmethod
    def supported_capabilities() -> List[str]:
        """List supported S2 capabilities."""
        return ["search", "recommend", "get_paper", "citations"]
