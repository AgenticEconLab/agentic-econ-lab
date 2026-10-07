# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Tavily provider — HTTP adapter for the Tavily Extract / Crawl APIs.

Full capability matrix: scrape (via Extract API), crawl (via Crawl API),
extract (batch Extract), map (URL discovery — returned by Crawl).

Does NOT support the ``map`` capability as a standalone call today; Tavily's
URL-discovery surface is bundled into the crawl endpoint. ``map`` raises
``UnsupportedCapability`` so the fallback chain can skip past it.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, FrozenSet, List, Optional

from shared.webcrawl.base import UnsupportedCapability, WebCrawlResult


TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"
TAVILY_CRAWL_URL = "https://api.tavily.com/crawl"


class TavilyProvider:
    name = "tavily"
    capabilities: FrozenSet[str] = frozenset({"scrape", "crawl", "extract"})

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY", "")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> Dict[str, str]:
        return {"Content-Type": "application/json"}

    def _post(self, url: str, body: Dict[str, Any], collector: Any, agent: str):
        from shared.observability import tracked_post

        return tracked_post(
            url=url,
            collector=collector,
            agent=agent,
            headers=self._headers(),
            json=body,
        )

    def scrape(
        self,
        url: str,
        *,
        formats: Optional[List[str]] = None,
        collector: Any = None,
        agent: str = "",
    ) -> WebCrawlResult:
        if not self.api_key:
            return WebCrawlResult(
                success=False,
                error="TAVILY_API_KEY not configured",
                provider=self.name,
            )
        want_markdown = not formats or "markdown" in formats
        start = time.time()
        try:
            resp = self._post(
                TAVILY_EXTRACT_URL,
                {
                    "api_key": self.api_key,
                    "urls": [url],
                    "format": "markdown" if want_markdown else "text",
                },
                collector,
                agent,
            )
            payload = resp.json()
            results = payload.get("results") or []
            first = results[0] if results else {}
            markdown = first.get("raw_content") if want_markdown else None
            return WebCrawlResult(
                success=True,
                data=payload,
                markdown=markdown,
                provider=self.name,
                latency_ms=round((time.time() - start) * 1000, 2),
            )
        except Exception as e:
            return WebCrawlResult(
                success=False,
                error=f"{type(e).__name__}: {e}",
                provider=self.name,
                latency_ms=round((time.time() - start) * 1000, 2),
            )

    def crawl(
        self,
        url: str,
        *,
        limit: int = 10,
        collector: Any = None,
        agent: str = "",
    ) -> WebCrawlResult:
        if not self.api_key:
            return WebCrawlResult(
                success=False,
                error="TAVILY_API_KEY not configured",
                provider=self.name,
            )
        start = time.time()
        try:
            resp = self._post(
                TAVILY_CRAWL_URL,
                {
                    "api_key": self.api_key,
                    "url": url,
                    "max_depth": 1,
                    "limit": limit,
                },
                collector,
                agent,
            )
            payload = resp.json()
            pages = payload.get("results") or []
            links = [p.get("url") for p in pages if p.get("url")]
            return WebCrawlResult(
                success=True,
                data=payload,
                links=links,
                provider=self.name,
                latency_ms=round((time.time() - start) * 1000, 2),
            )
        except Exception as e:
            return WebCrawlResult(
                success=False,
                error=f"{type(e).__name__}: {e}",
                provider=self.name,
                latency_ms=round((time.time() - start) * 1000, 2),
            )

    def extract(
        self,
        urls: List[str],
        *,
        schema: Optional[Dict[str, Any]] = None,
        collector: Any = None,
        agent: str = "",
    ) -> WebCrawlResult:
        if not self.api_key:
            return WebCrawlResult(
                success=False,
                error="TAVILY_API_KEY not configured",
                provider=self.name,
            )
        start = time.time()
        try:
            resp = self._post(
                TAVILY_EXTRACT_URL,
                {"api_key": self.api_key, "urls": list(urls)},
                collector,
                agent,
            )
            return WebCrawlResult(
                success=True,
                data=resp.json(),
                provider=self.name,
                latency_ms=round((time.time() - start) * 1000, 2),
            )
        except Exception as e:
            return WebCrawlResult(
                success=False,
                error=f"{type(e).__name__}: {e}",
                provider=self.name,
                latency_ms=round((time.time() - start) * 1000, 2),
            )

    def map(
        self,
        url: str,
        *,
        collector: Any = None,
        agent: str = "",
    ) -> WebCrawlResult:
        raise UnsupportedCapability(self.name, "map")
