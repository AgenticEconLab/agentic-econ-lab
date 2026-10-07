# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Firecrawl provider — thin adapter around the official ``firecrawl`` SDK.

Full capability matrix: scrape, crawl, extract, map.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, FrozenSet, List, Optional

from shared.webcrawl.base import UnsupportedCapability, WebCrawlResult


class FirecrawlProvider:
    name = "firecrawl"
    capabilities: FrozenSet[str] = frozenset({"scrape", "crawl", "extract", "map"})

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("FIRECRAWL_API_KEY", "")
        self._app = None

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _client(self):
        if self._app is not None:
            return self._app
        if not self.api_key:
            raise RuntimeError("FIRECRAWL_API_KEY not configured")
        from firecrawl import FirecrawlApp  # deferred import — SDK optional at install time

        self._app = FirecrawlApp(api_key=self.api_key)
        return self._app

    def scrape(
        self,
        url: str,
        *,
        formats: Optional[List[str]] = None,
        collector: Any = None,
        agent: str = "",
    ) -> WebCrawlResult:
        formats = formats or ["markdown"]
        start = time.time()
        try:
            resp = self._client().scrape_url(url, formats=formats)
            latency_ms = (time.time() - start) * 1000
            markdown = getattr(resp, "markdown", None)
            return WebCrawlResult(
                success=True,
                data=resp,
                markdown=markdown,
                provider=self.name,
                latency_ms=round(latency_ms, 2),
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
        start = time.time()
        try:
            resp = self._client().crawl_url(url, limit=limit)
            return WebCrawlResult(
                success=True,
                data=resp,
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
        start = time.time()
        try:
            client = self._client()
            resp = client.extract(urls, schema=schema) if schema else client.extract(urls)
            return WebCrawlResult(
                success=True,
                data=resp,
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
        start = time.time()
        try:
            resp = self._client().map_url(url)
            links = list(getattr(resp, "links", None) or [])
            return WebCrawlResult(
                success=True,
                data=resp,
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
