# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Jina AI Reader provider — markdown-rendering scrape and structured extract.

Capabilities: scrape, extract. Crawl and map raise ``UnsupportedCapability``
so the fallback chain advances past them.

Jina's Reader API renders any URL as clean markdown by prefixing the URL
with ``https://r.jina.ai/``. JINA_API_KEY is optional — unauthenticated
requests are rate-limited rather than rejected.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, FrozenSet, List, Optional

from shared.webcrawl.base import UnsupportedCapability, WebCrawlResult


JINA_READER_PREFIX = "https://r.jina.ai/"


class JinaProvider:
    name = "jina"
    capabilities: FrozenSet[str] = frozenset({"scrape", "extract"})

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("JINA_API_KEY", "")

    def is_available(self) -> bool:
        # Jina works without a key (rate-limited); treat as always available.
        return True

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "text/markdown"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def scrape(
        self,
        url: str,
        *,
        formats: Optional[List[str]] = None,
        collector: Any = None,
        agent: str = "",
    ) -> WebCrawlResult:
        from shared.observability import tracked_get

        start = time.time()
        try:
            resp = tracked_get(
                url=JINA_READER_PREFIX + url,
                collector=collector,
                agent=agent,
                headers=self._headers(),
            )
            markdown = resp.text
            return WebCrawlResult(
                success=True,
                data={"text": markdown},
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

    def extract(
        self,
        urls: List[str],
        *,
        schema: Optional[Dict[str, Any]] = None,
        collector: Any = None,
        agent: str = "",
    ) -> WebCrawlResult:
        # Jina has no batch extract endpoint — iterate scrape() and collect markdown.
        start = time.time()
        results: List[Dict[str, Any]] = []
        errors: List[str] = []
        for u in urls:
            r = self.scrape(u, collector=collector, agent=agent)
            if r.success:
                results.append({"url": u, "markdown": r.markdown})
            else:
                errors.append(f"{u}: {r.error}")
        success = bool(results) and not errors
        return WebCrawlResult(
            success=success,
            data={"results": results, "errors": errors},
            provider=self.name,
            latency_ms=round((time.time() - start) * 1000, 2),
            error="; ".join(errors) if errors else None,
        )

    def crawl(
        self,
        url: str,
        *,
        limit: int = 10,
        collector: Any = None,
        agent: str = "",
    ) -> WebCrawlResult:
        raise UnsupportedCapability(self.name, "crawl")

    def map(
        self,
        url: str,
        *,
        collector: Any = None,
        agent: str = "",
    ) -> WebCrawlResult:
        raise UnsupportedCapability(self.name, "map")
