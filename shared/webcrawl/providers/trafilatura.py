# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Trafilatura provider — OSS, pure-Python, no browser.

Fast, research-grade main-content extraction to Markdown via the ``trafilatura``
library (HTTP + lxml; no JavaScript rendering). Ideal *default* for static
academic / gov pages — ~20x faster than a browser crawler and cleaner output
(strips nav/boilerplate). JS-rendered pages return little, so ``crawl`` (deep /
dynamic) raises ``UnsupportedCapability`` and the fallback chain defers to a
browser provider (crawl4ai), then commercial.

Capabilities: scrape, extract.
"""

from __future__ import annotations

import time
from typing import Any, Dict, FrozenSet, List, Optional

from shared.webcrawl.base import UnsupportedCapability, WebCrawlResult


def _lib_importable() -> bool:
    try:
        import trafilatura  # noqa: F401
        return True
    except Exception:
        return False


class TrafilaturaProvider:
    name = "trafilatura"
    capabilities: FrozenSet[str] = frozenset({"scrape", "extract"})

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        # No configuration needed (pure library). Args accepted for a uniform interface.
        pass

    def is_available(self) -> bool:
        return _lib_importable()

    @staticmethod
    def _extract(url: str) -> str:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return ""
        return trafilatura.extract(
            downloaded, output_format="markdown",
            include_links=True, include_tables=True,
        ) or ""

    def scrape(self, url: str, *, formats: Optional[List[str]] = None,
               collector: Any = None, agent: str = "") -> WebCrawlResult:
        start = time.time()
        try:
            md = self._extract(url)
            return WebCrawlResult(
                success=bool(md), markdown=md, provider=self.name,
                error=None if md else "no content extracted (empty or JS-only page)",
                latency_ms=round((time.time() - start) * 1000, 2),
            )
        except Exception as e:
            return WebCrawlResult(success=False, error=f"{type(e).__name__}: {e}",
                                  provider=self.name, latency_ms=round((time.time() - start) * 1000, 2))

    def extract(self, urls: List[str], *, schema: Optional[Dict[str, Any]] = None,
                collector: Any = None, agent: str = "") -> WebCrawlResult:
        start = time.time()
        try:
            docs = {u: self._extract(u) for u in urls}
            md = "\n\n".join(v for v in docs.values() if v)
            return WebCrawlResult(
                success=bool(md), data={"documents": docs}, markdown=md, provider=self.name,
                error=None if md else "no content extracted",
                latency_ms=round((time.time() - start) * 1000, 2),
            )
        except Exception as e:
            return WebCrawlResult(success=False, error=f"{type(e).__name__}: {e}",
                                  provider=self.name, latency_ms=round((time.time() - start) * 1000, 2))

    def crawl(self, url: str, *, limit: int = 10,
              collector: Any = None, agent: str = "") -> WebCrawlResult:
        # No deep-crawl / JS rendering -> defer to a browser provider in the chain.
        raise UnsupportedCapability(self.name, "crawl")

    def map(self, url: str, *, collector: Any = None, agent: str = "") -> WebCrawlResult:
        raise UnsupportedCapability(self.name, "map")
