# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Crawl4AI provider — OSS (MIT), fully local web crawl. Two backends:

* **library mode (default)** — when ``CRAWL4AI_BASE_URL`` is NOT set, runs the
  ``crawl4ai`` library in-process (Playwright/Chromium). No server/Docker needed;
  this is the open, zero-quota primary used on the HPC.
* **HTTP mode** — when ``CRAWL4AI_BASE_URL`` (e.g. ``http://localhost:11235``) is
  set, POSTs to a self-hosted Crawl4AI server.

Capabilities: scrape, crawl, extract. ``map`` raises ``UnsupportedCapability``.
Chromium location is taken from ``PLAYWRIGHT_BROWSERS_PATH`` (set it to scratch on HPC).
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, FrozenSet, List, Optional

from shared.webcrawl.base import UnsupportedCapability, WebCrawlResult


def _lib_importable() -> bool:
    try:
        import crawl4ai  # noqa: F401
        return True
    except Exception:
        return False


class Crawl4AIProvider:
    name = "crawl4ai"
    capabilities: FrozenSet[str] = frozenset({"scrape", "crawl", "extract"})

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.base_url = (
            base_url or os.environ.get("CRAWL4AI_BASE_URL", "")
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("CRAWL4AI_API_KEY", "")

    def is_available(self) -> bool:
        # HTTP server configured, OR the library is importable (library mode).
        return bool(self.base_url) or _lib_importable()

    # -- HTTP backend (self-hosted server) ----------------------------------- #
    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, path: str, body: Dict[str, Any], collector: Any, agent: str):
        from shared.observability import tracked_post
        return tracked_post(
            url=f"{self.base_url}{path}",
            collector=collector, agent=agent,
            headers=self._headers(), json=body,
        )

    # -- library backend (in-process, no server) ----------------------------- #
    @staticmethod
    def _lib_crawl(url: str):
        """Run Crawl4AI in-process. Returns (markdown:str, links:List[str], raw:dict)."""
        import asyncio
        from crawl4ai import AsyncWebCrawler

        async def _run():
            async with AsyncWebCrawler(verbose=False) as c:
                r = await c.arun(url=url)
                md = getattr(r, "markdown", "") or ""
                if not isinstance(md, str):
                    md = getattr(md, "raw_markdown", "") or str(md)
                links: List[str] = []
                lk = getattr(r, "links", None)
                if isinstance(lk, dict):
                    for group in ("internal", "external"):
                        for item in (lk.get(group) or []):
                            href = item.get("href") if isinstance(item, dict) else item
                            if href:
                                links.append(href)
                return md, links, {"success": bool(getattr(r, "success", True))}

        return asyncio.run(_run())

    # -- capabilities -------------------------------------------------------- #
    def scrape(self, url: str, *, formats: Optional[List[str]] = None,
               collector: Any = None, agent: str = "") -> WebCrawlResult:
        if not self.is_available():
            return WebCrawlResult(success=False, provider=self.name,
                                  error="crawl4ai unavailable (no CRAWL4AI_BASE_URL and library not importable)")
        start = time.time()
        try:
            if self.base_url:
                resp = self._post("/crawl", {"urls": [url], "output_format": "markdown"}, collector, agent)
                payload = resp.json()
                first = (payload.get("results") or [{}])[0]
                markdown = first.get("markdown") or first.get("content")
                return WebCrawlResult(success=True, data=payload, markdown=markdown,
                                      provider=self.name, latency_ms=round((time.time()-start)*1000, 2))
            md, _links, raw = self._lib_crawl(url)
            return WebCrawlResult(success=bool(md), data=raw, markdown=md, provider=self.name,
                                  error=None if md else "empty markdown",
                                  latency_ms=round((time.time()-start)*1000, 2))
        except Exception as e:
            return WebCrawlResult(success=False, error=f"{type(e).__name__}: {e}",
                                  provider=self.name, latency_ms=round((time.time()-start)*1000, 2))

    def crawl(self, url: str, *, limit: int = 10,
              collector: Any = None, agent: str = "") -> WebCrawlResult:
        if not self.is_available():
            return WebCrawlResult(success=False, provider=self.name,
                                  error="crawl4ai unavailable (no CRAWL4AI_BASE_URL and library not importable)")
        start = time.time()
        try:
            if self.base_url:
                resp = self._post("/crawl", {"urls": [url], "max_pages": limit, "deep_crawl": True}, collector, agent)
                payload = resp.json()
                links = [r.get("url") for r in (payload.get("results") or []) if r.get("url")]
                return WebCrawlResult(success=True, data=payload, links=links,
                                      provider=self.name, latency_ms=round((time.time()-start)*1000, 2))
            md, links, raw = self._lib_crawl(url)   # library: single-page link discovery
            return WebCrawlResult(success=True, data=raw, links=links[:limit], markdown=md,
                                  provider=self.name, latency_ms=round((time.time()-start)*1000, 2))
        except Exception as e:
            return WebCrawlResult(success=False, error=f"{type(e).__name__}: {e}",
                                  provider=self.name, latency_ms=round((time.time()-start)*1000, 2))

    def extract(self, urls: List[str], *, schema: Optional[Dict[str, Any]] = None,
                collector: Any = None, agent: str = "") -> WebCrawlResult:
        if not self.is_available():
            return WebCrawlResult(success=False, provider=self.name,
                                  error="crawl4ai unavailable (no CRAWL4AI_BASE_URL and library not importable)")
        start = time.time()
        try:
            if self.base_url:
                body: Dict[str, Any] = {"urls": list(urls), "output_format": "json"}
                if schema:
                    body["extraction_schema"] = schema
                resp = self._post("/crawl", body, collector, agent)
                return WebCrawlResult(success=True, data=resp.json(), provider=self.name,
                                      latency_ms=round((time.time()-start)*1000, 2))
            docs = {u: self._lib_crawl(u)[0] for u in urls}     # library: markdown per url
            return WebCrawlResult(success=True, data={"documents": docs},
                                  markdown="\n\n".join(docs.values()), provider=self.name,
                                  latency_ms=round((time.time()-start)*1000, 2))
        except Exception as e:
            return WebCrawlResult(success=False, error=f"{type(e).__name__}: {e}",
                                  provider=self.name, latency_ms=round((time.time()-start)*1000, 2))

    def map(self, url: str, *, collector: Any = None, agent: str = "") -> WebCrawlResult:
        raise UnsupportedCapability(self.name, "map")
