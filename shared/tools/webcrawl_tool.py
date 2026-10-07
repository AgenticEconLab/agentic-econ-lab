# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
WebCrawl Tool — multi-provider web-crawl client + ToolRegistry registration.

Defines ``WebCrawlClient``, which presents a single interface across four
providers (Firecrawl / Tavily / Jina / Crawl4AI) with an ordered fallback
chain. Mirrors the Tavily→Brave→Serper fallback pattern already in
``shared.tools.web_search_tool``.

Selection:
  - primary: ``WEBCRAWL_PROVIDER`` env var, else ``"trafilatura"`` (``DEFAULT_PRIMARY``).
  - fallback chain: ``WEBCRAWL_FALLBACK_CHAIN`` env var (comma-separated),
    else empty.

Also registers four ``webcrawl_*`` tools (scrape / crawl / extract / map)
in ``ToolRegistry`` so any agent that discovers tools via the registry
picks them up automatically.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from shared.tools.tool_registry import ToolRegistry
from shared.webcrawl.base import UnsupportedCapability, WebCrawlResult
from shared.webcrawl.providers import get_provider


DEFAULT_PRIMARY = "trafilatura"  # open-first; was "firecrawl" (wrong default)

# Hard per-provider crawl timeout. A browser provider (crawl4ai/playwright) can hang
# indefinitely on a JS-heavy or navigating page with no internal timeout, stalling the
# whole pipeline (observed: a full-pipeline run hung ~3h on one DOI page). Cap each
# provider call so a hang is abandoned and the fallback chain continues. Override via
# WEBCRAWL_TIMEOUT (seconds).
_CRAWL_TIMEOUT = float(os.environ.get("WEBCRAWL_TIMEOUT", "60"))


# ---------------------------------------------------------------------------
# DOI resolution. doi.org links (e.g. NBER 10.3386/w####) can come back as a
# 174-byte interstitial the anti-bot classifier flagged, losing the paper entirely. Known
# publisher DOIs are rewritten to their canonical landing page; other doi.org links are
# resolved by following redirects once (cheap HEAD) before hitting the crawl chain.
# ---------------------------------------------------------------------------
import re as _re

_NBER_DOI_RE = _re.compile(r"doi\.org/10\.3386/(w\d+)", _re.IGNORECASE)


def _resolve_doi_url(url: str) -> str:
    """Rewrite/resolve doi.org URLs to their publisher landing page. Never raises."""
    if "doi.org/" not in (url or ""):
        return url
    m = _NBER_DOI_RE.search(url)
    if m:
        return f"https://www.nber.org/papers/{m.group(1)}"
    try:
        import requests
        resp = requests.head(url, allow_redirects=True, timeout=10)
        final = str(resp.url or "")
        if final and "doi.org/" not in final:
            return final
    except Exception:
        pass
    return url


# ---------------------------------------------------------------------------
# PDF extraction. A working paper (.pdf on an author site) would otherwise be
# misclassified as "anti-bot: minimal_text" because no provider in the chain reads PDF.
# Economics content is PDF-heavy (working papers), so .pdf URLs are served by a direct
# fetch + pypdf text extraction BEFORE the HTML chain.
# ---------------------------------------------------------------------------
_PDF_MAX_PAGES = 30


def _looks_like_pdf(url: str) -> bool:
    path = (url or "").split("?", 1)[0].split("#", 1)[0]
    return path.lower().endswith(".pdf")


def _scrape_pdf(url: str, collector=None, agent: str = "") -> Optional[WebCrawlResult]:
    """Fetch + extract a PDF's text. Returns None on any failure (chain then proceeds)."""
    start = time.time()
    try:
        from shared.observability import tracked_get
        resp = tracked_get(url, collector=collector, agent=agent or "webcrawl_pdf", retries=1)
        blob = resp.content or b""
        ctype = str(resp.headers.get("content-type", "")).lower()
        if not (blob[:5] == b"%PDF-" or "pdf" in ctype):
            return None
        import io
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(blob))
        pages = reader.pages[:_PDF_MAX_PAGES]
        text = "\n\n".join((p.extract_text() or "") for p in pages).strip()
        if len(text) < 200:                      # scanned/image PDF — nothing extractable
            return None
        note = (f"\n\n[pdf extraction: {min(len(reader.pages), _PDF_MAX_PAGES)}"
                f"/{len(reader.pages)} pages]")
        return WebCrawlResult(
            success=True, data={"text": text}, markdown=text + note,
            provider="pdf", latency_ms=round((time.time() - start) * 1000, 2),
        )
    except Exception:
        return None


def _call_with_timeout(fn, timeout: float, *args, **kwargs):
    """Run ``fn`` in a daemon thread and abandon it if it exceeds ``timeout``.

    A hung provider (e.g. playwright never returning) can't be force-killed, but the
    daemon thread is abandoned so the pipeline proceeds; it dies at process exit. Raises
    TimeoutError on expiry, else returns the result / re-raises the callee's exception.
    """
    box: Dict[str, Any] = {}

    def _run():
        try:
            box["result"] = fn(*args, **kwargs)
        except BaseException as e:  # capture to re-raise on the caller's thread
            box["error"] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"crawl provider exceeded {timeout}s")
    if "error" in box:
        raise box["error"]
    return box.get("result")


def _resolve_chain(
    primary: Optional[str], fallback_chain: Optional[Sequence[str]]
) -> List[str]:
    chosen_primary = (primary or os.environ.get("WEBCRAWL_PROVIDER") or DEFAULT_PRIMARY).lower()
    if fallback_chain is None:
        raw = os.environ.get("WEBCRAWL_FALLBACK_CHAIN", "").strip()
        fallback_chain = [p.strip().lower() for p in raw.split(",") if p.strip()]
    chain = [chosen_primary] + [p for p in fallback_chain if p != chosen_primary]
    return chain


class WebCrawlClient:
    """Facade over one or more ``WebCrawlProvider`` instances with fallback.

    Construction is cheap — providers defer their real init (e.g. SDK import,
    HTTP client build) until the first call. A stage that holds one
    ``WebCrawlClient`` can call ``scrape`` / ``crawl`` / ``extract`` / ``map``
    without knowing which backend serves it.
    """

    def __init__(
        self,
        primary: Optional[str] = None,
        fallback_chain: Optional[Sequence[str]] = None,
        collector: Any = None,
        agent: str = "TopicCrawler",
        provider_keys: Optional[Dict[str, str]] = None,
    ):
        self.chain = _resolve_chain(primary, fallback_chain)
        self.collector = collector
        self.agent = agent
        # Per-provider credential overrides — takes precedence over env vars
        # for that provider. Missing keys fall back to the env-read default.
        self._provider_keys: Dict[str, str] = {
            k.lower(): v for k, v in (provider_keys or {}).items() if v
        }
        self._instances: Dict[str, Any] = {}

    # ---- introspection --------------------------------------------------

    def is_available(self) -> bool:
        """True if at least one provider in the chain reports availability."""
        for name in self.chain:
            try:
                provider = self._resolve(name)
            except KeyError:
                continue
            if provider.is_available():
                return True
        return False

    def active_provider(self) -> Optional[str]:
        """First provider in the chain that reports as available."""
        for name in self.chain:
            try:
                provider = self._resolve(name)
            except KeyError:
                continue
            if provider.is_available():
                return name
        return None

    def _resolve(self, name: str):
        if name not in self._instances:
            cls = get_provider(name)
            kwargs: Dict[str, Any] = {}
            if name in self._provider_keys:
                kwargs["api_key"] = self._provider_keys[name]
            self._instances[name] = cls(**kwargs)
        return self._instances[name]

    # ---- core dispatch --------------------------------------------------

    def _dispatch(self, capability: str, *args, **kwargs) -> WebCrawlResult:
        attempts: List[Dict[str, Any]] = []
        last_result: Optional[WebCrawlResult] = None
        overall_start = time.time()

        for name in self.chain:
            try:
                provider = self._resolve(name)
            except KeyError:
                attempts.append(
                    {"provider": name, "success": False, "error": "unknown provider"}
                )
                continue

            if not provider.is_available():
                attempts.append(
                    {"provider": name, "success": False, "error": "not configured"}
                )
                continue

            method = getattr(provider, capability)
            try:
                result = _call_with_timeout(
                    method, _CRAWL_TIMEOUT,
                    *args, collector=self.collector, agent=self.agent, **kwargs
                )
            except UnsupportedCapability as e:
                attempts.append(
                    {"provider": name, "success": False, "error": str(e)}
                )
                continue
            except TimeoutError as e:
                attempts.append(
                    {"provider": name, "success": False, "error": str(e)}
                )
                continue
            except Exception as e:  # pragma: no cover — defensive
                attempts.append(
                    {
                        "provider": name,
                        "success": False,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
                continue

            attempts.append(
                {
                    "provider": name,
                    "success": result.success,
                    "error": result.error,
                    "latency_ms": result.latency_ms,
                }
            )
            if result.success:
                result.attempts = attempts
                return result
            last_result = result

        total_ms = round((time.time() - overall_start) * 1000, 2)
        if last_result is not None:
            last_result.attempts = attempts
            last_result.latency_ms = total_ms
            return last_result
        return WebCrawlResult(
            success=False,
            error="No configured web-crawl provider available",
            provider="",
            latency_ms=total_ms,
            attempts=attempts,
        )

    # ---- public API -----------------------------------------------------

    def scrape(
        self, url: str, *, formats: Optional[List[str]] = None
    ) -> WebCrawlResult:
        url = _resolve_doi_url(url)                       # land on the publisher page
        if _looks_like_pdf(url):
            pdf = _scrape_pdf(url, collector=self.collector, agent=self.agent)
            if pdf is not None:                           # PDFs never reach the HTML chain
                return pdf
        return self._dispatch("scrape", url, formats=formats)

    def crawl(self, url: str, *, limit: int = 10) -> WebCrawlResult:
        return self._dispatch("crawl", url, limit=limit)

    def extract(
        self, urls: List[str], *, schema: Optional[Dict[str, Any]] = None
    ) -> WebCrawlResult:
        return self._dispatch("extract", urls, schema=schema)

    def map(self, url: str) -> WebCrawlResult:
        return self._dispatch("map", url)

    # ---- back-compat shim for the old FirecrawlApp.scrape_url() shape ---

    def scrape_url(self, url: str, formats: Optional[List[str]] = None):
        """Drop-in replacement for ``FirecrawlApp.scrape_url``.

        Returns a lightweight object with a ``.markdown`` attribute so existing
        call sites like ``result.markdown`` keep working. Raises on failure
        to match the SDK semantics at stage boundaries.
        """
        result = self.scrape(url, formats=formats or ["markdown"])
        if not result.success:
            raise RuntimeError(result.error or "webcrawl scrape failed")

        class _ScrapeShim:
            def __init__(self, md: Optional[str], raw: Any):
                self.markdown = md
                self.data = raw

        return _ScrapeShim(result.markdown, result.data)


# ---------------------------------------------------------------------------
# Pydantic input schemas for the 4 registered tools
# ---------------------------------------------------------------------------


class WebCrawlScrapeInput(BaseModel):
    url: str = Field(description="URL to scrape")
    formats: List[str] = Field(default_factory=lambda: ["markdown"])


class WebCrawlCrawlInput(BaseModel):
    url: str = Field(description="Start URL for the crawl")
    limit: int = Field(default=10, description="Maximum pages to fetch")


class WebCrawlExtractInput(BaseModel):
    urls: List[str] = Field(description="URLs to extract structured data from")
    schema_: Optional[Dict[str, Any]] = Field(
        default=None,
        alias="schema",
        description="Optional JSON schema describing fields to extract",
    )


class WebCrawlMapInput(BaseModel):
    url: str = Field(description="URL whose site map to retrieve")


# ---------------------------------------------------------------------------
# Handler factories — one WebCrawlClient per handler invocation keeps
# state clean across invocations in the same process.
# ---------------------------------------------------------------------------


def _scrape_handler(url: str, formats: Optional[List[str]] = None, collector=None, agent: str = ""):
    client = WebCrawlClient(collector=collector, agent=agent)
    return client.scrape(url, formats=formats).model_dump()


def _crawl_handler(url: str, limit: int = 10, collector=None, agent: str = ""):
    client = WebCrawlClient(collector=collector, agent=agent)
    return client.crawl(url, limit=limit).model_dump()


def _extract_handler(
    urls: List[str],
    schema_: Optional[Dict[str, Any]] = None,
    collector=None,
    agent: str = "",
    **kwargs,
):
    # Pydantic alias-populated kwargs may arrive as ``schema`` too — accept both.
    if schema_ is None and "schema" in kwargs:
        schema_ = kwargs["schema"]
    client = WebCrawlClient(collector=collector, agent=agent)
    return client.extract(list(urls), schema=schema_).model_dump()


def _map_handler(url: str, collector=None, agent: str = ""):
    client = WebCrawlClient(collector=collector, agent=agent)
    return client.map(url).model_dump()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


ToolRegistry.register(
    name="webcrawl_scrape",
    description=(
        "Scrape a single URL and return its content. Multi-provider: Firecrawl / "
        "Tavily / Jina / Crawl4AI with WEBCRAWL_PROVIDER env var selection."
    ),
    input_schema=WebCrawlScrapeInput,
    handler=_scrape_handler,
    category="web",
)

ToolRegistry.register(
    name="webcrawl_crawl",
    description=(
        "Crawl a site starting at the given URL, returning up to ``limit`` pages. "
        "Multi-provider; Jina skips via UnsupportedCapability."
    ),
    input_schema=WebCrawlCrawlInput,
    handler=_crawl_handler,
    category="web",
)

ToolRegistry.register(
    name="webcrawl_extract",
    description=(
        "Batch-extract structured content from one or more URLs. Optional "
        "JSON schema describes the fields to extract."
    ),
    input_schema=WebCrawlExtractInput,
    handler=_extract_handler,
    category="web",
)

ToolRegistry.register(
    name="webcrawl_map",
    description=(
        "Return the site map (list of discoverable URLs) for a given URL. "
        "Only Firecrawl supports this natively today."
    ),
    input_schema=WebCrawlMapInput,
    handler=_map_handler,
    category="web",
)


__all__ = [
    "WebCrawlClient",
    "WebCrawlScrapeInput",
    "WebCrawlCrawlInput",
    "WebCrawlExtractInput",
    "WebCrawlMapInput",
]
