# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Open-first news/web search for ideation.

Discovery layer (query -> result URLs), distinct from web-crawl (URL -> content).
Mirrors the web-crawl chain's open-default philosophy:

    SearXNG (self-hosted, open) -> DDGS (open library) -> Tavily -> Brave (commercial)

Two open engines first: SearXNG (meta-search service at ``SEARXNG_BASE_URL``, default
``http://localhost:8899``) then DDGS (``ddgs`` library — no key, no service). Commercial
providers (Tavily/Brave, free-tier) are tried only if both open engines fail or return
nothing. Returns a list of ``{"title","url","content"}`` dicts.

Pair with ``shared.tools.webcrawl_tool.WebCrawlClient`` to crawl the returned URLs
for full article text (trafilatura -> crawl4ai -> ...).
"""

from __future__ import annotations

import os
from typing import Dict, List

import requests

DEFAULT_SEARXNG = "http://localhost:8899"


def _searxng(query: str, max_results: int, categories: str) -> List[Dict[str, str]]:
    base = os.environ.get("SEARXNG_BASE_URL", DEFAULT_SEARXNG).rstrip("/")
    params = {"q": query, "format": "json"}
    if categories:
        params["categories"] = categories
    resp = requests.get(f"{base}/search", params=params, timeout=20)
    resp.raise_for_status()
    out: List[Dict[str, str]] = []
    for it in (resp.json().get("results") or [])[:max_results]:
        url = it.get("url") or ""
        if url:
            out.append({"title": it.get("title") or "", "url": url, "content": it.get("content") or ""})
    return out


def _tavily(query: str, max_results: int, categories: str) -> List[Dict[str, str]]:
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        return []
    resp = requests.post(
        "https://api.tavily.com/search",
        headers={"Content-Type": "application/json"},
        json={"api_key": key, "query": query, "max_results": max_results},
        timeout=20,
    )
    resp.raise_for_status()
    return [
        {"title": r.get("title") or "", "url": r.get("url") or "", "content": r.get("content") or ""}
        for r in (resp.json().get("results") or [])[:max_results] if r.get("url")
    ]


def _brave(query: str, max_results: int, categories: str) -> List[Dict[str, str]]:
    key = os.environ.get("BRAVE_API_KEY")
    if not key:
        return []
    resp = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        headers={"Accept": "application/json", "X-Subscription-Token": key},
        params={"q": query, "count": max_results},
        timeout=20,
    )
    resp.raise_for_status()
    web = (resp.json().get("web") or {}).get("results") or []
    return [
        {"title": r.get("title") or "", "url": r.get("url") or "", "content": r.get("description") or ""}
        for r in web[:max_results] if r.get("url")
    ]


def _ddgs(query: str, max_results: int, categories: str) -> List[Dict[str, str]]:
    """DDGS (Dux Distributed Global Search; formerly duckduckgo_search) — open-source
    library, no API key, no service. Queries DuckDuckGo/Brave/Mojeek/etc. directly."""
    from ddgs import DDGS
    out: List[Dict[str, str]] = []
    with DDGS() as d:
        method = d.news if categories == "news" else d.text
        for r in list(method(query, max_results=max_results)):
            url = r.get("url") or r.get("href") or ""
            if url:
                out.append({"title": r.get("title") or "", "url": url, "content": r.get("body") or ""})
    return out


# Two OPEN engines first — SearXNG (self-hosted meta-search) then DDGS (library,
# no key/service) — commercial (Tavily/Brave) only as last-resort fallback.
_CHAIN = [("searxng", _searxng), ("ddgs", _ddgs), ("tavily", _tavily), ("brave", _brave)]


def search_news(query: str, max_results: int = 5, categories: str = "news") -> List[Dict[str, str]]:
    """Open-first news search. Returns [{title,url,content}] from the first provider
    in the chain (SearXNG -> Tavily -> Brave) that yields results; [] if all fail."""
    for name, fn in _CHAIN:
        try:
            res = fn(query, max_results, categories)
            if res:
                return res
        except Exception:
            continue
    return []


def active_search_provider() -> str:
    """First search provider in the chain that responds (for diagnostics/logging)."""
    for name, fn in _CHAIN:
        try:
            if fn("economics", 1, "news"):
                return name
        except Exception:
            continue
    return "none"
