# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Concrete WebCrawl providers."""

from shared.webcrawl.providers.crawl4ai import Crawl4AIProvider
from shared.webcrawl.providers.firecrawl import FirecrawlProvider
from shared.webcrawl.providers.jina import JinaProvider
from shared.webcrawl.providers.tavily import TavilyProvider
from shared.webcrawl.providers.trafilatura import TrafilaturaProvider

__all__ = [
    "Crawl4AIProvider",
    "FirecrawlProvider",
    "JinaProvider",
    "TavilyProvider",
    "TrafilaturaProvider",
]


def get_provider(name: str):
    """Resolve a provider name to its concrete class. Raises KeyError if unknown."""
    mapping = {
        "trafilatura": TrafilaturaProvider,   # OSS, pure-Python, no browser (default)
        "crawl4ai": Crawl4AIProvider,         # OSS, browser (JS fallback)
        "firecrawl": FirecrawlProvider,
        "tavily": TavilyProvider,
        "jina": JinaProvider,
    }
    return mapping[name.lower()]
