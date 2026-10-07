# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Web-crawl abstraction — provider-agnostic scrape/crawl/extract/map.

Introduced in the 2026-04-20 Firecrawl → WebCrawl refactor. Legacy code that imported
``from firecrawl import FirecrawlApp`` can continue doing so against the real
Firecrawl SDK; stages that want provider-agnostic behaviour should import
``WebCrawlClient`` from ``shared.tools.webcrawl_tool``.
"""

from shared.webcrawl.base import (
    UnsupportedCapability,
    WebCrawlProvider,
    WebCrawlResult,
)

__all__ = [
    "UnsupportedCapability",
    "WebCrawlProvider",
    "WebCrawlResult",
]
