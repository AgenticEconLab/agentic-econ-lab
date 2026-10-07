# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Backward-compatibility shim for code that used to do:

    from firecrawl import FirecrawlApp
    app = FirecrawlApp(api_key=...)
    resp = app.scrape_url(url, formats=['markdown'])

After the 2026-04-20 refactor, stage code should switch to::

    from shared.tools.webcrawl_tool import WebCrawlClient
    client = WebCrawlClient()
    resp = client.scrape_url(url)  # returns an object with .markdown

This module provides ``FirecrawlApp`` as an alias around ``WebCrawlClient``
so any straggler that still imports through this shim keeps working. Emits a
one-shot ``DeprecationWarning``. Scheduled for removal in a future release.
"""

from __future__ import annotations

import warnings
from typing import Any, List, Optional

from shared.tools.webcrawl_tool import WebCrawlClient

_WARNED = False


def _warn_once() -> None:
    global _WARNED
    if _WARNED:
        return
    _WARNED = True
    warnings.warn(
        "shared.webcrawl._firecrawl_compat.FirecrawlApp is a compatibility "
        "shim; import WebCrawlClient from shared.tools.webcrawl_tool instead. "
        "Scheduled for removal in a future release.",
        DeprecationWarning,
        stacklevel=3,
    )


class FirecrawlApp:
    """Compat alias around WebCrawlClient pinned to the Firecrawl provider."""

    def __init__(self, api_key: Optional[str] = None, **_ignored: Any):
        _warn_once()
        # Pin to firecrawl explicitly — callers of this shim specifically
        # expected Firecrawl behaviour.
        self._client = WebCrawlClient(primary="firecrawl", fallback_chain=[])
        # Preserve the user's api_key on the provider instance if provided.
        if api_key:
            fc = self._client._resolve("firecrawl")  # noqa: SLF001 — intentional
            fc.api_key = api_key

    def scrape_url(self, url: str, formats: Optional[List[str]] = None):
        return self._client.scrape_url(url, formats=formats)
