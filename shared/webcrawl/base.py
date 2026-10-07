# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Base types for the WebCrawl abstraction.

``WebCrawlResult`` intentionally mirrors ``shared.tools.tool_registry.ToolResult``
so a WebCrawl result can be returned from a ToolRegistry handler without
shape adaptation. The one added field is ``provider`` — which of the four
backends actually served the call.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class UnsupportedCapability(RuntimeError):
    """Raised when a provider is asked for a capability it does not offer.

    The ``WebCrawlClient`` catches this and advances to the next provider in
    the fallback chain rather than surfacing it to the caller.
    """

    def __init__(self, provider: str, capability: str):
        self.provider = provider
        self.capability = capability
        super().__init__(
            f"Provider {provider!r} does not support capability {capability!r}"
        )


class WebCrawlResult(BaseModel):
    """Unified result shape for scrape / crawl / extract / map calls."""

    success: bool = Field(description="Whether the call succeeded")
    data: Any = Field(default=None, description="Provider-native payload on success")
    markdown: Optional[str] = Field(
        default=None,
        description=(
            "Convenience field for scrape calls. When a provider returns "
            "markdown, it is copied here so call sites can do "
            "``result.markdown`` without reaching into .data."
        ),
    )
    links: Optional[List[str]] = Field(
        default=None, description="Links returned by map/crawl calls"
    )
    error: Optional[str] = Field(default=None, description="Error message on failure")
    latency_ms: float = Field(default=0.0, description="End-to-end latency in ms")
    provider: str = Field(default="", description="Provider that served the call")
    cost_usd: float = Field(default=0.0, description="Estimated USD cost if known")
    attempts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "One entry per provider tried, in order. Each entry is "
            "``{'provider': str, 'success': bool, 'error': str | None, "
            "'latency_ms': float}``. Populated by the fallback client; "
            "individual providers leave it empty."
        ),
    )


@runtime_checkable
class WebCrawlProvider(Protocol):
    """Provider protocol. Concrete providers live in ``shared/webcrawl/providers/``."""

    name: str
    capabilities: FrozenSet[str]

    def is_available(self) -> bool:
        """Return True iff the provider has the env vars / config it needs."""
        ...

    def scrape(
        self,
        url: str,
        *,
        formats: Optional[List[str]] = None,
        collector: Any = None,
        agent: str = "",
    ) -> WebCrawlResult:
        ...

    def crawl(
        self,
        url: str,
        *,
        limit: int = 10,
        collector: Any = None,
        agent: str = "",
    ) -> WebCrawlResult:
        ...

    def extract(
        self,
        urls: List[str],
        *,
        schema: Optional[Dict[str, Any]] = None,
        collector: Any = None,
        agent: str = "",
    ) -> WebCrawlResult:
        ...

    def map(
        self,
        url: str,
        *,
        collector: Any = None,
        agent: str = "",
    ) -> WebCrawlResult:
        ...
