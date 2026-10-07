# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Web Search Tool — MCP-aligned wrapper around tracked_get for search APIs.

Registers the web_search tool in the ToolRegistry with a validated
Pydantic input schema. Supports Tavily (primary), Brave, and Serper search APIs.
"""

import os
import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from shared.tools.tool_registry import ToolRegistry


class WebSearchInput(BaseModel):
    """Input schema for web search."""
    query: str = Field(description="Search query string")
    max_results: int = Field(default=10, description="Maximum number of results")
    engine: str = Field(default="tavily", description="Search engine: tavily, brave, or serper")


def web_search_handler(
    query: str,
    max_results: int = 10,
    engine: str = "tavily",
    collector: object = None,
    agent: str = "",
) -> List[Dict[str, Any]]:
    """
    Perform a web search using Tavily (primary), Brave, or Serper API.

    When engine is "tavily" (default), falls back to Brave then Serper
    if the TAVILY_API_KEY is not configured.
    """
    if engine == "tavily":
        if os.environ.get("TAVILY_API_KEY"):
            return _search_tavily(query, max_results, collector, agent)
        # Fallback: tavily key missing → try brave → serper
        if os.environ.get("BRAVE_API_KEY"):
            return _search_brave(query, max_results, collector, agent)
        if os.environ.get("SERPER_API_KEY"):
            return _search_serper(query, max_results, collector, agent)
        return [{"error": "No search API key configured (TAVILY_API_KEY, BRAVE_API_KEY, or SERPER_API_KEY)"}]
    elif engine == "serper":
        return _search_serper(query, max_results, collector, agent)
    else:
        return _search_brave(query, max_results, collector, agent)


def _search_tavily(
    query: str,
    max_results: int,
    collector: object,
    agent: str,
) -> List[Dict[str, Any]]:
    """Search using Tavily Search API."""
    from shared.observability import tracked_post

    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        return [{"error": "TAVILY_API_KEY not configured"}]

    response = tracked_post(
        url="https://api.tavily.com/search",
        collector=collector,
        agent=agent,
        headers={"Content-Type": "application/json"},
        json={"api_key": api_key, "query": query, "max_results": min(max_results, 20)},
    )

    if response.status_code != 200:
        return [{"error": f"Tavily API returned {response.status_code}"}]

    data = response.json()
    results = []
    for item in data.get("results", [])[:max_results]:
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "description": item.get("content", ""),
        })
    return results


def _search_brave(
    query: str,
    max_results: int,
    collector: object,
    agent: str,
) -> List[Dict[str, Any]]:
    """Search using Brave Search API."""
    from shared.observability import tracked_get

    api_key = os.environ.get("BRAVE_API_KEY", "")
    if not api_key:
        return [{"error": "BRAVE_API_KEY not configured"}]

    response = tracked_get(
        url="https://api.search.brave.com/res/v1/web/search",
        collector=collector,
        agent=agent,
        headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
        params={"q": query, "count": min(max_results, 20)},
    )

    if response.status_code != 200:
        return [{"error": f"Brave API returned {response.status_code}"}]

    data = response.json()
    results = []
    for item in data.get("web", {}).get("results", [])[:max_results]:
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "description": item.get("description", ""),
        })
    return results


def _search_serper(
    query: str,
    max_results: int,
    collector: object,
    agent: str,
) -> List[Dict[str, Any]]:
    """Search using Serper API."""
    from shared.observability import tracked_post

    api_key = os.environ.get("SERPER_API_KEY", "")
    if not api_key:
        return [{"error": "SERPER_API_KEY not configured"}]

    response = tracked_post(
        url="https://api.serper.dev/search",
        collector=collector,
        agent=agent,
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        json={"q": query, "num": min(max_results, 20)},
    )

    if response.status_code != 200:
        return [{"error": f"Serper API returned {response.status_code}"}]

    data = response.json()
    results = []
    for item in data.get("organic", [])[:max_results]:
        results.append({
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "description": item.get("snippet", ""),
        })
    return results


# Register in the global tool registry
ToolRegistry.register(
    name="web_search",
    description="Search the web using Tavily (primary), Brave, or Serper search API",
    input_schema=WebSearchInput,
    handler=web_search_handler,
    category="search",
)
