# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
ArXiv Search Tool — MCP-aligned wrapper around tracked_arxiv_search.

Registers the arxiv_search tool in the ToolRegistry with a validated
Pydantic input schema.
"""

from typing import List, Optional

from pydantic import BaseModel, Field

from shared.tools.tool_registry import ToolRegistry


class ArxivSearchInput(BaseModel):
    """Input schema for ArXiv search."""
    query: str = Field(description="Search query string")
    max_results: int = Field(default=10, description="Maximum number of results")
    sort_by: str = Field(default="relevance", description="Sort by: relevance or submittedDate")


def arxiv_search_handler(
    query: str,
    max_results: int = 10,
    sort_by: str = "relevance",
    collector: object = None,
    agent: str = "",
) -> List[dict]:
    """
    Search ArXiv for academic papers.

    Delegates to shared.observability.tracked_arxiv_search for the actual
    API call and observability tracking.
    """
    from shared.observability import tracked_arxiv_search

    results = tracked_arxiv_search(
        query=query,
        max_results=max_results,
        sort_by=sort_by,
        collector=collector,
        agent=agent,
    )
    # Convert arxiv Result objects to dicts
    items = []
    for r in results:
        items.append({
            "title": getattr(r, "title", str(r)),
            "authors": [str(a) for a in getattr(r, "authors", [])],
            "abstract": getattr(r, "summary", ""),
            "url": getattr(r, "entry_id", ""),
            "published": str(getattr(r, "published", "")),
            "updated": str(getattr(r, "updated", "")),
            "categories": getattr(r, "categories", []),
        })
    return items


# Register in the global tool registry
ToolRegistry.register(
    name="arxiv_search",
    description="Search ArXiv for academic papers by keyword or topic",
    input_schema=ArxivSearchInput,
    handler=arxiv_search_handler,
    category="search",
)
