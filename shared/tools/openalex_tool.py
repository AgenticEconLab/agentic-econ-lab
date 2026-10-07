# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""OpenAlex search tool — open scholarly index with abstracts.

Since 2026-02-24 OpenAlex uses usage-based pricing: an API key (free registration at
openalex.org/settings/api) is required for production use via OPENALEX_API_KEY. Keyless
requests are budget-throttled to ~$0.10/day (~100 search calls); the free keyed tier is
$1/day (~1,000 search calls). OPENALEX_MAILTO ("polite pool" email) remains optional.
OpenAlex is the open-first primary for economics literature discovery and abstracts;
keyed sources (RePEc, Elsevier) are last-resort enhancements.

Abstracts come back as an ``abstract_inverted_index`` (word -> positions) which we
reconstruct into plain text — this recovers abstracts that Crossref/OpenAlex-from-Crossref
would otherwise lack, though publisher-withheld (e.g. Elsevier) abstracts still need the
institutional path.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from shared.tools.tool_registry import ToolRegistry

OPENALEX_URL = "https://api.openalex.org/works"
# OpenAlex topic taxonomy: field "Economics, Econometrics and Finance" = fields/20.
ECON_FIELD_FILTER = "primary_topic.field.id:fields/20"

_warned_keyless = False


def _auth_params() -> Dict[str, str]:
    """OPENALEX_API_KEY (required for production tiers) + optional OPENALEX_MAILTO."""
    global _warned_keyless
    params: Dict[str, str] = {}
    key = os.getenv("OPENALEX_API_KEY")
    if key:
        params["api_key"] = key
    elif not _warned_keyless:
        _warned_keyless = True
        print("  ⚠ OPENALEX_API_KEY not set — keyless OpenAlex is throttled to ~100 "
              "search calls/day. Free key: openalex.org/settings/api")
    mail = os.getenv("OPENALEX_MAILTO")
    if mail:
        params["mailto"] = mail
    return params


def reconstruct_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    """Rebuild abstract text from OpenAlex's abstract_inverted_index (word -> [positions])."""
    if not inverted_index:
        return ""
    pairs = [(pos, word) for word, positions in inverted_index.items() for pos in positions]
    pairs.sort()
    return " ".join(word for _, word in pairs)


class OpenAlexSearchInput(BaseModel):
    query: str = Field(description="Search query")
    max_results: int = Field(default=10, description="Number of works to return (<=50)")
    econ_only: bool = Field(default=True, description="Restrict to the Economics field")


def openalex_search_handler(
    query: str,
    max_results: int = 10,
    econ_only: bool = True,
    collector: object = None,
    agent: str = "",
) -> List[Dict[str, Any]]:
    """Search OpenAlex; return works with reconstructed abstracts."""
    from shared.observability import tracked_get

    params: Dict[str, Any] = {
        "search": query,
        "per_page": max(1, min(max_results, 50)),
        "select": ("id,doi,title,authorships,publication_year,abstract_inverted_index,"
                   "primary_location,cited_by_count,type"),
    }
    params.update(_auth_params())
    if econ_only:
        params["filter"] = ECON_FIELD_FILTER

    resp = tracked_get(OPENALEX_URL, params=params, collector=collector, agent=agent, retries=3)
    data = resp.json() if hasattr(resp, "json") else {}
    items: List[Dict[str, Any]] = []
    for w in (data.get("results") or []):
        loc = w.get("primary_location") or {}
        src = (loc.get("source") or {}) if isinstance(loc, dict) else {}
        items.append({
            "title": w.get("title") or "",
            "authors": [
                (a.get("author") or {}).get("display_name", "")
                for a in (w.get("authorships") or [])
            ],
            "abstract": reconstruct_abstract(w.get("abstract_inverted_index")),
            "url": w.get("doi") or w.get("id") or "",
            "doi": w.get("doi"),
            "year": w.get("publication_year"),
            "citation_count": w.get("cited_by_count"),
            "venue": src.get("display_name"),
            "type": w.get("type"),
            "source": "OpenAlex",
        })
    return items


ToolRegistry.register(
    name="openalex_search",
    description="Search OpenAlex (open, keyed via OPENALEX_API_KEY) for scholarly works with abstracts",
    input_schema=OpenAlexSearchInput,
    handler=openalex_search_handler,
    category="search",
)
