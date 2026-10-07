# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
HTTP Tools — MCP-aligned wrappers around tracked_get / tracked_post.

Registers web_fetch and web_post tools in the ToolRegistry for generic
HTTP operations. Stage files that need raw HTTP access (e.g., Semantic Scholar,
World Bank, OpenAlex) use these tools instead of tracked_get/tracked_post.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from shared.tools.tool_registry import ToolRegistry


class WebFetchInput(BaseModel):
    """Input schema for HTTP GET requests."""
    url: str = Field(description="URL to fetch")
    params: Dict[str, Any] = Field(default_factory=dict, description="Query parameters")
    headers: Dict[str, str] = Field(default_factory=dict, description="HTTP headers")
    retries: int = Field(default=0, description="Number of retries on failure")


class WebPostInput(BaseModel):
    """Input schema for HTTP POST requests."""
    url: str = Field(description="URL to post to")
    json_body: Dict[str, Any] = Field(default_factory=dict, description="JSON body payload")
    headers: Dict[str, str] = Field(default_factory=dict, description="HTTP headers")
    retries: int = Field(default=0, description="Number of retries on failure")


def web_fetch_handler(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    retries: int = 0,
    collector: object = None,
    agent: str = "",
) -> Dict[str, Any]:
    """
    Perform an HTTP GET request and return parsed JSON response.

    Returns dict with keys: status_code, data (parsed JSON or text), url.
    """
    from shared.observability import tracked_get

    resp = tracked_get(
        url=url,
        params=params or {},
        headers=headers or {},
        collector=collector,
        agent=agent,
        retries=retries,
    )

    result = {"status_code": resp.status_code, "url": str(resp.url)}
    try:
        result["data"] = resp.json()
    except Exception:
        result["data"] = resp.text[:5000] if resp.text else ""
    return result


def web_post_handler(
    url: str,
    json_body: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    retries: int = 0,
    collector: object = None,
    agent: str = "",
) -> Dict[str, Any]:
    """
    Perform an HTTP POST request and return parsed JSON response.

    Returns dict with keys: status_code, data (parsed JSON or text), url.
    """
    from shared.observability import tracked_post

    resp = tracked_post(
        url=url,
        json=json_body or {},
        headers=headers or {},
        collector=collector,
        agent=agent,
        retries=retries,
    )

    result = {"status_code": resp.status_code, "url": str(resp.url)}
    try:
        result["data"] = resp.json()
    except Exception:
        result["data"] = resp.text[:5000] if resp.text else ""
    return result


# Register in the global tool registry
ToolRegistry.register(
    name="web_fetch",
    description="Perform an HTTP GET request and return JSON or text response",
    input_schema=WebFetchInput,
    handler=web_fetch_handler,
    category="web",
)

ToolRegistry.register(
    name="web_post",
    description="Perform an HTTP POST request and return JSON or text response",
    input_schema=WebPostInput,
    handler=web_post_handler,
    category="web",
)
