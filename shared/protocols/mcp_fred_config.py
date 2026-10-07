# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
FRED MCP Configuration — Configure DataTeam to consume FRED data via MCP.

Provides a pre-configured MCPClient for the FRED (Federal Reserve Economic Data)
MCP server. Falls back gracefully to direct FRED API if MCP server is unavailable.

Usage:
    from shared.protocols.mcp_fred_config import get_fred_mcp_client, fred_mcp_fetch

    # Get configured client (or None if not available)
    client = get_fred_mcp_client()

    # Fetch data via MCP (falls back to None on error)
    result = fred_mcp_fetch("GDP", start_date="2020-01-01")
"""

import os
from typing import Any, Dict, Optional

# Default FRED MCP server URL (configurable via environment)
FRED_MCP_DEFAULT_URL = os.environ.get(
    "FRED_MCP_SERVER_URL",
    "http://localhost:8080/mcp",
)

# Whether FRED MCP is enabled (disable in environments without MCP server)
FRED_MCP_ENABLED = os.environ.get("FRED_MCP_ENABLED", "false").lower() in ("true", "1", "yes")


def get_fred_mcp_client(
    server_url: Optional[str] = None,
    timeout: float = 30.0,
) -> Optional[Any]:
    """
    Get a configured MCPClient for the FRED MCP server.

    Returns None if FRED MCP is not enabled or httpx is unavailable.

    Args:
        server_url: Override server URL (default: FRED_MCP_SERVER_URL env var).
        timeout: Request timeout in seconds.

    Returns:
        MCPClient instance or None.
    """
    if not FRED_MCP_ENABLED:
        return None

    try:
        from shared.protocols.mcp_client import MCPClient
        url = server_url or FRED_MCP_DEFAULT_URL
        client = MCPClient(server_url=url, timeout=timeout)
        return client
    except Exception:
        return None


def fred_mcp_fetch(
    series_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    server_url: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Fetch a FRED data series via MCP protocol.

    Falls back to None if MCP is unavailable or the call fails.

    Args:
        series_id: FRED series ID (e.g., "GDP", "CPIAUCSL").
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        server_url: Override MCP server URL.

    Returns:
        Dict with series data, or None on failure.
    """
    client = get_fred_mcp_client(server_url=server_url)
    if client is None:
        return None

    try:
        arguments: Dict[str, Any] = {"series_id": series_id}
        if start_date:
            arguments["start_date"] = start_date
        if end_date:
            arguments["end_date"] = end_date

        result = client.call_tool("fred_get_series", arguments)
        return result
    except Exception:
        return None


# FRED MCP tool descriptions (for registration with ToolRegistry if needed)
FRED_MCP_TOOLS = [
    {
        "name": "fred_get_series",
        "description": "Fetch a FRED economic data series by series ID",
        "inputSchema": {
            "type": "object",
            "properties": {
                "series_id": {"type": "string", "description": "FRED series ID (e.g., GDP, CPIAUCSL)"},
                "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"},
            },
            "required": ["series_id"],
        },
    },
    {
        "name": "fred_search_series",
        "description": "Search FRED for data series matching a query",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Max results (default 10)"},
            },
            "required": ["query"],
        },
    },
]
