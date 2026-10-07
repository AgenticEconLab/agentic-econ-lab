# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
MCP Client — Consume external MCP tool servers from AEL workflows.

Enables AEL teams (especially DataTeam) to access tools provided
by external MCP servers (e.g., FRED MCP server, OpenEcon Data).

Usage:
    from shared.protocols.mcp_client import MCPClient

    # Connect to an external MCP server
    client = MCPClient("http://fred-mcp.example.com")

    # Discover tools
    tools = client.list_tools()

    # Invoke a tool
    result = client.call_tool("fred_get_series", {"series_id": "GDP"})
"""

import uuid
from typing import Any, Dict, List, Optional

import httpx


class MCPClientError(Exception):
    """Error communicating with an MCP server."""

    def __init__(self, message: str, code: int = -1):
        super().__init__(message)
        self.code = code


class MCPClient:
    """
    Client for consuming external MCP-compliant tool servers.

    Sends JSON-RPC requests to discover and invoke tools on external
    MCP servers.
    """

    def __init__(
        self,
        server_url: str,
        timeout: float = 60.0,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Args:
            server_url: Base URL of the MCP server.
            timeout: HTTP request timeout in seconds.
            headers: Optional extra headers.
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.headers = headers or {}
        self._server_info: Optional[Dict[str, Any]] = None
        self._tools_cache: Optional[List[Dict[str, Any]]] = None

    def initialize(self) -> Dict[str, Any]:
        """
        Initialize the MCP connection and negotiate capabilities.

        Returns:
            Server info dict with name, version, capabilities.
        """
        request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "clientInfo": {
                    "name": "AEL-Pipeline",
                    "version": "0.5.0",
                },
                "capabilities": {},
            },
            "id": str(uuid.uuid4()),
        }
        self._server_info = self._send_rpc(request)
        return self._server_info

    def list_tools(self, refresh: bool = False) -> List[Dict[str, Any]]:
        """
        List tools available on the MCP server.

        Args:
            refresh: Force refresh the tools cache.

        Returns:
            List of tool description dicts.
        """
        if self._tools_cache is not None and not refresh:
            return self._tools_cache

        request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": str(uuid.uuid4()),
        }
        result = self._send_rpc(request)
        self._tools_cache = result.get("tools", [])
        return self._tools_cache

    def call_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Invoke a tool on the MCP server.

        Args:
            name: Tool name.
            arguments: Tool arguments dict.

        Returns:
            Tool result with content blocks.
        """
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments or {},
            },
            "id": str(uuid.uuid4()),
        }
        return self._send_rpc(request)

    def get_tool_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get the input schema for a specific tool.

        Args:
            name: Tool name.

        Returns:
            JSON Schema dict, or None if tool not found.
        """
        tools = self.list_tools()
        for tool in tools:
            if tool.get("name") == name:
                return tool.get("inputSchema", {})
        return None

    def _send_rpc(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send a JSON-RPC request and return the result."""
        try:
            resp = httpx.post(
                self.server_url,
                json=request,
                headers={"Content-Type": "application/json", **self.headers},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            if "error" in data and data["error"] is not None:
                error = data["error"]
                raise MCPClientError(
                    error.get("message", "Unknown MCP error"),
                    code=error.get("code", -1),
                )

            return data.get("result", {})

        except httpx.HTTPError as e:
            raise MCPClientError(f"MCP request failed: {e}") from e
