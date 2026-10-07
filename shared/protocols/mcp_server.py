# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
MCP Server — Expose AEL ToolRegistry tools via Model Context Protocol.

Wraps the existing ToolRegistry as an MCP-compliant server, enabling
any MCP-compatible client (Claude, GPT, etc.) to discover and invoke
AEL tools (arxiv_search, fred_get_series, web_fetch, etc.).

The MCP protocol uses JSON-RPC 2.0 format for tool listing and invocation.

Usage:
    from shared.protocols.mcp_server import AELMCPServer

    server = AELMCPServer()

    # List available tools (MCP tools/list)
    tools = server.list_tools()

    # Invoke a tool (MCP tools/call)
    result = server.call_tool("arxiv_search", {"query": "AI economics", "max_results": 5})

    # Handle raw JSON-RPC request
    response = server.handle_request({
        "jsonrpc": "2.0",
        "method": "tools/list",
        "id": "req-001",
    })
"""

import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# V0.7 — ASI07 Log-To-Leak defense
try:
    from shared.security.log_sanitizer import LogSanitizer as _LogSanitizer
    _DEFAULT_SANITIZER: Optional["_LogSanitizer"] = _LogSanitizer()
except Exception:
    _DEFAULT_SANITIZER = None


# ---------------------------------------------------------------------------
# MCP Protocol Data Models
# ---------------------------------------------------------------------------

class MCPToolDescription(BaseModel):
    """MCP tool description (tools/list response item)."""
    name: str = Field(description="Tool name")
    description: str = Field(description="What the tool does")
    inputSchema: Dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}},
        description="JSON Schema for tool arguments",
    )


class MCPToolResult(BaseModel):
    """MCP tool invocation result (tools/call response)."""
    content: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Result content blocks",
    )
    isError: bool = Field(default=False)


class MCPServerInfo(BaseModel):
    """MCP server information (initialize response)."""
    name: str = Field(default="AEL-ToolServer")
    version: str = Field(default="0.7.0")
    # Default protocol stays V0.5-compatible for clients that don't negotiate.
    # V0.7 advertises support for the 2026-03-26 revision via .well-known.
    protocolVersion: str = Field(default="2024-11-05")
    capabilities: Dict[str, Any] = Field(
        default_factory=lambda: {"tools": {"listChanged": False}, "tasks": {}},
    )


# ---------------------------------------------------------------------------
# AEL MCP Server
# ---------------------------------------------------------------------------

class AELMCPServer:
    """
    MCP-compliant server wrapping AEL's ToolRegistry.

    Provides tool discovery (list_tools) and invocation (call_tool)
    via the MCP protocol. Handles JSON-RPC requests for:
    - initialize: Server capability negotiation
    - tools/list: List available tools
    - tools/call: Invoke a tool by name
    """

    def __init__(
        self,
        collector: Optional[Any] = None,
        *,
        sanitizer: Optional["_LogSanitizer"] = None,
        sanitize_io: bool = True,
        authorizer: Optional[Any] = None,
        task_registry: Optional[Any] = None,
    ):
        """
        Args:
            collector: Optional MetricsCollector for tracking tool calls.
            sanitizer: Optional LogSanitizer override (defaults to module-level
                instance). Pass ``None`` + ``sanitize_io=False`` to disable.
            sanitize_io: When True (default), tool arguments and result text
                are run through ``LogSanitizer`` to defeat ASI07 Log-To-Leak.
            authorizer: Optional V0.7 :class:`OAuth21Authorizer`. When present,
                ``handle_request`` enforces bearer-token auth. Legacy V0.5
                clients continue to work when ``authorizer is None``.
            task_registry: Optional V0.7 :class:`MCPTaskRegistry`. When present,
                ``tasks/*`` JSON-RPC methods are routed here.
        """
        self._collector = collector
        self._sanitizer = sanitizer if sanitizer is not None else _DEFAULT_SANITIZER
        self._sanitize_io = bool(sanitize_io) and self._sanitizer is not None
        # V0.7 extensions — both optional, back-compat preserved when None
        self._authorizer = authorizer
        self._task_registry = task_registry
        # Ensure the ToolRegistry is populated: a standalone MCP server (external client
        # following the module docstring) never runs a MasterOrchestrator, so without this
        # tools/list would be empty. register_all_tools is idempotent.
        try:
            from shared.tools.register_all import register_all_tools
            register_all_tools()
        except Exception:  # never let tool registration block server construction
            pass

    def get_server_info(self) -> MCPServerInfo:
        """Return MCP server metadata."""
        return MCPServerInfo()

    def list_tools(self) -> List[MCPToolDescription]:
        """
        List all tools available in the ToolRegistry as MCP tool descriptions.

        Returns:
            List of MCPToolDescription objects.
        """
        from shared.tools.tool_registry import ToolRegistry

        tools = ToolRegistry.list_tools()
        result = []
        for tool in tools:
            # input_schema is a Pydantic model class — convert to JSON Schema dict
            schema_dict = {"type": "object", "properties": {}}
            if hasattr(tool, "input_schema") and tool.input_schema is not None:
                try:
                    schema_dict = tool.input_schema.model_json_schema()
                except Exception:
                    pass
            result.append(MCPToolDescription(
                name=tool.name,
                description=getattr(tool, "description", tool.name),
                inputSchema=schema_dict,
            ))
        return result

    def call_tool(
        self,
        name: str,
        arguments: Dict[str, Any],
        agent: str = "mcp_client",
    ) -> MCPToolResult:
        """
        Invoke a tool by name with given arguments.

        Args:
            name: Tool name (must match a registered tool).
            arguments: Tool arguments as a dict.
            agent: Agent name for metrics attribution.

        Returns:
            MCPToolResult with content blocks.
        """
        from shared.tools.tool_registry import ToolRegistry

        # V0.7 — sanitize tool arguments before invocation to defeat
        # Log-To-Leak (ASI07). Sanitizer is a no-op for plain data.
        sanitized_args = arguments
        if self._sanitize_io and self._sanitizer is not None:
            try:
                sanitized_args = self._sanitizer.sanitize(arguments)
            except Exception:
                sanitized_args = arguments

        result = ToolRegistry.invoke(
            name, sanitized_args,
            collector=self._collector,
            agent=agent,
        )

        if result.success:
            text = str(result.data) if result.data is not None else ""
            if self._sanitize_io and self._sanitizer is not None:
                try:
                    text = self._sanitizer.sanitize_string(text)
                except Exception:
                    pass
            return MCPToolResult(
                content=[{"type": "text", "text": text}],
                isError=False,
            )
        else:
            err_text = result.error or "Tool invocation failed"
            if self._sanitize_io and self._sanitizer is not None:
                try:
                    err_text = self._sanitizer.sanitize_string(err_text)
                except Exception:
                    pass
            return MCPToolResult(
                content=[{"type": "text", "text": err_text}],
                isError=True,
            )

    def handle_request(
        self,
        request: Dict[str, Any],
        *,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Handle a raw JSON-RPC request per MCP protocol.

        Supported methods:
        - initialize: Server capability negotiation
        - tools/list: List available tools
        - tools/call: Invoke a tool
        - tasks/create, tasks/status, tasks/list: V0.7 Tasks primitive
          (active only when ``task_registry`` was passed to the constructor)

        Args:
            request: JSON-RPC request dict.
            headers: Optional HTTP headers. When ``authorizer`` is configured,
                a valid ``Authorization: Bearer <token>`` is required (V0.7).

        Returns:
            JSON-RPC 2.0 response dict.
        """
        method = request.get("method", "")
        req_id = request.get("id", str(uuid.uuid4()))
        params = request.get("params", {})

        # V0.7 OAuth 2.1 enforcement (only when the server was constructed
        # with an authorizer; legacy V0.5 clients continue unchanged).
        if self._authorizer is not None and method != "initialize":
            try:
                self._authorizer.authorize(headers or {})
            except PermissionError as exc:
                return _rpc_error(req_id, -32001, f"Unauthorized: {exc}")

        if method == "initialize":
            info = self.get_server_info()
            return _rpc_response(req_id, info.model_dump())

        elif method == "tools/list":
            tools = self.list_tools()
            return _rpc_response(req_id, {
                "tools": [t.model_dump() for t in tools],
            })

        elif method == "tools/call":
            name = params.get("name", "")
            arguments = params.get("arguments", {})
            if not name:
                return _rpc_error(req_id, -32602, "Missing tool name")
            result = self.call_tool(name, arguments)
            return _rpc_response(req_id, result.model_dump())

        # V0.7 Tasks primitive (SEP-1686) — only when a registry is attached
        elif method == "tasks/create" and self._task_registry is not None:
            tool = params.get("tool_name") or params.get("name")
            args = params.get("arguments", {})
            if not tool:
                return _rpc_error(req_id, -32602, "Missing tool_name")
            task = self._task_registry.create(tool, args)
            return _rpc_response(req_id, task.to_dict())

        elif method == "tasks/status" and self._task_registry is not None:
            tid = params.get("task_id")
            if not tid:
                return _rpc_error(req_id, -32602, "Missing task_id")
            task = self._task_registry.get(tid)
            if task is None:
                return _rpc_error(req_id, -32000, f"Unknown task: {tid}")
            return _rpc_response(req_id, task.to_dict())

        elif method == "tasks/list" and self._task_registry is not None:
            status = params.get("status")
            tasks = self._task_registry.list(status=status)
            return _rpc_response(req_id, {"tasks": [t.to_dict() for t in tasks]})

        else:
            return _rpc_error(req_id, -32601, f"Method not found: {method}")

    # V0.7 — OAuth 2.1 .well-known/oauth-protected-resource handler
    def well_known_oauth_protected_resource(self) -> Dict[str, Any]:
        """Return metadata describing the MCP resource for OAuth 2.1 clients.

        Returns an empty dict when no authorizer is attached so that the
        server's JSON shape is always well-formed.
        """
        if self._authorizer is None:
            return {}
        try:
            return self._authorizer.well_known_payload()
        except Exception:
            return {}


# ---------------------------------------------------------------------------
# JSON-RPC helpers
# ---------------------------------------------------------------------------

def _rpc_response(req_id: str, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "result": result, "id": req_id}


def _rpc_error(req_id: str, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "error": {"code": code, "message": message}, "id": req_id}
