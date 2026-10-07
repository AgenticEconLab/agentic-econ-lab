# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Entry points that invoke the ToolRegistry must ensure it is populated.

A standalone MCP server (an external client following the module docstring) never runs a
MasterOrchestrator, so without self-registration `tools/list` would be empty — the same
0-results failure class as an earlier pipeline bug. AELMCPServer must register on construction."""


def test_mcp_server_populates_registry_on_construction():
    from shared.tools.tool_registry import ToolRegistry
    from shared.protocols.mcp_server import AELMCPServer

    ToolRegistry._tools.clear()          # simulate a fresh process — no orchestrator has run
    server = AELMCPServer()              # must self-register (register_all_tools, idempotent)
    tools = server.list_tools()
    assert len(tools) > 0, "MCP server exposed an EMPTY tool list — registry was not populated"
    names = {t.name for t in tools}
    # a couple of the always-registered search tools should be present
    assert names & {"openalex_search", "arxiv_search", "web_fetch"}
