# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Phase 2 (V0.5) Integration Tests — Interoperability Standards (A2A + MCP).

Tests A2A server/client, MCP server/client, A2A agent cards for all 4 teams,
MessageBus A2A serialization, and backward compatibility.
"""

import sys
from pathlib import Path
from typing import Dict, Any

import pytest


# ============================================================================
# Test 1: A2A Server — Core
# ============================================================================

class TestA2AServerCore:
    """A2ATeamServer initialization and agent card generation."""

    def test_import(self):
        from shared.protocols.a2a_server import A2ATeamServer
        assert A2ATeamServer is not None

    def test_init(self):
        from shared.protocols.a2a_server import A2ATeamServer
        server = A2ATeamServer("IdeationTeam")
        assert server.team_name == "IdeationTeam"

    def test_get_agent_card(self):
        from shared.protocols.a2a_server import A2ATeamServer
        server = A2ATeamServer("IdeationTeam")
        card = server.get_agent_card()
        assert card.name == "AEL-IdeationTeam"
        assert card.version == "0.5.0"
        assert card.protocol_version == "0.3"
        assert len(card.skills) >= 3

    def test_agent_card_for_each_team(self):
        from shared.protocols.a2a_server import A2ATeamServer
        for team in ["IdeationTeam", "LiteratureTeam", "ModelTeam", "DataTeam"]:
            server = A2ATeamServer(team)
            card = server.get_agent_card()
            assert card.name == f"AEL-{team}"
            assert len(card.skills) >= 3

    def test_agent_card_has_url(self):
        from shared.protocols.a2a_server import A2ATeamServer
        server = A2ATeamServer("IdeationTeam", host="myhost", port=9000)
        card = server.get_agent_card()
        assert "myhost:9000" in card.url

    def test_agent_card_capabilities(self):
        from shared.protocols.a2a_server import A2ATeamServer
        server = A2ATeamServer("IdeationTeam")
        card = server.get_agent_card()
        assert "streaming" in card.capabilities
        assert card.capabilities["stateTransitionHistory"] is True


# ============================================================================
# Test 2: A2A Server — Task Handling
# ============================================================================

class TestA2AServerTasks:
    """A2A task submission, status, and cancellation."""

    def test_task_send(self):
        from shared.protocols.a2a_server import A2ATeamServer
        server = A2ATeamServer("IdeationTeam")
        response = server.handle_request({
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {
                "message": {"role": "user", "parts": [{"text": "AI economics"}]},
            },
            "id": "req-001",
        })
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == "req-001"
        assert response["result"]["state"] == "submitted"
        assert "id" in response["result"]

    def test_task_get(self):
        from shared.protocols.a2a_server import A2ATeamServer
        server = A2ATeamServer("IdeationTeam")
        # First create a task
        send_resp = server.handle_request({
            "jsonrpc": "2.0", "method": "tasks/send",
            "params": {"message": {"role": "user", "parts": []}},
            "id": "req-001",
        })
        task_id = send_resp["result"]["id"]

        # Then get it
        get_resp = server.handle_request({
            "jsonrpc": "2.0", "method": "tasks/get",
            "params": {"id": task_id},
            "id": "req-002",
        })
        assert get_resp["result"]["id"] == task_id
        assert get_resp["result"]["state"] == "submitted"

    def test_task_cancel(self):
        from shared.protocols.a2a_server import A2ATeamServer
        server = A2ATeamServer("IdeationTeam")
        send_resp = server.handle_request({
            "jsonrpc": "2.0", "method": "tasks/send",
            "params": {"message": {}},
            "id": "r1",
        })
        task_id = send_resp["result"]["id"]

        cancel_resp = server.handle_request({
            "jsonrpc": "2.0", "method": "tasks/cancel",
            "params": {"id": task_id},
            "id": "r2",
        })
        assert cancel_resp["result"]["state"] == "canceled"

    def test_task_get_not_found(self):
        from shared.protocols.a2a_server import A2ATeamServer
        server = A2ATeamServer("IdeationTeam")
        resp = server.handle_request({
            "jsonrpc": "2.0", "method": "tasks/get",
            "params": {"id": "nonexistent"},
            "id": "r1",
        })
        assert resp["error"] is not None
        assert resp["error"]["code"] == -32602

    def test_unknown_method(self):
        from shared.protocols.a2a_server import A2ATeamServer
        server = A2ATeamServer("IdeationTeam")
        resp = server.handle_request({
            "jsonrpc": "2.0", "method": "unknown/method",
            "params": {},
            "id": "r1",
        })
        assert resp["error"]["code"] == -32601

    def test_complete_task(self):
        from shared.protocols.a2a_server import A2ATeamServer
        server = A2ATeamServer("IdeationTeam")
        send_resp = server.handle_request({
            "jsonrpc": "2.0", "method": "tasks/send",
            "params": {"message": {}},
            "id": "r1",
        })
        task_id = send_resp["result"]["id"]

        server.complete_task(task_id, [{"type": "data", "data": {"questions": []}}])

        get_resp = server.handle_request({
            "jsonrpc": "2.0", "method": "tasks/get",
            "params": {"id": task_id},
            "id": "r2",
        })
        assert get_resp["result"]["state"] == "completed"
        assert len(get_resp["result"]["artifacts"]) == 1

    def test_fail_task(self):
        from shared.protocols.a2a_server import A2ATeamServer
        server = A2ATeamServer("IdeationTeam")
        send_resp = server.handle_request({
            "jsonrpc": "2.0", "method": "tasks/send",
            "params": {"message": {}},
            "id": "r1",
        })
        task_id = send_resp["result"]["id"]

        server.fail_task(task_id, "Stage 1 failed")

        get_resp = server.handle_request({
            "jsonrpc": "2.0", "method": "tasks/get",
            "params": {"id": task_id},
            "id": "r2",
        })
        assert get_resp["result"]["state"] == "failed"


# ============================================================================
# Test 3: A2A Data Models
# ============================================================================

class TestA2ADataModels:
    """A2A protocol data models."""

    def test_a2a_skill(self):
        from shared.protocols.a2a_server import A2ASkill
        skill = A2ASkill(id="test", name="Test Skill", description="Desc")
        assert skill.id == "test"

    def test_a2a_agent_card(self):
        from shared.protocols.a2a_server import A2AAgentCard
        card = A2AAgentCard(name="Test", description="Test agent", url="http://test")
        assert card.protocol_version == "0.3"

    def test_task_state_enum(self):
        from shared.protocols.a2a_server import TaskState
        assert TaskState.SUBMITTED == "submitted"
        assert TaskState.COMPLETED == "completed"
        assert TaskState.FAILED == "failed"
        assert TaskState.CANCELED == "canceled"

    def test_a2a_task(self):
        from shared.protocols.a2a_server import A2ATask, TaskState
        task = A2ATask()
        assert task.state == TaskState.SUBMITTED
        assert len(task.id) > 0

    def test_json_rpc_request(self):
        from shared.protocols.a2a_server import A2AJsonRpcRequest
        req = A2AJsonRpcRequest(method="tasks/send")
        assert req.jsonrpc == "2.0"

    def test_json_rpc_response(self):
        from shared.protocols.a2a_server import A2AJsonRpcResponse
        resp = A2AJsonRpcResponse(id="r1", result={"key": "val"})
        assert resp.error is None


# ============================================================================
# Test 4: A2A Client
# ============================================================================

class TestA2AClient:
    """A2A client module importability and structure."""

    def test_import(self):
        from shared.protocols.a2a_client import A2AClient
        assert A2AClient is not None

    def test_init(self):
        from shared.protocols.a2a_client import A2AClient
        client = A2AClient("http://localhost:8000/a2a/ideation")
        assert client.agent_url == "http://localhost:8000/a2a/ideation"

    def test_error_class(self):
        from shared.protocols.a2a_client import A2AClientError
        err = A2AClientError("test error", code=-32600)
        assert err.code == -32600

    def test_methods_exist(self):
        from shared.protocols.a2a_client import A2AClient
        client = A2AClient("http://test")
        assert hasattr(client, "get_agent_card")
        assert hasattr(client, "send_task")
        assert hasattr(client, "get_task")
        assert hasattr(client, "cancel_task")


# ============================================================================
# Test 5: MCP Server — Core
# ============================================================================

class TestMCPServerCore:
    """AELMCPServer initialization and tool listing."""

    def test_import(self):
        from shared.protocols.mcp_server import AELMCPServer
        assert AELMCPServer is not None

    def test_init(self):
        from shared.protocols.mcp_server import AELMCPServer
        server = AELMCPServer()
        assert server is not None

    def test_server_info(self):
        # V0.7 bumped MCPServerInfo.version to "0.7.0". Name +
        # capabilities contract unchanged.
        from shared.protocols.mcp_server import AELMCPServer
        server = AELMCPServer()
        info = server.get_server_info()
        assert info.name == "AEL-ToolServer"
        assert info.version == "0.7.0"
        assert "tools" in info.capabilities

    def test_list_tools(self):
        from shared.protocols.mcp_server import AELMCPServer
        from shared.tools.tool_registry import ToolRegistry
        from shared.tools.register_all import register_all_tools
        import shared.tools.register_all as reg_mod

        ToolRegistry.clear()
        reg_mod._registered = False
        register_all_tools()

        server = AELMCPServer()
        tools = server.list_tools()
        assert len(tools) >= 3
        tool_names = {t.name for t in tools}
        assert "arxiv_search" in tool_names

    def test_list_tools_mcp_format(self):
        from shared.protocols.mcp_server import AELMCPServer
        server = AELMCPServer()
        tools = server.list_tools()
        for tool in tools:
            assert hasattr(tool, "name")
            assert hasattr(tool, "description")
            assert hasattr(tool, "inputSchema")


# ============================================================================
# Test 6: MCP Server — JSON-RPC Handling
# ============================================================================

class TestMCPServerRPC:
    """MCP server JSON-RPC request handling."""

    def test_initialize(self):
        from shared.protocols.mcp_server import AELMCPServer
        server = AELMCPServer()
        resp = server.handle_request({
            "jsonrpc": "2.0", "method": "initialize", "id": "r1"
        })
        assert resp["jsonrpc"] == "2.0"
        assert resp["result"]["name"] == "AEL-ToolServer"

    def test_tools_list(self):
        from shared.protocols.mcp_server import AELMCPServer
        server = AELMCPServer()
        resp = server.handle_request({
            "jsonrpc": "2.0", "method": "tools/list", "id": "r1"
        })
        assert "tools" in resp["result"]
        assert isinstance(resp["result"]["tools"], list)

    def test_tools_call_nonexistent(self):
        from shared.protocols.mcp_server import AELMCPServer
        server = AELMCPServer()
        resp = server.handle_request({
            "jsonrpc": "2.0", "method": "tools/call",
            "params": {"name": "nonexistent_tool", "arguments": {}},
            "id": "r1",
        })
        result = resp["result"]
        assert result["isError"] is True

    def test_tools_call_missing_name(self):
        from shared.protocols.mcp_server import AELMCPServer
        server = AELMCPServer()
        resp = server.handle_request({
            "jsonrpc": "2.0", "method": "tools/call",
            "params": {},
            "id": "r1",
        })
        assert "error" in resp
        assert resp["error"]["code"] == -32602

    def test_unknown_method(self):
        from shared.protocols.mcp_server import AELMCPServer
        server = AELMCPServer()
        resp = server.handle_request({
            "jsonrpc": "2.0", "method": "resources/list", "id": "r1"
        })
        assert "error" in resp


# ============================================================================
# Test 7: MCP Client
# ============================================================================

class TestMCPClient:
    """MCP client module importability and structure."""

    def test_import(self):
        from shared.protocols.mcp_client import MCPClient
        assert MCPClient is not None

    def test_init(self):
        from shared.protocols.mcp_client import MCPClient
        client = MCPClient("http://localhost:8080")
        assert client.server_url == "http://localhost:8080"

    def test_error_class(self):
        from shared.protocols.mcp_client import MCPClientError
        err = MCPClientError("test", code=-1)
        assert err.code == -1

    def test_methods_exist(self):
        from shared.protocols.mcp_client import MCPClient
        client = MCPClient("http://test")
        assert hasattr(client, "initialize")
        assert hasattr(client, "list_tools")
        assert hasattr(client, "call_tool")
        assert hasattr(client, "get_tool_schema")

    def test_tools_cache(self):
        from shared.protocols.mcp_client import MCPClient
        client = MCPClient("http://test")
        assert client._tools_cache is None


# ============================================================================
# Test 8: A2A Agent Cards for All 4 Teams
# ============================================================================

class TestA2AAgentCards:
    """A2A agent card generation from TeamCards."""

    def test_get_a2a_card_ideation(self):
        from shared.protocols.agent_card import get_a2a_agent_card
        card = get_a2a_agent_card("IdeationTeam")
        assert card is not None
        assert card["name"] == "AEL-IdeationTeam"
        assert card["protocolVersion"] == "0.3"
        assert len(card["skills"]) == 3

    def test_get_a2a_card_literature(self):
        from shared.protocols.agent_card import get_a2a_agent_card
        card = get_a2a_agent_card("LiteratureTeam")
        assert card is not None
        assert card["name"] == "AEL-LiteratureTeam"

    def test_get_a2a_card_model(self):
        from shared.protocols.agent_card import get_a2a_agent_card
        card = get_a2a_agent_card("ModelTeam")
        assert card is not None
        assert card["name"] == "AEL-ModelTeam"

    def test_get_a2a_card_data(self):
        from shared.protocols.agent_card import get_a2a_agent_card
        card = get_a2a_agent_card("DataTeam")
        assert card is not None
        assert card["name"] == "AEL-DataTeam"

    def test_a2a_card_has_schemas(self):
        from shared.protocols.agent_card import get_a2a_agent_card
        card = get_a2a_agent_card("IdeationTeam")
        assert "inputSchema" in card
        assert "outputSchema" in card

    def test_a2a_card_nonexistent_team(self):
        from shared.protocols.agent_card import get_a2a_agent_card
        card = get_a2a_agent_card("NonexistentTeam")
        assert card is None

    def test_all_4_teams_have_a2a_cards(self):
        from shared.protocols.agent_card import get_a2a_agent_card
        for team in ["IdeationTeam", "LiteratureTeam", "ModelTeam", "DataTeam"]:
            card = get_a2a_agent_card(team)
            assert card is not None, f"{team} missing A2A card"
            assert "skills" in card
            assert "version" in card


# ============================================================================
# Test 9: MessageBus — A2A Serialization
# ============================================================================

class TestMessageBusA2A:
    """MessageBus A2A message serialization/deserialization."""

    def test_to_a2a_messages_empty(self):
        from shared.protocols.message_bus import MessageBus
        bus = MessageBus(pipeline_run_id="test-run")
        msgs = bus.to_a2a_messages()
        assert msgs == []

    def test_to_a2a_messages(self):
        from shared.protocols.message_bus import MessageBus, ArtifactMessage
        bus = MessageBus(pipeline_run_id="test-run")
        bus.publish(ArtifactMessage(
            source_team="IdeationTeam",
            target_team="LiteratureTeam",
            artifact_name="research_questions",
            artifact_data={"questions": ["Q1", "Q2"]},
            schema_name="IntegrationStageOutput",
        ))
        msgs = bus.to_a2a_messages()
        assert len(msgs) == 1
        assert msgs[0]["jsonrpc"] == "2.0"
        assert msgs[0]["method"] == "tasks/send"
        assert msgs[0]["params"]["metadata"]["source_team"] == "IdeationTeam"

    def test_from_a2a_message(self):
        from shared.protocols.message_bus import MessageBus, ArtifactMessage
        a2a_msg = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {
                "message": {
                    "role": "agent",
                    "parts": [{"type": "data", "data": {"questions": ["Q1"]}}],
                },
                "metadata": {
                    "source_team": "IdeationTeam",
                    "target_team": "LiteratureTeam",
                    "artifact_name": "research_questions",
                    "schema_name": "IntegrationStageOutput",
                    "pipeline_run_id": "run-001",
                },
            },
            "id": "msg-001",
        }
        artifact = MessageBus.from_a2a_message(a2a_msg)
        assert artifact.source_team == "IdeationTeam"
        assert artifact.target_team == "LiteratureTeam"
        assert artifact.artifact_data == {"questions": ["Q1"]}

    def test_roundtrip_a2a_serialization(self):
        """Publish → to_a2a → from_a2a preserves data."""
        from shared.protocols.message_bus import MessageBus, ArtifactMessage
        bus = MessageBus(pipeline_run_id="test")
        original_data = {"key": "value", "nested": {"a": 1}}
        bus.publish(ArtifactMessage(
            source_team="ModelTeam",
            target_team="DataTeam",
            artifact_name="model_spec",
            artifact_data=original_data,
        ))
        a2a_msgs = bus.to_a2a_messages()
        recovered = MessageBus.from_a2a_message(a2a_msgs[0])
        assert recovered.artifact_data == original_data
        assert recovered.source_team == "ModelTeam"


# ============================================================================
# Test 10: MCP Server — Tool Invocation
# ============================================================================

class TestMCPServerInvocation:
    """MCP server tool invocation via call_tool."""

    def test_call_tool_nonexistent(self):
        from shared.protocols.mcp_server import AELMCPServer
        server = AELMCPServer()
        result = server.call_tool("nonexistent", {})
        assert result.isError is True

    def test_call_tool_arxiv_no_network(self):
        """arxiv_search may fail without network, but should return MCPToolResult."""
        from shared.protocols.mcp_server import AELMCPServer
        from shared.tools.register_all import register_all_tools
        import shared.tools.register_all as reg_mod
        from shared.tools.tool_registry import ToolRegistry

        ToolRegistry.clear()
        reg_mod._registered = False
        register_all_tools()

        server = AELMCPServer()
        result = server.call_tool("arxiv_search", {"query": "test", "max_results": 1})
        # May succeed or fail depending on network, but should be MCPToolResult
        assert hasattr(result, "isError")
        assert hasattr(result, "content")


# ============================================================================
# Test 11: MCP Data Models
# ============================================================================

class TestMCPDataModels:
    """MCP protocol data models."""

    def test_tool_description(self):
        from shared.protocols.mcp_server import MCPToolDescription
        td = MCPToolDescription(name="test", description="A test tool")
        assert td.name == "test"
        assert isinstance(td.inputSchema, dict)

    def test_tool_result(self):
        from shared.protocols.mcp_server import MCPToolResult
        tr = MCPToolResult(content=[{"type": "text", "text": "hello"}])
        assert tr.isError is False

    def test_server_info(self):
        from shared.protocols.mcp_server import MCPServerInfo
        info = MCPServerInfo()
        assert info.protocolVersion == "2024-11-05"


# ============================================================================
# Test 12: Team Skills
# ============================================================================

class TestTeamSkills:
    """A2A team skills definitions."""

    def test_all_4_teams_have_skills(self):
        from shared.protocols.a2a_server import TEAM_SKILLS
        for team in ["IdeationTeam", "LiteratureTeam", "ModelTeam", "DataTeam"]:
            assert team in TEAM_SKILLS, f"{team} missing skills"
            assert len(TEAM_SKILLS[team]) >= 3

    def test_skills_have_required_fields(self):
        from shared.protocols.a2a_server import TEAM_SKILLS
        for team, skills in TEAM_SKILLS.items():
            for skill in skills:
                assert skill.id, f"{team} skill missing id"
                assert skill.name, f"{team} skill missing name"
                assert skill.description, f"{team} skill missing description"


# ============================================================================
# Test 13: Backward Compatibility
# ============================================================================

class TestProtocolBackwardCompat:
    """Existing protocol imports and functions still work."""

    def test_teamcard_import(self):
        from shared.protocols import TeamCard, get_team_card, list_team_cards
        assert TeamCard is not None
        assert callable(get_team_card)
        assert callable(list_team_cards)

    def test_team_cards_still_work(self):
        from shared.protocols.agent_card import list_team_cards
        cards = list_team_cards()
        assert len(cards) == 4

    def test_message_bus_still_works(self):
        from shared.protocols.message_bus import MessageBus, ArtifactMessage
        bus = MessageBus(pipeline_run_id="test")
        bus.publish(ArtifactMessage(
            source_team="Test", artifact_name="test", artifact_data={"a": 1},
        ))
        assert bus.message_count() == 1

    def test_get_pipeline_order(self):
        from shared.protocols.agent_card import get_pipeline_order
        order = get_pipeline_order()
        assert order == ["IdeationTeam", "LiteratureTeam", "ModelTeam", "DataTeam"]

    def test_get_team_dependencies(self):
        from shared.protocols.agent_card import get_team_dependencies
        deps = get_team_dependencies("LiteratureTeam")
        assert "IdeationTeam" in deps["upstream"]
