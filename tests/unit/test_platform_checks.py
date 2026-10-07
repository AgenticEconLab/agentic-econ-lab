# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Platform modernization tests."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Qwen routing collision
# ---------------------------------------------------------------------------

class TestQwenRouting:
    def test_qwen35_routes_to_qwen(self):
        from shared.provider_map import detect_provider
        assert detect_provider("qwen3.5-plus") == "qwen"
        assert detect_provider("qwen3.5-flash") == "qwen"
        assert detect_provider("qwen3.5-397b-a17b") == "qwen"

    def test_qwen3_legacy_still_openrouter(self):
        from shared.provider_map import detect_provider
        assert detect_provider("qwen3-8b") == "openrouter"
        assert detect_provider("qwen3-32b") == "openrouter"


# ---------------------------------------------------------------------------
# ConstrainedLLM retries issue fresh LLM calls
# ---------------------------------------------------------------------------

class _Schema(BaseModel):
    name: str
    count: int = Field(ge=0)


class TestConstrainedLLMRetry:
    def test_second_call_on_semantic_failure(self):
        from shared.llm_helpers import ConstrainedLLM
        calls: list[str] = []

        class Stub:
            def __init__(self, responses):
                self.responses = list(responses)
            def invoke(self, prompt, schema=None, json_mode=None):
                calls.append(prompt)
                return self.responses.pop(0)

        stub = Stub([
            '{"name": "x", "count": -1}',   # violates ge=0
            '{"name": "x", "count": 5}',    # valid on retry
        ])
        out = ConstrainedLLM(stub).invoke("p", _Schema, max_retries=1)
        assert out.count == 5
        # Must have issued two distinct LLM calls
        assert len(calls) == 2
        # Second prompt should include the prior validation error
        assert "failed" in calls[1].lower()

    def test_exhausted_retries_raises(self):
        from shared.llm_helpers import ConstrainedLLM, ConstrainedValidationError

        class Stub:
            def invoke(self, prompt, schema=None, json_mode=None):
                return '{"name": "x", "count": -5}'
        with pytest.raises(ConstrainedValidationError):
            ConstrainedLLM(Stub()).invoke("p", _Schema, max_retries=1)


# ---------------------------------------------------------------------------
# Compaction uses role=user for summary
# ---------------------------------------------------------------------------

class TestCompactionRoleFix:
    def test_summary_is_user_role(self):
        from shared.llm_helpers import compact_conversation
        msgs = [{"role": "system", "content": "sys"}]
        msgs += [{"role": "user", "content": "x" * 300} for _ in range(20)]
        result = compact_conversation(msgs, budget_tokens=100)
        # Find the compaction-summary message (contains "compaction summary")
        summary_msgs = [m for m in result.messages if "compaction summary" in m.get("content", "").lower()]
        assert summary_msgs, "expected a compaction-summary message"
        # The summary must NOT be a second system message
        for m in summary_msgs:
            assert m["role"] != "system"
            assert m["role"] == "user"


# ---------------------------------------------------------------------------
# MCP OAuth 2.1 + Tasks + .well-known
# ---------------------------------------------------------------------------

class TestMCPOAuthIntegration:
    def test_server_without_authorizer_is_legacy(self):
        from shared.protocols.mcp_server import AELMCPServer
        server = AELMCPServer()
        # Legacy client with no Authorization header should work
        resp = server.handle_request({"jsonrpc": "2.0", "method": "tools/list", "id": "1"})
        assert "result" in resp

    def test_server_with_authorizer_rejects_missing_token(self):
        from shared.protocols.mcp_server import AELMCPServer
        from shared.protocols.mcp_v07_extensions import (
            OAuth21Authorizer, OAuth21ResourceMetadata,
        )
        auth = OAuth21Authorizer(
            metadata=OAuth21ResourceMetadata("r", ["https://auth.example.com"]),
            allowed_tokens=["t1"],
        )
        server = AELMCPServer(authorizer=auth)
        resp = server.handle_request(
            {"jsonrpc": "2.0", "method": "tools/list", "id": "1"},
        )
        assert "error" in resp
        assert resp["error"]["code"] == -32001

    def test_server_with_authorizer_accepts_valid_bearer(self):
        from shared.protocols.mcp_server import AELMCPServer
        from shared.protocols.mcp_v07_extensions import (
            OAuth21Authorizer, OAuth21ResourceMetadata,
        )
        auth = OAuth21Authorizer(
            metadata=OAuth21ResourceMetadata("r", ["https://auth.example.com"]),
            allowed_tokens=["t1"],
        )
        server = AELMCPServer(authorizer=auth)
        resp = server.handle_request(
            {"jsonrpc": "2.0", "method": "tools/list", "id": "1"},
            headers={"Authorization": "Bearer t1"},
        )
        assert "result" in resp

    def test_initialize_bypasses_auth(self):
        from shared.protocols.mcp_server import AELMCPServer
        from shared.protocols.mcp_v07_extensions import (
            OAuth21Authorizer, OAuth21ResourceMetadata,
        )
        auth = OAuth21Authorizer(
            metadata=OAuth21ResourceMetadata("r", ["https://auth.example.com"]),
            allowed_tokens=["t1"],
        )
        server = AELMCPServer(authorizer=auth)
        resp = server.handle_request(
            {"jsonrpc": "2.0", "method": "initialize", "id": "1"},
        )
        assert "result" in resp

    def test_well_known_payload(self):
        from shared.protocols.mcp_server import AELMCPServer
        from shared.protocols.mcp_v07_extensions import (
            OAuth21Authorizer, OAuth21ResourceMetadata,
        )
        auth = OAuth21Authorizer(
            metadata=OAuth21ResourceMetadata(
                "https://ael/mcp", ["https://auth.example.com"],
                scopes_supported=["mcp:read"],
            ),
        )
        server = AELMCPServer(authorizer=auth)
        payload = server.well_known_oauth_protected_resource()
        assert payload["resource"] == "https://ael/mcp"
        assert payload["scopes_supported"] == ["mcp:read"]

    def test_well_known_no_authorizer_returns_empty(self):
        from shared.protocols.mcp_server import AELMCPServer
        assert AELMCPServer().well_known_oauth_protected_resource() == {}

    def test_tasks_create_and_status(self):
        from shared.protocols.mcp_server import AELMCPServer
        from shared.protocols.mcp_v07_extensions import MCPTaskRegistry
        reg = MCPTaskRegistry()
        server = AELMCPServer(task_registry=reg)
        # Create task
        resp = server.handle_request({
            "jsonrpc": "2.0", "method": "tasks/create", "id": "1",
            "params": {"tool_name": "echo", "arguments": {"x": 1}},
        })
        assert "result" in resp
        tid = resp["result"]["id"]
        # Status
        status_resp = server.handle_request({
            "jsonrpc": "2.0", "method": "tasks/status", "id": "2",
            "params": {"task_id": tid},
        })
        assert status_resp["result"]["id"] == tid

    def test_tasks_without_registry_returns_method_not_found(self):
        from shared.protocols.mcp_server import AELMCPServer
        server = AELMCPServer()  # no task registry
        resp = server.handle_request({
            "jsonrpc": "2.0", "method": "tasks/create", "id": "1",
            "params": {"tool_name": "x"},
        })
        assert "error" in resp
        assert resp["error"]["code"] == -32601
