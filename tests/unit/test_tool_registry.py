# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Tests for shared.tools.tool_registry.ToolRegistry.

Validates that:
- Tools can be registered and discovered
- Input validation works
- Invocation succeeds with valid inputs
- Invocation fails gracefully with invalid inputs
- Observability recording works
- Unknown tool names return errors
"""

import pytest
from pydantic import BaseModel, Field

from shared.tools.tool_registry import ToolRegistry, ToolSpec, ToolResult


# ============================================================================
# Test tool definitions
# ============================================================================

class AddInput(BaseModel):
    """Input for a simple addition tool."""
    a: int = Field(description="First number")
    b: int = Field(description="Second number")


def add_handler(a: int, b: int, **kwargs) -> int:
    """Add two numbers."""
    return a + b


class FailInput(BaseModel):
    """Input for a tool that always fails."""
    message: str = Field(description="Error message")


def fail_handler(message: str, **kwargs):
    """Always raises an exception."""
    raise RuntimeError(message)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def clean_registry():
    """Clear the registry before and after each test."""
    ToolRegistry.clear()
    yield
    ToolRegistry.clear()


# ============================================================================
# Tests
# ============================================================================

class TestRegistration:
    """Test tool registration and discovery."""

    def test_register_tool(self):
        ToolRegistry.register(
            name="add",
            description="Add two numbers",
            input_schema=AddInput,
            handler=add_handler,
        )
        assert "add" in ToolRegistry.tool_names()

    def test_list_tools_empty(self):
        assert ToolRegistry.list_tools() == []

    def test_list_tools(self):
        ToolRegistry.register("add", "Add", AddInput, add_handler, category="math")
        tools = ToolRegistry.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "add"
        assert tools[0].category == "math"

    def test_list_tools_filter_by_category(self):
        ToolRegistry.register("add", "Add", AddInput, add_handler, category="math")
        ToolRegistry.register("fail", "Fail", FailInput, fail_handler, category="test")
        assert len(ToolRegistry.list_tools(category="math")) == 1
        assert len(ToolRegistry.list_tools(category="test")) == 1
        assert len(ToolRegistry.list_tools(category="other")) == 0

    def test_get_tool(self):
        ToolRegistry.register("add", "Add", AddInput, add_handler)
        spec = ToolRegistry.get_tool("add")
        assert spec is not None
        assert spec.description == "Add"

    def test_get_unknown_tool(self):
        assert ToolRegistry.get_tool("nonexistent") is None

    def test_unregister(self):
        ToolRegistry.register("add", "Add", AddInput, add_handler)
        ToolRegistry.unregister("add")
        assert "add" not in ToolRegistry.tool_names()


class TestInvocation:
    """Test tool invocation."""

    def test_invoke_success(self):
        ToolRegistry.register("add", "Add", AddInput, add_handler)
        result = ToolRegistry.invoke("add", {"a": 3, "b": 4})
        assert result.success
        assert result.data == 7
        assert result.latency_ms >= 0  # May be 0.0 for trivial handlers

    def test_invoke_unknown_tool(self):
        result = ToolRegistry.invoke("nonexistent", {})
        assert not result.success
        assert "not found" in result.error

    def test_invoke_invalid_input(self):
        ToolRegistry.register("add", "Add", AddInput, add_handler)
        result = ToolRegistry.invoke("add", {"a": "not_a_number", "b": 4})
        assert not result.success
        assert "validation failed" in result.error.lower()

    def test_invoke_missing_required_field(self):
        ToolRegistry.register("add", "Add", AddInput, add_handler)
        result = ToolRegistry.invoke("add", {"a": 3})
        assert not result.success

    def test_invoke_handler_exception(self):
        ToolRegistry.register("fail", "Fail", FailInput, fail_handler)
        result = ToolRegistry.invoke("fail", {"message": "oops"})
        assert not result.success
        assert "oops" in result.error


class TestObservability:
    """Test observability recording during invocation."""

    def test_invoke_records_to_collector(self):
        """Verify that successful invocations record a ToolCallRecord."""
        ToolRegistry.register("add", "Add", AddInput, add_handler)

        # Simple mock collector
        recorded_calls = []

        class MockCollector:
            def record_tool_call(self, record):
                recorded_calls.append(record)

        collector = MockCollector()
        result = ToolRegistry.invoke("add", {"a": 1, "b": 2}, collector=collector, agent="TestAgent")

        assert result.success
        assert len(recorded_calls) == 1
        record = recorded_calls[0]
        assert record.tool_name == "add"
        assert record.agent == "TestAgent"
        assert record.success is True

    def test_invoke_records_failure_to_collector(self):
        ToolRegistry.register("fail", "Fail", FailInput, fail_handler)

        recorded_calls = []

        class MockCollector:
            def record_tool_call(self, record):
                recorded_calls.append(record)

        collector = MockCollector()
        result = ToolRegistry.invoke("fail", {"message": "boom"}, collector=collector)

        assert not result.success
        assert len(recorded_calls) == 1
        assert recorded_calls[0].success is False

    def test_invoke_without_collector(self):
        """Invocation works fine without a collector."""
        ToolRegistry.register("add", "Add", AddInput, add_handler)
        result = ToolRegistry.invoke("add", {"a": 1, "b": 2})
        assert result.success
        assert result.data == 3
