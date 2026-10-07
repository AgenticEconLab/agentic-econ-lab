# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
MCP-Aligned Tool Registry — Discover and invoke tools through a standard interface.

Inspired by the Model Context Protocol (MCP), this registry provides:
- Registration: Tools declare name, description, input schema, handler
- Discovery: Agents list available tools and their schemas
- Invocation: Agents invoke tools with validated inputs and auto-observability

Usage:
    from shared.tools import ToolRegistry

    # List available tools
    tools = ToolRegistry.list_tools()

    # Invoke a tool
    result = ToolRegistry.invoke("arxiv_search", {"query": "DSGE models"}, collector=collector)
    if result.success:
        papers = result.data
"""

import time
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel, Field, ValidationError

from shared.tools.redact import redact_secrets


class ToolSpec(BaseModel):
    """Specification for a registered tool."""
    model_config = {"arbitrary_types_allowed": True}

    name: str = Field(description="Unique tool name")
    description: str = Field(description="Human-readable description")
    input_schema: Any = Field(description="Pydantic model class for input validation")
    handler: Any = Field(description="Callable handler function", exclude=True)
    category: str = Field(default="general", description="Tool category: search, data, web, compute")


class ToolResult(BaseModel):
    """Result of a tool invocation."""
    success: bool = Field(description="Whether invocation succeeded")
    data: Any = Field(default=None, description="Result data on success")
    error: Optional[str] = Field(default=None, description="Error message on failure")
    latency_ms: float = Field(default=0.0, description="Invocation latency in milliseconds")
    tool_name: str = Field(default="", description="Name of the invoked tool")


class ToolRegistry:
    """
    MCP-inspired tool registry for agent tool discovery and invocation.

    Tools are registered globally (class-level) and available to all agents.
    Each invocation validates inputs via Pydantic and optionally records
    metrics to a MetricsCollector.
    """

    _tools: Dict[str, ToolSpec] = {}

    @classmethod
    def register(
        cls,
        name: str,
        description: str,
        input_schema: Type[BaseModel],
        handler: Callable,
        category: str = "general",
    ):
        """
        Register a tool in the registry.

        Args:
            name: Unique tool name (e.g., "arxiv_search").
            description: Human-readable description.
            input_schema: Pydantic model class for input validation.
            handler: Callable that implements the tool logic.
            category: Tool category for grouping.
        """
        cls._tools[name] = ToolSpec(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler,
            category=category,
        )

    @classmethod
    def unregister(cls, name: str):
        """Remove a tool from the registry."""
        cls._tools.pop(name, None)

    @classmethod
    def invoke(
        cls,
        name: str,
        params: Dict[str, Any],
        collector: Any = None,
        agent: str = "",
    ) -> ToolResult:
        """
        Invoke a registered tool with automatic input validation and observability.

        Args:
            name: Tool name to invoke.
            params: Input parameters (validated against tool's input_schema).
            collector: Optional MetricsCollector for observability tracking.
            agent: Agent name for attribution.

        Returns:
            ToolResult with success/data or error details.
        """
        if name not in cls._tools:
            return ToolResult(
                success=False,
                error=f"Tool '{name}' not found. Available: {list(cls._tools.keys())}",
                tool_name=name,
            )

        tool = cls._tools[name]

        # Validate input
        try:
            validated_input = tool.input_schema.model_validate(params)
        except ValidationError as e:
            return ToolResult(
                success=False,
                error=f"Input validation failed: {e}",
                tool_name=name,
            )

        # Invoke handler — pass collector/agent through for nested tracking
        start = time.time()
        try:
            result = tool.handler(
                **validated_input.model_dump(), collector=collector, agent=agent
            )
            latency_ms = (time.time() - start) * 1000

            # Record to MetricsCollector
            if collector is not None:
                cls._record_tool_call(collector, name, agent, latency_ms, success=True)

            return ToolResult(
                success=True,
                data=result,
                latency_ms=round(latency_ms, 2),
                tool_name=name,
            )
        except Exception as e:
            latency_ms = (time.time() - start) * 1000

            if collector is not None:
                cls._record_tool_call(
                    collector, name, agent, latency_ms, success=False, error=redact_secrets(e)
                )

            return ToolResult(
                success=False,
                error=redact_secrets(e),  # no credentials in logged errors
                latency_ms=round(latency_ms, 2),
                tool_name=name,
            )

    @classmethod
    def list_tools(cls, category: Optional[str] = None) -> List[ToolSpec]:
        """
        List all registered tools (MCP-style discovery).

        Args:
            category: Optional filter by category.

        Returns:
            List of ToolSpec objects.
        """
        tools = list(cls._tools.values())
        if category:
            tools = [t for t in tools if t.category == category]
        return tools

    @classmethod
    def get_tool(cls, name: str) -> Optional[ToolSpec]:
        """Get a specific tool's spec."""
        return cls._tools.get(name)

    @classmethod
    def tool_names(cls) -> List[str]:
        """Return list of registered tool names."""
        return list(cls._tools.keys())

    @classmethod
    def search(cls, query: str, max_tools: int = 5) -> List[ToolSpec]:
        """Dynamic tool discovery — return relevant tools for a task query.

        Uses keyword matching against tool names and descriptions to find
        the most relevant tools, reducing token usage (V0.6 Phase 5).

        Args:
            query: Natural language task description.
            max_tools: Maximum tools to return.

        Returns:
            List of ToolSpec sorted by relevance.
        """
        if not query or not cls._tools:
            return []

        query_words = set(query.lower().split())
        scored: List[tuple] = []

        for tool in cls._tools.values():
            # Score by keyword overlap in name + description
            tool_words = set(tool.name.lower().replace("_", " ").split())
            tool_words |= set(tool.description.lower().split())
            overlap = len(query_words & tool_words)
            if overlap > 0:
                scored.append((overlap, tool))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:max_tools]]

    @classmethod
    def get_stage_tools(
        cls, team: str, stage: str, task_description: str = "", max_tools: int = 10
    ) -> List[ToolSpec]:
        """Get tools relevant to a specific team/stage context.

        Combines category filtering with search-based relevance (V0.6 Phase 5).

        Args:
            team: Team name (e.g., "IdeationTeam").
            stage: Stage name (e.g., "SourcingStage").
            task_description: Optional task context for relevance ranking.
            max_tools: Maximum tools to return.

        Returns:
            List of relevant ToolSpec objects.
        """
        # Map teams to likely categories
        team_categories = {
            "IdeationTeam": ["search", "web"],
            "LiteratureTeam": ["search", "data"],
            "DataTeam": ["data", "compute"],
            "ModelTeam": ["compute", "data"],
        }
        categories = team_categories.get(team, [])

        # Get category-matched tools
        category_tools = []
        for tool in cls._tools.values():
            if tool.category in categories:
                category_tools.append(tool)

        # If task description provided, also search by relevance
        if task_description:
            search_results = cls.search(task_description, max_tools=max_tools)
            # Merge: category tools first, then search results (dedup)
            seen = {t.name for t in category_tools}
            for t in search_results:
                if t.name not in seen:
                    category_tools.append(t)
                    seen.add(t.name)

        return category_tools[:max_tools]

    @classmethod
    def lazy_load(cls, tool_id: str) -> Optional[ToolSpec]:
        """Load a tool only when needed (deferred initialization).

        Returns the tool spec if registered, None otherwise.
        Useful for stages that declare tool dependencies but don't
        load all tools at initialization time (V0.6 Phase 5).

        Args:
            tool_id: Tool name to load.

        Returns:
            ToolSpec if found, None otherwise.
        """
        return cls._tools.get(tool_id)

    @classmethod
    def clear(cls):
        """Clear all registered tools. Useful for testing."""
        cls._tools.clear()

    @staticmethod
    def _record_tool_call(
        collector: Any,
        tool_name: str,
        agent: str,
        latency_ms: float,
        success: bool,
        error: Optional[str] = None,
    ):
        """Record a tool call to the MetricsCollector."""
        try:
            from shared.observability import ToolCallRecord
            record = ToolCallRecord(
                agent=agent,
                tool_name=tool_name,
                latency_seconds=latency_ms / 1000.0,
                success=success,
                error=error,
            )
            collector.record_tool_call(record)
        except Exception as e:
            import logging
            logging.getLogger("ael.tools").debug("Tool call recording failed: %s", e)
