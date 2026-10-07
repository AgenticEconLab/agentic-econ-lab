# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Pydantic models for observability data.

These schemas define the type-safe structure for observability metrics
captured by the MetricsCollector and stored in execution_log.json.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class LLMCallSchema(BaseModel):
    """Schema for a single LLM API call record."""
    agent: str = Field(default="", description="Agent that made this call")
    stage: str = Field(default="", description="Stage during which this call was made")
    model: str = Field(default="", description="LLM model name")
    prompt_tokens: int = Field(default=0, description="Number of prompt tokens")
    completion_tokens: int = Field(default=0, description="Number of completion tokens")
    total_tokens: int = Field(default=0, description="Total tokens used")
    cost_usd: float = Field(default=0.0, description="Estimated cost in USD")
    latency_seconds: float = Field(default=0.0, description="Call latency in seconds")
    timestamp: str = Field(default="", description="ISO-format timestamp")
    error: Optional[str] = Field(default=None, description="Error message if call failed")


class ToolCallSchema(BaseModel):
    """Schema for a single external tool/API call record."""
    agent: str = Field(default="", description="Agent that made this call")
    stage: str = Field(default="", description="Stage during which this call was made")
    tool_name: str = Field(default="", description="Human-readable tool name")
    url: str = Field(default="", description="Request URL")
    http_method: str = Field(default="GET", description="HTTP method")
    status_code: Optional[int] = Field(default=None, description="HTTP status code")
    latency_seconds: float = Field(default=0.0, description="Call latency in seconds")
    success: bool = Field(default=True, description="Whether the call succeeded")
    timestamp: str = Field(default="", description="ISO-format timestamp")
    error: Optional[str] = Field(default=None, description="Error message if call failed")


class LLMSummary(BaseModel):
    """Aggregated LLM usage summary."""
    total_calls: int = Field(default=0, description="Total number of LLM calls")
    total_prompt_tokens: int = Field(default=0, description="Total prompt tokens")
    total_completion_tokens: int = Field(default=0, description="Total completion tokens")
    total_tokens: int = Field(default=0, description="Total tokens")
    total_cost_usd: float = Field(default=0.0, description="Total cost in USD")
    avg_latency_seconds: float = Field(default=0.0, description="Average call latency")
    error_count: int = Field(default=0, description="Number of failed calls")


class ToolSummary(BaseModel):
    """Aggregated tool/API usage summary."""
    total_calls: int = Field(default=0, description="Total number of tool calls")
    success_count: int = Field(default=0, description="Number of successful calls")
    error_count: int = Field(default=0, description="Number of failed calls")
    success_rate: float = Field(default=0.0, description="Success rate (0-1)")
    avg_latency_seconds: float = Field(default=0.0, description="Average call latency")


class AgentSummary(BaseModel):
    """Per-agent observability summary."""
    llm: LLMSummary = Field(default_factory=LLMSummary)
    tools: ToolSummary = Field(default_factory=ToolSummary)


class ObservabilityReport(BaseModel):
    """
    Top-level observability report embedded in execution_log.json.

    This is the Pydantic equivalent of MetricsCollector.get_summary().
    """
    schema_version: str = Field(default="1.0.0", description="Observability schema version")
    llm: LLMSummary = Field(default_factory=LLMSummary, description="Overall LLM usage summary")
    tools: ToolSummary = Field(default_factory=ToolSummary, description="Overall tool usage summary")
    by_agent: Dict[str, AgentSummary] = Field(
        default_factory=dict,
        description="Per-agent breakdown of LLM and tool usage"
    )
    by_stage: Dict[str, AgentSummary] = Field(
        default_factory=dict,
        description="Per-stage breakdown of LLM and tool usage"
    )
