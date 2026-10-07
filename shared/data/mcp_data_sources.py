# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
MCP Data Source Registry — Unified configuration for all MCP-based data sources (V0.6 Phase 2).

Provides a central registry of financial and research MCP server configurations
with automatic fallback to direct API when MCP is unavailable.

Usage:
    from shared.data.mcp_data_sources import MCPDataSourceRegistry

    registry = MCPDataSourceRegistry()
    sources = registry.list_sources(workflow="open_source_api")
    client = registry.get_client("alpha_vantage")
"""

import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MCPDataSourceConfig(BaseModel):
    """Configuration for a single MCP data source."""

    name: str
    server_url: str = ""
    env_key: str = ""
    enabled_env: str = ""
    workflow: str = "all"
    description: str = ""
    capabilities: List[str] = Field(default_factory=list)
    fallback_type: str = ""  # "rest_api", "sdk", or "" (no fallback)
    fallback_url: str = ""
    tools: List[Dict[str, Any]] = Field(default_factory=list)


# Pre-defined MCP data sources
MCP_DATA_SOURCES: Dict[str, MCPDataSourceConfig] = {
    "alpha_vantage": MCPDataSourceConfig(
        name="alpha_vantage",
        server_url=os.environ.get("ALPHA_VANTAGE_MCP_URL", "http://localhost:8081/mcp"),
        env_key="ALPHA_VANTAGE_API_KEY",
        enabled_env="ALPHA_VANTAGE_MCP_ENABLED",
        workflow="open_source_api",
        description="GDP, CPI, Treasury yields, unemployment, 50+ technical indicators",
        capabilities=["macro_indicators", "technical_indicators", "forex", "commodities"],
        fallback_type="rest_api",
        fallback_url="https://www.alphavantage.co/query",
        tools=[
            {
                "name": "av_get_macro_indicator",
                "description": "Fetch macro indicator (GDP, CPI, TREASURY_YIELD, etc.)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "function": {"type": "string"},
                        "interval": {"type": "string", "default": "annual"},
                    },
                    "required": ["function"],
                },
            },
            {
                "name": "av_get_technical",
                "description": "Fetch technical indicator for a symbol",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "function": {"type": "string"},
                        "symbol": {"type": "string"},
                        "interval": {"type": "string"},
                    },
                    "required": ["function", "symbol"],
                },
            },
        ],
    ),
    "openbb": MCPDataSourceConfig(
        name="openbb",
        server_url=os.environ.get("OPENBB_MCP_URL", "http://localhost:8082/mcp"),
        env_key="OPENBB_TOKEN",
        enabled_env="OPENBB_MCP_ENABLED",
        workflow="all",
        description="350+ datasets — financial statements, market data, economic indicators",
        capabilities=["equity", "economy", "fixedincome", "crypto", "etf"],
        fallback_type="sdk",
        tools=[
            {
                "name": "openbb_economy_indicators",
                "description": "Fetch economic indicators (GDP, CPI, unemployment, etc.)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "provider": {"type": "string", "default": "fred"},
                    },
                    "required": ["symbol"],
                },
            },
        ],
    ),
    "financial_datasets": MCPDataSourceConfig(
        name="financial_datasets",
        server_url=os.environ.get("FINANCIAL_DATASETS_MCP_URL", "http://localhost:8083/mcp"),
        env_key="FINANCIAL_DATASETS_API_KEY",
        enabled_env="FINANCIAL_DATASETS_MCP_ENABLED",
        workflow="premium_subscribed",
        description="Income statements, balance sheets, cash flow, stock prices, market news",
        capabilities=["financials", "prices", "news", "insider_trades"],
        fallback_type="rest_api",
        fallback_url="https://api.financialdatasets.ai",
        tools=[
            {
                "name": "fd_get_financials",
                "description": "Fetch financial statements for a ticker",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "statement": {"type": "string", "enum": ["income", "balance", "cashflow"]},
                        "period": {"type": "string", "default": "annual"},
                    },
                    "required": ["ticker"],
                },
            },
        ],
    ),
    "semantic_scholar": MCPDataSourceConfig(
        name="semantic_scholar",
        server_url=os.environ.get("S2_MCP_URL", "http://localhost:8084/mcp"),
        env_key="S2_API_KEY",
        enabled_env="S2_MCP_ENABLED",
        workflow="all",
        description="225M+ papers, 2.8B citations, SPECTER2 embeddings, recommendations",
        capabilities=["search", "recommend", "embeddings", "citations"],
        fallback_type="rest_api",
        fallback_url="https://api.semanticscholar.org/graph/v1",
        tools=[
            {
                "name": "s2_search_papers",
                "description": "Search Semantic Scholar for papers",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "default": 10},
                        "fields": {"type": "string", "default": "title,abstract,year,citationCount"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "s2_recommend",
                "description": "Get paper recommendations based on a seed paper",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "paper_id": {"type": "string"},
                        "limit": {"type": "integer", "default": 10},
                    },
                    "required": ["paper_id"],
                },
            },
        ],
    ),
}


class MCPDataSourceRegistry:
    """Registry of MCP data sources with status checking and fallback.

    Args:
        sources: Optional custom source configs to override defaults.
    """

    def __init__(self, sources: Optional[Dict[str, MCPDataSourceConfig]] = None):
        self._sources = sources or dict(MCP_DATA_SOURCES)

    def list_sources(self, workflow: Optional[str] = None) -> List[MCPDataSourceConfig]:
        """List available data sources, optionally filtered by workflow.

        Args:
            workflow: Filter by workflow type (e.g., "open_source_api").

        Returns:
            List of matching data source configs.
        """
        sources = list(self._sources.values())
        if workflow:
            sources = [s for s in sources if s.workflow in (workflow, "all")]
        return sources

    def get_config(self, name: str) -> Optional[MCPDataSourceConfig]:
        """Get configuration for a named data source."""
        return self._sources.get(name)

    def is_enabled(self, name: str) -> bool:
        """Check if a data source is enabled via environment variable."""
        config = self._sources.get(name)
        if not config:
            return False
        if not config.enabled_env:
            return False
        return os.environ.get(config.enabled_env, "false").lower() in ("true", "1", "yes")

    def has_api_key(self, name: str) -> bool:
        """Check if the required API key is set."""
        config = self._sources.get(name)
        if not config or not config.env_key:
            return False
        return bool(os.environ.get(config.env_key))

    def get_client(self, name: str) -> Optional[Any]:
        """Get an MCPClient for a data source (if enabled).

        Args:
            name: Data source name.

        Returns:
            MCPClient instance or None if not enabled.
        """
        if not self.is_enabled(name):
            return None

        config = self._sources.get(name)
        if not config:
            return None

        try:
            from shared.protocols.mcp_client import MCPClient

            headers = {}
            api_key = os.environ.get(config.env_key, "")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            return MCPClient(
                server_url=config.server_url,
                timeout=30.0,
                headers=headers,
            )
        except Exception:
            return None

    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all data sources."""
        status = {}
        for name, config in self._sources.items():
            status[name] = {
                "enabled": self.is_enabled(name),
                "has_api_key": self.has_api_key(name),
                "workflow": config.workflow,
                "capabilities": config.capabilities,
                "fallback": config.fallback_type or "none",
            }
        return status

    def register(self, config: MCPDataSourceConfig) -> None:
        """Register a new data source configuration."""
        self._sources[config.name] = config
