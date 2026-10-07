# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Register All Tools — Convenience module for MasterOrchestrator startup.

Importing this module triggers registration of all tool wrappers in the
ToolRegistry, making them available for agent discovery and invocation.

Usage:
    from shared.tools.register_all import register_all_tools
    register_all_tools()  # Call at MasterOrchestrator startup
"""

import importlib
import sys

_registered = False

_TOOL_MODULES = [
    "shared.tools.arxiv_tool",       # arxiv_search
    "shared.tools.openalex_tool",    # openalex_search (open, keyed: OPENALEX_API_KEY, abstracts)
    "shared.tools.elsevier_tool",    # elsevier_abstract (institutional, optional; Elsevier gap)
    "shared.tools.fred_tool",        # fred_get_series
    "shared.tools.web_search_tool",  # web_search
    "shared.tools.webcrawl_tool",    # webcrawl_scrape/crawl/extract/map
    "shared.tools.yfinance_tool",    # yfinance_history
    "shared.tools.http_tool",        # web_fetch, web_post
]


def register_all_tools():
    """
    Import all tool modules to trigger their ToolRegistry.register() calls.

    Safe to call multiple times — only registers once.  Uses reload() when
    modules are already cached so registration works after ToolRegistry.clear().
    """
    global _registered
    # Honor the docstring ("works after ToolRegistry.clear()"): skip only if we have
    # registered AND the registry is still populated. After a clear() the flag alone would
    # wrongly no-op, leaving an EMPTY registry (zero search results) — re-register
    # via the reload() path instead.
    from shared.tools.tool_registry import ToolRegistry
    if _registered and ToolRegistry._tools:
        return

    for mod_name in _TOOL_MODULES:
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])
        else:
            __import__(mod_name)

    _registered = True
