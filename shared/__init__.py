# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Shared utilities for AEL (Agentic Econ Lab) workflows.

This module provides common functionality used across all teams:
- WorkflowLogger: Execution instrumentation for evaluation
- LLMClient: Lightweight OpenAI wrapper with observability
- MetricsCollector: Observability and performance monitoring
- tracked_get, tracked_post, tracked_arxiv_search: Instrumented API wrappers
"""

from .instrumentation import WorkflowLogger
from .llm import LLMClient
from .observability import (
    MetricsCollector,
    tracked_get,
    tracked_post,
    tracked_arxiv_search,
    tracked_fred_get_series,
    tracked_yfinance_history,
)

__all__ = [
    "WorkflowLogger",
    "LLMClient",
    "MetricsCollector",
    "tracked_get",
    "tracked_post",
    "tracked_arxiv_search",
    "tracked_fred_get_series",
    "tracked_yfinance_history",
]
