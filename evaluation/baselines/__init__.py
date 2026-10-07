# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Baseline comparison framework for agentic workflow evaluation.

Provides single-prompt LLM baseline generation and comparison
against multi-agent workflow outputs.
"""

from .baseline_generator import BaselineGenerator
from .baseline_comparator import BaselineComparator, ComparisonResult

__all__ = [
    "BaselineGenerator",
    "BaselineComparator",
    "ComparisonResult",
]
