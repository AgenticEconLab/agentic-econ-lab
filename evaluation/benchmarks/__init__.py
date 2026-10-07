# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Economics-specific benchmarks module.

Provides reference outputs and comparison logic for calibrating
evaluation scores against curated exemplars.
"""

from evaluation.benchmarks.econ_benchmarks import (
    BenchmarkComparator,
    BenchmarkResult,
    get_team_benchmark,
)

__all__ = [
    "BenchmarkComparator",
    "BenchmarkResult",
    "get_team_benchmark",
]
