# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Core evaluation infrastructure.

This module contains the fundamental components for workflow evaluation:
- Dimension definitions and metrics
- Metric calculation functions
- Data collection utilities
- Report generation
"""

from .dimensions import (
    DIMENSIONS,
    DIMENSION_CLUSTERS,
    DimensionDefinition,
    MetricDefinition,
    get_dimension,
    get_dimensions_by_cluster,
)
from .metrics import MetricCalculator
from .reporter import EvaluationReporter

__all__ = [
    "DIMENSIONS",
    "DIMENSION_CLUSTERS",
    "DimensionDefinition",
    "MetricDefinition",
    "get_dimension",
    "get_dimensions_by_cluster",
    "MetricCalculator",
    "EvaluationReporter",
]
