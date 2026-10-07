# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Data schemas for the evaluation framework.

This module contains Pydantic models for:
- Execution traces and stage information
- Metric results and dimension scores
- Report formats
"""

from .execution import (
    ExecutionTrace,
    StageExecution,
    ErrorInfo,
    TimingInfo,
    WorkflowOutputs,
)
from .metrics import (
    MetricResult,
    DimensionScore,
    EvaluationResult,
    TeamEvaluation,
    FullEvaluation,
)

__all__ = [
    # Execution schemas
    "ExecutionTrace",
    "StageExecution",
    "ErrorInfo",
    "TimingInfo",
    "WorkflowOutputs",
    # Metric schemas
    "MetricResult",
    "DimensionScore",
    "EvaluationResult",
    "TeamEvaluation",
    "FullEvaluation",
]
