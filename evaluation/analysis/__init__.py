# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Analysis and comparison tools for workflow evaluation.

Provides batch evaluation, dimension analysis, visualization,
and statistical aggregation capabilities.
"""

from .batch_evaluator import BatchEvaluator, TeamEvaluation, ModeComparison, FullEvaluation
from .dimension_analyzer import DimensionAnalyzer, DimensionAnalysis, RadarChartData
from .visualizer import EvaluationVisualizer
from .statistical_aggregator import StatisticalAggregator, AggregatedScores, FactorialEffects
from .output_comparator import OutputComparator
from .innovation_analyzer import InnovationAnalyzer

__all__ = [
    "BatchEvaluator",
    "TeamEvaluation",
    "ModeComparison",
    "FullEvaluation",
    "DimensionAnalyzer",
    "DimensionAnalysis",
    "RadarChartData",
    "EvaluationVisualizer",
    "StatisticalAggregator",
    "AggregatedScores",
    "FactorialEffects",
    "OutputComparator",
    "InnovationAnalyzer",
]
