# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
AEL Workflow Evaluation Framework

A comprehensive evaluation codebase for assessing agentic workflows across
10 dimensions: Reliability, Correctness, Soundness, Efficiency, Scalability,
Robustness, Transparency, Traceability, Reproducibility, and Innovation Potential.

Usage:
    from evaluation import BatchEvaluator, AELParser, AELRunner

    # Analyze existing outputs
    parser = AELParser()
    trace = parser.build_execution_trace("IdeationTeam", "ModeNoWcNoHITL", output_dir)

    # Run batch evaluation
    evaluator = BatchEvaluator(framework="ael")
    evaluation = evaluator.evaluate_team("IdeationTeam", n_runs=3)

    # Analyze dimensions
    analyzer = DimensionAnalyzer()
    analysis = analyzer.analyze_dimension("reliability", results)

    # Generate reports
    reporter = EvaluationReporter()
    reporter.generate_markdown_report(evaluation)
"""

__version__ = "0.1.0"
__author__ = "AgenticEconLab"

# Core components
from .core.dimensions import (
    DIMENSIONS,
    DIMENSION_CLUSTERS,
    DimensionDefinition,
    MetricDefinition,
    get_dimension,
    get_dimensions_by_cluster,
)
from .core.metrics import MetricCalculator
from .core.reporter import EvaluationReporter

# Schemas
from .schemas.execution import ExecutionTrace, StageExecution, ErrorInfo, TimingInfo, WorkflowOutputs
from .schemas.metrics import MetricResult, DimensionScore, EvaluationResult

# Parsers
from .parsers.base import BaseWorkflowParser
from .parsers.ael_parser import AELParser

# Runners
from .runners.base import BaseWorkflowRunner, RunConfig, ExecutionResult
from .runners.ael_runner import AELRunner

# Analysis
from .analysis.batch_evaluator import BatchEvaluator, TeamEvaluation, ModeComparison, FullEvaluation
from .analysis.dimension_analyzer import DimensionAnalyzer, DimensionAnalysis, RadarChartData
from .analysis.visualizer import EvaluationVisualizer

__all__ = [
    # Version
    "__version__",

    # Core - Dimensions
    "DIMENSIONS",
    "DIMENSION_CLUSTERS",
    "DimensionDefinition",
    "MetricDefinition",
    "get_dimension",
    "get_dimensions_by_cluster",

    # Core - Metrics
    "MetricCalculator",

    # Core - Reporter
    "EvaluationReporter",

    # Schemas - Execution
    "ExecutionTrace",
    "StageExecution",
    "ErrorInfo",
    "TimingInfo",
    "WorkflowOutputs",

    # Schemas - Metrics
    "MetricResult",
    "DimensionScore",
    "EvaluationResult",

    # Parsers
    "BaseWorkflowParser",
    "AELParser",

    # Runners
    "BaseWorkflowRunner",
    "RunConfig",
    "ExecutionResult",
    "AELRunner",

    # Analysis
    "BatchEvaluator",
    "TeamEvaluation",
    "ModeComparison",
    "FullEvaluation",
    "DimensionAnalyzer",
    "DimensionAnalysis",
    "RadarChartData",
    "EvaluationVisualizer",
]
