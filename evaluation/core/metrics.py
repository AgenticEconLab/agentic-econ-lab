# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Metric calculation functions for workflow evaluation.

Each function calculates a specific metric from execution traces and outputs.
Functions are organized by dimension cluster.
"""

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..schemas.execution import ExecutionTrace, StageExecution, ErrorInfo
from ..schemas.metrics import MetricResult
from .dimensions import DIMENSIONS, get_dimension


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    if denominator == 0:
        return default
    return numerator / denominator


def compute_coefficient_of_variation(values: List[float]) -> float:
    """Compute coefficient of variation (CV) for a list of values."""
    if not values or len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    std_dev = math.sqrt(variance)
    return std_dev / mean


def compute_entropy(probabilities: List[float]) -> float:
    """Compute entropy for a probability distribution."""
    if not probabilities:
        return 0.0
    # Filter out zeros and normalize
    probs = [p for p in probabilities if p > 0]
    total = sum(probs)
    if total == 0:
        return 0.0
    probs = [p / total for p in probs]
    return -sum(p * math.log2(p) for p in probs if p > 0)


# =============================================================================
# QUALITY CLUSTER METRICS
# =============================================================================

def calculate_output_consistency_cv(
    traces: List[ExecutionTrace],
    output_key: str = "item_count"
) -> MetricResult:
    """
    Calculate output consistency across multiple execution traces.

    When multiple traces are available, uses OutputComparator for a richer
    reliability measurement that considers text similarity, structural
    overlap, and item count stability. Falls back to simple CV of item
    counts when OutputComparator is not available or traces lack rich data.
    """
    dimension = get_dimension("reliability")
    metric_def = dimension.metrics[0]  # output_consistency_cv

    # Extract item counts per trace (total across stages)
    per_trace_counts = []
    for trace in traces:
        total = sum(
            float(s.item_count) for s in trace.stages
            if s.item_count is not None
        )
        per_trace_counts.append(total)

    # Try rich comparison via OutputComparator when we have >= 2 traces
    if len(traces) >= 2:
        try:
            from ..analysis.output_comparator import OutputComparator

            comparator = OutputComparator()

            # Build run_results from traces for the comparator
            run_results = []
            for trace in traces:
                result = {
                    "total_items": sum(
                        s.item_count or 0 for s in trace.stages
                    ),
                    "summary": {
                        "total_items": sum(
                            s.item_count or 0 for s in trace.stages
                        ),
                        "total_stages": len(trace.stages),
                        "successful_stages": sum(
                            1 for s in trace.stages if s.status == "success"
                        ),
                    },
                    "stages": {
                        s.stage_name: {
                            "status": s.status,
                            "item_count": s.item_count,
                            "duration": s.duration_seconds,
                        }
                        for s in trace.stages
                    },
                }
                run_results.append(result)

            breakdown = comparator.compute_reliability_breakdown(run_results)
            composite = breakdown["composite"]
            normalized = metric_def.normalize(1.0 - composite)

            return MetricResult(
                metric_id="output_consistency_cv",
                metric_name="Output Consistency (Multi-Run)",
                dimension="reliability",
                value=composite,
                normalized_value=composite,  # already 0-1 scale
                raw_data={
                    "method": "output_comparator",
                    "per_trace_counts": per_trace_counts,
                    "n_traces": len(traces),
                    "breakdown": breakdown,
                },
                notes=(
                    f"Composite reliability {composite:.3f} from "
                    f"{len(traces)} runs via OutputComparator"
                ),
            )
        except (ImportError, Exception):
            pass  # Fall back to simple CV below

    # Fallback: simple CV of per-stage item counts
    values = []
    for trace in traces:
        for stage in trace.stages:
            if stage.item_count is not None:
                values.append(float(stage.item_count))

    cv = compute_coefficient_of_variation(values) if values else 0.0
    normalized = metric_def.normalize(cv)

    return MetricResult(
        metric_id="output_consistency_cv",
        metric_name="Output Consistency (CV)",
        dimension="reliability",
        value=cv,
        normalized_value=normalized,
        raw_data={"values": values, "n_traces": len(traces)},
        notes=f"Computed from {len(values)} measurements across {len(traces)} runs"
    )


def calculate_execution_path_stability(traces: List[ExecutionTrace]) -> MetricResult:
    """
    Calculate execution path stability across multiple runs.

    Measures proportion of runs with identical stage sequences.
    """
    dimension = get_dimension("reliability")
    metric_def = dimension.metrics[1]  # execution_path_stability

    if not traces:
        return MetricResult(
            metric_id="execution_path_stability",
            metric_name="Execution Path Stability",
            dimension="reliability",
            value=0.0,
            normalized_value=0.0,
            notes="No traces provided"
        )

    # Extract stage sequences
    sequences = []
    for trace in traces:
        seq = tuple(s.stage_name for s in trace.stages)
        sequences.append(seq)

    # Find most common sequence
    from collections import Counter
    seq_counts = Counter(sequences)
    most_common_count = seq_counts.most_common(1)[0][1] if seq_counts else 0

    stability = safe_divide(most_common_count, len(traces))
    normalized = metric_def.normalize(stability)

    return MetricResult(
        metric_id="execution_path_stability",
        metric_name="Execution Path Stability",
        dimension="reliability",
        value=stability,
        normalized_value=normalized,
        raw_data={"sequences": [list(s) for s in sequences], "unique_sequences": len(seq_counts)},
        notes=f"{most_common_count}/{len(traces)} runs followed the same path"
    )


def calculate_error_rate(trace: ExecutionTrace) -> MetricResult:
    """Calculate error rate for a single execution."""
    dimension = get_dimension("correctness")
    metric_def = dimension.metrics[0]  # error_rate

    error_count = trace.error_count
    normalized = metric_def.normalize(error_count)

    return MetricResult(
        metric_id="error_rate",
        metric_name="Error Rate",
        dimension="correctness",
        value=float(error_count),
        normalized_value=normalized,
        raw_data={"errors": [e.model_dump() for e in trace.errors]},
        notes=f"{error_count} errors detected"
    )


def calculate_stage_completion_rate(trace: ExecutionTrace) -> MetricResult:
    """Calculate proportion of stages completing successfully."""
    dimension = get_dimension("correctness")
    metric_def = dimension.metrics[1]  # stage_completion_rate

    if not trace.stages:
        return MetricResult(
            metric_id="stage_completion_rate",
            metric_name="Stage Completion Rate",
            dimension="correctness",
            value=0.0,
            normalized_value=0.0,
            notes="No stages in trace"
        )

    successful = sum(1 for s in trace.stages if s.status == "success")
    rate = safe_divide(successful, len(trace.stages))
    normalized = metric_def.normalize(rate)

    return MetricResult(
        metric_id="stage_completion_rate",
        metric_name="Stage Completion Rate",
        dimension="correctness",
        value=rate,
        normalized_value=normalized,
        raw_data={"successful": successful, "total": len(trace.stages)},
        notes=f"{successful}/{len(trace.stages)} stages completed successfully"
    )


def calculate_stage_ordering_validity(trace: ExecutionTrace) -> MetricResult:
    """Check if stages executed in correct methodological order."""
    dimension = get_dimension("soundness")
    metric_def = dimension.metrics[0]  # stage_ordering_validity

    # Check stage numbers are in order
    stage_numbers = [s.stage_number for s in trace.stages]
    is_valid = stage_numbers == sorted(stage_numbers)
    value = 1.0 if is_valid else 0.0

    return MetricResult(
        metric_id="stage_ordering_validity",
        metric_name="Stage Ordering Validity",
        dimension="soundness",
        value=value,
        normalized_value=value,
        raw_data={"stage_numbers": stage_numbers, "is_valid": is_valid},
        notes="Valid ordering" if is_valid else "Invalid stage ordering detected"
    )


# =============================================================================
# OPERATIONAL CLUSTER METRICS
# =============================================================================

def calculate_execution_time(trace: ExecutionTrace) -> MetricResult:
    """Calculate total execution time."""
    dimension = get_dimension("efficiency")
    metric_def = dimension.metrics[0]  # execution_time

    duration = trace.total_duration_seconds or 0.0
    normalized = metric_def.normalize(duration)

    return MetricResult(
        metric_id="execution_time",
        metric_name="Execution Time",
        dimension="efficiency",
        value=duration,
        normalized_value=normalized,
        raw_data={
            "start_time": trace.start_time.isoformat() if trace.start_time else None,
            "end_time": trace.end_time.isoformat() if trace.end_time else None
        },
        notes=f"Total duration: {duration:.1f} seconds"
    )


def calculate_time_per_item(trace: ExecutionTrace) -> MetricResult:
    """Calculate average processing time per item."""
    dimension = get_dimension("efficiency")
    metric_def = dimension.metrics[1]  # time_per_item

    total_items = sum(s.item_count or 0 for s in trace.stages)
    duration = trace.total_duration_seconds or 0.0

    time_per_item = safe_divide(duration, total_items)
    normalized = metric_def.normalize(time_per_item)

    return MetricResult(
        metric_id="time_per_item",
        metric_name="Time per Item",
        dimension="efficiency",
        value=time_per_item,
        normalized_value=normalized,
        raw_data={"total_items": total_items, "total_time": duration},
        notes=f"{time_per_item:.2f} seconds per item ({total_items} items)"
    )


def calculate_throughput(trace: ExecutionTrace) -> MetricResult:
    """Calculate items processed per minute."""
    dimension = get_dimension("scalability")
    metric_def = dimension.metrics[0]  # throughput

    total_items = sum(s.item_count or 0 for s in trace.stages)
    duration_minutes = (trace.total_duration_seconds or 0.0) / 60.0

    throughput = safe_divide(total_items, duration_minutes)
    normalized = metric_def.normalize(throughput)

    return MetricResult(
        metric_id="throughput",
        metric_name="Throughput",
        dimension="scalability",
        value=throughput,
        normalized_value=normalized,
        raw_data={"total_items": total_items, "duration_minutes": duration_minutes},
        notes=f"{throughput:.1f} items/minute"
    )


def calculate_error_recovery_rate(trace: ExecutionTrace) -> MetricResult:
    """Calculate proportion of errors that were recovered from."""
    dimension = get_dimension("robustness")
    metric_def = dimension.metrics[0]  # error_recovery_rate

    if not trace.errors:
        return MetricResult(
            metric_id="error_recovery_rate",
            metric_name="Error Recovery Rate",
            dimension="robustness",
            value=1.0,
            normalized_value=1.0,
            notes="No errors occurred"
        )

    recovered = trace.recovered_error_count
    total = trace.error_count
    rate = safe_divide(recovered, total)
    normalized = metric_def.normalize(rate)

    return MetricResult(
        metric_id="error_recovery_rate",
        metric_name="Error Recovery Rate",
        dimension="robustness",
        value=rate,
        normalized_value=normalized,
        raw_data={"recovered": recovered, "total": total},
        notes=f"{recovered}/{total} errors recovered"
    )


def calculate_failure_tolerance(traces: List[ExecutionTrace]) -> MetricResult:
    """Calculate proportion of runs completing despite errors."""
    dimension = get_dimension("robustness")
    metric_def = dimension.metrics[1]  # failure_tolerance

    if not traces:
        return MetricResult(
            metric_id="failure_tolerance",
            metric_name="Failure Tolerance",
            dimension="robustness",
            value=0.0,
            normalized_value=0.0,
            notes="No traces provided"
        )

    # Count runs that had errors but still completed
    completed_with_errors = sum(
        1 for t in traces if t.success and t.error_count > 0
    )
    total_with_errors = sum(1 for t in traces if t.error_count > 0)

    rate = safe_divide(completed_with_errors, total_with_errors, default=1.0)
    normalized = metric_def.normalize(rate)

    return MetricResult(
        metric_id="failure_tolerance",
        metric_name="Failure Tolerance",
        dimension="robustness",
        value=rate,
        normalized_value=normalized,
        raw_data={
            "completed_with_errors": completed_with_errors,
            "total_with_errors": total_with_errors
        },
        notes=f"{completed_with_errors}/{total_with_errors} error runs completed"
    )


# =============================================================================
# EPISTEMIC CLUSTER METRICS
# =============================================================================

def calculate_documentation_coverage(trace: ExecutionTrace) -> MetricResult:
    """Calculate proportion of stages with documentation."""
    dimension = get_dimension("transparency")
    metric_def = dimension.metrics[0]  # documentation_coverage

    # Check for output files (proxy for documentation)
    stages_with_docs = sum(1 for s in trace.stages if s.output_files)
    total_stages = len(trace.stages)

    coverage = safe_divide(stages_with_docs, total_stages)
    normalized = metric_def.normalize(coverage)

    return MetricResult(
        metric_id="documentation_coverage",
        metric_name="Documentation Coverage",
        dimension="transparency",
        value=coverage,
        normalized_value=normalized,
        raw_data={"stages_with_docs": stages_with_docs, "total_stages": total_stages},
        notes=f"{stages_with_docs}/{total_stages} stages have output files"
    )


def calculate_output_file_completeness(
    trace: ExecutionTrace,
    expected_files: Optional[List[str]] = None
) -> MetricResult:
    """Calculate proportion of expected output files generated."""
    dimension = get_dimension("transparency")
    metric_def = dimension.metrics[1]  # output_file_completeness

    all_outputs = trace.outputs.all_files()
    if expected_files:
        found = sum(1 for f in expected_files if f in all_outputs)
        completeness = safe_divide(found, len(expected_files))
    else:
        # Default: check if any outputs were generated
        completeness = 1.0 if all_outputs else 0.0

    normalized = metric_def.normalize(completeness)

    return MetricResult(
        metric_id="output_file_completeness",
        metric_name="Output File Completeness",
        dimension="transparency",
        value=completeness,
        normalized_value=normalized,
        raw_data={"output_files": all_outputs, "expected_files": expected_files},
        notes=f"{len(all_outputs)} output files generated"
    )


def calculate_provenance_completeness(trace: ExecutionTrace) -> MetricResult:
    """Calculate proportion of outputs with complete provenance metadata."""
    dimension = get_dimension("traceability")
    metric_def = dimension.metrics[0]  # provenance_completeness

    # Check for key provenance fields
    required_fields = ["run_id", "team", "mode", "start_time"]
    present_fields = sum(1 for f in required_fields if getattr(trace, f, None) is not None)

    completeness = safe_divide(present_fields, len(required_fields))
    normalized = metric_def.normalize(completeness)

    return MetricResult(
        metric_id="provenance_completeness",
        metric_name="Provenance Completeness",
        dimension="traceability",
        value=completeness,
        normalized_value=normalized,
        raw_data={"present_fields": present_fields, "required_fields": required_fields},
        notes=f"{present_fields}/{len(required_fields)} provenance fields present"
    )


def calculate_replication_success_rate(
    traces: List[ExecutionTrace],
    similarity_threshold: float = 0.9
) -> MetricResult:
    """Calculate proportion of runs producing consistent outputs."""
    dimension = get_dimension("reproducibility")
    metric_def = dimension.metrics[0]  # replication_success_rate

    if len(traces) < 2:
        return MetricResult(
            metric_id="replication_success_rate",
            metric_name="Replication Success Rate",
            dimension="reproducibility",
            value=1.0,
            normalized_value=1.0,
            notes="Need at least 2 runs for replication analysis"
        )

    # Compare item counts as a proxy for output similarity
    item_counts = []
    for trace in traces:
        total = sum(s.item_count or 0 for s in trace.stages)
        item_counts.append(total)

    # Calculate pairwise similarity
    similarities = []
    for i in range(len(item_counts)):
        for j in range(i + 1, len(item_counts)):
            if item_counts[i] == 0 and item_counts[j] == 0:
                sim = 1.0
            elif item_counts[i] == 0 or item_counts[j] == 0:
                sim = 0.0
            else:
                sim = min(item_counts[i], item_counts[j]) / max(item_counts[i], item_counts[j])
            similarities.append(sim)

    avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
    success_rate = 1.0 if avg_similarity >= similarity_threshold else avg_similarity
    normalized = metric_def.normalize(success_rate)

    return MetricResult(
        metric_id="replication_success_rate",
        metric_name="Replication Success Rate",
        dimension="reproducibility",
        value=success_rate,
        normalized_value=normalized,
        raw_data={"item_counts": item_counts, "similarities": similarities},
        notes=f"Average similarity: {avg_similarity:.2f}"
    )


# =============================================================================
# INNOVATION CLUSTER METRICS
# =============================================================================

def calculate_output_diversity(trace: ExecutionTrace) -> MetricResult:
    """Calculate diversity of outputs using entropy."""
    dimension = get_dimension("innovation_potential")
    metric_def = dimension.metrics[0]  # output_diversity

    # Use stage item counts as a proxy for diversity
    item_counts = [s.item_count or 0 for s in trace.stages]
    total = sum(item_counts)

    if total == 0:
        diversity = 0.0
    else:
        probabilities = [c / total for c in item_counts if c > 0]
        # Normalize entropy to 0-1 (max entropy for uniform distribution)
        max_entropy = math.log2(len(probabilities)) if len(probabilities) > 1 else 1.0
        raw_entropy = compute_entropy(probabilities)
        diversity = safe_divide(raw_entropy, max_entropy)

    normalized = metric_def.normalize(diversity)

    return MetricResult(
        metric_id="output_diversity",
        metric_name="Output Diversity",
        dimension="innovation_potential",
        value=diversity,
        normalized_value=normalized,
        raw_data={"item_counts": item_counts},
        notes=f"Normalized entropy: {diversity:.2f}"
    )


def calculate_coverage_breadth(trace: ExecutionTrace) -> MetricResult:
    """Calculate breadth of coverage based on total items."""
    dimension = get_dimension("innovation_potential")
    metric_def = dimension.metrics[1]  # coverage_breadth

    total_items = sum(s.item_count or 0 for s in trace.stages)
    normalized = metric_def.normalize(total_items)

    return MetricResult(
        metric_id="coverage_breadth",
        metric_name="Coverage Breadth",
        dimension="innovation_potential",
        value=float(total_items),
        normalized_value=normalized,
        raw_data={"total_items": total_items},
        notes=f"Total items: {total_items}"
    )


def calculate_innovation_embedding(
    trace: ExecutionTrace,
    output_texts: Optional[List[str]] = None,
    reference_texts: Optional[List[str]] = None,
) -> MetricResult:
    """
    Calculate embedding-based innovation metrics for a single execution.

    Uses InnovationAnalyzer to compute novelty, cross-domain integration,
    and diversity scores. Falls back to keyword heuristics when the OpenAI
    API is unavailable.

    Args:
        trace: Execution trace (used for metadata).
        output_texts: Workflow output texts. If None, tries to extract from trace.
        reference_texts: Reference literature for novelty comparison. Optional.

    Returns:
        MetricResult with composite innovation score.
    """
    dimension = get_dimension("innovation_potential")
    # Use embedding_novelty metric definition (index 2)
    metric_def = dimension.get_metric("embedding_novelty")
    if metric_def is None:
        metric_def = dimension.metrics[0]  # fallback

    # Build output_texts from trace output files if not provided
    if output_texts is None:
        output_texts = []
        for stage in trace.stages:
            if stage.output_files:
                for fpath in stage.output_files:
                    try:
                        p = Path(fpath)
                        if p.exists() and p.suffix in (".json", ".txt", ".md"):
                            output_texts.append(p.read_text(encoding="utf-8"))
                    except (IOError, OSError):
                        pass

    if not output_texts:
        return MetricResult(
            metric_id="embedding_novelty",
            metric_name="Embedding Novelty (Innovation)",
            dimension="innovation_potential",
            value=0.0,
            normalized_value=0.0,
            notes="No output texts available for innovation analysis",
        )

    try:
        from ..analysis.innovation_analyzer import InnovationAnalyzer

        analyzer = InnovationAnalyzer()
        result = analyzer.compute_composite_innovation(
            outputs=output_texts,
            reference_corpus=reference_texts,
        )

        composite = result["composite"]
        normalized = metric_def.normalize(composite)

        return MetricResult(
            metric_id="embedding_novelty",
            metric_name="Embedding Novelty (Innovation)",
            dimension="innovation_potential",
            value=composite,
            normalized_value=normalized,
            raw_data={
                "method": "innovation_analyzer",
                "novelty": result["novelty"],
                "cross_domain": result["cross_domain"],
                "diversity": result["diversity"],
                "composite": result["composite"],
                "n_outputs": result["n_outputs"],
                "n_references": result["n_references"],
                "subfields_matched": result["subfields_matched"],
            },
            notes=(
                f"Composite innovation {composite:.3f} "
                f"(novelty={result['novelty']:.3f}, "
                f"cross_domain={result['cross_domain']:.3f}, "
                f"diversity={result['diversity']:.3f})"
            ),
        )
    except (ImportError, Exception) as e:
        return MetricResult(
            metric_id="embedding_novelty",
            metric_name="Embedding Novelty (Innovation)",
            dimension="innovation_potential",
            value=0.0,
            normalized_value=0.0,
            notes=f"Innovation analysis unavailable: {str(e)}",
        )


# =============================================================================
# METRIC REGISTRY
# =============================================================================

# Single-trace metrics (one trace required)
SINGLE_TRACE_METRICS = {
    # Quality
    "error_rate": calculate_error_rate,
    "stage_completion_rate": calculate_stage_completion_rate,
    "stage_ordering_validity": calculate_stage_ordering_validity,
    # Operational
    "execution_time": calculate_execution_time,
    "time_per_item": calculate_time_per_item,
    "throughput": calculate_throughput,
    "error_recovery_rate": calculate_error_recovery_rate,
    # Epistemic
    "documentation_coverage": calculate_documentation_coverage,
    "output_file_completeness": calculate_output_file_completeness,
    "provenance_completeness": calculate_provenance_completeness,
    # Innovation
    "output_diversity": calculate_output_diversity,
    "coverage_breadth": calculate_coverage_breadth,
    "embedding_novelty": calculate_innovation_embedding,
}

# Multi-trace metrics (multiple traces required)
MULTI_TRACE_METRICS = {
    # Quality
    "output_consistency_cv": calculate_output_consistency_cv,
    "execution_path_stability": calculate_execution_path_stability,
    # Operational
    "failure_tolerance": calculate_failure_tolerance,
    # Epistemic
    "replication_success_rate": calculate_replication_success_rate,
}


def calculate_all_single_trace_metrics(trace: ExecutionTrace) -> List[MetricResult]:
    """Calculate all metrics that require only a single trace."""
    results = []
    for metric_id, calc_func in SINGLE_TRACE_METRICS.items():
        try:
            result = calc_func(trace)
            results.append(result)
        except Exception as e:
            results.append(MetricResult(
                metric_id=metric_id,
                metric_name=metric_id,
                dimension="unknown",
                value=0.0,
                notes=f"Calculation error: {str(e)}"
            ))
    return results


def calculate_all_multi_trace_metrics(traces: List[ExecutionTrace]) -> List[MetricResult]:
    """Calculate all metrics that require multiple traces."""
    results = []
    for metric_id, calc_func in MULTI_TRACE_METRICS.items():
        try:
            result = calc_func(traces)
            results.append(result)
        except Exception as e:
            results.append(MetricResult(
                metric_id=metric_id,
                metric_name=metric_id,
                dimension="unknown",
                value=0.0,
                notes=f"Calculation error: {str(e)}"
            ))
    return results


# =============================================================================
# METRIC CALCULATOR CLASS
# =============================================================================

class MetricCalculator:
    """
    Main calculator class for computing evaluation metrics.

    Provides a unified interface for calculating metrics from execution traces.
    """

    def __init__(self):
        """Initialize the metric calculator."""
        self.single_trace_metrics = SINGLE_TRACE_METRICS
        self.multi_trace_metrics = MULTI_TRACE_METRICS

    def calculate_single(self, trace: ExecutionTrace) -> List[MetricResult]:
        """
        Calculate all single-trace metrics for an execution.

        Args:
            trace: The execution trace to analyze.

        Returns:
            List of MetricResult objects.
        """
        return calculate_all_single_trace_metrics(trace)

    # Alias for backward compatibility
    def calculate_single_trace_metrics(self, trace: ExecutionTrace) -> Dict[str, float]:
        """
        Calculate single-trace metrics and return as dictionary.

        Alias method for backward compatibility with batch_evaluator.

        Args:
            trace: The execution trace to analyze.

        Returns:
            Dictionary mapping metric_id to value.
        """
        results = calculate_all_single_trace_metrics(trace)
        return {r.metric_id: r.value for r in results}

    def calculate_multi(self, traces: List[ExecutionTrace]) -> List[MetricResult]:
        """
        Calculate all multi-trace metrics from multiple executions.

        Args:
            traces: List of execution traces to compare.

        Returns:
            List of MetricResult objects.
        """
        return calculate_all_multi_trace_metrics(traces)

    # Alias for backward compatibility
    def calculate_multi_trace_metrics(self, traces: List[ExecutionTrace]) -> Dict[str, float]:
        """
        Calculate multi-trace metrics and return as dictionary.

        Alias method for backward compatibility with batch_evaluator.

        Args:
            traces: List of execution traces to compare.

        Returns:
            Dictionary mapping metric_id to value.
        """
        results = calculate_all_multi_trace_metrics(traces)
        return {r.metric_id: r.value for r in results}

    def calculate_all(
        self,
        traces: List[ExecutionTrace]
    ) -> Dict[str, List[MetricResult]]:
        """
        Calculate all metrics from a list of traces.

        Args:
            traces: List of execution traces.

        Returns:
            Dictionary with 'single' and 'multi' keys containing metric results.
        """
        results = {
            "single": [],
            "multi": []
        }

        # Single-trace metrics for each trace
        for trace in traces:
            results["single"].extend(self.calculate_single(trace))

        # Multi-trace metrics if we have multiple traces
        if len(traces) >= 2:
            results["multi"] = self.calculate_multi(traces)

        return results

    def calculate_metric(
        self,
        metric_id: str,
        trace: Optional[ExecutionTrace] = None,
        traces: Optional[List[ExecutionTrace]] = None
    ) -> Optional[MetricResult]:
        """
        Calculate a specific metric by ID.

        Args:
            metric_id: The ID of the metric to calculate.
            trace: Single trace (for single-trace metrics).
            traces: List of traces (for multi-trace metrics).

        Returns:
            MetricResult or None if metric not found.
        """
        if metric_id in self.single_trace_metrics:
            if trace is None:
                return None
            return self.single_trace_metrics[metric_id](trace)
        elif metric_id in self.multi_trace_metrics:
            if traces is None:
                return None
            return self.multi_trace_metrics[metric_id](traces)
        return None

    def get_dimension_scores(
        self,
        traces: List[ExecutionTrace]
    ) -> Dict[str, float]:
        """
        Get normalized scores for each dimension.

        Args:
            traces: List of execution traces.

        Returns:
            Dictionary mapping dimension_id to average normalized score.
        """
        all_results = self.calculate_all(traces)
        all_metrics = all_results["single"] + all_results["multi"]

        # Group by dimension
        dimension_scores: Dict[str, List[float]] = {}
        for result in all_metrics:
            dim = result.dimension
            if dim not in dimension_scores:
                dimension_scores[dim] = []
            if result.normalized_value is not None:
                dimension_scores[dim].append(result.normalized_value)

        # Average scores per dimension
        return {
            dim: sum(scores) / len(scores) if scores else 0.0
            for dim, scores in dimension_scores.items()
        }
