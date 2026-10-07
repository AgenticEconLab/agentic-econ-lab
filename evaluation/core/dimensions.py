# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Evaluation dimension definitions.

Defines the 14 evaluation dimensions organized into 5 clusters:
- Quality: Reliability, Correctness, Soundness, Decision Quality, Economic Rigor
- Operational: Efficiency, Scalability, Robustness
- Epistemic: Transparency, Traceability, Reproducibility
- Innovation: Innovation Potential
- Economics: Simulation Fidelity, Causal Validity  (V0.7)

Based on the theoretical framework in the paper:
- Distributed Cognition Theory
- Epistemic Reliability Theory
- Scientific Norm Theory
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any
from enum import Enum


class DimensionCluster(str, Enum):
    """Dimension clusters (V0.6: 4 clusters; V0.7 adds Economics)."""
    QUALITY = "Quality"
    OPERATIONAL = "Operational"
    EPISTEMIC = "Epistemic"
    INNOVATION = "Innovation"
    DOMAIN = "Domain"           # V0.6 — decision_quality, economic_rigor
    ECONOMICS = "Economics"      # V0.7 — simulation_fidelity, causal_validity


@dataclass
class MetricDefinition:
    """Definition of a single metric within a dimension."""

    metric_id: str
    name: str
    description: str
    unit: str = ""
    higher_is_better: bool = True
    min_value: float = 0.0
    max_value: float = 1.0
    calculation_method: str = ""
    requires_multiple_runs: bool = False

    def normalize(self, value: float) -> float:
        """Normalize a raw value to 0-1 scale."""
        if self.max_value == self.min_value:
            return 0.5
        normalized = (value - self.min_value) / (self.max_value - self.min_value)
        if not self.higher_is_better:
            normalized = 1.0 - normalized
        return max(0.0, min(1.0, normalized))


@dataclass
class DimensionDefinition:
    """Definition of an evaluation dimension."""

    dimension_id: str
    name: str
    cluster: DimensionCluster
    description: str
    theoretical_basis: str
    design_features: List[str]
    metrics: List[MetricDefinition]
    weight: float = 1.0

    def get_metric(self, metric_id: str) -> Optional[MetricDefinition]:
        """Get a metric by ID."""
        for metric in self.metrics:
            if metric.metric_id == metric_id:
                return metric
        return None


# =============================================================================
# DIMENSION DEFINITIONS
# =============================================================================

DIMENSIONS: Dict[str, DimensionDefinition] = {

    # -------------------------------------------------------------------------
    # QUALITY CLUSTER
    # -------------------------------------------------------------------------

    "reliability": DimensionDefinition(
        dimension_id="reliability",
        name="Reliability",
        cluster=DimensionCluster.QUALITY,
        description="Consistency and dependability of workflow outputs under varying conditions",
        theoretical_basis="Consistency under varying conditions including changes in input characteristics, environmental factors, and system state",
        design_features=[
            "Explicit agent specifications that constrain behavior",
            "Structured communication protocols",
            "Version control integration",
            "Deterministic orchestration logic"
        ],
        metrics=[
            MetricDefinition(
                metric_id="output_consistency_cv",
                name="Output Consistency (CV)",
                description="Coefficient of variation of outputs across repeated runs",
                unit="CV",
                higher_is_better=False,
                min_value=0.0,
                max_value=1.0,
                calculation_method="embedding_similarity",
                requires_multiple_runs=True
            ),
            MetricDefinition(
                metric_id="execution_path_stability",
                name="Execution Path Stability",
                description="Proportion of runs following identical stage sequences",
                unit="ratio",
                higher_is_better=True,
                min_value=0.0,
                max_value=1.0,
                calculation_method="sequence_comparison",
                requires_multiple_runs=True
            ),
        ]
    ),

    "correctness": DimensionDefinition(
        dimension_id="correctness",
        name="Correctness",
        cluster=DimensionCluster.QUALITY,
        description="Accuracy and validity of workflow outputs relative to domain standards",
        theoretical_basis="Accuracy relative to domain standards and ground truth including factual accuracy, logical validity, and methodological appropriateness",
        design_features=[
            "Specialized agents with domain-focused capabilities",
            "Verification agents checking outputs",
            "Human-in-the-loop checkpoints",
            "Error escalation pathways"
        ],
        metrics=[
            MetricDefinition(
                metric_id="error_rate",
                name="Error Rate",
                description="Frequency of detected errors per execution",
                unit="errors/run",
                higher_is_better=False,
                min_value=0.0,
                max_value=10.0,
                calculation_method="log_parsing"
            ),
            MetricDefinition(
                metric_id="stage_completion_rate",
                name="Stage Completion Rate",
                description="Proportion of stages completing successfully",
                unit="ratio",
                higher_is_better=True,
                min_value=0.0,
                max_value=1.0,
                calculation_method="status_check"
            ),
            MetricDefinition(
                metric_id="output_validation_score",
                name="Output Validation Score",
                description="Score from automated output validation checks",
                unit="score",
                higher_is_better=True,
                min_value=0.0,
                max_value=1.0,
                calculation_method="schema_validation"
            ),
        ]
    ),

    "soundness": DimensionDefinition(
        dimension_id="soundness",
        name="Soundness",
        cluster=DimensionCluster.QUALITY,
        description="Logical and methodological validity of reasoning processes",
        theoretical_basis="Logical validity independent of factual accuracy - whether inferences follow validly from premises",
        design_features=[
            "Staged workflow structure enforcing logical ordering",
            "Explicit reasoning chains across agent interactions",
            "Methodological evaluation agents",
            "Documentation requirements"
        ],
        metrics=[
            MetricDefinition(
                metric_id="stage_ordering_validity",
                name="Stage Ordering Validity",
                description="Whether stages execute in methodologically correct order",
                unit="binary",
                higher_is_better=True,
                min_value=0.0,
                max_value=1.0,
                calculation_method="sequence_validation"
            ),
            MetricDefinition(
                metric_id="reasoning_chain_coherence",
                name="Reasoning Chain Coherence",
                description="Coherence of reasoning across stages",
                unit="score",
                higher_is_better=True,
                min_value=0.0,
                max_value=1.0,
                calculation_method="llm_evaluation"
            ),
        ]
    ),

    # -------------------------------------------------------------------------
    # OPERATIONAL CLUSTER
    # -------------------------------------------------------------------------

    "efficiency": DimensionDefinition(
        dimension_id="efficiency",
        name="Efficiency",
        cluster=DimensionCluster.OPERATIONAL,
        description="Relationship between resources consumed and outputs produced",
        theoretical_basis="Resource-output optimization including researcher time, computational resources, and cognitive effort",
        design_features=[
            "Parallelization capabilities",
            "Task automation for routine operations",
            "Caching and reuse mechanisms",
            "Adaptive resource allocation"
        ],
        metrics=[
            MetricDefinition(
                metric_id="execution_time",
                name="Execution Time",
                description="Total wall-clock duration for workflow completion",
                unit="seconds",
                higher_is_better=False,
                min_value=0.0,
                max_value=3600.0,
                calculation_method="timing_extraction"
            ),
            MetricDefinition(
                metric_id="time_per_item",
                name="Time per Item",
                description="Average processing time per output item",
                unit="seconds/item",
                higher_is_better=False,
                min_value=0.0,
                max_value=60.0,
                calculation_method="throughput_calculation"
            ),
            MetricDefinition(
                metric_id="stage_efficiency",
                name="Stage Efficiency",
                description="Ratio of productive stage time to total time",
                unit="ratio",
                higher_is_better=True,
                min_value=0.0,
                max_value=1.0,
                calculation_method="timing_analysis"
            ),
        ]
    ),

    "scalability": DimensionDefinition(
        dimension_id="scalability",
        name="Scalability",
        cluster=DimensionCluster.OPERATIONAL,
        description="Ability to maintain performance as workload increases",
        theoretical_basis="Performance maintenance under increasing workload, complexity, or scope",
        design_features=[
            "Modular agent design for independent scaling",
            "Parallelization capabilities",
            "Distributed execution support",
            "Incremental processing strategies"
        ],
        metrics=[
            MetricDefinition(
                metric_id="throughput",
                name="Throughput",
                description="Number of items processed per unit time",
                unit="items/minute",
                higher_is_better=True,
                min_value=0.0,
                max_value=100.0,
                calculation_method="throughput_calculation"
            ),
            MetricDefinition(
                metric_id="scaling_linearity",
                name="Scaling Linearity",
                description="How linearly performance scales with workload",
                unit="coefficient",
                higher_is_better=True,
                min_value=0.0,
                max_value=1.0,
                calculation_method="scaling_analysis",
                requires_multiple_runs=True
            ),
        ]
    ),

    "robustness": DimensionDefinition(
        dimension_id="robustness",
        name="Robustness",
        cluster=DimensionCluster.OPERATIONAL,
        description="Ability to handle errors and unexpected conditions",
        theoretical_basis="Graceful handling of errors, edge cases, unexpected inputs, and system failures",
        design_features=[
            "Error escalation pathways",
            "Validation checkpoints",
            "Fallback mechanisms",
            "Graceful degradation strategies",
            "Retry and recovery mechanisms"
        ],
        metrics=[
            MetricDefinition(
                metric_id="error_recovery_rate",
                name="Error Recovery Rate",
                description="Proportion of errors recovered from successfully",
                unit="ratio",
                higher_is_better=True,
                min_value=0.0,
                max_value=1.0,
                calculation_method="error_analysis"
            ),
            MetricDefinition(
                metric_id="failure_tolerance",
                name="Failure Tolerance",
                description="Proportion of runs completing despite errors",
                unit="ratio",
                higher_is_better=True,
                min_value=0.0,
                max_value=1.0,
                calculation_method="completion_analysis"
            ),
        ]
    ),

    # -------------------------------------------------------------------------
    # EPISTEMIC CLUSTER
    # -------------------------------------------------------------------------

    "transparency": DimensionDefinition(
        dimension_id="transparency",
        name="Transparency",
        cluster=DimensionCluster.EPISTEMIC,
        description="Inspectability of system operations and decision rationales",
        theoretical_basis="Degree to which observers can understand what the system does, how, and why",
        design_features=[
            "Explicit agent definitions",
            "Logged inter-agent communications",
            "Documented decision points",
            "Human-readable output formats"
        ],
        metrics=[
            MetricDefinition(
                metric_id="documentation_coverage",
                name="Documentation Coverage",
                description="Proportion of operations with documentation",
                unit="ratio",
                higher_is_better=True,
                min_value=0.0,
                max_value=1.0,
                calculation_method="file_analysis"
            ),
            MetricDefinition(
                metric_id="output_file_completeness",
                name="Output File Completeness",
                description="Proportion of expected output files generated",
                unit="ratio",
                higher_is_better=True,
                min_value=0.0,
                max_value=1.0,
                calculation_method="file_check"
            ),
        ]
    ),

    "traceability": DimensionDefinition(
        dimension_id="traceability",
        name="Traceability",
        cluster=DimensionCluster.EPISTEMIC,
        description="Ability to track the provenance of outputs",
        theoretical_basis="Reconstructing the historical sequence of operations, inputs, and transformations",
        design_features=[
            "Comprehensive logging",
            "Version control for artifacts",
            "Explicit provenance metadata",
            "Checkpoint preservation"
        ],
        metrics=[
            MetricDefinition(
                metric_id="provenance_completeness",
                name="Provenance Completeness",
                description="Proportion of outputs with complete provenance metadata",
                unit="ratio",
                higher_is_better=True,
                min_value=0.0,
                max_value=1.0,
                calculation_method="metadata_analysis"
            ),
            MetricDefinition(
                metric_id="stage_linkage",
                name="Stage Linkage",
                description="Whether outputs can be traced to input stages",
                unit="ratio",
                higher_is_better=True,
                min_value=0.0,
                max_value=1.0,
                calculation_method="dependency_analysis"
            ),
        ]
    ),

    "reproducibility": DimensionDefinition(
        dimension_id="reproducibility",
        name="Reproducibility",
        cluster=DimensionCluster.EPISTEMIC,
        description="Ability to obtain consistent results when repeating procedures",
        theoretical_basis="Obtaining consistent results with identical procedures and inputs",
        design_features=[
            "Deterministic workflow orchestration",
            "Comprehensive configuration documentation",
            "Version-controlled dependencies",
            "Automated replication package generation"
        ],
        metrics=[
            MetricDefinition(
                metric_id="replication_success_rate",
                name="Replication Success Rate",
                description="Proportion of runs producing consistent outputs",
                unit="ratio",
                higher_is_better=True,
                min_value=0.0,
                max_value=1.0,
                calculation_method="output_comparison",
                requires_multiple_runs=True
            ),
            MetricDefinition(
                metric_id="configuration_completeness",
                name="Configuration Completeness",
                description="Coverage of documented configuration parameters",
                unit="ratio",
                higher_is_better=True,
                min_value=0.0,
                max_value=1.0,
                calculation_method="config_analysis"
            ),
        ]
    ),

    # -------------------------------------------------------------------------
    # INNOVATION CLUSTER
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # DOMAIN CLUSTER
    # -------------------------------------------------------------------------

    "decision_quality": DimensionDefinition(
        dimension_id="decision_quality",
        name="Decision Quality",
        cluster=DimensionCluster.QUALITY,
        description="Quality of agent decision sequences during workflow execution",
        theoretical_basis="Evaluates whether agents make well-justified decisions at each step, including tool selection, information routing, and output formatting",
        design_features=[
            "Logged decision points in execution traces",
            "Agent-as-Judge trajectory evaluation",
            "Cross-stage decision coherence checks",
            "Explicit justification requirements at decision points",
        ],
        metrics=[
            MetricDefinition(
                metric_id="decision_coherence",
                name="Decision Coherence",
                description="Whether sequential decisions follow logically from prior context",
                unit="score",
                higher_is_better=True,
                min_value=0.0,
                max_value=1.0,
                calculation_method="trajectory_analysis",
            ),
            MetricDefinition(
                metric_id="tool_selection_appropriateness",
                name="Tool Selection Appropriateness",
                description="Whether agents chose appropriate tools for each task step",
                unit="score",
                higher_is_better=True,
                min_value=0.0,
                max_value=1.0,
                calculation_method="llm_evaluation",
            ),
        ],
    ),

    "economic_rigor": DimensionDefinition(
        dimension_id="economic_rigor",
        name="Economic Rigor",
        cluster=DimensionCluster.QUALITY,
        description="Domain-specific validity of economics content produced by agents",
        theoretical_basis="Assesses whether outputs reflect established economics methodology: proper use of terminology, appropriate theoretical frameworks, valid empirical approaches, and adherence to field norms",
        design_features=[
            "Economics-specific evaluation rubrics",
            "Benchmark comparison against curated exemplars",
            "Domain terminology validation",
            "Methodological appropriateness checks",
        ],
        metrics=[
            MetricDefinition(
                metric_id="terminology_accuracy",
                name="Terminology Accuracy",
                description="Correct use of economics terminology and concepts",
                unit="score",
                higher_is_better=True,
                min_value=0.0,
                max_value=1.0,
                calculation_method="llm_evaluation",
            ),
            MetricDefinition(
                metric_id="methodology_appropriateness",
                name="Methodology Appropriateness",
                description="Whether proposed methods match standard economics practice",
                unit="score",
                higher_is_better=True,
                min_value=0.0,
                max_value=1.0,
                calculation_method="llm_evaluation",
            ),
        ],
    ),

    # -------------------------------------------------------------------------
    # INNOVATION CLUSTER
    # -------------------------------------------------------------------------

    "innovation_potential": DimensionDefinition(
        dimension_id="innovation_potential",
        name="Innovation Potential",
        cluster=DimensionCluster.INNOVATION,
        description="Capacity for generating novel, valuable ideas or approaches",
        theoretical_basis="Generation of novel, valuable ideas through analogical reasoning, cross-domain transfer, and creative recombination",
        design_features=[
            "Multi-agent exploration of diverse sources",
            "Explicit gap identification agents",
            "Ideation workflows",
            "Cross-domain integration capabilities"
        ],
        metrics=[
            MetricDefinition(
                metric_id="output_diversity",
                name="Output Diversity",
                description="Diversity of generated outputs (entropy-based)",
                unit="entropy",
                higher_is_better=True,
                min_value=0.0,
                max_value=1.0,
                calculation_method="diversity_calculation"
            ),
            MetricDefinition(
                metric_id="coverage_breadth",
                name="Coverage Breadth",
                description="Breadth of topics/sources covered",
                unit="count",
                higher_is_better=True,
                min_value=0.0,
                max_value=100.0,
                calculation_method="coverage_analysis"
            ),
            MetricDefinition(
                metric_id="embedding_novelty",
                name="Embedding Novelty",
                description="Cosine distance from output embeddings to reference literature embeddings",
                unit="distance",
                higher_is_better=True,
                min_value=0.0,
                max_value=1.0,
                calculation_method="embedding_comparison"
            ),
            MetricDefinition(
                metric_id="cross_domain_score",
                name="Cross-Domain Score",
                description="Count of distinct economic subfields meaningfully referenced",
                unit="score",
                higher_is_better=True,
                min_value=0.0,
                max_value=1.0,
                calculation_method="keyword_taxonomy"
            ),
            MetricDefinition(
                metric_id="diversity_index",
                name="Diversity Index",
                description="Pairwise embedding distance entropy across generated ideas",
                unit="index",
                higher_is_better=True,
                min_value=0.0,
                max_value=1.0,
                calculation_method="embedding_pairwise"
            ),
        ]
    ),

    # -------------------------------------------------------------------------
    # ECONOMICS CLUSTER (V0.7)
    # -------------------------------------------------------------------------

    "simulation_fidelity": DimensionDefinition(
        dimension_id="simulation_fidelity",
        name="Simulation Fidelity",
        cluster=DimensionCluster.ECONOMICS,
        description=(
            "How closely an economic simulation's outputs match closed-form "
            "or published benchmarks for the chosen scenario."
        ),
        theoretical_basis=(
            "Benchmark-anchored validation of agent-based and DSGE simulations. "
            "Reference: ABIDES-Economist (arXiv 2402.09563v2), LLM-Economist "
            "(arXiv 2507.15815), MALLES (arXiv 2603.17694)."
        ),
        design_features=[
            "Compares SimulationResult.metrics to BENCHMARKS anchors",
            "Per-metric normalised deviation with configurable tolerance",
            "Fraction of metrics within tolerance becomes the dimension score",
        ],
        metrics=[
            MetricDefinition(
                metric_id="benchmark_match_rate",
                name="Benchmark Match Rate",
                description="Fraction of scenario metrics within tolerance of benchmark",
                unit="ratio",
                higher_is_better=True,
                min_value=0.0,
                max_value=1.0,
                calculation_method="simulation_fidelity_evaluator",
            ),
        ],
    ),

    "causal_validity": DimensionDefinition(
        dimension_id="causal_validity",
        name="Causal Validity",
        cluster=DimensionCluster.ECONOMICS,
        description=(
            "How well the pipeline's causal estimate (from CausalFM, DoWhy, "
            "or the OLS/2SLS/front-door fallback) matches ground-truth ATE "
            "(synthetic benchmarks) or a prior mean within the bootstrap CI."
        ),
        theoretical_basis=(
            "Identification robustness under stated adjustment sets. "
            "Reference: CausalFM (ICLR 2026), DoWhy, EconML."
        ),
        design_features=[
            "Compares CausalEstimate.ate to true_ate when available",
            "Falls back to prior-mean-in-CI coverage check",
            "Linear falloff outside the CI, clamped to [0, 1]",
        ],
        metrics=[
            MetricDefinition(
                metric_id="causal_validity_score",
                name="Causal Validity Score",
                description="|ate - true_ate| / tolerance, or CI coverage of prior",
                unit="ratio",
                higher_is_better=True,
                min_value=0.0,
                max_value=1.0,
                calculation_method="causal_validity_evaluator",
            ),
        ],
    ),
}


# =============================================================================
# CLUSTER DEFINITIONS
# =============================================================================

DIMENSION_CLUSTERS: Dict[str, List[str]] = {
    "Quality": ["reliability", "correctness", "soundness"],
    "Operational": ["efficiency", "scalability", "robustness"],
    "Epistemic": ["transparency", "traceability", "reproducibility"],
    "Innovation": ["innovation_potential"],
    "Domain": ["decision_quality", "economic_rigor"],
    "Economics": ["simulation_fidelity", "causal_validity"],
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_dimension(dimension_id: str) -> Optional[DimensionDefinition]:
    """Get a dimension definition by ID."""
    return DIMENSIONS.get(dimension_id)


def get_dimensions_by_cluster(cluster: str) -> List[DimensionDefinition]:
    """Get all dimensions in a cluster."""
    dimension_ids = DIMENSION_CLUSTERS.get(cluster, [])
    return [DIMENSIONS[did] for did in dimension_ids if did in DIMENSIONS]


def get_all_metrics() -> List[MetricDefinition]:
    """Get all metric definitions across all dimensions."""
    metrics = []
    for dimension in DIMENSIONS.values():
        metrics.extend(dimension.metrics)
    return metrics


def get_metrics_requiring_multiple_runs() -> List[MetricDefinition]:
    """Get metrics that require multiple runs to calculate."""
    return [m for m in get_all_metrics() if m.requires_multiple_runs]
