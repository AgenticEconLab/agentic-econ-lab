# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
ModelTeam stage output schemas.

Defines the validated output contracts for each stage in the Modeling pipeline.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from shared.schema_coerce import LLMCoercedModel


# ============================================================================
# Stage 1: Theory — Output Models
# ============================================================================

class TheoreticalAssumption(BaseModel):
    """A theoretical assumption in the framework."""
    assumption_id: str = Field(description="Unique assumption identifier")
    assumption_statement: str = Field(description="Clear statement of the assumption")
    justification: str = Field(description="Justification from literature or theory")
    type: str = Field(description="Type: behavioral, structural, parametric, distributional")
    criticality: str = Field(description="Critical/Important/Standard")
    supporting_literature: List[str] = Field(description="Papers supporting this assumption")
    potential_relaxations: List[str] = Field(description="How this could be relaxed")


class MathematicalFormulation(BaseModel):
    """A mathematical formulation in the framework."""
    equation_id: str = Field(description="Unique equation identifier")
    equation_name: str = Field(description="Name/description of equation")
    equation_latex: str = Field(description="LaTeX representation")
    equation_plain: str = Field(description="Plain text representation")
    variables: List[str] = Field(description="Variables in the equation")
    parameters: List[str] = Field(description="Parameters in the equation")
    interpretation: str = Field(description="Economic interpretation")
    derivation_notes: str = Field(description="Brief derivation notes")


class ConceptualComponent(BaseModel):
    """A conceptual component of the theoretical framework."""
    component_id: str = Field(description="Unique component identifier")
    component_name: str = Field(description="Name of the component")
    component_type: str = Field(description="Type: agent, market, institution, mechanism, constraint")
    description: str = Field(description="Detailed description")
    key_features: List[str] = Field(description="Key features (3-5 points)")
    interactions: List[str] = Field(description="Interactions with other components")
    literature_basis: List[str] = Field(description="Literature supporting this component")


class TheoreticalFramework(BaseModel):
    """A complete theoretical framework."""
    framework_title: str = Field(description="Title of the framework")
    framework_overview: str = Field(description="High-level overview")
    research_question: str = Field(description="Primary research question addressed")
    theoretical_approach: str = Field(description="Approach (e.g., DSGE, ABM, Game Theory)")
    assumptions: List[TheoreticalAssumption] = Field(description="Theoretical assumptions")
    conceptual_components: List[ConceptualComponent] = Field(description="Conceptual components")
    mathematical_formulations: List[MathematicalFormulation] = Field(description="Key equations")
    model_structure: str = Field(description="Overall model structure description")
    equilibrium_concept: Optional[str] = Field(default="", description="Equilibrium concept")
    solution_approach: str = Field(description="Proposed solution approach")
    literature_connections: List[str] = Field(description="Connections to existing literature")
    innovation_points: List[str] = Field(description="Novel aspects of the framework")
    testable_predictions: List[str] = Field(description="Testable predictions (3-5)")
    metadata: Dict = Field(default_factory=dict, description="Framework metadata")


class TheoryStageOutput(BaseModel):
    """Complete output of Stage 1: Theory."""
    research_questions: List[Dict] = Field(description="Input research questions")
    theoretical_frameworks: List[TheoreticalFramework] = Field(description="Developed frameworks")
    framework_comparison: Optional[str] = Field(default="", description="Comparison if multiple")
    metadata: Dict = Field(default_factory=dict, description="Output metadata")


# ============================================================================
# Stage 2: Model Design — Output Models
# ============================================================================

class ModelVariable(BaseModel):
    """A variable in the mathematical model."""
    variable_id: str = Field(description="Unique variable identifier")
    variable_symbol: str = Field(description="Mathematical symbol (e.g., 'c_t', 'Y')")
    variable_name: str = Field(description="Full name")
    variable_type: str = Field(description="Type: endogenous, exogenous, state, control, parameter")
    description: str = Field(description="Description of the variable")
    domain: str = Field(description="Domain/range (e.g., 'R+', '[0,1]')")
    time_subscript: bool = Field(description="Whether variable has time subscript")
    agent_subscript: Optional[str] = Field(default="", description="Agent subscript if applicable")


class ModelParameter(LLMCoercedModel):
    """A parameter in the mathematical model. LLM-parsed -> coerce richer shapes
    (e.g. list/dict arriving for a str field) to the declared scalar/list type."""
    parameter_id: str = Field(description="Unique parameter identifier")
    parameter_symbol: str = Field(description="Mathematical symbol (e.g., 'beta')")
    parameter_name: str = Field(description="Full name")
    description: str = Field(description="Economic interpretation")
    typical_range: str = Field(description="Typical range in literature")
    calibration_source: str = Field(description="How to calibrate/estimate")


class ModelEquation(BaseModel):
    """An equation in the mathematical model."""
    equation_id: str = Field(description="Unique equation identifier")
    equation_type: str = Field(description="Type: behavioral, equilibrium, identity, constraint, law_of_motion")
    equation_name: str = Field(description="Name of the equation")
    equation_latex: str = Field(description="LaTeX representation")
    equation_plain: str = Field(description="Plain text representation")
    variables_used: List[str] = Field(description="Variable symbols used")
    parameters_used: List[str] = Field(description="Parameter symbols used")
    interpretation: str = Field(description="Economic interpretation")
    derivation: str = Field(description="How equation is derived")
    timing: Optional[str] = Field(default="", description="Timing convention")


class ModelConstraint(BaseModel):
    """A constraint in the mathematical model."""
    constraint_id: str = Field(description="Unique constraint identifier")
    constraint_type: str = Field(description="Type: budget, resource, feasibility, non_negativity, boundary")
    constraint_latex: str = Field(description="LaTeX representation")
    constraint_plain: str = Field(description="Plain text representation")
    description: str = Field(description="Description of the constraint")
    binding_conditions: str = Field(description="When constraint binds")


class ModelDynamics(BaseModel):
    """Dynamic structure of the model."""
    time_structure: str = Field(description="Discrete/Continuous time")
    time_horizon: str = Field(description="Finite/Infinite horizon")
    state_variables: List[str] = Field(description="State variable symbols")
    control_variables: List[str] = Field(description="Control variable symbols")
    transition_equations: List[str] = Field(description="State transition equation IDs")
    initial_conditions: List[str] = Field(description="Initial conditions")
    terminal_conditions: List[str] = Field(description="Terminal conditions if applicable")


class EquilibriumDefinition(BaseModel):
    """Equilibrium definition for the model."""
    equilibrium_type: str = Field(description="Type: competitive, Nash, rational_expectations, etc.")
    equilibrium_conditions: List[str] = Field(description="Conditions defining equilibrium")
    market_clearing: List[str] = Field(description="Market clearing conditions")
    optimality_conditions: List[str] = Field(description="Optimality conditions (FOCs, etc.)")
    consistency_requirements: List[str] = Field(description="Consistency requirements")


class SolutionMethod(BaseModel):
    """Solution methodology for the model."""
    solution_approach: str = Field(description="Analytical/Numerical/Simulation")
    solution_steps: List[str] = Field(description="Steps to solve the model")
    computational_methods: List[str] = Field(description="Computational methods needed")
    software_requirements: List[str] = Field(description="Software/packages needed")
    expected_challenges: List[str] = Field(description="Expected computational challenges")


class FormalMathematicalModel(BaseModel):
    """Complete formal mathematical model specification."""
    model_title: str = Field(description="Title of the model")
    model_summary: str = Field(description="High-level summary")
    based_on_framework: str = Field(description="Source theoretical framework title")
    variables: List[ModelVariable] = Field(description="All variables")
    parameters: List[ModelParameter] = Field(description="All parameters")
    equations: List[ModelEquation] = Field(description="Complete equation system")
    constraints: List[ModelConstraint] = Field(description="All constraints")
    dynamics: ModelDynamics = Field(description="Dynamic structure")
    equilibrium: EquilibriumDefinition = Field(description="Equilibrium definition")
    solution_method: SolutionMethod = Field(description="Solution methodology")
    model_notation: str = Field(description="Notation conventions")
    model_assumptions_recap: List[str] = Field(description="Key assumptions recap")
    model_extensions: List[str] = Field(description="Possible extensions")
    metadata: Dict = Field(default_factory=dict, description="Model metadata")


class ModelDesignStageOutput(BaseModel):
    """Complete output of Stage 2: Model Design."""
    theoretical_frameworks: List[Dict] = Field(description="Input frameworks (simplified)")
    formal_models: List[FormalMathematicalModel] = Field(description="Designed formal models")
    metadata: Dict = Field(default_factory=dict, description="Output metadata")


# ============================================================================
# Stage 3: Calibration — Output Models
# ============================================================================

class EmpiricalTarget(LLMCoercedModel):
    """An empirical target or moment for calibration."""
    target_id: str = Field(description="Unique target identifier")
    target_name: str = Field(description="Name of the target")
    target_type: str = Field(description="Type: moment, ratio, elasticity, correlation, volatility")
    description: str = Field(description="Description of what this measures")
    empirical_value: float = Field(description="Empirical value from data/literature")
    empirical_source: str = Field(default="unspecified", description="Source of empirical value")
    standard_error: Optional[float] = Field(default=None, description="Standard error")
    related_variables: List[str] = Field(description="Model variables related to this target")
    related_parameters: List[str] = Field(description="Parameters that affect this target")
    importance: str = Field(description="High/Medium/Low importance")


class CalibrationStrategy(BaseModel):
    """Calibration strategy for the model."""
    strategy_type: str = Field(description="Type: direct, indirect, SMM, GMM, maximum_likelihood")
    description: str = Field(description="Description of the strategy")
    parameters_to_calibrate: List[str] = Field(description="Parameter symbols to calibrate")
    targets_to_match: List[str] = Field(description="Target IDs to match")
    calibration_order: List[str] = Field(description="Order of parameter calibration")
    identification_notes: str = Field(description="Notes on parameter identification")


class CalibratedParameter(LLMCoercedModel):
    """A calibrated parameter value. LLM-parsed -> coerce richer shapes (e.g. a
    list like [0.4, 0.8] arriving for the str field ``literature_range``) to the
    declared scalar/list type instead of raising a ValidationError."""
    parameter_symbol: str = Field(description="Parameter symbol")
    parameter_name: str = Field(description="Parameter name")
    calibrated_value: float = Field(description="Calibrated value")
    calibration_method: str = Field(description="How it was calibrated")
    target_matched: Optional[str] = Field(default="", description="Target matched")
    literature_range: str = Field(description="Typical range in literature")
    justification: str = Field(description="Justification for this value")
    sensitivity: str = Field(description="High/Medium/Low sensitivity")


class ModelMoment(BaseModel):
    """A moment generated by the calibrated model."""
    moment_name: str = Field(description="Name of the moment")
    model_value: float = Field(description="Value from calibrated model")
    empirical_value: float = Field(description="Empirical target value")
    absolute_error: float = Field(description="Absolute error")
    relative_error: float = Field(description="Relative error (percentage)")
    fit_quality: str = Field(description="Excellent/Good/Fair/Poor")


class FitMetrics(BaseModel):
    """Overall fit metrics for calibration."""
    total_targets: int = Field(description="Total number of targets")
    targets_matched: int = Field(description="Number of well-matched targets")
    mean_absolute_error: float = Field(description="Mean absolute error across targets")
    mean_relative_error: float = Field(description="Mean relative error (percentage)")
    rmse: float = Field(description="Root mean squared error")
    fit_score: float = Field(description="Overall fit score (0-1)")
    fit_assessment: str = Field(description="Overall assessment")
    harness_details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured deterministic-harness result: verdict, parameters the harness "
                    "estimated, cited targets it matched, simulator used and whether it is a "
                    "canonical stand-in (archetype). The authoritative source for reports.")


class SandboxValidation(BaseModel):
    """Result of sandbox code execution for calibration validation."""
    validated: bool = Field(description="Whether the validation PASSED — the code ran AND its printed verdict was 'pass' (NOT merely that the process exited 0)")
    verdict: str = Field(description="Parsed sandbox verdict from stdout: pass / fail / unknown / error", default="")
    validation_method: str = Field(description="Validation method used", default="sandbox_execution")
    code: str = Field(description="The generated validation code", default="")
    stdout: str = Field(description="Sandbox stdout output", default="")
    stderr: str = Field(description="Sandbox stderr output", default="")
    error: str = Field(description="Error message if validation failed", default="")
    execution_time_sec: float = Field(description="Execution time in seconds", default=0.0)


class CalibratedModel(BaseModel):
    """Complete calibrated model output."""
    model_title: str = Field(description="Title of the calibrated model")
    based_on_model: str = Field(description="Source formal model title")
    calibration_summary: str = Field(description="Summary of calibration")
    empirical_targets: List[EmpiricalTarget] = Field(description="Empirical targets")
    calibration_strategy: CalibrationStrategy = Field(description="Calibration strategy")
    calibrated_parameters: List[CalibratedParameter] = Field(description="Calibrated parameters")
    model_moments: List[ModelMoment] = Field(description="Model-generated moments")
    fit_metrics: FitMetrics = Field(description="Fit metrics")
    sandbox_validation: Optional[SandboxValidation] = Field(
        default=None, description="Result of sandbox-based calibration validation")
    calibration_notes: str = Field(description="Additional calibration notes")
    robustness_checks: List[str] = Field(description="Suggested robustness checks")
    next_steps: List[str] = Field(description="Next steps for model implementation")
    metadata: Dict = Field(default_factory=dict, description="Calibration metadata")


class CalibrationStageOutput(BaseModel):
    """Complete output of Stage 3: Calibration."""
    formal_models: List[Dict] = Field(description="Input formal models (simplified)")
    calibrated_models: List[CalibratedModel] = Field(description="Calibrated models")
    metadata: Dict = Field(default_factory=dict, description="Output metadata")


# Backward-compatible aliases (inline stage files used shorter names)
ModelDesignOutput = ModelDesignStageOutput
CalibrationOutput = CalibrationStageOutput
