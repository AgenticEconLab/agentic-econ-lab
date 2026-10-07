# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
ModelTeam AEL Schemas — Canonical Pydantic models for stage inputs/outputs.
"""

from ModelTeam.ael.schemas.stage_inputs import (
    ResearchQuestionInput,
    LiteratureInput,
    TheoryStageInput,
    ModelDesignStageInput,
    CalibrationStageInput,
    # Backward-compatible aliases
    ResearchQuestion,
    LiteratureItem,
)
from ModelTeam.ael.schemas.stage_outputs import (
    TheoreticalAssumption,
    MathematicalFormulation,
    ConceptualComponent,
    TheoreticalFramework,
    TheoryStageOutput,
    ModelVariable,
    ModelParameter,
    ModelEquation,
    ModelConstraint,
    ModelDynamics,
    EquilibriumDefinition,
    SolutionMethod,
    FormalMathematicalModel,
    ModelDesignStageOutput,
    EmpiricalTarget,
    CalibrationStrategy,
    CalibratedParameter,
    ModelMoment,
    FitMetrics,
    SandboxValidation,
    CalibratedModel,
    CalibrationStageOutput,
    # Backward-compatible aliases
    ModelDesignOutput,
    CalibrationOutput,
)

__all__ = [
    # Inputs
    "ResearchQuestionInput",
    "LiteratureInput",
    "TheoryStageInput",
    "ModelDesignStageInput",
    "CalibrationStageInput",
    # Input aliases
    "ResearchQuestion",
    "LiteratureItem",
    # Outputs
    "TheoreticalAssumption",
    "MathematicalFormulation",
    "ConceptualComponent",
    "TheoreticalFramework",
    "TheoryStageOutput",
    "ModelVariable",
    "ModelParameter",
    "ModelEquation",
    "ModelConstraint",
    "ModelDynamics",
    "EquilibriumDefinition",
    "SolutionMethod",
    "FormalMathematicalModel",
    "ModelDesignStageOutput",
    "EmpiricalTarget",
    "CalibrationStrategy",
    "CalibratedParameter",
    "ModelMoment",
    "FitMetrics",
    "SandboxValidation",
    "CalibratedModel",
    "CalibrationStageOutput",
    # Output aliases
    "ModelDesignOutput",
    "CalibrationOutput",
]
