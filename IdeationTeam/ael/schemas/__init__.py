# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
IdeationTeam AEL Schemas — Canonical Pydantic models for stage inputs/outputs.

These schemas define the validated data contracts between stages.
All stage files MUST import from here — no inline model definitions (V0.4).
"""

from IdeationTeam.ael.schemas.stage_inputs import (
    SourcingStageInput,
    RefinementStageInput,
    IntegrationStageInput,
)
from IdeationTeam.ael.schemas.stage_outputs import (
    LiteratureItem,
    SearchQuery,
    SourcingFeedback,
    SourcingStageOutput,
    ResearchConcept,
    EvolutionTrace,
    ResearchQuestion,
    ConceptList,
    QuestionList,
    HumanFeedback,
    StageTransitionSummary,
    RefinementStageOutput,
    ContextualizedQuestion,
    PrioritizedQuestion,
    IntegrationFeedback,
    ContextualizedQuestionList,
    PrioritizedQuestionList,
    IntegrationStageOutput,
)

__all__ = [
    # Inputs
    "SourcingStageInput",
    "RefinementStageInput",
    "IntegrationStageInput",
    # Stage 1 outputs
    "LiteratureItem",
    "SearchQuery",
    "SourcingFeedback",
    "SourcingStageOutput",
    # Stage 2 outputs
    "ResearchConcept",
    "EvolutionTrace",
    "ResearchQuestion",
    "ConceptList",
    "QuestionList",
    "HumanFeedback",
    "StageTransitionSummary",
    "RefinementStageOutput",
    # Stage 3 outputs
    "ContextualizedQuestion",
    "PrioritizedQuestion",
    "IntegrationFeedback",
    "ContextualizedQuestionList",
    "PrioritizedQuestionList",
    "IntegrationStageOutput",
]
