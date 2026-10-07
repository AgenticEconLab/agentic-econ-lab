# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
LiteratureTeam AEL Schemas — Canonical Pydantic models for stage inputs/outputs.
"""

from LiteratureTeam.ael.schemas.stage_inputs import (
    GatheringStageInput,
    GapDetectionStageInput,
    SynthesisStageInput,
)
from LiteratureTeam.ael.schemas.stage_outputs import (
    LiteratureItem,
    TrendAnalysis,
    CitationEntry,
    KnowledgeInsight,
    LiteratureBatch,
    GatheringStageOutput,
    PaperStructure,
    ResearchGap,
    KnowledgeNode,
    KnowledgeEdge,
    KnowledgeGraph,
    GapAnalysisResult,
    GapDetectionStageOutput,
    LiteratureSection,
    LiteratureReview,
    ResearchObjective,
    ResearchPlan,
    FormattedReference,
    Bibliography,
    CrossDomainConnection,
    SynthesisResult,
    SynthesisStageOutput,
)

__all__ = [
    # Inputs
    "GatheringStageInput",
    "GapDetectionStageInput",
    "SynthesisStageInput",
    # Outputs
    "LiteratureItem",
    "TrendAnalysis",
    "CitationEntry",
    "KnowledgeInsight",
    "LiteratureBatch",
    "GatheringStageOutput",
    "PaperStructure",
    "ResearchGap",
    "KnowledgeNode",
    "KnowledgeEdge",
    "KnowledgeGraph",
    "GapAnalysisResult",
    "GapDetectionStageOutput",
    "LiteratureSection",
    "LiteratureReview",
    "ResearchObjective",
    "ResearchPlan",
    "FormattedReference",
    "Bibliography",
    "CrossDomainConnection",
    "SynthesisResult",
    "SynthesisStageOutput",
]
