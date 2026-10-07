# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
ModelTeam stage input schemas.

Defines the validated input contracts for each stage in the Modeling pipeline.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ResearchQuestionInput(BaseModel):
    """A research question from IdeationTeam output."""
    question: str = Field(description="The research question")
    priority_rank: Optional[int] = Field(default=None, description="Priority ranking")
    priority_score: Optional[float] = Field(default=None, description="Priority score")
    theoretical_framework: Optional[str] = Field(default="", description="Theoretical framework")
    methodology: Optional[List[str]] = Field(default_factory=list, description="Suggested methodologies")


class LiteratureInput(BaseModel):
    """Simplified literature item from LiteratureTeam output."""
    title: str = Field(description="Title of the paper/document")
    authors: List[str] = Field(description="List of authors")
    abstract: str = Field(description="Abstract or summary")
    key_findings: Optional[List[str]] = Field(default_factory=list, description="Key findings")
    methodologies: Optional[List[str]] = Field(default_factory=list, description="Methodologies used")


class TheoryStageInput(BaseModel):
    """Input for Stage 1: Theory Development."""
    research_questions: List[ResearchQuestionInput] = Field(
        description="Research questions from IdeationTeam")
    literature_items: List[LiteratureInput] = Field(
        default_factory=list, description="Literature from LiteratureTeam (optional)")
    model_type: str = Field(default="dsge", description="Model type: dsge, abm, micro, macro")


class ModelDesignStageInput(BaseModel):
    """Input for Stage 2: Model Design. Consumes Stage 1 output."""
    theory_output_path: str = Field(
        description="Path to theory_output.json from Stage 1")


class CalibrationStageInput(BaseModel):
    """Input for Stage 3: Calibration. Consumes Stage 2 output."""
    model_design_path: str = Field(
        description="Path to model_design_output.json from Stage 2")


# Backward-compatible aliases (inline stage files used shorter names)
ResearchQuestion = ResearchQuestionInput
LiteratureItem = LiteratureInput
