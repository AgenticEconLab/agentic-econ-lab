# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
LiteratureTeam stage input schemas.

Defines the validated input contracts for each stage in the Literature pipeline.
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


class GatheringStageInput(BaseModel):
    """Input for Stage 1: Literature Gathering."""
    research_questions: List[ResearchQuestionInput] = Field(
        description="Research questions from IdeationTeam")
    max_results_per_question: int = Field(default=15, description="Max papers per question")
    use_firecrawl: bool = Field(default=False, description="Whether to use Firecrawl")


class GapDetectionStageInput(BaseModel):
    """Input for Stage 2: Gap Detection. Consumes Stage 1 output."""
    literature_batch_path: str = Field(
        description="Path to literature_batch.json from Stage 1")


class SynthesisStageInput(BaseModel):
    """Input for Stage 3: Synthesis. Consumes Stage 2 output."""
    gap_analysis_path: str = Field(
        description="Path to gap_analysis_results.json from Stage 2")
    literature_batch_path: str = Field(
        description="Path to literature_batch.json from Stage 1")
