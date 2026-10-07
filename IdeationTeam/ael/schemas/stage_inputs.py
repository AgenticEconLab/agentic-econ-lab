# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
IdeationTeam stage input schemas.

Defines the validated input contracts for each stage in the Ideation pipeline.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class SourcingStageInput(BaseModel):
    """Input for Stage 1: Sourcing."""
    research_topic: str = Field(description="Research topic or area to explore")
    max_results_per_agent: int = Field(default=15, description="Maximum results per sourcing agent")
    use_firecrawl: bool = Field(default=False, description="Whether to use Firecrawl for web scraping")


class RefinementStageInput(BaseModel):
    """Input for Stage 2: Refinement. Consumes Stage 1 output."""
    literature_csv_path: str = Field(description="Path to literature_results CSV from Stage 1")
    num_concepts: int = Field(default=10, description="Number of research concepts to generate")
    num_questions: int = Field(default=8, description="Number of research questions to refine")
    enable_hitl: bool = Field(default=False, description="Whether human-in-the-loop feedback is enabled")


class IntegrationStageInput(BaseModel):
    """Input for Stage 3: Integration. Consumes Stage 2 output."""
    refinement_json_path: str = Field(description="Path to refinement_results JSON from Stage 2")
    max_final_questions: int = Field(default=5, description="Maximum number of final prioritized questions")
    enable_hitl: bool = Field(default=False, description="Whether human-in-the-loop feedback is enabled")
