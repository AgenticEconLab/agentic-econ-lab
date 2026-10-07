# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
DataTeam stage input schemas.

Defines the validated input contracts for each stage in the Data pipeline.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class DataSourceStageInput(BaseModel):
    """Input for Stage 1: Data Source Discovery & Retrieval."""
    research_question: str = Field(description="Research question requiring data")
    data_requirements: List[Dict] = Field(
        default_factory=list,
        description="Specific data requirements (optional; agents will derive from question)")
    workflow_type: str = Field(
        default="open_source_api",
        description="Workflow type: open_source_api, premium_subscribed, user_uploaded")
    file_path: Optional[str] = Field(
        default=None, description="Path to uploaded file (for user_uploaded workflow)")


class DataCleaningStageInput(BaseModel):
    """Input for Stage 2: Data Cleaning. Consumes Stage 1 output."""
    data_source_output_path: str = Field(
        description="Path to data source output JSON from Stage 1")


class QualityAssuranceStageInput(BaseModel):
    """Input for Stage 3: Quality Assurance. Consumes Stage 2 output."""
    data_cleaning_output_path: str = Field(
        description="Path to data cleaning output JSON from Stage 2")
