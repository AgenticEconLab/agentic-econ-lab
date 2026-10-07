# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Execution trace schemas for workflow evaluation.

These models capture the execution characteristics of workflow runs,
including timing, errors, and output artifacts.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
import uuid


class ErrorInfo(BaseModel):
    """Information about an error that occurred during execution."""

    error_type: str = Field(..., description="Type/class of the error")
    error_message: str = Field(..., description="Error message")
    stage: Optional[str] = Field(None, description="Stage where error occurred")
    timestamp: datetime = Field(default_factory=datetime.now)
    traceback: Optional[str] = Field(None, description="Full traceback if available")
    recovered: bool = Field(False, description="Whether the error was recovered from")

    class Config:
        json_schema_extra = {
            "example": {
                "error_type": "APIError",
                "error_message": "Rate limit exceeded",
                "stage": "Sourcing",
                "recovered": True
            }
        }


class TimingInfo(BaseModel):
    """Timing information for a workflow or stage execution."""

    start_time: datetime = Field(..., description="Start timestamp")
    end_time: Optional[datetime] = Field(None, description="End timestamp")
    duration_seconds: Optional[float] = Field(None, description="Duration in seconds")

    def compute_duration(self) -> float:
        """Compute duration if end_time is set."""
        if self.end_time and self.start_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        return self.duration_seconds or 0.0


class StageExecution(BaseModel):
    """Execution information for a single workflow stage."""

    stage_name: str = Field(..., description="Name of the stage")
    stage_number: int = Field(..., description="Stage number (1, 2, 3)")
    timing: TimingInfo = Field(..., description="Timing information")
    status: Literal["pending", "running", "success", "error", "partial"] = Field(
        "pending", description="Execution status"
    )
    error_message: Optional[str] = Field(None, description="Error message if failed")
    output_files: List[str] = Field(default_factory=list, description="Generated output files")
    item_count: Optional[int] = Field(None, description="Number of items processed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "stage_name": "Sourcing",
                "stage_number": 1,
                "status": "success",
                "output_files": ["literature_results_automated.csv"],
                "item_count": 42
            }
        }


class WorkflowOutputs(BaseModel):
    """Collection of output files from a workflow run."""

    json_files: Dict[str, str] = Field(default_factory=dict, description="JSON output files")
    csv_files: Dict[str, str] = Field(default_factory=dict, description="CSV output files")
    text_files: Dict[str, str] = Field(default_factory=dict, description="Text/Markdown files")
    other_files: List[str] = Field(default_factory=list, description="Other output files")

    def all_files(self) -> List[str]:
        """Get list of all output files."""
        return (
            list(self.json_files.values()) +
            list(self.csv_files.values()) +
            list(self.text_files.values()) +
            self.other_files
        )


class ExecutionTrace(BaseModel):
    """Complete execution trace for a workflow run."""

    # Identification
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    team: str = Field(..., description="Team name (IdeationTeam, LiteratureTeam, etc.)")
    mode: str = Field(..., description="Mode (ModeNoWcNoHITL, ModeWithWcWithHITL, etc.)")
    framework: Literal["ael"] = Field(
        "ael", description="Framework used (AEL — the project's stage-based framework)"
    )

    # Timing
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = Field(None)
    total_duration_seconds: Optional[float] = Field(None)

    # Execution details
    stages: List[StageExecution] = Field(default_factory=list)
    errors: List[ErrorInfo] = Field(default_factory=list)

    # Input/Output
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Input parameters")
    outputs: WorkflowOutputs = Field(default_factory=WorkflowOutputs)

    # Metadata
    schema_version: str = Field("1.0.0", description="Schema version for compatibility")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def complete(self) -> None:
        """Mark the execution as complete and compute duration."""
        self.end_time = datetime.now()
        if self.start_time:
            self.total_duration_seconds = (self.end_time - self.start_time).total_seconds()

    def add_stage(self, stage: StageExecution) -> None:
        """Add a stage execution record."""
        self.stages.append(stage)

    def add_error(self, error: ErrorInfo) -> None:
        """Add an error record."""
        self.errors.append(error)

    @property
    def success(self) -> bool:
        """Check if execution completed successfully."""
        return all(s.status == "success" for s in self.stages)

    @property
    def error_count(self) -> int:
        """Count total errors."""
        return len(self.errors)

    @property
    def recovered_error_count(self) -> int:
        """Count recovered errors."""
        return sum(1 for e in self.errors if e.recovered)

    class Config:
        json_schema_extra = {
            "example": {
                "run_id": "abc12345",
                "team": "IdeationTeam",
                "mode": "ModeNoWcNoHITL",
                "framework": "ael",
                "total_duration_seconds": 125.5
            }
        }
