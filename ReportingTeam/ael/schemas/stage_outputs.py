# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Stage output contracts for ReportingTeam (§4.7) — thin envelopes over report_harness types."""

from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, Field

from ReportingTeam.ael.report_harness.types import (
    ConsistencyReport,
    DraftingResult,
    InterpretationResult,
    QualityResult,
)


class InterpretationStageOutput(BaseModel):
    """Stage 1: deterministic magnitudes + figures (+ LLM narration of computed numbers)."""
    interpretation: InterpretationResult
    metadata: Dict = Field(default_factory=dict)


class DraftingStageOutput(BaseModel):
    """Stage 2: assembled report + the narrative blocks that went into it."""
    drafting: DraftingResult
    report_markdown: str = ""
    metadata: Dict = Field(default_factory=dict)


class QualityStageOutput(BaseModel):
    """Stage 3: consistency verdict over the final draft."""
    quality: QualityResult
    report_markdown: str = ""
    metadata: Dict = Field(default_factory=dict)
