# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Stage output contracts for EstimationTeam (§4.6).

The heavy lifting lives in ``estim_harness.types`` (already pydantic); these wrappers add the
per-stage envelopes the orchestrator saves and downstream stages/pipeline consume.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from EstimationTeam.ael.estim_harness.types import (
    DiagnosticsReport,
    EstimationOutcome,
    InferenceReport,
    PerformanceProfile,
)


class EstimationStageOutput(BaseModel):
    """Stage 1: the proposed spec estimated by the deterministic harness."""
    research_question: str = ""
    outcome: EstimationOutcome
    loader_notes: List[str] = Field(default_factory=list)
    spec_source: str = "llm"          # "llm" | "llm_revised" — provenance of the spec
    # Wall-clock timing (§4.6 Optimizer's raw material), keyed "<step>_det_sec"/"_llm_sec".
    timing_sec: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict = Field(default_factory=dict)


class ValidationStageOutput(BaseModel):
    """Stage 2: diagnostics battery + (possibly downgraded) verdict."""
    diagnostics: DiagnosticsReport
    outcome: EstimationOutcome        # verdict may have been downgraded to 'fragile'
    timing_sec: Dict[str, float] = Field(default_factory=dict)   # cumulative: stage 1 + stage 2
    metadata: Dict = Field(default_factory=dict)


class InferenceStageOutput(BaseModel):
    """Stage 3: theory-derived hypothesis tests + robustness sweeps."""
    inference: InferenceReport
    outcome: EstimationOutcome        # final verdict carried through unchanged
    summary: str = ""
    # §4.6 Optimizer: aggregated timing across all 3 stages.
    performance_profile: PerformanceProfile = Field(default_factory=PerformanceProfile)
    metadata: Dict = Field(default_factory=dict)
