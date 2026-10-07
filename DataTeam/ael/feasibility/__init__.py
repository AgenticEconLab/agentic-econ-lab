# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Model<->Data feasibility loop.

Public API:
  - Contracts: DataRequirementsSpec (DRS), DataAvailabilityReport (DAR),
    ModelRevisionRequest (MRR), AvailabilityFinding, ProxyCandidate,
    RequirementVerdict, FeasibilityAction, FeasibilityDecision.
  - Agents/stages: AvailabilityScout, AvailabilityReporter, FeasibilityReview.
  - Orchestration: FeasibilityLoopController, LoopResult.
"""

from DataTeam.ael.feasibility.contracts import (
    AvailabilityFinding,
    DataAvailabilityReport,
    DataRequirementsSpec,
    FeasibilityAction,
    FeasibilityDecision,
    ModelRevisionRequest,
    ProxyCandidate,
    RequirementVerdict,
)
from DataTeam.ael.feasibility.scout import AvailabilityScout, AvailableSeries, similarity
from DataTeam.ael.feasibility.reporter import AvailabilityReporter
from DataTeam.ael.feasibility.adapters import (
    available_series_from_source,
    drs_from_model_artifacts,
    format_revision_directive,
)
from DataTeam.ael.feasibility.review import FeasibilityReview
from DataTeam.ael.feasibility.loop import (
    CycleRecord,
    FeasibilityLoopController,
    LoopResult,
)

__all__ = [
    "DataRequirementsSpec",
    "DataAvailabilityReport",
    "ModelRevisionRequest",
    "AvailabilityFinding",
    "ProxyCandidate",
    "RequirementVerdict",
    "FeasibilityAction",
    "FeasibilityDecision",
    "AvailabilityScout",
    "AvailableSeries",
    "similarity",
    "AvailabilityReporter",
    "drs_from_model_artifacts",
    "available_series_from_source",
    "format_revision_directive",
    "FeasibilityReview",
    "FeasibilityLoopController",
    "LoopResult",
    "CycleRecord",
]
