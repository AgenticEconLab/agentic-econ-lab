# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
AvailabilityReporter — assemble the Data Availability Report (DAR) from findings.

Deterministic aggregation lives in ``DataAvailabilityReport.from_findings``; this
thin wrapper adds an optional LLM narrative seam and is the object the pipeline
calls, mirroring the DataTeam's other reporter agents.
"""

from __future__ import annotations

from typing import Callable, List, Optional

from DataTeam.ael.feasibility.contracts import (
    AvailabilityFinding,
    DataAvailabilityReport,
)

# (research_question, tier, findings) -> narrative string
NarrateFn = Callable[[str, str, List[AvailabilityFinding]], str]


class AvailabilityReporter:
    """Builds the DAR; an optional ``narrate_fn`` supplies prose for the summary."""

    def __init__(self, narrate_fn: Optional[NarrateFn] = None):
        self._narrate_fn = narrate_fn

    def build(
        self,
        research_question: str,
        tier: str,
        findings: List[AvailabilityFinding],
        metadata: Optional[dict] = None,
    ) -> DataAvailabilityReport:
        summary = ""
        if self._narrate_fn is not None:
            try:
                summary = self._narrate_fn(research_question, tier, findings) or ""
            except Exception:
                summary = ""  # narrative is best-effort; aggregation still stands
        return DataAvailabilityReport.from_findings(
            research_question=research_question,
            tier=tier,
            findings=findings,
            summary=summary,
            metadata=metadata,
        )
