# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Contracts for the Model<->Data feasibility loop.

See README.md, "Running". Three artifacts flow around
the loop:

  * DRS — DataRequirementsSpec: what the model needs (reuses DataTeam's
    ``DataRequirement`` rows).
  * DAR — DataAvailabilityReport: per-requirement verdict (feasible / proxy-only /
    unavailable) with candidate proxies and bias caveats.
  * MRR — ModelRevisionRequest: emitted when the Feasibility Review decides to
    revise the model; re-enters ModelDesign and makes the pipeline cyclic.

Plus the human trade-off decision (FeasibilityDecision) taken at the Feasibility
Review gate.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from DataTeam.ael.schemas.stage_outputs import DataRequirement


# --------------------------------------------------------------------------- #
# DRS — Data Requirements Spec
# --------------------------------------------------------------------------- #
class DataRequirementsSpec(BaseModel):
    """What the model needs from the data world."""

    research_question: str = Field(description="The research question being served")
    requirements: List[DataRequirement] = Field(
        default_factory=list, description="Per-variable data requirements (the DRS rows)"
    )
    source_model: str = Field(default="", description="Model spec / cycle that produced this DRS")
    cycle: int = Field(default=0, description="Feasibility-loop cycle index (0 = first pass)")

    def essential(self) -> List[DataRequirement]:
        """Requirements the study cannot proceed without (High priority)."""
        return [r for r in self.requirements if str(r.priority).strip().lower() == "high"]


# --------------------------------------------------------------------------- #
# DAR — Data Availability Report
# --------------------------------------------------------------------------- #
class RequirementVerdict(str, Enum):
    """Availability verdict for a single requirement in the searched tier."""

    FEASIBLE = "feasible"          # a good direct match exists
    PROXY_ONLY = "proxy_only"      # only an approximate/proxy series exists
    UNAVAILABLE = "unavailable"    # nothing in the searched tier


class ProxyCandidate(BaseModel):
    """An approximate series offered in place of an exact construct."""

    variable_name: str = Field(description="The proxy variable/series construct")
    source_name: str = Field(description="Source/API offering the proxy")
    series_id: str = Field(default="", description="Series identifier if known")
    closeness: float = Field(
        default=0.0, ge=0.0, le=1.0, description="0..1 how close the proxy is to the required construct"
    )
    bias_caveat: str = Field(
        default="", description="What bias/limitation substituting this proxy introduces"
    )


class AvailabilityFinding(BaseModel):
    """The availability verdict for one requirement."""

    requirement_id: str
    variable_name: str
    priority: str = Field(default="Medium", description="High/Medium/Low (from the DRS row)")
    verdict: RequirementVerdict
    matched_source: str = Field(default="", description="Source of the direct match (if FEASIBLE)")
    matched_series: str = Field(default="", description="Series id of the direct match (if FEASIBLE)")
    coverage_note: str = Field(default="", description="Coverage/quality note for the match")
    proxy_candidates: List[ProxyCandidate] = Field(default_factory=list)
    notes: str = Field(default="")
    # FEASIBLE via the curated connector universe, not the retrieved pool — the series
    # is fetchable on demand and the pipeline schedules a supplemental fetch for it.
    fetchable: bool = Field(
        default=False,
        description="True when the match is a curated connector id absent from the retrieved "
                    "dataset (supplemental fetch required)")

    @property
    def essential(self) -> bool:
        return str(self.priority).strip().lower() == "high"

    @property
    def met(self) -> bool:
        return self.verdict == RequirementVerdict.FEASIBLE


class DataAvailabilityReport(BaseModel):
    """DAR — the DataTeam's answer to the DRS."""

    research_question: str
    tier: str = Field(description="Tier searched: open_source_api | premium_subscribed | user_uploaded")
    findings: List[AvailabilityFinding] = Field(default_factory=list)
    n_feasible: int = 0
    n_proxy: int = 0
    n_unavailable: int = 0
    unmet_essential: List[str] = Field(
        default_factory=list, description="requirement_ids of essential requirements not FEASIBLE"
    )
    overall_feasible: bool = Field(
        default=True, description="True iff every essential requirement is FEASIBLE"
    )
    summary: str = Field(default="")
    metadata: Dict = Field(default_factory=dict)

    def find(self, requirement_id: str) -> Optional[AvailabilityFinding]:
        for f in self.findings:
            if f.requirement_id == requirement_id:
                return f
        return None

    @classmethod
    def from_findings(
        cls, research_question: str, tier: str, findings: List[AvailabilityFinding],
        summary: str = "", metadata: Optional[Dict] = None,
    ) -> "DataAvailabilityReport":
        n_feasible = sum(1 for f in findings if f.verdict == RequirementVerdict.FEASIBLE)
        n_proxy = sum(1 for f in findings if f.verdict == RequirementVerdict.PROXY_ONLY)
        n_unavailable = sum(1 for f in findings if f.verdict == RequirementVerdict.UNAVAILABLE)
        unmet_essential = [f.requirement_id for f in findings if f.essential and not f.met]
        return cls(
            research_question=research_question,
            tier=tier,
            findings=findings,
            n_feasible=n_feasible,
            n_proxy=n_proxy,
            n_unavailable=n_unavailable,
            unmet_essential=unmet_essential,
            overall_feasible=len(unmet_essential) == 0,
            summary=summary or (
                f"{n_feasible} feasible / {n_proxy} proxy-only / {n_unavailable} unavailable "
                f"of {len(findings)} requirements in tier '{tier}'; "
                f"{len(unmet_essential)} essential requirement(s) unmet."
            ),
            metadata=metadata or {},
        )


# --------------------------------------------------------------------------- #
# Feasibility Review — the human trade-off gate
# --------------------------------------------------------------------------- #
class FeasibilityAction(str, Enum):
    """The trade-off menu at the Feasibility Review checkpoint."""

    ACCEPT_PROXY = "accept_proxy"          # substitute a proxy (with documented bias)
    ESCALATE_SOURCE = "escalate_source"    # try premium / user-uploaded tier
    REVISE_MODEL = "revise_model"          # change the model -> emits an MRR
    NARROW_QUESTION = "narrow_question"    # scope the claim to what data supports
    ACCEPT_AS_IS = "accept_as_is"          # proceed with a documented data limitation


class FeasibilityDecision(BaseModel):
    """One resolved trade-off for one unmet requirement."""

    requirement_id: str
    action: FeasibilityAction
    rationale: str = ""
    decider: str = Field(default="llm_economist_committee", description="who decided (or 'human'/'auto')")
    contested: bool = False
    chosen_proxy: Optional[ProxyCandidate] = None


# --------------------------------------------------------------------------- #
# MRR — Model Revision Request
# --------------------------------------------------------------------------- #
class ModelRevisionRequest(BaseModel):
    """Sent back to ModelDesign when the review decides to revise the model."""

    reason: str = Field(description="Why the model must change")
    unmet_requirements: List[str] = Field(
        default_factory=list, description="requirement_ids that drove the revision"
    )
    requested_change: str = Field(
        default="", description="What the model should change (drop var, relax construct, ...)"
    )
    decisions: List[FeasibilityDecision] = Field(default_factory=list)
    cycle: int = Field(default=0, description="Cycle that produced this MRR")
    provenance: Dict = Field(default_factory=dict)
