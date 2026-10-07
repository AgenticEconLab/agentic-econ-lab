# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
FeasibilityLoopController — the bounded Model<->Data cycle.

Runs: design -> availability -> review, and if the review requests a model
revision (MRR), re-runs design with that MRR and repeats. Bounded (default 3
cycles), monotone (a revision that does not strictly shrink the unmet-essential
set stops the loop), and always terminating (the review's terminal ACCEPT_AS_IS
default guarantees resolution).

The two workflow steps are injected so the controller is testable offline and does
not hard-depend on the ModelTeam/DataTeam orchestrators:

  * ``design_step(cycle, mrr) -> DataRequirementsSpec`` — run/re-run ModelDesign
    (cycle 1 gets ``mrr=None``; later cycles get the previous MRR to apply).
  * ``availability_step(drs) -> DataAvailabilityReport`` — run scout + reporter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from DataTeam.ael.feasibility.contracts import (
    DataAvailabilityReport,
    DataRequirementsSpec,
    FeasibilityDecision,
    ModelRevisionRequest,
)
from DataTeam.ael.feasibility.review import FeasibilityReview

DesignStep = Callable[[int, Optional[ModelRevisionRequest]], DataRequirementsSpec]
AvailabilityStep = Callable[[DataRequirementsSpec], DataAvailabilityReport]


@dataclass
class CycleRecord:
    """One trip around the loop, for provenance."""

    cycle: int
    tier: str
    n_requirements: int
    n_unmet_essential: int
    overall_feasible: bool
    decisions: List[FeasibilityDecision] = field(default_factory=list)
    mrr: Optional[ModelRevisionRequest] = None


@dataclass
class LoopResult:
    """Outcome of the bounded feasibility loop."""

    status: str                       # feasible | resolved_without_revision | not_converging | max_cycles
    cycles: int
    final_drs: DataRequirementsSpec
    final_dar: DataAvailabilityReport
    decisions: List[FeasibilityDecision] = field(default_factory=list)
    mrrs: List[ModelRevisionRequest] = field(default_factory=list)
    history: List[CycleRecord] = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        return self.status == "feasible"


class FeasibilityLoopController:
    def __init__(
        self,
        design_step: DesignStep,
        availability_step: AvailabilityStep,
        review: Optional[FeasibilityReview] = None,
        max_cycles: int = 3,
    ):
        self.design_step = design_step
        self.availability_step = availability_step
        self.review = review or FeasibilityReview()
        self.max_cycles = max(1, max_cycles)

    def run(self) -> LoopResult:
        history: List[CycleRecord] = []
        mrrs: List[ModelRevisionRequest] = []
        mrr: Optional[ModelRevisionRequest] = None
        prev_unmet: Optional[int] = None
        drs: Optional[DataRequirementsSpec] = None
        dar: Optional[DataAvailabilityReport] = None
        last_decisions: List[FeasibilityDecision] = []

        for cycle in range(1, self.max_cycles + 1):
            drs = self.design_step(cycle, mrr)
            drs.cycle = cycle
            dar = self.availability_step(drs)
            unmet = len(dar.unmet_essential)

            rec = CycleRecord(
                cycle=cycle, tier=dar.tier, n_requirements=len(dar.findings),
                n_unmet_essential=unmet, overall_feasible=dar.overall_feasible,
            )

            # Success: the model's needs are fully met.
            if dar.overall_feasible:
                history.append(rec)
                return LoopResult("feasible", cycle, drs, dar, [], mrrs, history)

            # A prior revision failed to shrink the unmet set -> stop looping and
            # resolve the remainder without asking for yet another model cycle.
            not_converging = prev_unmet is not None and unmet >= prev_unmet
            last_cycle = cycle == self.max_cycles
            allow_revision = not (not_converging or last_cycle)

            decisions, cyc_mrr = self.review.review(dar, drs, allow_revision=allow_revision)
            rec.decisions = decisions
            rec.mrr = cyc_mrr
            history.append(rec)
            last_decisions = decisions

            if cyc_mrr is None:
                # Review resolved via proxy / narrow / accept-as-is — no new cycle.
                status = "not_converging" if not_converging else (
                    "max_cycles" if last_cycle else "resolved_without_revision"
                )
                return LoopResult(status, cycle, drs, dar, decisions, mrrs, history)

            # Otherwise a revision was requested — loop again with the MRR.
            mrr = cyc_mrr
            mrrs.append(cyc_mrr)
            prev_unmet = unmet

        # Exhausted the cycle budget with an outstanding revision request.
        return LoopResult("max_cycles", self.max_cycles, drs, dar, last_decisions, mrrs, history)
