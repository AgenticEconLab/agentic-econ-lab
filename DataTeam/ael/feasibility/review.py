# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Feasibility Review — the human trade-off gate.

For each unmet essential requirement, decide among the trade-off menu (accept
proxy / escalate source / revise model / narrow question / accept-as-is) and, if
any decision is REVISE_MODEL, emit a Model Revision Request (MRR).

The decision is resolved as a **priority-ordered cascade of yes/no propositions**
actions are proposed in order and the first to clear a committee
2/3 vote carries; ACCEPT_AS_IS is the terminal default so the cascade always
resolves. Who votes depends on the HITL resolution mode:

  * ``llm_economist`` -> the 3-model committee (reasoned stand-in; testing/examples);
  * ``interactive``   -> a real human (deployment) via auto_input;
  * ``auto``          -> infrastructure smoke-test: takes the terminal default,
                         ACCEPT_AS_IS (no economic meaning).

A ``resolver`` seam (proposition -> bool) is injectable for testing.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from DataTeam.ael.feasibility.contracts import (
    AvailabilityFinding,
    DataAvailabilityReport,
    DataRequirementsSpec,
    FeasibilityAction,
    FeasibilityDecision,
    ModelRevisionRequest,
)

# resolver(question, context) -> True if the committee/human approves the proposition
Resolver = Callable[[str, dict], bool]


def _cascade(
    finding: AvailabilityFinding, escalation_available: bool, allow_revision: bool = True
) -> List[FeasibilityAction]:
    """Priority order of actions proposed for one unmet requirement.

    Least-disruptive-first, but ACCEPT_AS_IS is always the terminal default so the
    cascade cannot fail to resolve. ESCALATE_SOURCE is only offered when a higher
    tier is actually reachable (in an unattended HPC full run it is not), which
    naturally forces the substantive proxy-vs-revise-model trade-off. REVISE_MODEL is
    omitted when the loop's cycle budget is exhausted (``allow_revision=False``), so
    the gate resolves without requesting another (impossible) model cycle.
    """
    order: List[FeasibilityAction] = []
    if finding.proxy_candidates:
        order.append(FeasibilityAction.ACCEPT_PROXY)
    if escalation_available:
        order.append(FeasibilityAction.ESCALATE_SOURCE)
    if allow_revision:
        order.append(FeasibilityAction.REVISE_MODEL)
    order.append(FeasibilityAction.NARROW_QUESTION)
    order.append(FeasibilityAction.ACCEPT_AS_IS)  # terminal default
    return order


_QUESTION = {
    FeasibilityAction.ACCEPT_PROXY:
        "Accept the best available proxy for requirement '{var}' "
        "(bias: {bias}), substituting it for the exact construct?",
    FeasibilityAction.ESCALATE_SOURCE:
        "Escalate requirement '{var}' to a premium/subscribed or user-uploaded source?",
    FeasibilityAction.REVISE_MODEL:
        "Revise the model to remove or relax its dependence on requirement '{var}'?",
    FeasibilityAction.NARROW_QUESTION:
        "Narrow the research question so it no longer requires '{var}'?",
    FeasibilityAction.ACCEPT_AS_IS:
        "Proceed as-is, documenting '{var}' as an accepted data limitation?",
}


class FeasibilityReview:
    """Resolves the trade-off for every unmet essential requirement in a DAR."""

    def __init__(self, resolver: Optional[Resolver] = None, escalation_available: bool = False):
        self.escalation_available = escalation_available
        self._resolver = resolver or self._default_resolver
        self._decider_label = "custom" if resolver else _current_decider_label()

    def review(
        self, dar: DataAvailabilityReport, drs: Optional[DataRequirementsSpec] = None,
        allow_revision: bool = True,
    ) -> Tuple[List[FeasibilityDecision], Optional[ModelRevisionRequest]]:
        decisions: List[FeasibilityDecision] = []
        for rid in dar.unmet_essential:
            finding = dar.find(rid)
            if finding is None:
                continue
            decisions.append(self._decide(finding, dar, allow_revision))

        mrr = self._maybe_mrr(decisions, dar)
        return decisions, mrr

    def _decide(
        self, finding: AvailabilityFinding, dar: DataAvailabilityReport, allow_revision: bool = True
    ) -> FeasibilityDecision:
        best_proxy = finding.proxy_candidates[0] if finding.proxy_candidates else None
        bias = best_proxy.bias_caveat if best_proxy else "n/a"
        context = {
            "research_question": dar.research_question,
            "tier": dar.tier,
            "requirement": finding.variable_name,
            "verdict": finding.verdict.value,
            "proxy_candidates": [p.model_dump() for p in finding.proxy_candidates],
        }
        for action in _cascade(finding, self.escalation_available, allow_revision):
            question = _QUESTION[action].format(var=finding.variable_name, bias=bias)
            if action == FeasibilityAction.ACCEPT_AS_IS:
                # terminal default — always taken if nothing earlier passed
                return FeasibilityDecision(
                    requirement_id=finding.requirement_id, action=action,
                    rationale="terminal default: no earlier action approved; documented limitation",
                    decider=self._decider_label,
                )
            if self._resolver(question, context):
                return FeasibilityDecision(
                    requirement_id=finding.requirement_id, action=action,
                    rationale=f"committee/human approved: {action.value}",
                    decider=self._decider_label,
                    chosen_proxy=best_proxy if action == FeasibilityAction.ACCEPT_PROXY else None,
                )
        # Unreachable (ACCEPT_AS_IS terminates), but keep a safe fallback.
        return FeasibilityDecision(
            requirement_id=finding.requirement_id, action=FeasibilityAction.ACCEPT_AS_IS,
            rationale="fallback terminal default", decider=self._decider_label,
        )

    @staticmethod
    def _maybe_mrr(
        decisions: List[FeasibilityDecision], dar: DataAvailabilityReport
    ) -> Optional[ModelRevisionRequest]:
        revise = [d for d in decisions if d.action == FeasibilityAction.REVISE_MODEL]
        if not revise:
            return None
        rids = [d.requirement_id for d in revise]
        vars_ = [dar.find(r).variable_name for r in rids if dar.find(r)]
        return ModelRevisionRequest(
            reason="Feasibility Review chose to revise the model for unmet essential requirement(s).",
            unmet_requirements=rids,
            requested_change=(
                "Remove or relax the model's dependence on: " + ", ".join(vars_)
                + " (no adequate data or acceptable proxy in the searched tier)."
            ),
            decisions=revise,
            provenance={"tier": dar.tier, "research_question": dar.research_question},
        )

    # -- default resolver: routes by HITL mode ------------------------------ #
    def _default_resolver(self, question: str, context: dict) -> bool:
        from shared.auto_input import get_hitl_mode

        mode = get_hitl_mode()
        if mode == "llm_economist":
            try:
                from shared.llm_economist import LLMEconomistCommittee
                return LLMEconomistCommittee().vote_proposition(question, context).passed
            except Exception:
                return False  # committee unavailable -> don't approve; fall through cascade
        if mode == "auto":
            return False  # infra smoke-test: approve nothing -> terminal ACCEPT_AS_IS
        # interactive: a real human answers yes/no
        try:
            from shared.auto_input import auto_input
            ans = auto_input(f"{question} (yes/no)", default="no", input_type="yes_no")
            return str(ans).strip().lower() in ("yes", "y", "true")
        except Exception:
            return False


def _current_decider_label() -> str:
    """Provenance label for who resolved the gate, based on the HITL mode."""
    try:
        from shared.auto_input import get_hitl_mode

        mode = get_hitl_mode()
    except Exception:
        mode = "interactive"
    return {
        "llm_economist": "llm_economist_committee",
        "auto": "auto",
        "interactive": "human",
    }.get(mode, "human")
