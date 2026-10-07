# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
LLM-Economist committee — a reasoned human stand-in for WithHITL checkpoints.

This implements the ``llm_economist`` HITL *resolution* mode (see
README.md, "Running"). It is used ONLY for
testing/evaluation and the paper's worked examples, where no real economist can
sit an autonomous run. It is **not** a change to the framework's design:

  * real deployments keep a real human in the loop (``interactive``);
  * ``auto`` (``AUTO_HITL_MODE``) stays a plumbing-only smoke-test with **no
    economic meaning** (it rubber-stamps the canned approve/no-feedback defaults);
  * ``llm_economist`` is the reasoned stand-in for runs that must *mean* something.

Mechanism
---------
A checkpoint is resolved by a committee of N=3 **distinct open-weight families**
(Qwen + Mistral + Gemma — the same "no model grades its own work" diversity used by
the dual-judge evaluator). Each member casts a **binary yes/no vote**
(approve/proceed vs. object) with a one-line economist rationale, and the decision
passes on a **2/3 majority**. Because N is odd and the vote is binary, the outcome
is always decisive — no ties, no deliberative second round. The **full ballot** is
recorded to a provenance log stamped ``decider=llm_economist_committee`` so that
evaluation, audits, and the manuscript can separate synthetic decisions from a real
human's.

A checkpoint is handled by kind: **decision** gates (yes/no, choices, approve) are voted
2/3; **review/feedback** prompts (empty-default free text — "which papers are
irrelevant?", "what's missing?") get a **reasoned economist response** grounded in the
supplied ``context`` (giving review feedback is the economist's judgment, not
fabrication — the members are told not to invent papers/data/citations); only
**mechanical** inputs (file paths, seed values) keep their default.

Env
---
    AEL_HITL_MODE=llm_economist        enable this mode (see shared.auto_input.get_hitl_mode)
    AEL_HITL_COMMITTEE=m1,m2,m3        member models (default gpt-oss-20b, Phi-4-mini, Olmo-3.1-32B on vLLM)
    AEL_HITL_LOG=/path/ballots.jsonl   provenance log (default ./hitl_committee_ballots.jsonl)
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel

from shared.selection_aggregation import (
    aggregate_selection,
    count_listed_items,
    selection_kind,
)

# Default committee: one model per open-weight family we serve on vLLM. Distinct
# families so no single vendor decides. Override via AEL_HITL_COMMITTEE.
# The committee shares no family with the
# generating model (Qwen) or the Tier-2 judges (Mistral, Gemma), so no model approves its
# own proposals and no judge scores work it helped decide. Families: OpenAI, Microsoft,
# AllenAI (served names as in hpc/full_pipeline.sbatch).
DEFAULT_COMMITTEE = [
    "vllm/gpt-oss-20b",
    "vllm/phi-4-mini",
    "vllm/olmo-3.1-32b-instruct",
]
# Families that must not sit on the committee: the generating model and the judges.
EXCLUDED_COMMITTEE_FAMILIES = ("qwen", "mistral", "gemma")

# A checkpoint whose default answer is one of these (or whose input_type is a
# decision type) is a *decision* gate the committee votes on. Everything else
# (free-text feedback, file paths, empty defaults) is a non-decision: the default
# stands and no vote is cast.
_APPROVE_TOKENS = {
    "yes", "y", "approve", "approved", "ok", "okay", "proceed",
    "a", "accept", "accepted", "continue", "confirm", "true",
}
_DECISION_INPUT_TYPES = {"yes_no", "choice", "confirm"}

_DECIDER = "llm_economist_committee"


# --------------------------------------------------------------------------- #
# The economist decision rubric (shared by every member)
# --------------------------------------------------------------------------- #
RUBRIC_SYSTEM = (
    "You are a domain economist standing in for a human reviewer at a "
    "human-in-the-loop checkpoint of an automated economic-research workflow. "
    "Decide whether to APPROVE and let the workflow proceed with the default "
    "action, or to OBJECT.\n\n"
    "Apply an economist's judgment, not a rubber stamp:\n"
    "  - Approve only when the work meets economic standards (identification is "
    "credible, the reasoning is structurally coherent, and any empirical claim is "
    "supported by the data actually available).\n"
    "  - Weigh proxy bias against loss of scope; protect identification integrity.\n"
    "  - Prefer the smallest change that restores soundness; escalate cost only "
    "when justified.\n"
    "  - Refuse to proceed on a claim the data cannot support.\n\n"
    "Vote is BINARY: approve = true (proceed with the default) or false (object). "
    "If you object, put the concrete answer you would enter instead in \"answer\" "
    "(choose from any options shown in the checkpoint text). Keep the rationale to "
    "one sentence. Respond with ONLY a JSON object: "
    '{"approve": true|false, "answer": "<token or empty>", "rationale": "<one sentence>"}'
)


# Proposition voting (used by the feasibility-gate cascade): each member votes
# YES/NO on one concrete proposition, framed directly (not "approve the default").
PROPOSITION_SYSTEM = (
    "You are a domain economist on a review committee for an automated "
    "economic-research workflow. Vote YES or NO on the specific proposition below "
    "about how to handle a data-feasibility problem. Apply an economist's judgment: "
    "protect identification and the credibility of any empirical claim, weigh proxy "
    "bias against loss of scope, and prefer the smallest change that keeps the study "
    "sound. Vote YES only if the proposition is the right call. "
    'Respond with ONLY a JSON object: {"yes": true|false, "rationale": "<one sentence>"}'
)


# --------------------------------------------------------------------------- #
# Per-member review lenses. A full run with a shared rubric showed 0% within-ballot dissent
# across 119 ballots — three models reading the same context through the same
# rubric converge, so the 2/3 rule never binds and the committee adds compute,
# not discrimination. Each member now reviews through a DISTINCT economist lens,
# so votes carry independent information. The vote stays binary and the 2/3
# aggregation is unchanged. Disable with AEL_HITL_LENSES=0 (identical rubric).
# --------------------------------------------------------------------------- #
MEMBER_LENSES: List[tuple] = [
    ("theory", "Your review lens is THEORETICAL SOUNDNESS: judge identification, internal "
               "coherence, and whether the assumptions can bear the conclusions. Object when "
               "the theory or identification is weak, even if the work looks polished."),
    ("data", "Your review lens is EMPIRICAL / DATA REALISM: judge whether the data actually "
             "available (sources, series, coverage, frequency) can support each claim, and "
             "whether measurement and proxies are credible. Object when claims outrun the data."),
    ("policy", "Your review lens is POLICY & PRACTICAL RELEVANCE: judge whether the work answers "
               "an economically meaningful question, the scope/cost trade-offs are sensible, and "
               "the output would be useful to a policymaker or referee. Object when relevance or "
               "practicality is weak."),
]


def _lenses_enabled() -> bool:
    return os.environ.get("AEL_HITL_LENSES", "1").strip().lower() not in ("0", "false", "off")


def _lens_for(index: int) -> tuple:
    """(lens_name, lens_directive) for member ``index``; ('', '') when lenses are disabled."""
    if not _lenses_enabled() or not MEMBER_LENSES:
        return ("", "")
    return MEMBER_LENSES[index % len(MEMBER_LENSES)]


class _MemberBallot(BaseModel):
    """Structured vote returned by one committee member (approve-the-default framing)."""

    approve: bool
    answer: str = ""
    rationale: str = ""


class _PropositionVote(BaseModel):
    """Structured YES/NO vote on a single proposition."""

    yes: bool
    rationale: str = ""


# Review/feedback generation (empty free-text checkpoints). The economist gives
# steering feedback grounded in the material — but must NOT invent facts.
FEEDBACK_SYSTEM = (
    "You are a domain economist standing in for a human reviewer at a human-in-the-loop "
    "checkpoint of an automated economic-research workflow. Answer the checkpoint the way a "
    "human reviewer would, to steer the next round.\n"
    "FORMAT — this matters: if the checkpoint asks for a specific answer format (e.g. "
    "comma-separated item numbers like '1,3', pairs like '1,2 3,4', a yes/no, or a letter), "
    "reply in EXACTLY that format and nothing else, referring to the numbered items in the "
    "context. Only when the checkpoint is open-ended (e.g. 'general comments', 'additional "
    "guidance', 'missing topics') should you give concise prose.\n"
    "Base your answer ONLY on the material in the context; do NOT invent papers, data, "
    "citations, or numbers. If nothing applies, return empty. "
    'Respond with ONLY a JSON object: {"feedback": "<answer in the requested format, or empty>"}'
)


class _FeedbackResponse(BaseModel):
    """Structured economist review feedback for an open-ended checkpoint."""

    feedback: str = ""


# Chair synthesis: merge the members' individual reviews into one consolidated feedback.
SYNTHESIS_SYSTEM = (
    "You are the chair of a committee of domain economists. Several members have each "
    "answered the same checkpoint. Consolidate their answers into ONE.\n"
    "FORMAT: preserve the answer format the checkpoint requested — if the members answered "
    "in a terse format (comma-separated numbers like '1,3', pairs, yes/no, a letter), return "
    "the consolidated answer in that SAME format and nothing else (e.g. only the item numbers "
    "a majority of members chose, never the union). Only for open-ended prose should you write consolidated prose: keep "
    "well-supported points, merge overlaps, note genuine disagreements neutrally, and drop "
    "anything that invents facts. "
    'Respond with ONLY a JSON object: {"feedback": "<consolidated answer in the requested format>"}'
)


@dataclass
class MemberVote:
    """One member's cast vote (or abstention on error)."""

    model: str
    approve: Optional[bool]          # None => abstained (call failed)
    answer: str = ""
    rationale: str = ""
    error: Optional[str] = None
    lens: str = ""                   # the member's review lens (dissent telemetry)

    @property
    def abstained(self) -> bool:
        return self.approve is None


@dataclass
class CommitteeDecision:
    """Aggregated committee decision + full ballot, for provenance."""

    answer: str
    approve: bool
    voted: bool                      # False => non-decision input, default used verbatim
    contested: bool = False
    agreement: float = 1.0
    approve_votes: int = 0
    object_votes: int = 0
    ballot: List[MemberVote] = field(default_factory=list)
    note: str = ""
    decider: str = _DECIDER

    def to_record(self, prompt: str, default: str, input_type: str) -> Dict[str, Any]:
        """Serialisable provenance record."""
        return {
            "decider": self.decider,
            "kind": "decision",     # kind-less ballots would skew every census
            "prompt": (prompt or "")[:500],
            "input_type": input_type,
            "default": default,
            "decision": self.answer,
            "approve": self.approve,
            "voted": self.voted,
            "contested": self.contested,
            "agreement": round(self.agreement, 4),
            "approve_votes": self.approve_votes,
            "object_votes": self.object_votes,
            "note": self.note,
            "ballot": [
                {
                    "model": v.model,
                    "approve": v.approve,
                    "answer": v.answer,
                    "rationale": v.rationale,
                    "error": v.error,
                    "lens": v.lens,
                }
                for v in self.ballot
            ],
        }


@dataclass
class PropositionResult:
    """Outcome of a single YES/NO proposition vote by the committee."""

    question: str
    passed: bool
    yes_votes: int = 0
    no_votes: int = 0
    agreement: float = 1.0
    contested: bool = False
    ballot: List[MemberVote] = field(default_factory=list)
    note: str = ""
    decider: str = _DECIDER

    def to_record(self) -> Dict[str, Any]:
        return {
            "decider": self.decider,
            "kind": "proposition",
            "question": (self.question or "")[:500],
            "passed": self.passed,
            "yes_votes": self.yes_votes,
            "no_votes": self.no_votes,
            "agreement": round(self.agreement, 4),
            "contested": self.contested,
            "note": self.note,
            "ballot": [
                {"model": v.model, "yes": v.approve, "rationale": v.rationale,
                 "error": v.error, "lens": v.lens}
                for v in self.ballot
            ],
        }


@dataclass
class MemberFeedback:
    """One member's review-feedback draft (before synthesis)."""

    model: str
    feedback: str = ""
    error: Optional[str] = None


@dataclass
class FeedbackResult:
    """Consolidated committee feedback + each member's draft, for provenance."""

    feedback: str                                   # the synthesized (chair) feedback
    drafts: List[MemberFeedback] = field(default_factory=list)
    note: str = ""
    decider: str = _DECIDER


@dataclass
class RefineRound:
    """One iteration of the develop-until-approved loop."""

    round: int
    approved: bool
    yes_votes: int
    no_votes: int
    feedback: str = ""


@dataclass
class RefineResult:
    """Outcome of refine_until_approved."""

    approved: bool
    rounds: int
    artifact: Any
    history: List[RefineRound] = field(default_factory=list)


def get_committee_models() -> List[str]:
    """Resolve the committee member models from AEL_HITL_COMMITTEE (or the default).

    A member from an excluded family (the generating model's or a judge's) is logged as a
    warning, not silently accepted: an override may be deliberate, but it breaks the
    'no model reviews its own work' property the evaluation relies on."""
    raw = os.environ.get("AEL_HITL_COMMITTEE", "").strip()
    models = [m.strip() for m in raw.split(",") if m.strip()] if raw else list(DEFAULT_COMMITTEE)
    overlap = [m for m in models if any(f in m.lower() for f in EXCLUDED_COMMITTEE_FAMILIES)]
    if overlap:
        import logging
        logging.getLogger("ael.llm_economist").warning(
            "committee member(s) %s share a family with the generating model or a Tier-2 "
            "judge; independence of checkpoint decisions is not guaranteed", overlap)
    return models


def _is_decision(default: str, input_type: str) -> bool:
    """Is this checkpoint a decision gate (votable) rather than free-text input?"""
    if input_type in _DECISION_INPUT_TYPES:
        return True
    return str(default).strip().lower() in _APPROVE_TOKENS


_PLACEHOLDER_FEEDBACK = {"empty", "none", "n/a", "na", "no feedback", "no comment", "-"}

# Prompts that legitimately expect a terse/selection answer ("1,4") rather than prose.
_TERSE_PROMPT_RE = re.compile(
    r"comma[- ]separated|e\.g\.|\(yes/no|enter\s|select\s|choose\s", re.IGNORECASE)


def _expects_terse(prompt: str) -> bool:
    return bool(_TERSE_PROMPT_RE.search(prompt or ""))


def _is_substantive_feedback(text: str) -> bool:
    """A feedback draft counts only if it is actual prose — not a placeholder token and
    not a bare digit/selection string ('1,2' is a vote, not review feedback)."""
    t = (text or "").strip()
    if not t or t.lower().strip(".!") in _PLACEHOLDER_FEEDBACK:
        return False
    return len(re.findall(r"[A-Za-z]{3,}", t)) >= 2


def _is_feedback(default: str, input_type: str) -> bool:
    """Is this a review/feedback prompt (an empty free-text slot the human fills in)?

    Empty-default, non-path free text is where a reviewer writes steering feedback
    (relevant/irrelevant items, missing topics, priorities). Reached only for
    non-decision inputs. Non-empty defaults (seeds like a research topic, a max count)
    and paths are mechanical and keep their default.
    """
    if input_type == "path":
        return False
    return str(default).strip() == ""


def _approves(approve_count: int, n: int) -> bool:
    """2/3-majority rule: approve iff approve_count / n >= 2/3 (i.e. >= ceil(2n/3))."""
    if n <= 0:
        return False
    return approve_count >= math.ceil(2 * n / 3)


def _negative_token(default: str) -> str:
    """Best-effort 'do not proceed' token when objectors give no explicit alternative."""
    d = str(default).strip().lower()
    if d == "y":
        return "n"
    if d in ("yes", "approve", "approved", "ok", "okay", "proceed", "confirm", "true"):
        return "no"
    return ""  # can't infer a valid alternative token


class LLMEconomistCommittee:
    """A 3-model committee that resolves WithHITL checkpoints by 2/3 binary vote.

    Parameters
    ----------
    members:
        Model ids (default: ``get_committee_models()``).
    client_factory:
        ``model -> client`` where client exposes
        ``format_and_invoke(system, user, parse_as=...)``. Defaults to
        :class:`shared.llm.LLMClient`. Injected in tests to avoid network calls.
    log_path:
        Provenance JSONL path (default: env ``AEL_HITL_LOG`` or
        ``./hitl_committee_ballots.jsonl``). Set to ``""`` to disable logging.
    """

    def __init__(
        self,
        members: Optional[List[str]] = None,
        client_factory: Optional[Callable[[str], Any]] = None,
        log_path: Optional[str] = None,
        collector: Optional[Any] = None,
    ):
        self.members = members if members is not None else get_committee_models()
        self._client_factory = client_factory or self._default_client_factory
        self._collector = collector
        if log_path is None:
            log_path = os.environ.get("AEL_HITL_LOG", "hitl_committee_ballots.jsonl")
        self.log_path = log_path

    # -- LLM plumbing ------------------------------------------------------- #
    def _default_client_factory(self, model: str):
        from shared.llm import LLMClient  # lazy: keep httpx off the import path

        short = model.split("/")[-1][:12]
        return LLMClient(
            model=model,
            temperature=0.0,
            agent_name=f"LLMEconomist:{short}",
            collector=self._collector,
            # Pin the member's model — do NOT let AEL_MODEL (the agents' model) override it,
            # else all 3 distinct committee members collapse onto one model.
            allow_env_override=False,
        )

    def _member_vote(
        self, model: str, prompt: str, default: str, input_type: str, context: Optional[Dict],
        member_index: int = 0,
    ) -> MemberVote:
        user = self._build_user_prompt(prompt, default, input_type, context)
        lens_name, lens_directive = _lens_for(member_index)
        system = RUBRIC_SYSTEM + (f"\n\n{lens_directive}" if lens_directive else "")
        try:
            ballot = self._client_factory(model).format_and_invoke(
                system, user, parse_as=_MemberBallot
            )
            return MemberVote(
                model=model,
                approve=bool(ballot.approve),
                answer=(ballot.answer or "").strip(),
                rationale=(ballot.rationale or "").strip(),
                lens=lens_name,
            )
        except Exception as e:  # a member that errors abstains — never crash the run
            return MemberVote(model=model, approve=None, error=str(e), lens=lens_name)

    @staticmethod
    def _build_user_prompt(
        prompt: str, default: str, input_type: str, context: Optional[Dict]
    ) -> str:
        parts = [
            "HITL CHECKPOINT",
            "----------------",
            f"The workflow is asking:\n{prompt}",
            f"\nDefault (approve/proceed) answer: {default!r}",
            f"Answer type: {input_type}",
        ]
        if context:
            try:
                ctx = json.dumps(context, indent=2, default=str)[:4000]
            except Exception:
                ctx = str(context)[:4000]
            parts.append(f"\nContext for your decision:\n{ctx}")
        parts.append(
            "\nApprove and proceed with the default, or object. Reply with the JSON object only."
        )
        return "\n".join(parts)

    # -- decision ----------------------------------------------------------- #
    def decide(
        self,
        prompt: str,
        default: str = "",
        input_type: str = "general",
        context: Optional[Dict] = None,
    ) -> CommitteeDecision:
        """Resolve one checkpoint. Never raises — falls back to ``default`` on trouble."""
        if not _is_decision(default, input_type):
            # Review/feedback prompt: the economist gives reasoned, context-grounded
            # steering feedback (not a vote). Mechanical inputs keep their default.
            if _is_feedback(default, input_type):
                return self._feedback_decision(prompt, default, context)
            return CommitteeDecision(
                answer=default, approve=True, voted=False,
                note="mechanical input; default used (no vote)",
            )

        votes = [
            self._member_vote(m, prompt, default, input_type, context, member_index=i)
            for i, m in enumerate(self.members)
        ]
        cast = [v for v in votes if not v.abstained]
        n = len(cast)

        if n == 0:
            decision = CommitteeDecision(
                answer=default, approve=True, voted=True, contested=True,
                agreement=0.0, ballot=votes,
                note="all members abstained (calls failed); fell back to default",
            )
            self._log(decision, prompt, default, input_type)
            return decision

        approve_votes = sum(1 for v in cast if v.approve)
        object_votes = n - approve_votes
        approved = _approves(approve_votes, n)
        agreement = max(approve_votes, object_votes) / n
        contested = approve_votes > 0 and object_votes > 0

        if approved:
            answer, note = default, ""
        else:
            answer, note = self._objection_answer(default, input_type, cast)

        decision = CommitteeDecision(
            answer=answer,
            approve=approved,
            voted=True,
            contested=contested,
            agreement=agreement,
            approve_votes=approve_votes,
            object_votes=object_votes,
            ballot=votes,
            note=note,
        )
        self._log(decision, prompt, default, input_type)
        return decision

    # -- review/feedback generation (synthesized from all members) ---------- #
    # Feedback ballots can answer a PROSE question with a bare digit string ("1,2") or a
    # placeholder ("empty"); without a guard one member's degenerate draft can become the
    # committee's official guidance. A draft must contain actual prose to enter synthesis.
    def _feedback_decision(
        self, prompt: str, default: str, context: Optional[Dict]
    ) -> CommitteeDecision:
        """Economist feedback for an open-ended checkpoint, synthesized from all members."""
        res = self.synthesize_feedback(prompt, context)
        return CommitteeDecision(
            answer=res.feedback if res.feedback else default,
            approve=True, voted=False, note=res.note, decider="llm_economist_feedback",
        )

    def synthesize_feedback(self, prompt: str, context: Optional[Dict] = None) -> FeedbackResult:
        """Collect each member's review feedback and synthesize into one (chair) feedback.

        All members draft independently (grounded in ``context``, told not to invent
        facts); the chair (first member) consolidates. Falls back to a single draft if
        only one member responded, or "" if none did. Never raises.
        """
        drafts = [self._member_feedback(m, prompt, context, member_index=i)
                  for i, m in enumerate(self.members)]
        # A selection prompt (pick/discard/merge numbered items) is
        # aggregated deterministically by majority, not by the chair, which can return the
        # UNION of the members' picks and out-of-range indices.
        kind = selection_kind(prompt)
        if kind:
            res = self._aggregate_selection_drafts(kind, drafts, context)
            self._log_record({
                "decider": _DECIDER, "kind": "feedback", "prompt": (prompt or "")[:500],
                "feedback": res.feedback[:1500], "note": res.note,
                "aggregation": "deterministic_selection",
                "drafts": [{"model": d.model, "feedback": d.feedback[:800], "error": d.error}
                           for d in drafts],
            })
            return res
        # For a prose question, digit-string/placeholder drafts are degenerate; a prompt
        # that itself asks for a terse format ("comma-separated", "e.g., '1,3'") is exempt.
        if _expects_terse(prompt):
            non_empty = [d for d in drafts if d.feedback]
        else:
            non_empty = [d for d in drafts if _is_substantive_feedback(d.feedback)]
        n_degenerate = sum(1 for d in drafts if d.feedback and d not in non_empty)
        if not non_empty:
            note = "no feedback generated"
            if n_degenerate:
                note += f" ({n_degenerate} degenerate draft(s) discarded)"
            res = FeedbackResult(feedback="", drafts=drafts, note=note)
        elif len(non_empty) == 1:
            res = FeedbackResult(feedback=non_empty[0].feedback, drafts=drafts,
                                 note="single member provided feedback (no synthesis)")
        else:
            combined = self._synthesize_feedback([d.feedback for d in non_empty], prompt, context)
            res = FeedbackResult(feedback=combined or non_empty[0].feedback, drafts=drafts,
                                 note=f"synthesized from {len(non_empty)} members")
        self._log_record({
            "decider": _DECIDER, "kind": "feedback", "prompt": (prompt or "")[:500],
            "feedback": res.feedback[:1500], "note": res.note,
            "drafts": [{"model": d.model, "feedback": d.feedback[:800], "error": d.error} for d in drafts],
        })
        return res

    @staticmethod
    def _aggregate_selection_drafts(kind: str, drafts: List[MemberFeedback],
                                    context: Optional[Dict]) -> FeedbackResult:
        """Majority-aggregate the members' selection answers (shared.selection_aggregation).

        A member whose call failed abstains; a member that answered empty voted for no item."""
        answered = [d for d in drafts if d.error is None]
        if not answered:
            return FeedbackResult(feedback="", drafts=drafts,
                                  note="no feedback generated (all members failed)")
        answer, details = aggregate_selection(kind, [d.feedback for d in answered],
                                              n_items=count_listed_items(context))
        note = f"deterministic {kind} selection: {details['rule']}"
        if details["rejected_out_of_range"]:
            note += f"; rejected out-of-range {details['rejected_out_of_range']}"
        return FeedbackResult(feedback=answer, drafts=drafts, note=note)

    def _member_feedback(self, model: str, prompt: str, context: Optional[Dict],
                         member_index: int = 0) -> MemberFeedback:
        parts = ["HITL FEEDBACK REQUEST", "---------------------", f"The workflow is asking:\n{prompt}"]
        if context:
            try:
                parts.append("\nMaterial under review:\n" + json.dumps(context, indent=2, default=str)[:6000])
            except Exception:
                parts.append("\nMaterial under review:\n" + str(context)[:6000])
        else:
            parts.append(
                "\n(No structured material was provided. Give brief, general economist "
                "guidance for this prompt; do not invent specific papers, data, or numbers.)"
            )
        parts.append("\nReturn the JSON object only.")
        _, lens_directive = _lens_for(member_index)
        system = FEEDBACK_SYSTEM + (f"\n\n{lens_directive}" if lens_directive else "")
        try:
            resp = self._client_factory(model).format_and_invoke(
                system, "\n".join(parts), parse_as=_FeedbackResponse
            )
            return MemberFeedback(model=model, feedback=(resp.feedback or "").strip())
        except Exception as e:
            return MemberFeedback(model=model, error=str(e))

    def _synthesize_feedback(self, drafts: List[str], prompt: str, context: Optional[Dict]) -> str:
        body = ["WORK / QUESTION:", (prompt or "")]
        for i, d in enumerate(drafts, 1):
            body.append(f"\nMEMBER {i} FEEDBACK:\n{d}")
        body.append("\nConsolidate into one feedback. Return the JSON object only.")
        try:
            resp = self._client_factory(self.members[0]).format_and_invoke(
                SYNTHESIS_SYSTEM, "\n".join(body), parse_as=_FeedbackResponse
            )
            return (resp.feedback or "").strip()
        except Exception:
            return ""

    # -- develop-until-approved loop --------------------------------------- #
    def refine_until_approved(
        self,
        approve_question: str,
        produce: Callable[[], Any],
        revise: Callable[[str], Any],
        context_of: Callable[[Any], Optional[Dict]],
        max_rounds: int = 3,
    ) -> RefineResult:
        """Iterate produce -> committee-approve? -> synthesized feedback -> revise, until
        the committee approves (2/3) or ``max_rounds`` is reached.

        ``produce()`` returns the initial artifact; ``revise(feedback)`` returns a new
        artifact given the synthesized feedback; ``context_of(artifact)`` yields the dict
        the committee reviews. All are workflow callbacks (e.g. a team's round method).
        """
        artifact = produce()
        history: List[RefineRound] = []
        approved = False
        rounds_done = 0
        rounds = max(1, max_rounds)
        for rnd in range(1, rounds + 1):
            rounds_done = rnd
            ctx = context_of(artifact)
            vote = self.vote_proposition(approve_question, ctx)
            fb_text = ""
            if not vote.passed and rnd < rounds:
                fb_text = self.synthesize_feedback(
                    "What must change for this work to be approved?", ctx
                ).feedback
            history.append(RefineRound(round=rnd, approved=vote.passed,
                                       yes_votes=vote.yes_votes, no_votes=vote.no_votes, feedback=fb_text))
            if vote.passed:
                approved = True
                break
            if rnd < rounds:
                artifact = revise(fb_text)
        return RefineResult(approved=approved, rounds=rounds_done, artifact=artifact, history=history)

    @staticmethod
    def _objection_answer(default: str, input_type: str, cast: List[MemberVote]):
        """Pick the answer token when the committee objects (majority said no)."""
        alts = [v.answer for v in cast if (v.approve is False and v.answer)]
        if alts:
            top, _ = Counter(alts).most_common(1)[0]
            return top, "objection: majority alternative"
        neg = _negative_token(default)
        if neg:
            return neg, "objection: no explicit alternative; inferred negative token"
        # Cannot infer a valid alternative — proceed with default but flag it.
        return default, "objection without actionable alternative; proceeded with default"

    # -- proposition voting (feasibility-gate cascade) ---------------------- #
    def vote_proposition(
        self, question: str, context: Optional[Dict] = None
    ) -> PropositionResult:
        """Vote YES/NO on one proposition; passes on a 2/3 majority. Never raises.

        Used by the Feasibility Review cascade: actions are proposed in
        priority order and the first proposition to pass carries.
        """
        votes = [self._member_proposition_vote(m, question, context, member_index=i)
                 for i, m in enumerate(self.members)]
        cast = [v for v in votes if not v.abstained]
        n = len(cast)
        if n == 0:
            res = PropositionResult(
                question=question, passed=False, contested=True, agreement=0.0,
                ballot=votes, note="all members abstained (calls failed); proposition not passed",
            )
            self._log_record(res.to_record())
            return res
        yes_votes = sum(1 for v in cast if v.approve)
        no_votes = n - yes_votes
        res = PropositionResult(
            question=question,
            passed=_approves(yes_votes, n),
            yes_votes=yes_votes,
            no_votes=no_votes,
            agreement=max(yes_votes, no_votes) / n,
            contested=yes_votes > 0 and no_votes > 0,
            ballot=votes,
        )
        self._log_record(res.to_record())
        return res

    def _member_proposition_vote(
        self, model: str, question: str, context: Optional[Dict], member_index: int = 0
    ) -> MemberVote:
        parts = ["FEASIBILITY PROPOSITION", "-----------------------", f"Vote YES or NO on:\n{question}"]
        if context:
            try:
                parts.append("\nContext:\n" + json.dumps(context, indent=2, default=str)[:4000])
            except Exception:
                parts.append("\nContext:\n" + str(context)[:4000])
        parts.append("\nReply with the JSON object only.")
        user = "\n".join(parts)
        lens_name, lens_directive = _lens_for(member_index)
        system = PROPOSITION_SYSTEM + (f"\n\n{lens_directive}" if lens_directive else "")
        try:
            vote = self._client_factory(model).format_and_invoke(
                system, user, parse_as=_PropositionVote
            )
            return MemberVote(model=model, approve=bool(vote.yes),
                              rationale=(vote.rationale or "").strip(), lens=lens_name)
        except Exception as e:
            return MemberVote(model=model, approve=None, error=str(e), lens=lens_name)

    # -- provenance --------------------------------------------------------- #
    def _log(self, decision: CommitteeDecision, prompt: str, default: str, input_type: str) -> None:
        self._log_record(decision.to_record(prompt, default, input_type))

    def _log_record(self, record: Dict[str, Any]) -> None:
        if not self.log_path:
            return
        try:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except Exception:
            pass  # provenance logging is best-effort; never break a run over it


def summarize_ballots(log_path: str) -> Dict[str, Any]:
    """Dissent telemetry over a ballot log: how often did members actually disagree?

    Reads the jsonl ballot log and returns counts: total multi-member ballots, unanimous vs
    split (2-1), per-member vote balance, and per-lens objection counts — the numbers that
    tell whether the 2/3 rule ever binds. Never raises (empty summary on trouble)."""
    out: Dict[str, Any] = {"ballots": 0, "unanimous": 0, "split": 0, "dissent_rate": 0.0,
                           "by_model": {}, "objections_by_lens": {}, "kinds": {}}
    try:
        with open(log_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                kind = r.get("kind", "decision")
                out["kinds"][kind] = out["kinds"].get(kind, 0) + 1
                ballot = r.get("ballot") or []
                votes = [b.get("approve", b.get("yes")) for b in ballot]
                cast = [v for v in votes if v is not None]
                if len(cast) >= 2:
                    out["ballots"] += 1
                    if len(set(cast)) == 1:
                        out["unanimous"] += 1
                    else:
                        out["split"] += 1
                for b in ballot:
                    m = b.get("model", "?")
                    out["by_model"][m] = out["by_model"].get(m, 0) + 1
                    v = b.get("approve", b.get("yes"))
                    if v is False:
                        lens = b.get("lens") or "?"
                        out["objections_by_lens"][lens] = out["objections_by_lens"].get(lens, 0) + 1
        if out["ballots"]:
            out["dissent_rate"] = round(out["split"] / out["ballots"], 4)
    except Exception:
        pass
    return out


def resolve_checkpoint(
    prompt: str,
    default: str = "",
    input_type: str = "general",
    context: Optional[Dict] = None,
) -> str:
    """Module-level convenience used by ``shared.auto_input``: return the answer string."""
    return LLMEconomistCommittee().decide(prompt, default, input_type, context).answer
