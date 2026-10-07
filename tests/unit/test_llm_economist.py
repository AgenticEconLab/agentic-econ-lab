# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Tests for the LLM-Economist committee HITL resolution mode.

Covers the design:
2/3 binary majority, decisive with N=3, non-decision fall-through, provenance
ballot logging, fail-safe on member errors, and the auto_input routing/back-compat.
"""

import json

import pytest

from shared import auto_input
from shared.llm_economist import (
    LLMEconomistCommittee,
    CommitteeDecision,
    resolve_checkpoint,
    get_committee_models,
    _approves,
    _is_decision,
)


# --------------------------------------------------------------------------- #
# Fake LLM client — returns a scripted vote, or raises, per model
# --------------------------------------------------------------------------- #
class _FakeClient:
    def __init__(self, vote):
        self._vote = vote  # dict(approve/answer/rationale) OR an Exception instance

    def format_and_invoke(self, system_prompt, user_prompt, parse_as=None, response_format=None):
        if isinstance(self._vote, Exception):
            raise self._vote
        return parse_as(**self._vote)


def _committee(votes_by_model, **kwargs):
    """Committee whose members are the keys of votes_by_model, stubbed."""
    members = list(votes_by_model.keys())
    factory = lambda model: _FakeClient(votes_by_model[model])  # noqa: E731
    kwargs.setdefault("log_path", "")  # logging off unless a test asks for it
    return LLMEconomistCommittee(members=members, client_factory=factory, **kwargs)


def _yes(ans="", why="ok"):
    return {"approve": True, "answer": ans, "rationale": why}


def _no(ans="", why="unsound"):
    return {"approve": False, "answer": ans, "rationale": why}


# --------------------------------------------------------------------------- #
# 2/3 majority arithmetic
# --------------------------------------------------------------------------- #
class TestMajorityRule:
    def test_two_thirds_of_three(self):
        assert _approves(3, 3) and _approves(2, 3)
        assert not _approves(1, 3) and not _approves(0, 3)

    def test_reduced_quorum_on_abstention(self):
        # 2 members voting => need both; 1 member => need it
        assert _approves(2, 2) and not _approves(1, 2)
        assert _approves(1, 1)

    def test_no_votes_never_approves(self):
        assert not _approves(0, 0)


# --------------------------------------------------------------------------- #
# Voting outcomes
# --------------------------------------------------------------------------- #
class TestCommitteeVote:
    def test_unanimous_approve_returns_default(self):
        c = _committee({"m1": _yes(), "m2": _yes(), "m3": _yes()})
        d = c.decide("Approve this stage? (yes/no)", default="yes", input_type="yes_no")
        assert d.answer == "yes"
        assert d.approve and d.voted and not d.contested
        assert d.approve_votes == 3 and d.agreement == 1.0

    def test_two_one_approve_still_passes_but_contested(self):
        c = _committee({"m1": _yes(), "m2": _yes(), "m3": _no("no")})
        d = c.decide("Approve this stage? (yes/no)", default="yes", input_type="yes_no")
        assert d.answer == "yes" and d.approve
        assert d.contested and d.approve_votes == 2 and d.object_votes == 1
        assert d.agreement == pytest.approx(2 / 3)

    def test_two_one_object_yields_negative_token(self):
        # majority objects; objectors gave no explicit alt -> inferred negative
        c = _committee({"m1": _no(), "m2": _no(), "m3": _yes()})
        d = c.decide("Approve this stage? (yes/no)", default="yes", input_type="yes_no")
        assert d.answer == "no" and not d.approve and d.contested

    def test_objection_uses_majority_alternative_for_choice(self):
        c = _committee({"m1": _no("B"), "m2": _no("B"), "m3": _yes("A")})
        d = c.decide("A) approve  B) revise model  C) narrow", default="A", input_type="choice")
        assert d.answer == "B" and not d.approve

    def test_unanimous_object_not_contested(self):
        c = _committee({"m1": _no("no"), "m2": _no("no"), "m3": _no("no")})
        d = c.decide("Approve? (yes/no)", default="yes", input_type="yes_no")
        assert not d.approve and not d.contested and d.object_votes == 3


def _pv(yes, why="ok"):
    return {"yes": yes, "rationale": why}


def _fb(text):
    return {"feedback": text}


class TestPropositionVote:
    def test_passes_on_two_thirds(self):
        c = _committee({"m1": _pv(True), "m2": _pv(True), "m3": _pv(False)})
        r = c.vote_proposition("Revise the model to drop variable X?", {"var": "X"})
        assert r.passed and r.yes_votes == 2 and r.no_votes == 1 and r.contested

    def test_fails_when_minority_yes(self):
        c = _committee({"m1": _pv(False), "m2": _pv(False), "m3": _pv(True)})
        r = c.vote_proposition("Accept the proxy?")
        assert not r.passed and r.yes_votes == 1

    def test_abstentions_do_not_pass(self):
        err = RuntimeError("down")
        c = _committee({"m1": err, "m2": err, "m3": err})
        r = c.vote_proposition("Accept the proxy?")
        assert not r.passed and r.agreement == 0.0 and "abstained" in r.note

    def test_proposition_logged(self, tmp_path):
        log = tmp_path / "ballots.jsonl"
        c = _committee({"m1": _pv(True), "m2": _pv(True), "m3": _pv(True)}, log_path=str(log))
        c.vote_proposition("Narrow the question?")
        rec = json.loads(log.read_text().strip())
        assert rec["kind"] == "proposition" and rec["passed"] is True
        assert rec["decider"] == "llm_economist_committee" and len(rec["ballot"]) == 3


# --------------------------------------------------------------------------- #
# Non-decision inputs & fail-safety
# --------------------------------------------------------------------------- #
class TestNonDecisionAndFailSafe:
    def test_mechanical_input_uses_default_without_calling(self):
        # a file path (input_type=path) and a seeded non-empty default are mechanical:
        # keep the default, don't call any model.
        calls = {"n": 0}

        def factory(model):
            calls["n"] += 1
            return _FakeClient(_fb("x"))

        c = LLMEconomistCommittee(members=["m1"], client_factory=factory, log_path="")
        d = c.decide("Enter file path:", default="", input_type="path")
        assert d.answer == "" and not d.voted and calls["n"] == 0
        d2 = c.decide("Enter topic:", default="ABM in macro", input_type="general")
        assert d2.answer == "ABM in macro" and not d2.voted and calls["n"] == 0

    def test_all_members_abstain_falls_back_to_default(self):
        err = RuntimeError("vLLM down")
        c = _committee({"m1": err, "m2": err, "m3": err})
        d = c.decide("Approve? (yes/no)", default="yes", input_type="yes_no")
        assert d.answer == "yes" and d.voted and d.contested
        assert d.agreement == 0.0 and "abstained" in d.note

    def test_partial_abstention_still_decides(self):
        c = _committee({"m1": _yes(), "m2": _yes(), "m3": RuntimeError("boom")})
        d = c.decide("Approve? (yes/no)", default="yes", input_type="yes_no")
        # 2 cast votes, both approve => passes on 2/2
        assert d.approve and d.approve_votes == 2 and d.object_votes == 0

    def test_approve_token_default_is_treated_as_decision(self):
        # 'approved' is an approve token even though input_type is 'general'
        c = _committee({"m1": _no(), "m2": _no(), "m3": _yes()})
        d = c.decide("Review budget", default="approved", input_type="general")
        assert d.voted and not d.approve


class TestFeedbackSynthesis:
    def test_review_prompt_generates_synthesized_feedback(self):
        c = _committee({"m1": _fb("drop paper A"), "m2": _fb("add DSGE lit"), "m3": _fb("prioritize Q2")})
        d = c.decide("Which papers are irrelevant?", default="", input_type="general",
                     context={"papers": ["A", "B", "C"]})
        assert not d.voted and d.approve and d.decider == "llm_economist_feedback"
        assert d.answer  # non-empty synthesized feedback (not the empty default)

    def test_synthesize_collects_all_member_drafts(self):
        c = _committee({"m1": _fb("tighten the identification argument"),
                        "m2": _fb("add robustness for the data section"),
                        "m3": _fb("clarify the policy mechanism")})
        res = c.synthesize_feedback("review this", {"x": 1})
        assert len(res.drafts) == 3 and all(d.feedback for d in res.drafts)
        assert res.feedback and "synthesized from 3" in res.note

    def test_synthesize_empty_when_all_members_fail(self):
        err = RuntimeError("down")
        c = _committee({"m1": err, "m2": err, "m3": err})
        res = c.synthesize_feedback("review this")
        assert res.feedback == "" and "no feedback" in res.note

    def test_system_prompts_request_format_honoring(self):
        from shared.llm_economist import FEEDBACK_SYSTEM, SYNTHESIS_SYSTEM
        assert "format" in FEEDBACK_SYSTEM.lower()
        assert "format" in SYNTHESIS_SYSTEM.lower()

    def test_terse_format_answer_passes_through(self):
        # members answer a format-specific field tersely -> committee returns it, not prose
        c = _committee({"m1": _fb("1,4"), "m2": _fb("1,4"), "m3": _fb("1,3")})
        d = c.decide("Enter questions to prioritize (comma-separated, e.g., '1,3'):",
                     default="", input_type="general", context={"questions": ["a", "b", "c", "d"]})
        assert d.answer == "1,4"


# Artifact-aware fake: approves once the revised artifact ("v2") is under review.
class _ArtifactAwareClient:
    def format_and_invoke(self, system_prompt, user_prompt, parse_as=None, response_format=None):
        from shared.llm_economist import _PropositionVote, _FeedbackResponse
        if parse_as is _PropositionVote:
            return parse_as(yes="v2" in user_prompt)
        if parse_as is _FeedbackResponse:
            return parse_as(feedback="please improve X")
        return parse_as()


class TestRefineUntilApproved:
    def test_approves_first_round(self):
        c = _committee({"m1": _pv(True), "m2": _pv(True), "m3": _pv(True)})
        res = c.refine_until_approved(
            approve_question="Approve this?", produce=lambda: "v1",
            revise=lambda fb: "v2", context_of=lambda a: {"artifact": a},
        )
        assert res.approved and res.rounds == 1 and res.artifact == "v1"

    def test_revises_until_approved(self):
        from shared.llm_economist import LLMEconomistCommittee
        c = LLMEconomistCommittee(members=["m1", "m2", "m3"],
                                  client_factory=lambda m: _ArtifactAwareClient(), log_path="")
        state = {"v": "v1"}

        def revise(_fb_text):
            state["v"] = "v2"
            return state["v"]

        res = c.refine_until_approved(
            approve_question="Approve this?", produce=lambda: state["v"],
            revise=revise, context_of=lambda a: {"artifact": a}, max_rounds=3,
        )
        assert res.approved and res.rounds == 2 and res.artifact == "v2"
        assert res.history[0].approved is False and res.history[1].approved is True
        assert res.history[0].feedback  # synthesized feedback drove the revision

    def test_not_approved_after_max_rounds(self):
        c = _committee({"m1": _pv(False), "m2": _pv(False), "m3": _pv(False)})
        res = c.refine_until_approved(
            approve_question="Approve this?", produce=lambda: "v",
            revise=lambda fb: "v", context_of=lambda a: {}, max_rounds=2,
        )
        assert not res.approved and res.rounds == 2 and len(res.history) == 2


# --------------------------------------------------------------------------- #
# Provenance logging
# --------------------------------------------------------------------------- #
class TestProvenance:
    def test_ballot_written_to_log(self, tmp_path):
        log = tmp_path / "ballots.jsonl"
        c = _committee(
            {"m1": _yes("", "sound"), "m2": _yes(), "m3": _no("no", "weak id")},
            log_path=str(log),
        )
        c.decide("Approve? (yes/no)", default="yes", input_type="yes_no")
        lines = log.read_text().strip().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["decider"] == "llm_economist_committee"
        assert rec["decision"] == "yes" and rec["approve"] is True
        assert rec["contested"] is True and len(rec["ballot"]) == 3
        assert {b["model"] for b in rec["ballot"]} == {"m1", "m2", "m3"}

    def test_logging_disabled_with_empty_path(self, tmp_path):
        c = _committee({"m1": _yes(), "m2": _yes(), "m3": _yes()}, log_path="")
        c.decide("Approve? (yes/no)", default="yes", input_type="yes_no")
        assert not list(tmp_path.iterdir())


# --------------------------------------------------------------------------- #
# Committee membership config
# --------------------------------------------------------------------------- #
class TestCommitteeConfig:
    def test_default_is_three_distinct_families(self, monkeypatch):
        monkeypatch.delenv("AEL_HITL_COMMITTEE", raising=False)
        models = get_committee_models()
        assert len(models) == 3
        joined = " ".join(models).lower()
        assert "gpt-oss" in joined and "phi-4" in joined and "olmo" in joined

    def test_default_shares_no_family_with_subject_or_judges(self, monkeypatch):
        """No model approves its own proposals; no judge scores work it helped decide."""
        monkeypatch.delenv("AEL_HITL_COMMITTEE", raising=False)
        joined = " ".join(get_committee_models()).lower()
        for fam in ("qwen", "mistral", "gemma"):
            assert fam not in joined

    def test_overlapping_override_is_warned(self, monkeypatch, caplog):
        monkeypatch.setenv("AEL_HITL_COMMITTEE", "vllm/qwen3.6-27b-fp8,vllm/phi-4-mini,vllm/x")
        with caplog.at_level("WARNING", logger="ael.llm_economist"):
            get_committee_models()
        assert "share a family" in caplog.text

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("AEL_HITL_COMMITTEE", "vllm/a, vllm/b ,vllm/c")
        assert get_committee_models() == ["vllm/a", "vllm/b", "vllm/c"]


# --------------------------------------------------------------------------- #
# auto_input routing + back-compat
# --------------------------------------------------------------------------- #
class TestHitlModeRouting:
    def test_get_hitl_mode_default_interactive(self, monkeypatch):
        monkeypatch.delenv("AEL_HITL_MODE", raising=False)
        monkeypatch.delenv("AUTO_HITL_MODE", raising=False)
        assert auto_input.get_hitl_mode() == "interactive"

    def test_auto_hitl_mode_true_maps_to_auto(self, monkeypatch):
        monkeypatch.delenv("AEL_HITL_MODE", raising=False)
        monkeypatch.setenv("AUTO_HITL_MODE", "true")
        assert auto_input.get_hitl_mode() == "auto"

    def test_auto_hitl_mode_false_is_interactive(self, monkeypatch):
        monkeypatch.delenv("AEL_HITL_MODE", raising=False)
        monkeypatch.setenv("AUTO_HITL_MODE", "false")
        assert auto_input.get_hitl_mode() == "interactive"

    def test_explicit_llm_economist(self, monkeypatch):
        monkeypatch.setenv("AEL_HITL_MODE", "llm_economist")
        assert auto_input.get_hitl_mode() == "llm_economist"

    def test_auto_input_routes_to_committee(self, monkeypatch):
        monkeypatch.setenv("AEL_HITL_MODE", "llm_economist")
        monkeypatch.setattr(
            "shared.llm_economist.resolve_checkpoint",
            lambda prompt, default="", input_type="general", context=None: "committee-says-no",
        )
        out = auto_input.auto_input("Approve? (yes/no)", default="yes", input_type="yes_no")
        assert out == "committee-says-no"

    def test_auto_input_committee_failure_falls_back(self, monkeypatch):
        monkeypatch.setenv("AEL_HITL_MODE", "llm_economist")

        def boom(*a, **k):
            raise RuntimeError("no server")

        monkeypatch.setattr("shared.llm_economist.resolve_checkpoint", boom)
        out = auto_input.auto_input("Approve? (yes/no)", default="yes", input_type="yes_no")
        assert out == "yes"  # fail-safe to default

    def test_auto_input_auto_mode_returns_default(self, monkeypatch):
        monkeypatch.delenv("AEL_HITL_MODE", raising=False)
        monkeypatch.setenv("AUTO_HITL_MODE", "true")
        out = auto_input.auto_input("Approve? (yes/no)", default="yes", input_type="yes_no")
        assert out == "yes"


# --------------------------------------------------------------------------- #
# per-member review lenses + dissent telemetry
# --------------------------------------------------------------------------- #
class _LensCapturingClient:
    """Records the system prompt so tests can assert lens injection."""
    captured = []

    def __init__(self, vote):
        self._vote = vote

    def format_and_invoke(self, system_prompt, user_prompt, parse_as=None, response_format=None):
        _LensCapturingClient.captured.append(system_prompt)
        return parse_as(**self._vote)


class TestMemberLenses:
    def test_each_member_gets_a_distinct_lens(self, monkeypatch):
        monkeypatch.delenv("AEL_HITL_LENSES", raising=False)
        _LensCapturingClient.captured = []
        members = ["m1", "m2", "m3"]
        c = LLMEconomistCommittee(
            members=members,
            client_factory=lambda m: _LensCapturingClient(_yes()),
            log_path="",
        )
        d = c.decide("Approve? (yes/no)", default="yes", input_type="yes_no")
        assert d.approve
        sys_prompts = _LensCapturingClient.captured
        assert len(sys_prompts) == 3
        assert "THEORETICAL SOUNDNESS" in sys_prompts[0]
        assert "DATA REALISM" in sys_prompts[1]
        assert "POLICY & PRACTICAL RELEVANCE" in sys_prompts[2]
        # lenses recorded on the ballot (telemetry)
        assert [v.lens for v in d.ballot] == ["theory", "data", "policy"]

    def test_lenses_disabled_by_env(self, monkeypatch):
        monkeypatch.setenv("AEL_HITL_LENSES", "0")
        _LensCapturingClient.captured = []
        c = LLMEconomistCommittee(
            members=["m1", "m2", "m3"],
            client_factory=lambda m: _LensCapturingClient(_yes()),
            log_path="",
        )
        d = c.decide("Approve? (yes/no)", default="yes", input_type="yes_no")
        assert all("REVIEW LENS" not in s.upper() or "lens" not in s.lower()
                   for s in _LensCapturingClient.captured)  # no lens text injected
        assert [v.lens for v in d.ballot] == ["", "", ""]

    def test_proposition_votes_carry_lenses_too(self, monkeypatch):
        monkeypatch.delenv("AEL_HITL_LENSES", raising=False)
        c = _committee({"m1": {"yes": True}, "m2": {"yes": True}, "m3": {"yes": False}})
        res = c.vote_proposition("Revise the model?")
        assert res.passed and res.contested
        assert [v.lens for v in res.ballot] == ["theory", "data", "policy"]


class TestDissentTelemetry:
    def test_summarize_ballots(self, tmp_path, monkeypatch):
        from shared.llm_economist import summarize_ballots
        monkeypatch.delenv("AEL_HITL_LENSES", raising=False)
        log = tmp_path / "ballots.jsonl"
        # one unanimous decision + one split proposition, through the REAL logging path
        c = _committee({"m1": _yes(), "m2": _yes(), "m3": _yes()}, log_path=str(log))
        c.decide("Approve? (yes/no)", default="yes", input_type="yes_no")
        c2 = _committee({"m1": {"yes": True}, "m2": {"yes": False}, "m3": {"yes": True}},
                        log_path=str(log))
        c2.vote_proposition("Narrow the question?")
        s = summarize_ballots(str(log))
        assert s["ballots"] == 2
        assert s["unanimous"] == 1 and s["split"] == 1
        assert s["dissent_rate"] == 0.5
        assert s["objections_by_lens"].get("data") == 1     # m2 (data lens) objected
        assert s["kinds"] == {"decision": 1, "proposition": 1}

    def test_summarize_missing_file_is_safe(self):
        from shared.llm_economist import summarize_ballots
        s = summarize_ballots("/nonexistent/ballots.jsonl")
        assert s["ballots"] == 0 and s["dissent_rate"] == 0.0


class TestSubstantiveFeedbackGuard:
    """In a live run 8/22 feedback ballots answered a prose question with '1,2' or
    'empty' — degenerate drafts must not become the committee's official guidance."""

    def test_digit_strings_and_placeholders_are_not_substantive(self):
        from shared.llm_economist import _is_substantive_feedback
        assert not _is_substantive_feedback("1,2")
        assert not _is_substantive_feedback("empty")
        assert not _is_substantive_feedback("N/A")
        assert not _is_substantive_feedback("")
        assert not _is_substantive_feedback("1, 2, 5")
        assert _is_substantive_feedback(
            "The model needs explicit micro-foundations for the discounting assumption.")

    def test_synthesis_discards_degenerate_drafts(self, monkeypatch):
        from shared import llm_economist as le
        committee = le.LLMEconomistCommittee.__new__(le.LLMEconomistCommittee)
        committee.members = ["m1", "m2", "m3"]
        committee._log_record = lambda rec: None
        drafts = {"m1": "", "m2": "empty", "m3": "1,2"}
        monkeypatch.setattr(
            le.LLMEconomistCommittee, "_member_feedback",
            lambda self, m, prompt, context, member_index=0: le.MemberFeedback(
                model=m, feedback=drafts[m]),
        )
        res = committee.synthesize_feedback("What must change in the model design?")
        assert res.feedback == ""                       # no fabricated guidance
        assert "degenerate" in res.note


def test_i10_decision_records_carry_kind():
    """Decision records carry a kind (kind-less ballots skewed every census)."""
    from shared.llm_economist import CommitteeDecision
    d = CommitteeDecision(answer="yes", approve=True, voted=True, note="", decider="x")
    rec = d.to_record("prompt", "yes", "yes_no")
    assert rec["kind"] == "decision"
