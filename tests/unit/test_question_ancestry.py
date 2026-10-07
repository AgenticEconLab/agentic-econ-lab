# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""The Integration stage's Contextualizer/Finalizer may only reformulate the
questions they are given. Each output must cite a source id; outputs that cite no valid,
unused id or that are not a reformulation of their source are replaced by the source.

No network: every LLM call is replaced by a fake.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from shared.question_ancestry import (
    REFORMULATION_MIN_SIMILARITY,
    parse_source_id,
    reformulation_similarity,
    validate_ancestry,
)

CODES = Path(__file__).resolve().parents[2]
ALL_MODES = ["ModeNoWcNoHITL", "ModeNoWcWithHITL", "ModeWithWcNoHITL", "ModeWithWcWithHITL"]

TOPIC = "Tariff pass-through to consumer import prices"
SRC = [
    "Do tariffs raise consumer import prices for durable goods?",
    "How do exchange-rate movements alter tariff pass-through to retail prices?",
    "Which firms absorb tariff costs in their margins rather than passing them on?",
]
# A wholly new question that carries the seed topic's vocabulary.
NEW = ("Do tariff pass-through rates to consumer import prices predict central-bank "
       "digital currency adoption by households?")


# --------------------------------------------------------------------------- #
# Generic helper
# --------------------------------------------------------------------------- #

class TestValidateAncestry:
    def test_parse_source_id(self):
        assert parse_source_id("Q2", 3) == 1
        assert parse_source_id("q3", 3) == 2
        assert parse_source_id("Question 1", 3) == 0
        assert parse_source_id(2, 3) == 1
        assert parse_source_id("2", 3) == 1
        for bad in (None, "", "Q0", "Q4", 7, "Q1a", True, 1.5):
            assert parse_source_id(bad, 3) is None

    def test_similarity_ignores_seed_topic_vocabulary(self):
        assert reformulation_similarity(SRC[0], SRC[0], TOPIC) == 1.0
        assert reformulation_similarity(SRC[1], NEW, TOPIC) < REFORMULATION_MIN_SIMILARITY
        reform = "By how much do tariffs raise consumer import prices of durable goods?"
        assert reformulation_similarity(SRC[0], reform, TOPIC) >= REFORMULATION_MIN_SIMILARITY

    def test_statuses_and_slots(self):
        outs = [
            (NEW, "Q2"),                                                  # new text: replaced
            ("By how much do tariffs raise consumer import prices of durable goods?", "Q1"),
            (SRC[2], "Q3"),                                               # unchanged
        ]
        entries, rejected = validate_ancestry(SRC, outs, topic=TOPIC)
        assert [(e["source_id"], e["status"]) for e in entries] == [
            ("Q2", "carried_forward"), ("Q1", "reformulated"), ("Q3", "accepted")]
        assert entries[0]["text"] == SRC[1]
        assert len(rejected) == 1 and rejected[0]["text"] == NEW

    def test_no_id_or_reused_id_dropped_and_uncited_source_fills(self):
        outs = [(SRC[0], "Q1"), (SRC[0] + " Really?", "Q1"), (NEW, None)]
        entries, rejected = validate_ancestry(SRC, outs, topic=TOPIC)
        assert [(e["source_id"], e["status"]) for e in entries] == [
            ("Q1", "accepted"), ("Q2", "carried_forward"), ("Q3", "carried_forward")]
        assert {r["reason"] for r in rejected} == {"source id Q1 already used", "no valid source id"}

    def test_never_more_outputs_than_inputs(self):
        outs = [(s, f"Q{i}") for i, s in enumerate(SRC + SRC + SRC, 1)]
        entries, _ = validate_ancestry(SRC, outs, slots=9)
        assert len(entries) == len(SRC)
        assert len(validate_ancestry(SRC, outs[:3], slots=2)[0]) == 2


# --------------------------------------------------------------------------- #
# Integration stage, all four modes
# --------------------------------------------------------------------------- #

@pytest.fixture(params=ALL_MODES)
def integ(request, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    path = CODES / "IdeationTeam" / "ael" / request.param / "3-IntegrationStage.py"
    spec = importlib.util.spec_from_file_location(f"_anc_integ_{request.param}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ctx_payload(items):
    return json.dumps({"questions": [
        dict({"theoretical_framework": "framework", "literature_gaps": ["g1", "g2"],
              "contribution": "contribution", "related_theories": ["t1", "t2"]}, **it)
        for it in items]})


def _fin_payload(items):
    return json.dumps({"questions": [
        dict({"theoretical_framework": "", "rationale": "", "methodology": [],
              "expected_impact": "", "feasibility": ""}, **it)
        for it in items]})


def _orch(integ, ctx_items, fin_items, seen=None):
    orch = integ.IntegrationOrchestrator(quiet=True)
    orch.research_topic = TOPIC

    def ctx(system_prompt, user_prompt, variables=None, **kw):
        if seen is not None:
            seen["ctx"] = user_prompt.format(**variables)
        return _ctx_payload(ctx_items)

    def fin(system_prompt, user_prompt, variables=None, **kw):
        if seen is not None:
            seen["fin"] = user_prompt.format(**variables)
        return _fin_payload(fin_items)

    orch.contextualizer.llm.format_and_invoke = ctx
    orch.finalizer.llm.format_and_invoke = fin
    return orch


def _rqs(integ):
    return [integ.ResearchQuestion(question=s, rationale="r", methodology_hints=[],
                                   related_concepts=[]) for s in SRC]


_CTX_OK = [{"source_id": f"Q{i}", "question": s} for i, s in enumerate(SRC, 1)]


class TestIntegrationAncestry:
    def test_new_question_with_topic_vocabulary_cannot_take_rank_one(self, integ):
        fin = [
            {"source_id": "Q2", "question": NEW, "priority_rank": 1},
            {"source_id": "Q1", "priority_rank": 2,
             "question": "By how much do tariffs raise consumer import prices of durable goods?"},
        ]
        orch = _orch(integ, _CTX_OK, fin)
        _, pr = orch.run_integration_round(_rqs(integ), round_number=1)
        assert NEW not in [q.question for q in pr]
        assert pr[0].question == SRC[1] and pr[0].source_id == "Q2"
        assert pr[0].ancestry == "carried_forward" and pr[0].priority_rank == 1
        assert pr[1].source_id == "Q1" and pr[1].ancestry == "reformulated"

    def test_uncited_output_dropped_and_uncovered_source_carried(self, integ):
        fin = [{"question": NEW, "priority_rank": 1},
               {"source_id": "Q3", "question": SRC[2], "priority_rank": 2},
               {"source_id": "Q3", "question": SRC[2] + " In which sectors?", "priority_rank": 3}]
        orch = _orch(integ, _CTX_OK, fin)
        _, pr = orch.run_integration_round(_rqs(integ), round_number=1)
        assert len(pr) == 3
        assert [(q.source_id, q.ancestry) for q in pr] == [
            ("Q3", "accepted"), ("Q1", "carried_forward"), ("Q2", "carried_forward")]
        assert [q.priority_rank for q in pr] == [1, 2, 3]
        rec = [r for r in orch.ancestry_log if r["step"] == "finalizer"][0]
        assert len(rec["rejected"]) == 2

    def test_never_more_final_questions_than_inputs(self, integ):
        fin = [{"source_id": f"Q{(i % 3) + 1}", "question": SRC[i % 3], "priority_rank": i + 1}
               for i in range(7)]
        orch = _orch(integ, _CTX_OK, fin)
        _, pr = orch.run_integration_round(_rqs(integ), round_number=1, max_final_questions=5)
        assert len(pr) == 3 and sorted(q.source_id for q in pr) == ["Q1", "Q2", "Q3"]

    def test_contextualizer_cannot_introduce_questions(self, integ):
        ctx = [{"source_id": "Q1", "question": NEW},                 # replaced by Q1
               {"source_id": "Q2", "question": SRC[1]},
               {"question": "An extra question nobody asked for?"}]   # dropped; Q3 carried
        fin_seen = {}
        orch = _orch(integ, ctx, [{"source_id": "Q1", "question": SRC[0], "priority_rank": 1}],
                     seen=fin_seen)
        cq, _ = orch.run_integration_round(_rqs(integ), round_number=1)
        assert [c.question for c in cq] == SRC
        assert [(c.source_id, c.ancestry) for c in cq] == [
            ("Q1", "carried_forward"), ("Q2", "accepted"), ("Q3", "carried_forward")]
        assert NEW not in fin_seen["fin"] and "An extra question" not in fin_seen["fin"]

    def test_prompts_carry_source_ids(self, integ):
        seen = {}
        orch = _orch(integ, _CTX_OK, [{"source_id": "Q1", "question": SRC[0], "priority_rank": 1}],
                     seen=seen)
        orch.run_integration_round(_rqs(integ), round_number=1)
        for key in ("ctx", "fin"):
            assert "Question Q1:" in seen[key] and "Question Q3:" in seen[key]
            assert "source_id" in seen[key]

    def test_final_file_records_ancestry(self, integ, tmp_path):
        fin = [{"source_id": "Q2", "question": NEW, "priority_rank": 1}]
        orch = _orch(integ, _CTX_OK, fin)
        orch.run_integration_round(_rqs(integ), round_number=1)
        out = tmp_path / "final.json"
        orch.save_final_questions(str(out))
        data = json.loads(out.read_text())
        q = data["final_questions"][0]
        assert (q["source_id"], q["ancestry"]) == ("Q2", "carried_forward")
        steps = [r["step"] for r in data["ancestry_checks"]]
        assert steps == ["contextualizer", "finalizer"]


def test_published_text_is_the_source_question():
    from shared.question_ancestry import validate_ancestry
    src = ["How does monetary policy affect household consumption in France?"]
    out = [("How does monetary policy affect household bankruptcy in Germany?", "Q1")]
    entries, _ = validate_ancestry(src, out, topic="Monetary policy")
    assert entries[0]["text"] == src[0]
    assert entries[0]["proposed_wording"] == out[0][0]


def test_reworded_output_publishes_the_source_in_every_mode():
    import importlib.util, sys
    from pathlib import Path
    from types import SimpleNamespace
    root = Path(__file__).resolve().parents[2] / "IdeationTeam" / "ael"
    for f in sorted(root.glob("Mode*/3-IntegrationStage.py")):
        spec = importlib.util.spec_from_file_location(f"integ_{f.parent.name}_r3", f)
        mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        src = mod.ContextualizedQuestion(question="How does monetary policy affect household consumption in France?",
                                         theoretical_framework="t", literature_gaps=[], contribution="c",
                                         related_theories=[])
        fin = mod.PrioritizedQuestion(question="How does monetary policy affect household bankruptcy in Germany?",
                                      theoretical_framework="x", rationale="r", methodology=["German insolvency data"],
                                      expected_impact="e", feasibility="f", priority_score=0.9, priority_rank=1,
                                      source_id="Q1")
        out, rec = mod.check_finalizer_ancestry([src], [fin], 5, research_topic="Monetary policy")
        assert out[0].question == src.question, f.parent.name
        assert out[0].methodology == []
        assert rec["questions"][0]["proposed_wording"] == fin.question


def test_sign_change_is_not_verbatim():
    from shared.question_ancestry import validate_ancestry, ACCEPTED
    src = ["How does a -2% policy rate affect consumption?"]
    entries, _ = validate_ancestry(src, [("How does a +2% policy rate affect consumption?", "Q1")])
    assert entries[0]["status"] != ACCEPTED and entries[0]["text"] == src[0]
    same, _ = validate_ancestry(src, [("How does a  -2% policy rate affect consumption? ", "Q1")])
    assert same[0]["status"] == ACCEPTED
