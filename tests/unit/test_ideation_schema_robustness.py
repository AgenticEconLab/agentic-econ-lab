# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Regression tests: IdeationTeam models must coerce richer LLM JSON instead of crashing.

Reproduces the real crashes ("30 validation errors for ConceptList", "14 validation errors
for QuestionList ... evolution_trace.alternatives_considered input_type=dict") + verifier-found
gaps (non-numeric priority_rank/score, single-object-where-list).
"""
from IdeationTeam.ael.schemas.stage_outputs import (
    SearchQuery, ResearchConcept, ConceptList, ResearchQuestion, QuestionList, PrioritizedQuestion,
)


def test_searchquery_list_of_dicts_coerced():
    sq = SearchQuery(queries=[{"query": "q1", "rationale": "r"}], keywords=[{"keyword": "k"}], focus_areas=[{"area": "a"}])
    assert sq.queries == ["q1"] and sq.keywords == ["k"] and sq.focus_areas == ["a"]


def test_researchconcept_dict_where_str():
    rc = ResearchConcept(concept_title={"text": "T"}, description={"summary": "s", "detail": "d"},
                         key_themes=[{"theme": "x"}], literature_support=[{"title": "p"}])
    assert rc.concept_title == "T" and isinstance(rc.description, str) and rc.key_themes == ["x"]


def test_questionlist_nested_evolution_trace_dicts():
    ql = QuestionList(questions=[{"question": "Q", "rationale": "w", "methodology_hints": ["m"],
        "related_concepts": ["c"], "evolution_trace": {"source_ideas": ["s"], "refinement_steps": ["r"],
        "alternatives_considered": [{"question": "alt1"}, {"question": "alt2"}]}}])
    assert ql.questions[0].evolution_trace.alternatives_considered == ["alt1", "alt2"]


def test_prioritized_question_non_numeric_rank_to_none():
    pq = PrioritizedQuestion(question="Q", theoretical_framework="F", rationale="R", methodology="M",
                             expected_impact="I", feasibility="High", priority_rank="high", priority_score="unknown")
    assert pq.priority_rank is None and pq.priority_score is None


def test_conceptlist_single_object_wrapped():
    # LLM returns a single concept object instead of a list
    cl = ConceptList(concepts={"concept_title": "T", "description": "d", "key_themes": ["x"], "literature_support": ["p"]})
    assert len(cl.concepts) == 1 and cl.concepts[0].concept_title == "T"


def test_freeform_audit_fields_keep_structure():
    rc = ResearchConcept(concept_title="T", description="d", key_themes=["x"], literature_support=["p"],
                         novelty_claim={"novelty": "n", "risk": "r"}, alternatives_considered=[{"framing": "a"}])
    assert isinstance(rc.novelty_claim, dict) and isinstance(rc.alternatives_considered[0], dict)


def test_questionlist_drops_stray_non_object_elements():
    # LLM emitted a stray string ('reasoning') among the question objects (real Stage-3 crash)
    from IdeationTeam.ael.schemas.stage_outputs import PrioritizedQuestionList
    pql = PrioritizedQuestionList(questions=[
        {"question": "Q1", "theoretical_framework": "F", "rationale": "R", "methodology": "M",
         "expected_impact": "I", "feasibility": "High"},
        "reasoning",  # stray non-object -> must be dropped, not crash
        {"question": "Q2", "theoretical_framework": "F", "rationale": "R", "methodology": "M",
         "expected_impact": "I", "feasibility": "High"},
    ])
    assert len(pql.questions) == 2 and all(q.question.startswith("Q") for q in pql.questions)


def test_questionlist_backfills_missing_required_fields():
    # Real crash: the LLM returned a question object missing the required 'rationale'
    # (questions.6.rationale: Field required). The mixin now backfills missing required
    # str/List[str] so a partial object degrades gracefully instead of crashing the stage.
    from IdeationTeam.ael.schemas.stage_outputs import QuestionList
    ql = QuestionList(questions=[
        {"question": "Q1", "rationale": "r1", "methodology_hints": ["m"], "related_concepts": ["c"]},
        {"question": "How do interconnections drive propagation of shocks."},  # missing required fields
    ])
    assert ql.questions[1].rationale == "" and ql.questions[1].methodology_hints == []
    assert ql.questions[0].rationale == "r1"  # present fields untouched
