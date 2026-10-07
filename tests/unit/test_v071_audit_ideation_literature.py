# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Fixes from the v0.7.1 run audit (IdeationTeam / LiteratureTeam / committee / references).

No network: every LLM / HTTP call is replaced by a fake.
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest

CODES = Path(__file__).resolve().parents[2]
ALL_MODES = ["ModeNoWcNoHITL", "ModeNoWcWithHITL", "ModeWithWcNoHITL", "ModeWithWcWithHITL"]


def _load(team, mode, stage_file, alias):
    path = CODES / team / "ael" / mode / stage_file
    spec = importlib.util.spec_from_file_location(f"{alias}_{mode}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# deterministic aggregation of selection answers
# --------------------------------------------------------------------------- #
from shared.selection_aggregation import (  # noqa: E402
    aggregate_items,
    aggregate_pairs,
    count_listed_items,
    is_on_topic,
    plan_selection,
    selection_kind,
)


class TestSelectionAggregation:
    def test_kind_detection(self):
        assert selection_kind("Enter pairs of questions to merge (e.g., '1,2 3,4'): ") == "pairs"
        assert selection_kind("Enter questions to discard (comma-separated, e.g., '2,5'): ") == "items"
        assert selection_kind("Enter questions to prioritize (comma-separated, e.g., '1,3'): ") == "items"
        assert selection_kind("Irrelevant paper titles OR numbers (comma-separated): ") == "items"
        assert selection_kind("Missing topics to explore (comma-separated): ") is None
        assert selection_kind("Additional guidance for refinement: ") is None

    def test_demo_discard_is_majority_not_union(self):
        # v0.7.1 demo: members '2', '2,3,4,5', '3,5' -> chair returned the union '2,3,4,5'
        ans, det = aggregate_items(["2", "2,3,4,5", "3,5"], n_items=5)
        assert ans == "2,3,5"
        assert det["support"]["4"] == 1

    def test_out_of_range_indices_rejected(self):
        ans, det = aggregate_items(["1,6", "1,6", "7"], n_items=5)
        assert ans == "1" and det["rejected_out_of_range"] == ["6", "7"]

    def test_demo_merges_valid_and_non_overlapping(self):
        # demo members: '1,4 2,5', '1,2 3,4 5,6', '1,5' -> chair '1,2 1,4 1,5 2,5 3,4 5,6'
        ans, _ = aggregate_pairs(["1,4 2,5", "1,2 3,4 5,6", "1,5"], n_items=5)
        assert ans == ""                      # no pair has 2 of 3 votes
        ans, det = aggregate_pairs(["1,2 3,4", "2,1 3,4 5,6", "1,2 2,3"], n_items=5)
        assert ans == "1,2 3,4"               # 5,6 out of range; 2,3 has 1 vote
        assert "5,6" in det["rejected_out_of_range"]
        ans, _ = aggregate_pairs(["1,2 2,3", "1,2 2,3", "2,3"], n_items=5)
        assert ans == "2,3"                   # 2,3 (3 votes) beats overlapping 1,2 (2 votes)

    def test_titles_majority(self):
        ans, _ = aggregate_items(["Paper A, Paper B", "paper a", "Paper C"])
        assert ans == "paper a"

    def test_count_listed_items(self):
        assert count_listed_items({"concepts": [1, 2], "questions": [1, 2, 3]}) == 5
        assert count_listed_items({"x": "y"}) is None

    def test_plan_selection_floor_and_original_numbering(self):
        # discard 2,3,5 of 5 (0-based 1,2,4) -> floor 3 restores the highest-ranked discard (2)
        plan = plan_selection(5, [1, 2, 4], [], min_keep=3)
        assert plan["keep"] == [0, 1, 3] and plan["restored"] == [1]
        # all discarded -> never empty
        plan = plan_selection(5, [0, 1, 2, 3, 4], [], min_keep=3)
        assert len(plan["keep"]) == 3
        # merges only between survivors, against ORIGINAL indices, no reuse
        plan = plan_selection(6, [5], [(0, 5), (0, 1), (1, 2), (3, 4)], min_keep=3)
        assert plan["merge"] == [(0, 1), (3, 4)]
        assert plan["keep"] == [0, 2, 3]

    def test_topic_check_generic(self):
        assert is_on_topic("Hospital competition and patient outcomes",
                           "How does hospital market concentration affect patient mortality?")
        assert is_on_topic("Tariff pass-through to import prices",
                           "Does monetary policy affect housing?") is False
        assert is_on_topic("economic research (upstream topic metadata unavailable)", "x") is None


class _FakeClient:
    def __init__(self, answers):
        self._answers = answers

    def format_and_invoke(self, system, user, parse_as=None):
        if parse_as.__name__ == "_FeedbackResponse":
            if "chair" in system:
                return parse_as(feedback="1,2 1,4 1,5 2,5 3,4 5,6")  # the old union behavior
            return parse_as(feedback=self._answers.pop(0))
        raise AssertionError("unexpected call")


class TestCommitteeSelection:
    def _committee(self, answers, tmp_path):
        from shared.llm_economist import LLMEconomistCommittee
        client = _FakeClient(list(answers))
        return LLMEconomistCommittee(members=["a", "b", "c"], client_factory=lambda m: client,
                                     log_path=str(tmp_path / "ballots.jsonl"))

    def test_discard_prompt_uses_majority(self, tmp_path):
        c = self._committee(["2", "2,3,4,5", "3,5"], tmp_path)
        ctx = {"prioritized_questions": ["q1", "q2", "q3", "q4", "q5"]}
        d = c.decide("Enter questions to discard (comma-separated, e.g., '2,5'): ", "", "general", ctx)
        assert d.answer == "2,3,5"
        rec = json.loads((tmp_path / "ballots.jsonl").read_text().splitlines()[-1])
        assert rec["aggregation"] == "deterministic_selection"

    def test_merge_prompt_never_returns_out_of_range_or_overlap(self, tmp_path):
        c = self._committee(["1,4 2,5", "1,2 3,4 5,6", "1,5 2,5 3,4"], tmp_path)
        ctx = {"prioritized_questions": ["q1", "q2", "q3", "q4", "q5"]}
        d = c.decide("Enter pairs of questions to merge (e.g., '1,2 3,4'): ", "", "general", ctx)
        assert d.answer == "2,5 3,4"

    def test_prose_prompt_still_synthesized(self, tmp_path):
        c = self._committee(["Consider the identification strategy carefully.",
                             "Tighten the data plan for the panel.", ""], tmp_path)
        d = c.decide("Additional guidance for refinement: ", "", "general", {"q": ["a"]})
        assert d.answer == "1,2 1,4 1,5 2,5 3,4 5,6"  # fake chair output: the chair path ran


# --------------------------------------------------------------------------- #
# IntegrationStage — floor, one numbering, no empty Finalizer, topic
# --------------------------------------------------------------------------- #

def _rq(mod, text, score=0.5):
    return mod.ResearchQuestion(question=text, rationale="r", methodology_hints=["m"],
                                related_concepts=[], feasibility_score=score)


@pytest.fixture(params=ALL_MODES)
def integ(request, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    return _load("IdeationTeam", request.param, "3-IntegrationStage.py", "_v071_integ")


class TestIntegrationStage:
    def test_apply_selection_feedback_floor_and_numbering(self, integ):
        qs = [_rq(integ, f"Q{i}") for i in range(1, 6)]
        fb = integ.IntegrationFeedback(round_number=1, questions_to_discard=[0, 1, 2, 3, 4],
                                       questions_to_merge=[(0, 1)])
        out, rec = integ.apply_selection_feedback(qs, fb)
        assert len(out) >= 3 and rec["n_out"] == len(out)
        fb = integ.IntegrationFeedback(round_number=1, questions_to_discard=[1],
                                       questions_to_merge=[(2, 3)], questions_to_prioritize=[4])
        out, rec = integ.apply_selection_feedback(qs, fb)
        # discard Q2; merge Q3+Q4 by ORIGINAL numbering; Q5 prioritized to the front
        assert [q.question.split(" (merged")[0] for q in out] == ["Q5", "Q1", "Q3"]
        assert "Q4" in out[2].question

    def _orch(self, integ, ctx_out, fin_out, calls):
        orch = integ.IntegrationOrchestrator(quiet=True)

        def ctx_fn(questions, feedback=None, research_topic=None):
            calls.append(("ctx", len(questions), research_topic))
            return ctx_out(questions)

        def fin_fn(contextualized_questions, feedback=None, max_questions=5, research_topic=None):
            calls.append(("fin", len(contextualized_questions), research_topic))
            return fin_out(contextualized_questions)

        orch.contextualizer.contextualize_questions = ctx_fn
        orch.finalizer.synthesize_questions = fin_fn
        return orch

    @staticmethod
    def _pq(integ, text, rank):
        return integ.PrioritizedQuestion(question=text, theoretical_framework="", rationale="",
                                         methodology=[], expected_impact="", feasibility="",
                                         priority_rank=rank)

    def test_empty_round_keeps_previous_set(self, integ):
        calls = []
        orch = self._orch(integ, lambda qs: [], lambda cq: [], calls)
        prev = [self._pq(integ, "kept question", 1)]
        orch.round_results[1] = {"contextualized": [], "prioritized": prev}
        _, prioritized = orch.run_integration_round([], round_number=2)
        assert prioritized == prev
        assert not [c for c in calls if c[0] == "fin"]   # Finalizer not called with nothing

    def test_contextualizer_parse_failure_does_not_starve_finalizer(self, integ):
        calls = []
        orch = self._orch(integ, lambda qs: [],
                          lambda cq: [self._pq(integ, c.question, i + 1) for i, c in enumerate(cq)],
                          calls)
        _, pr = orch.run_integration_round([_rq(integ, "Q1"), _rq(integ, "Q2")], round_number=1)
        assert [c[1] for c in calls if c[0] == "fin"] == [2] and len(pr) == 2

    def test_topic_passed_and_off_topic_ranked_last(self, integ):
        calls = []
        orch = self._orch(
            integ, lambda qs: [integ.ContextualizedQuestion(
                question=q.question, theoretical_framework="", literature_gaps=[],
                contribution="", related_theories=[]) for q in qs],
            lambda cq: [self._pq(integ, "How do interest rates affect housing?", 1),
                        self._pq(integ, "Do tariffs raise consumer import prices?", 2)],
            calls)
        orch.research_topic = "Tariff pass-through to consumer import prices"
        _, pr = orch.run_integration_round([_rq(integ, "a"), _rq(integ, "b")], round_number=1)
        assert all(c[2] == orch.research_topic for c in calls)
        assert pr[0].question.startswith("Do tariffs") and pr[0].on_topic is True
        assert pr[1].on_topic is False and pr[1].priority_rank == 2
        assert orch.off_topic and orch.off_topic[0]["question"].startswith("How do interest")

    def test_finalizer_prompt_has_topic_and_no_new_questions(self, integ):
        fin = integ.Finalizer("test")
        seen = {}

        def fake(system_prompt, user_prompt, variables=None, **kw):
            seen["prompt"] = user_prompt.format(**variables)
            return json.dumps({"questions": [
                {"question": f"q{i}", "theoretical_framework": "", "rationale": "",
                 "methodology": [], "expected_impact": "", "feasibility": "", "priority_rank": i}
                for i in range(1, 6)]})
        fin.llm.format_and_invoke = fake
        cq = [integ.ContextualizedQuestion(question="x", theoretical_framework="",
                                           literature_gaps=[], contribution="", related_theories=[])
              for _ in range(2)]
        out = fin.synthesize_questions(cq, max_questions=5, research_topic="Hospital mergers")
        assert "Hospital mergers" in seen["prompt"]
        assert "do not merge, reword or add questions" in seen["prompt"]
        assert "VERBATIM" in seen["prompt"]
        assert len(out) == 2                       # capped at the number of inputs
        assert fin.synthesize_questions([], max_questions=5) == []

    def test_feedback_file_goes_to_output_dir(self, integ, tmp_path, monkeypatch):
        if not hasattr(integ.IntegrationOrchestrator, "collect_integration_feedback"):
            pytest.skip("mode has no feedback checkpoint")
        monkeypatch.chdir(tmp_path)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        monkeypatch.setattr(integ, "auto_input", lambda *a, **k: "")
        orch = integ.IntegrationOrchestrator(quiet=True)
        orch.output_dir = str(run_dir)
        orch.collect_integration_feedback(round_number=1, num_questions=3)
        assert (run_dir / "round1_integration_feedback.json").exists()
        assert not (tmp_path / "round1_integration_feedback.json").exists()


# --------------------------------------------------------------------------- #
# synthesis never reads run data from the cwd
# --------------------------------------------------------------------------- #

def test_synthesis_does_not_read_cwd_batch(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    mod = _load("LiteratureTeam", "ModeWithWcWithHITL", "3-SynthesisStage.py", "_v071_syn")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "literature_batch.json").write_text(json.dumps(
        {"literature_items": [{"title": "Stale", "year": 1999, "authors": ["X"]}]}))
    orch = mod.SynthesisOrchestrator()
    seen = {}

    def fake_finalize(gap, review, plan, literature_metadata=None):
        seen["meta"] = literature_metadata
        raise StopIteration
    orch.cite_keeper.finalize_references = fake_finalize
    orch.knowledge_weaver.integrate_knowledge = lambda gap: ("rev", "plan", [])
    with pytest.raises(StopIteration):
        orch.run_synthesis_pipeline({"metadata": {}})
    assert seen["meta"] == {}                 # the cwd batch was NOT read


# --------------------------------------------------------------------------- #
# Crossref fields, headers, provider stats, retries
# --------------------------------------------------------------------------- #
from shared.tools import scholarly_search as ss  # noqa: E402


class TestScholarlySearch:
    def test_crossref_year_and_venue(self):
        item = {"posted": {"date-parts": [[2021, 5]]}, "container-title": ["SSRN Electronic Journal"]}
        assert ss.crossref_year(item) == 2021 and ss.crossref_venue(item) == "SSRN Electronic Journal"
        item = {"issued": {"date-parts": [[2019]]}, "published-online": {"date-parts": [[2018]]}}
        assert ss.crossref_year(item) == 2019
        assert ss.crossref_venue({"container-title": []}) is None
        for f in ("issued", "published-online", "posted", "container-title", "type"):
            assert f in ss.crossref_select()

    def test_headers_no_placeholder(self, monkeypatch):
        monkeypatch.delenv("CROSSREF_MAILTO", raising=False)
        monkeypatch.delenv("OPENALEX_MAILTO", raising=False)
        assert "example.com" not in ss.crossref_headers()["User-Agent"]
        assert "mailto" not in ss.crossref_headers()["User-Agent"]
        monkeypatch.setenv("OPENALEX_MAILTO", "me@uni.edu")
        assert "mailto:me@uni.edu" in ss.crossref_headers()["User-Agent"]

    def test_clean_venue(self):
        for p in ("Crossref", "OpenAlex", "arXiv", "Semantic Scholar", ""):
            assert ss.clean_venue(p) is None
        assert ss.clean_venue("Journal of Health Economics") == "Journal of Health Economics"

    def test_fetch_json_logs_and_counts_failures(self, monkeypatch):
        from shared.tools.tool_registry import ToolRegistry, ToolResult
        seen = {}

        def fake_invoke(name, payload, collector=None, agent=""):
            seen["retries"] = payload["retries"]
            return ToolResult(success=False, error="429 Too Many Requests", tool_name=name)
        monkeypatch.setattr(ToolRegistry, "invoke", staticmethod(fake_invoke))
        stats = ss.ProviderStats()
        assert ss.crossref_items("q", 5, stats=stats) == []
        assert seen["retries"] == ss.DEFAULT_RETRIES >= 2
        assert stats.as_dict()["Crossref"]["failures"] == 1
        assert "429" in stats.as_dict()["Crossref"]["last_error"]

    def test_crossref_items_parse(self, monkeypatch):
        from shared.tools.tool_registry import ToolRegistry, ToolResult
        body = {"message": {"items": [{
            "title": ["A health paper"], "author": [{"given": "Ann", "family": "Lee"}],
            "issued": {"date-parts": [[2020]]}, "container-title": ["J Health Econ"],
            "DOI": "10.1/x", "is-referenced-by-count": 3}]}}
        monkeypatch.setattr(ToolRegistry, "invoke", staticmethod(
            lambda name, payload, collector=None, agent="": ToolResult(
                success=True, data={"status_code": 200, "data": body}, tool_name=name)))
        stats = ss.ProviderStats()
        out = ss.crossref_items("q", 5, stats=stats)
        assert out[0]["year"] == 2020 and out[0]["venue"] == "J Health Econ"
        assert stats.as_dict()["Crossref"]["results"] == 1

    def test_dedupe_same_title_two_dois(self):
        papers = [
            {"title": "Trade Wars: Evidence!", "authors": ["Pablo Fajgelbaum"], "url": "10.2139/ssrn.1"},
            {"title": "trade wars evidence", "authors": ["Fajgelbaum, Pablo"], "url": "10.2139/ssrn.2",
             "year": 2020},
            {"title": "Trade Wars: Evidence", "authors": ["Someone Else"]},
        ]
        out = ss.dedupe_papers(papers)
        assert len(out) == 2 and out[0]["year"] == 2020   # duplicate filled the missing year

    def test_arxiv_econ_query(self):
        assert "cat:econ.*" in ss.arxiv_econ_query("hospital pricing")


@pytest.mark.parametrize("mode", ALL_MODES)
def test_trendsurfer_arxiv_uses_econ_categories(mode, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    mod = _load("IdeationTeam", mode, "1-SourcingStage.py", "_v071_src")
    seen = []
    from shared.tools.tool_registry import ToolResult

    def fake_invoke(name, payload, collector=None, agent=""):
        seen.append((name, payload))
        return ToolResult(success=True, data=[], tool_name=name)
    monkeypatch.setattr(mod.ToolRegistry, "invoke", staticmethod(fake_invoke))
    agent = mod.TrendSurfer("test") if "NoWc" in mode else mod.TrendSurfer("test", None)
    if hasattr(agent, "_search_open"):
        monkeypatch.setattr(agent, "_search_open", lambda *a, **k: [])
    agent.search("hospital competition", 5)
    q = [p["query"] for n, p in seen if n == "arxiv_search"]
    assert q and "cat:econ.*" in q[0]


@pytest.mark.parametrize("mode", ALL_MODES)
def test_scholar_searcher_counts_and_retries(mode, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    mod = _load("IdeationTeam", mode, "1-SourcingStage.py", "_v071_src2")
    from shared.tools.tool_registry import ToolResult
    seen = []

    def fake_invoke(name, payload, collector=None, agent=""):
        seen.append(payload)
        if "semanticscholar" in payload.get("url", ""):
            return ToolResult(success=False, error="HTTP 429", tool_name=name)
        return ToolResult(success=True, data={"status_code": 200, "data": {"message": {"items": [
            {"title": ["T"], "posted": {"date-parts": [[2022]]}, "container-title": ["SSRN"],
             "is-referenced-by-count": 99}]}}}, tool_name=name)
    monkeypatch.setattr(mod.ToolRegistry, "invoke", staticmethod(fake_invoke))  # one shared class
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    agent = mod.ScholarSearcher("test")
    agent.provider_stats = ss.ProviderStats()
    out = agent.search("q", 5)
    stats = agent.provider_stats.as_dict()
    assert stats["Semantic Scholar"]["failures"] == 1
    assert stats["Crossref"]["results"] == 1
    assert out and out[0].year == 2022 and out[0].venue == "SSRN"
    assert all(p.get("retries") == ss.DEFAULT_RETRIES for p in seen)


@pytest.mark.parametrize("mode", ALL_MODES)
def test_literature_crossref_and_openalex_counted(mode, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    mod = _load("LiteratureTeam", mode, "1-LiteratureGatheringStage.py", "_v071_gath")
    from shared.tools.tool_registry import ToolRegistry, ToolResult

    def fake_invoke(name, payload, collector=None, agent=""):
        if name == "openalex_search":
            return ToolResult(success=False, error="HTTP 503", tool_name=name)
        return ToolResult(success=True, data={"status_code": 200, "data": {"message": {"items": [
            {"title": ["A"], "issued": {"date-parts": [[2017]]},
             "container-title": ["Journal of Development Economics"]}]}}}, tool_name=name)
    monkeypatch.setattr(ToolRegistry, "invoke", staticmethod(fake_invoke))
    crawler = mod.TopicCrawler("test")
    res = crawler._search_crossref_fallback("q", 5)
    assert res[0].year == 2017 and res[0].venue == "Journal of Development Economics"
    assert crawler._search_openalex("q", 5) == []
    st = crawler.provider_stats.as_dict()
    assert st["OpenAlex"]["failures"] == 1 and st["Crossref"]["results"] == 1


# --------------------------------------------------------------------------- #
# report never prints the provider as the venue
# --------------------------------------------------------------------------- #

def test_reference_formatting_no_provider_venue():
    from ReportingTeam.ael.report_harness.assemble import _format_reference
    from ReportingTeam.ael.report_harness.formatting import _aea_reference
    item = {"title": "T", "authors": ["A B"], "year": 2020, "source": "Crossref", "venue": None}
    assert "Crossref" not in _format_reference(item)
    assert "Crossref" not in _aea_reference(item)
    item["venue"] = "OpenAlex"
    assert "OpenAlex" not in _format_reference(item)
    item["venue"] = "Journal of Health Economics"
    assert "*Journal of Health Economics*" in _format_reference(item)


# --------------------------------------------------------------------------- #
# pipeline runners — citation audit and ballot location
# --------------------------------------------------------------------------- #

def test_citation_audit_attached(tmp_path, monkeypatch):
    from pipeline import team_runners
    import shared.verification as ver
    f = tmp_path / "synthesis_results.json"
    f.write_text(json.dumps({"literature_review": {"sections": [{"content": "As shown [Smith2020]."}]}}))
    seen = {}

    def fake_pass(texts, collector=None):
        seen["texts"] = texts
        return {"overall_total": 1, "overall_hallucinated": 0, "overall_hallucination_rate": 0.0}
    monkeypatch.setattr(ver, "run_citation_verifier_pass", fake_pass)
    audit = team_runners._citation_audit(str(f), "LiteratureTeam")
    assert audit and seen["texts"] == ["As shown [Smith2020]."]
    assert json.loads(f.read_text())["citation_audit"]["overall_total"] == 1


def test_hitl_ballot_log_in_run_dir(tmp_path, monkeypatch):
    from pipeline import team_runners
    monkeypatch.delenv("AEL_HITL_LOG", raising=False)
    with team_runners._hitl_log_in(str(tmp_path)) as p:
        assert os.environ["AEL_HITL_LOG"] == str(tmp_path / "hitl_committee_ballots.jsonl") == p
    assert "AEL_HITL_LOG" not in os.environ
    monkeypatch.setenv("AEL_HITL_LOG", "/elsewhere.jsonl")
    with team_runners._hitl_log_in(str(tmp_path)):
        assert os.environ["AEL_HITL_LOG"] == "/elsewhere.jsonl"
