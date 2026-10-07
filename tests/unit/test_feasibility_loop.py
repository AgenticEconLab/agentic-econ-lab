# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Tests for the Model<->Data feasibility loop.

Covers the scout's verdicts, DAR aggregation, the reporter, the Feasibility
Review trade-off cascade + MRR, and the bounded/monotone loop controller.
"""

import pytest

from DataTeam.ael.schemas.stage_outputs import DataRequirement
from DataTeam.ael.feasibility import (
    AvailabilityFinding,
    AvailabilityReporter,
    AvailabilityScout,
    AvailableSeries,
    DataAvailabilityReport,
    DataRequirementsSpec,
    FeasibilityAction,
    FeasibilityLoopController,
    FeasibilityReview,
    ProxyCandidate,
    RequirementVerdict,
    available_series_from_source,
    drs_from_model_artifacts,
    format_revision_directive,
    similarity,
)

import importlib.util
from pathlib import Path
from types import SimpleNamespace


# --------------------------------------------------------------------------- #
# factories
# --------------------------------------------------------------------------- #
def _req(rid, var, priority="High"):
    return DataRequirement(
        requirement_id=rid, variable_name=var, description=f"{var} series",
        frequency="quarterly", time_period="1990-2023", geographic_coverage="US",
        unit_of_measurement="index", priority=priority, suggested_sources=[],
    )


def _series(var, source="FRED", tier="open_source_api", quality=1.0, sid=""):
    # a retrieved series whose source title is the variable itself (a direct
    # FEASIBLE needs a successful retrieval and a title concept match)
    return AvailableSeries(variable_name=var, source_name=source, series_id=sid, tier=tier,
                           quality=quality, retrieved=True, source_title=var)


def _finding(rid, var, verdict, priority="High", proxies=None):
    return AvailabilityFinding(
        requirement_id=rid, variable_name=var, priority=priority, verdict=verdict,
        matched_source="FRED" if verdict == RequirementVerdict.FEASIBLE else "",
        proxy_candidates=proxies or [],
    )


def _dar(tier="open_source_api", unmet=(), feasible=(), proxy=()):
    findings = []
    for i, v in enumerate(unmet):
        findings.append(_finding(f"U{i}", v, RequirementVerdict.UNAVAILABLE))
    for i, v in enumerate(feasible):
        findings.append(_finding(f"F{i}", v, RequirementVerdict.FEASIBLE))
    for i, v in enumerate(proxy):
        findings.append(_finding(
            f"P{i}", v, RequirementVerdict.PROXY_ONLY,
            proxies=[ProxyCandidate(variable_name=v + " proxy", source_name="FRED", closeness=0.4, bias_caveat="approx")],
        ))
    return DataAvailabilityReport.from_findings("RQ", tier, findings)


# --------------------------------------------------------------------------- #
# similarity + scout
# --------------------------------------------------------------------------- #
class TestScout:
    def test_similarity_bounds(self):
        assert similarity("unemployment rate", "unemployment rate") == pytest.approx(1.0)
        assert similarity("carbon emissions", "unemployment rate") == 0.0

    def test_feasible_direct_match(self):
        drs = DataRequirementsSpec(research_question="RQ", requirements=[_req("R1", "unemployment rate")])
        out = AvailabilityScout().grade(drs, [_series("Unemployment Rate", sid="UNRATE")])
        assert out[0].verdict == RequirementVerdict.FEASIBLE
        assert out[0].matched_series == "UNRATE"

    def test_proxy_only_for_approximate(self):
        drs = DataRequirementsSpec(research_question="RQ", requirements=[_req("R1", "consumer confidence index")])
        out = AvailabilityScout().grade(drs, [_series("consumer sentiment index")])
        assert out[0].verdict == RequirementVerdict.PROXY_ONLY
        assert out[0].proxy_candidates and out[0].proxy_candidates[0].bias_caveat

    def test_unavailable_when_no_match(self):
        drs = DataRequirementsSpec(research_question="RQ", requirements=[_req("R1", "carbon emissions")])
        out = AvailabilityScout().grade(drs, [_series("unemployment rate")])
        assert out[0].verdict == RequirementVerdict.UNAVAILABLE

    def test_low_quality_match_downgraded_to_proxy(self):
        drs = DataRequirementsSpec(research_question="RQ", requirements=[_req("R1", "gdp growth")])
        out = AvailabilityScout().grade(drs, [_series("gdp growth", quality=0.3)])
        assert out[0].verdict == RequirementVerdict.PROXY_ONLY
        assert "quality" in out[0].proxy_candidates[0].bias_caveat

    def test_other_tier_series_ignored(self):
        drs = DataRequirementsSpec(research_question="RQ", requirements=[_req("R1", "unemployment rate")])
        out = AvailabilityScout().grade(
            drs, [_series("unemployment rate", tier="premium_subscribed")], tier="open_source_api"
        )
        assert out[0].verdict == RequirementVerdict.UNAVAILABLE


class TestFetchableUniverse:
    """In a live run var:G_t Government Spending graded UNAVAILABLE — 29/34 unmet,
    not_converging — because the scout only saw the retrieved pool. With the curated
    fetchable_lookup seam, a pool-missing but connector-covered requirement is FEASIBLE
    (fetchable) and the MRR never asks the model to drop it."""

    @staticmethod
    def _lookup(name):
        return ("FRED", "GCE") if "government spending" in name.lower() else None

    def test_pool_miss_with_curated_hit_is_feasible_fetchable(self):
        drs = DataRequirementsSpec(research_question="RQ", requirements=[_req("R1", "Government Spending")])
        out = AvailabilityScout(fetchable_lookup=self._lookup).grade(
            drs, [_series("unemployment rate")])
        assert out[0].verdict == RequirementVerdict.FEASIBLE
        assert out[0].fetchable is True
        assert out[0].matched_series == "GCE" and out[0].matched_source == "FRED"
        assert "supplemental fetch" in out[0].notes and "open-connector" in out[0].notes

    def test_pool_direct_match_wins_without_consulting_lookup(self):
        calls = []
        def lookup(name):
            calls.append(name)
            return ("FRED", "GCE")
        drs = DataRequirementsSpec(research_question="RQ", requirements=[_req("R1", "Government Spending")])
        out = AvailabilityScout(fetchable_lookup=lookup).grade(
            drs, [_series("Government Spending", sid="W068RCQ027SBEA")])
        assert out[0].verdict == RequirementVerdict.FEASIBLE
        assert out[0].fetchable is False and out[0].matched_series == "W068RCQ027SBEA"
        assert calls == []   # pool match short-circuits: no curated consultation

    def test_curated_hit_outranks_pool_proxies(self):
        # a weak pool proxy must not shadow the exact curated id
        drs = DataRequirementsSpec(research_question="RQ", requirements=[_req("R1", "Government Spending")])
        out = AvailabilityScout(fetchable_lookup=self._lookup).grade(
            drs, [_series("government revenue")])   # proxy-grade similarity only
        assert out[0].verdict == RequirementVerdict.FEASIBLE and out[0].fetchable

    def test_lookup_miss_and_error_keep_prior_behavior(self):
        drs = DataRequirementsSpec(research_question="RQ", requirements=[_req("R1", "carbon emissions")])
        out = AvailabilityScout(fetchable_lookup=self._lookup).grade(drs, [_series("unemployment rate")])
        assert out[0].verdict == RequirementVerdict.UNAVAILABLE
        def boom(name):
            raise RuntimeError("curated table exploded")
        out = AvailabilityScout(fetchable_lookup=boom).grade(drs, [_series("unemployment rate")])
        assert out[0].verdict == RequirementVerdict.UNAVAILABLE   # seam failure never crashes grading

    def test_no_lookup_default_unchanged(self):
        drs = DataRequirementsSpec(research_question="RQ", requirements=[_req("R1", "Government Spending")])
        out = AvailabilityScout().grade(drs, [_series("unemployment rate")])
        assert out[0].verdict == RequirementVerdict.UNAVAILABLE


# --------------------------------------------------------------------------- #
# DAR aggregation + reporter
# --------------------------------------------------------------------------- #
class TestDARAndReporter:
    def test_aggregation_counts_and_unmet(self):
        findings = [
            _finding("R1", "a", RequirementVerdict.FEASIBLE, priority="High"),
            _finding("R2", "b", RequirementVerdict.PROXY_ONLY, priority="High"),
            _finding("R3", "c", RequirementVerdict.UNAVAILABLE, priority="Low"),
        ]
        dar = DataAvailabilityReport.from_findings("RQ", "open_source_api", findings)
        assert (dar.n_feasible, dar.n_proxy, dar.n_unavailable) == (1, 1, 1)
        # only essential (High) non-feasible requirements are "unmet"
        assert dar.unmet_essential == ["R2"]
        assert dar.overall_feasible is False

    def test_all_essential_feasible_is_overall_feasible(self):
        dar = _dar(feasible=["a", "b"], unmet=[])
        assert dar.overall_feasible is True and dar.unmet_essential == []

    def test_low_priority_gap_does_not_block(self):
        findings = [
            _finding("R1", "a", RequirementVerdict.FEASIBLE, priority="High"),
            _finding("R2", "b", RequirementVerdict.UNAVAILABLE, priority="Medium"),
        ]
        dar = DataAvailabilityReport.from_findings("RQ", "open_source_api", findings)
        assert dar.overall_feasible is True

    def test_reporter_builds_dar_with_summary(self):
        findings = [_finding("R1", "a", RequirementVerdict.FEASIBLE)]
        dar = AvailabilityReporter().build("RQ", "open_source_api", findings)
        assert dar.n_feasible == 1 and dar.summary

    def test_reporter_narrate_error_is_swallowed(self):
        def boom(*a):
            raise RuntimeError("llm down")

        dar = AvailabilityReporter(narrate_fn=boom).build("RQ", "open_source_api",
                                                           [_finding("R1", "a", RequirementVerdict.FEASIBLE)])
        assert dar.n_feasible == 1  # aggregation still works; summary falls back


# --------------------------------------------------------------------------- #
# Feasibility Review — trade-off cascade + MRR
# --------------------------------------------------------------------------- #
def _approve(substr):
    return lambda q, c: substr.lower() in q.lower()


def _approve_none():
    return lambda q, c: False


class TestFeasibilityReview:
    def test_accept_proxy_first_when_available(self):
        dar = _dar(proxy=["inflation"])
        decisions, mrr = FeasibilityReview(resolver=_approve("Accept the best available proxy")).review(dar)
        assert len(decisions) == 1
        assert decisions[0].action == FeasibilityAction.ACCEPT_PROXY
        assert decisions[0].chosen_proxy is not None
        assert mrr is None

    def test_revise_model_emits_mrr(self):
        dar = _dar(unmet=["capital stock"])  # no proxy -> cascade offers revise
        decisions, mrr = FeasibilityReview(resolver=_approve("Revise the model")).review(dar)
        assert decisions[0].action == FeasibilityAction.REVISE_MODEL
        assert mrr is not None
        assert "capital stock" in mrr.requested_change
        assert mrr.unmet_requirements == ["U0"]

    def test_escalate_only_when_available(self):
        dar = _dar(unmet=["tick data"])
        # escalation offered -> approved
        d1, _ = FeasibilityReview(resolver=_approve("Escalate"), escalation_available=True).review(dar)
        assert d1[0].action == FeasibilityAction.ESCALATE_SOURCE
        # escalation NOT offered -> that proposition never appears -> terminal default
        d2, _ = FeasibilityReview(resolver=_approve("Escalate"), escalation_available=False).review(dar)
        assert d2[0].action == FeasibilityAction.ACCEPT_AS_IS

    def test_terminal_default_when_all_declined(self):
        dar = _dar(unmet=["x"])
        decisions, mrr = FeasibilityReview(resolver=_approve_none()).review(dar)
        assert decisions[0].action == FeasibilityAction.ACCEPT_AS_IS
        assert mrr is None
        assert decisions[0].decider == "custom"

    def test_allow_revision_false_skips_model_revision(self):
        dar = _dar(unmet=["x"])
        decisions, mrr = FeasibilityReview(resolver=_approve("Revise the model")).review(dar, allow_revision=False)
        # revise is not offered -> falls through to narrow/accept-as-is
        assert decisions[0].action != FeasibilityAction.REVISE_MODEL
        assert mrr is None


# --------------------------------------------------------------------------- #
# Loop controller — bounded, monotone, terminating
# --------------------------------------------------------------------------- #
def _design_step():
    def step(cycle, mrr):
        return DataRequirementsSpec(research_question="RQ", requirements=[_req("R1", "x")], cycle=cycle)
    return step


def _availability_from(dars):
    def step(drs):
        return dars[drs.cycle - 1]
    return step


class TestAdapters:
    def test_drs_from_model_artifacts(self):
        model_design = {"formal_models": [{"model_title": "M", "variables": [
            {"variable_symbol": "Y", "variable_name": "output", "variable_type": "endogenous", "description": "y"},
            {"variable_symbol": "g", "variable_name": "government spending", "variable_type": "exogenous",
             "description": "gov", "domain": "R+"},
        ]}]}
        calibration = {"calibrated_models": [{"empirical_targets": [
            {"target_id": "T1", "target_name": "investment share", "importance": "High", "description": "I/Y"},
            {"target_id": "T2", "target_name": "labor share", "importance": "Low"},
        ]}]}
        drs = drs_from_model_artifacts("RQ", model_design, calibration)
        names = {r.variable_name for r in drs.requirements}
        assert "government spending" in names          # exogenous kept
        assert "output" not in names                   # endogenous excluded
        assert {"investment share", "labor share"} <= names
        by = {r.variable_name: r for r in drs.requirements}
        assert by["government spending"].priority == "High"
        assert by["investment share"].priority == "High"
        assert by["labor share"].priority == "Low"     # importance -> priority
        assert len(drs.essential()) == 2               # gov spending + investment share

    def test_drs_dedups_by_variable_name(self):
        md = {"formal_models": [{"variables": [
            {"variable_symbol": "x", "variable_name": "inflation", "variable_type": "exogenous"},
        ]}]}
        cal = {"calibrated_models": [{"empirical_targets": [
            {"target_id": "T", "target_name": "inflation", "importance": "High"},
        ]}]}
        drs = drs_from_model_artifacts("RQ", md, cal)
        assert len(drs.requirements) == 1

    def test_drs_tolerates_empty(self):
        assert drs_from_model_artifacts("RQ", None, None).requirements == []

    def test_drs_excludes_unobservable_shocks(self):
        """Shocks/noise/innovations are model PRIMITIVES, not observables — demanding them
        made the loop structurally non-convergent (a live run: Demand Shock, Signal Noise, ...
        as High-essential requirements). They must not become data requirements."""
        md = {"formal_models": [{"model_title": "HANK", "variables": [
            # the exact offenders from the real run — all typed "exogenous" by the LLM
            {"variable_symbol": "xi_t", "variable_name": "Cost-Push Shock", "variable_type": "exogenous"},
            {"variable_symbol": "eps_t", "variable_name": "Demand Shock", "variable_type": "exogenous"},
            {"variable_symbol": "eps_s", "variable_name": "Signal Noise", "variable_type": "exogenous",
             "description": "white noise measurement error"},
            {"variable_symbol": "a_R", "variable_name": "Routine AI Productivity Shock", "variable_type": "exogenous"},
            {"variable_symbol": "eta", "variable_name": "NLP Measurement Noise", "variable_type": "exogenous"},
            # genuine observables stay
            {"variable_symbol": "S_t", "variable_name": "NLP Sentiment Score", "variable_type": "exogenous"},
            {"variable_symbol": "r_n", "variable_name": "Natural Real Interest Rate", "variable_type": "exogenous"},
        ]}]}
        drs = drs_from_model_artifacts("RQ", md, None)
        names = {r.variable_name for r in drs.requirements}
        assert names == {"NLP Sentiment Score", "Natural Real Interest Rate"}   # shocks/noise all gone

    def test_available_series_from_selected(self):
        src = {"selected_series": [
            {"series_id": "UNRATE", "series_name": "Unemployment Rate", "source_name": "FRED",
             "frequency": "monthly", "units": "percent"},
        ]}
        av = available_series_from_source(src)
        assert len(av) == 1 and av[0].variable_name == "Unemployment Rate" and av[0].series_id == "UNRATE"

    def test_available_series_falls_back_to_retrieved(self):
        src = {"selected_series": [], "retrieved_data": [
            {"series_id": "GDP", "series_name": "GDP", "source_name": "FRED"},
        ]}
        assert len(available_series_from_source(src)) == 1

    def test_available_series_empty(self):
        assert available_series_from_source({}) == []
        assert available_series_from_source(None) == []


class TestRevisionDirective:
    def test_empty_for_none(self):
        assert format_revision_directive(None) == ""
        assert format_revision_directive({}) == ""

    def test_renders_reason_and_change(self):
        mrr = {"reason": "no data for K", "requested_change": "Remove capital stock"}
        note = format_revision_directive(mrr)
        assert "REVISION DIRECTIVE" in note
        assert "Remove capital stock" in note
        assert "Do not reintroduce" in note


def _load_design_module():
    path = Path(__file__).resolve().parents[2] / "ModelTeam/ael/ModeNoWcNoHITL/2-ModelDesignStage.py"
    spec = importlib.util.spec_from_file_location("_model_design_stage_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeLLM:
    def __init__(self):
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return '{"variables": []}'


class TestModelDesignRevisionInjection:
    """2b: the MRR directive is (only) injected into the variable-definition prompt."""

    def _designer(self):
        mod = _load_design_module()
        d = mod.ModelDesigner(openai_api_key="test")
        d.llm = _FakeLLM()
        return d

    def _framework(self):
        return SimpleNamespace(
            framework_title="F", theoretical_approach="A",
            conceptual_components=[], mathematical_formulations=[],
        )

    def test_directive_injected_when_present(self):
        d = self._designer()
        d._define_variables(self._framework(), revision_note="DROP capital stock proxy")
        user_msg = d.llm.messages[1]["content"]
        assert "DROP capital stock proxy" in user_msg

    def test_prompt_unchanged_when_empty(self):
        d = self._designer()
        d._define_variables(self._framework(), revision_note="")
        user_msg = d.llm.messages[1]["content"]
        assert "REVISION DIRECTIVE" not in user_msg
        assert user_msg.startswith("Define 10-20")  # byte-identical prefix to the original prompt


class TestLoopController:
    def test_feasible_on_first_cycle(self):
        ctrl = FeasibilityLoopController(
            _design_step(), _availability_from([_dar(feasible=["a"])]),
            review=FeasibilityReview(resolver=_approve_none()),
        )
        res = ctrl.run()
        assert res.status == "feasible" and res.cycles == 1 and res.feasible

    def test_resolved_without_revision_via_proxy(self):
        ctrl = FeasibilityLoopController(
            _design_step(), _availability_from([_dar(proxy=["inflation"])]),
            review=FeasibilityReview(resolver=_approve("Accept the best available proxy")),
        )
        res = ctrl.run()
        assert res.status == "resolved_without_revision" and res.cycles == 1
        assert res.mrrs == []

    def test_revise_then_feasible(self):
        dars = [_dar(unmet=["k"]), _dar(feasible=["k"])]  # cycle1 unmet -> revise; cycle2 feasible
        ctrl = FeasibilityLoopController(
            _design_step(), _availability_from(dars),
            review=FeasibilityReview(resolver=_approve("Revise the model")),
        )
        res = ctrl.run()
        assert res.status == "feasible" and res.cycles == 2
        assert len(res.mrrs) == 1

    def test_not_converging_stops(self):
        # unmet count fails to shrink between cycle1 and cycle2 -> stop
        dars = [_dar(unmet=["k1", "k2"]), _dar(unmet=["k1", "k2"])]
        ctrl = FeasibilityLoopController(
            _design_step(), _availability_from(dars),
            review=FeasibilityReview(resolver=_approve("Revise the model")),
        )
        res = ctrl.run()
        assert res.status == "not_converging" and res.cycles == 2

    def test_max_cycles_budget_exhausted(self):
        # strictly shrinking but never zero: 3 -> 2 -> 1, budget = 3
        dars = [_dar(unmet=["a", "b", "c"]), _dar(unmet=["a", "b"]), _dar(unmet=["a"])]
        ctrl = FeasibilityLoopController(
            _design_step(), _availability_from(dars),
            review=FeasibilityReview(resolver=_approve("Revise the model")), max_cycles=3,
        )
        res = ctrl.run()
        assert res.status == "max_cycles" and res.cycles == 3
        # last cycle forbids revision -> resolves the remaining requirement otherwise
        assert res.final_dar.unmet_essential == ["U0"]
        assert all(d.action != FeasibilityAction.REVISE_MODEL for d in res.decisions)

    def test_history_recorded_each_cycle(self):
        dars = [_dar(unmet=["k"]), _dar(feasible=["k"])]
        ctrl = FeasibilityLoopController(
            _design_step(), _availability_from(dars),
            review=FeasibilityReview(resolver=_approve("Revise the model")),
        )
        res = ctrl.run()
        assert [r.cycle for r in res.history] == [1, 2]
        assert res.history[0].mrr is not None and res.history[1].overall_feasible
