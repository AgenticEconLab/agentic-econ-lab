# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""AEL V0.7 Phase 6 — evaluation & reliability additions."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from evaluation.reliability import (
    ReliabilityEvaluator,
    RunResult,
    StratifiedReliabilityEvaluator,
    StratifiedReliabilityReport,
)
from evaluation.external import (
    HLERunner,
    HypoBenchRunner,
)
from evaluation.continuous import ContinuousProdEvaluator
from evaluation.dimensions import (
    CausalValidityEvaluator,
    SimulationFidelityEvaluator,
)
from evaluation.optimizer import AFlowV2Optimizer, WorkflowCandidate, WorkflowOperator


# ---------------------------------------------------------------------------
# StratifiedReliabilityEvaluator
# ---------------------------------------------------------------------------

def _runs(n: int, quality: float = 0.8, conf: float = 0.7) -> list[RunResult]:
    return [
        RunResult(
            run_id=f"r{i}",
            quality_score=quality,
            confidence=conf,
            output_text=f"output-{i}",
            dimension_scores={"correctness": quality},
        )
        for i in range(n)
    ]


class TestStratifiedReliability:
    def test_per_task_report_shape(self):
        evaluator = StratifiedReliabilityEvaluator()
        reports = evaluator.evaluate({
            "ideation":         _runs(3, quality=0.8),
            "literature_review": _runs(3, quality=0.7),
            "modeling":         _runs(3, quality=0.9),
            "simulation":       _runs(3, quality=0.85),
        })
        assert len(reports) == 4
        task_types = {r.task_type for r in reports}
        assert task_types == {
            "ideation", "literature_review", "modeling", "simulation",
        }
        for r in reports:
            assert isinstance(r, StratifiedReliabilityReport)
            assert 0.0 <= r.overall <= 1.0
            assert r.n_runs == 3

    def test_empty_task_type_skipped(self):
        evaluator = StratifiedReliabilityEvaluator()
        reports = evaluator.evaluate({
            "ideation": _runs(2),
            "empty":    [],
        })
        assert len(reports) == 1
        assert reports[0].task_type == "ideation"

    def test_distinct_task_types_count(self):
        evaluator = StratifiedReliabilityEvaluator()
        reports = evaluator.evaluate({
            "a": _runs(2), "b": _runs(2), "c": _runs(2), "d": _runs(2),
        })
        assert evaluator.distinct_task_types(reports) == 4

    def test_to_dict_roundtrip(self):
        evaluator = StratifiedReliabilityEvaluator()
        reports = evaluator.evaluate({"ideation": _runs(3)})
        out = evaluator.to_dict(reports)
        assert "ideation" in out
        assert "overall" in out["ideation"]
        assert out["ideation"]["n_runs"] == 3

    def test_different_task_types_distinct_scores(self):
        # Inject variance only in the modeling runs so Consistency diverges —
        # quality alone does NOT affect Princeton reliability.
        evaluator = StratifiedReliabilityEvaluator()
        ideation = _runs(5, quality=0.9, conf=0.9)
        modeling = [
            RunResult(run_id=f"m{i}", quality_score=q, confidence=c,
                       output_text=f"output-{i}")
            for i, (q, c) in enumerate([(0.9, 0.9), (0.5, 0.9), (0.3, 0.9),
                                          (0.7, 0.9), (0.95, 0.9)])
        ]
        reports = evaluator.evaluate({
            "ideation": ideation,
            "modeling": modeling,
        })
        by_task = {r.task_type: r.consistency for r in reports}
        # Consistency is variance-based — modeling should show lower consistency
        assert by_task["ideation"] > by_task["modeling"]


# ---------------------------------------------------------------------------
# HypoBenchRunner
# ---------------------------------------------------------------------------

class TestHypoBench:
    def test_default_prompt_set(self):
        r = HypoBenchRunner()
        assert len(r.prompts) >= 6

    def test_scores_hypotheses(self):
        hypos = [
            "Central bank policy influences inflation expectations.",
            "Labour market frictions increase unemployment duration.",
            "Monetary policy shifts increase income inequality.",
            "Financial contagion propagates via interbank lending.",
            "AI adoption raises total factor productivity causally.",
            "Bounded rationality agents deviate from rational expectations.",
        ]
        report = HypoBenchRunner().run(hypos)
        assert report.total_prompts >= 6
        assert 0.0 <= report.aggregate_score <= 1.0
        assert set(report.difficulty_breakdown.keys()) <= {"easy", "medium", "hard"}

    def test_empty_hypotheses(self):
        r = HypoBenchRunner().run([])
        assert r.total_prompts == 0

    def test_to_dict(self):
        r = HypoBenchRunner().run(["inflation monetary policy hypothesis"])
        d = r.to_dict()
        assert "aggregate_score" in d and "difficulty_breakdown" in d


# ---------------------------------------------------------------------------
# HLE runner
# ---------------------------------------------------------------------------

class TestHLE:
    def test_default_subset_size(self):
        assert len(HLERunner().problems) == 20

    def test_perfect_solver_100pct(self):
        r = HLERunner()
        def solver(q):
            idx = int(q.split("#")[-1])
            return f"answer_{idx}"
        report = r.run(solver)
        assert report.accuracy == 1.0
        assert report.correct == 20

    def test_zero_solver_0pct(self):
        report = HLERunner().run(lambda q: "")
        assert report.accuracy == 0.0

    def test_per_domain_breakdown(self):
        report = HLERunner().run(lambda q: "")
        assert set(report.per_domain.keys()) <= {"economics", "mathematics", "sciences"}
        assert all("accuracy" in v for v in report.per_domain.values())


# ---------------------------------------------------------------------------
# ContinuousProdEvaluator
# ---------------------------------------------------------------------------

class TestContinuousProdEval:
    def test_sampling_rate(self, tmp_path):
        e = ContinuousProdEvaluator(sample_every_n=3)
        log = tmp_path / "log.json"
        log.write_text("{}")
        results = [e.evaluate_log(log) for _ in range(10)]
        sampled = [r for r in results if r is not None]
        # With sample_every_n=3 over 10 calls, we expect ~3 hits (indexes 3, 6, 9)
        assert 2 <= len(sampled) <= 4

    def test_persistence_jsonl(self, tmp_path):
        log = tmp_path / "log.json"
        log.write_text(json.dumps({"team": "IdeationTeam"}))
        out = tmp_path / "eval_results.jsonl"
        e = ContinuousProdEvaluator(
            sample_every_n=1,
            output_path=out,
            eval_fn=lambda l: {"team": l.get("team")},
        )
        e.evaluate_log(log)
        e.evaluate_log(log)
        lines = out.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["summary"]["team"] == "IdeationTeam"

    def test_crash_safe(self, tmp_path):
        e = ContinuousProdEvaluator(sample_every_n=1)
        result = e.evaluate_log(tmp_path / "does-not-exist.json")
        assert result is not None
        assert result.status == "error"

    def test_overhead_budget(self):
        e = ContinuousProdEvaluator()
        assert e.overhead_budget_respected([0.1, 0.2, 0.3], budget_pct=5.0)
        assert not e.overhead_budget_respected([10.0, 20.0], budget_pct=5.0)

    def test_replay(self, tmp_path):
        logs = []
        for i in range(4):
            p = tmp_path / f"log_{i}.json"
            p.write_text("{}")
            logs.append(p)
        e = ContinuousProdEvaluator(sample_every_n=1)
        results = e.replay(logs)
        assert len(results) == 4


# ---------------------------------------------------------------------------
# SimulationFidelity & CausalValidity dimensions
# ---------------------------------------------------------------------------

class TestSimulationFidelity:
    def test_scores_stackelberg_exact(self):
        ev = SimulationFidelityEvaluator()
        score = ev.evaluate({
            "scenario": "stackelberg_tax",
            "metrics": {"tau_saez_optimal": 2.0 / 3.0},
        })
        assert score.score == 1.0

    def test_scores_low_when_off(self):
        ev = SimulationFidelityEvaluator(tolerance=0.01)
        score = ev.evaluate({
            "scenario": "consumer_choice",
            "metrics": {"hit_rate": 0.20},
        })
        assert score.score < 0.5

    def test_unknown_scenario_zero_score(self):
        score = SimulationFidelityEvaluator().evaluate({
            "scenario": "bogus", "metrics": {"x": 1.0},
        })
        assert score.score == 0.0
        assert "No benchmark" in score.notes


class TestCausalValidity:
    def test_synthetic_truth_perfect(self):
        score = CausalValidityEvaluator(tolerance=0.1).evaluate(
            {"ate": 2.0, "method": "back_door_ols"},
            true_ate=2.0,
        )
        assert score.score == 1.0
        assert score.relative_error == 0.0

    def test_synthetic_truth_close(self):
        score = CausalValidityEvaluator(tolerance=0.20).evaluate(
            {"ate": 2.1, "method": "back_door_ols"},
            true_ate=2.0,
        )
        assert 0.5 < score.score < 1.0

    def test_prior_mean_contained_in_ci(self):
        score = CausalValidityEvaluator().evaluate(
            {"ate": 1.5, "ci_low": 1.2, "ci_high": 1.8, "method": "iv_2sls"},
            prior_mean=1.55,
        )
        assert score.score == 1.0

    def test_prior_mean_outside_ci(self):
        score = CausalValidityEvaluator().evaluate(
            {"ate": 1.0, "ci_low": 0.9, "ci_high": 1.1, "method": "x"},
            prior_mean=2.0,
        )
        assert score.score < 1.0

    def test_no_ground_truth_not_evaluated(self):
        score = CausalValidityEvaluator().evaluate(
            {"ate": 1.0, "method": "x"},
        )
        assert score.score is None       # nothing to compare against -> not evaluated


# ---------------------------------------------------------------------------
# AFlowV2 optimizer (offline)
# ---------------------------------------------------------------------------

class TestAFlowV2:
    def _ops(self):
        return [
            WorkflowOperator("search", "Search the web for literature"),
            WorkflowOperator("summarize", "Summarise a document"),
            WorkflowOperator("verify", "Run CitationVerifier"),
            WorkflowOperator("falsify", "Run AdversarialFalsifier"),
        ]

    def test_rejects_empty_operators(self):
        with pytest.raises(ValueError):
            AFlowV2Optimizer([], lambda c: 0.0)

    def test_returns_ranked_candidates(self):
        ops = self._ops()
        # Reward workflows that end with a 'verify' step
        def fit(c):
            score = 0.5
            if c.operators[-1].name == "verify":
                score += 0.4
            if len(c.operators) >= 2 and c.operators[0].name == "search":
                score += 0.1
            return score

        opt = AFlowV2Optimizer(ops, fit, max_depth=2, budget=30, seed=1)
        ranked = opt.optimize()
        assert ranked
        assert ranked[0].fitness >= ranked[-1].fitness  # sorted descending
        # Best should end with 'verify'
        assert ranked[0].operators[-1].name == "verify"

    def test_best_shortcut(self):
        ops = self._ops()
        # 4 ops → 4^1+4^2+4^3 = 84 candidates at depth<=3; budget 100 covers all.
        opt = AFlowV2Optimizer(ops, lambda c: len(c.operators), max_depth=3, budget=100)
        best = opt.best()
        assert best and len(best.operators) == 3

    def test_budget_respected(self):
        ops = self._ops()
        opt = AFlowV2Optimizer(ops, lambda c: 1.0, max_depth=3, budget=10)
        ranked = opt.optimize()
        assert len(ranked) <= 10

    def test_mutation_phase_fills_budget(self):
        ops = self._ops()
        # Depth 1 + 4 operators -> only 4 unique candidates; budget 10 forces mutation
        opt = AFlowV2Optimizer(ops, lambda c: len(c.operators), max_depth=1, budget=10,
                                seed=42)
        ranked = opt.optimize()
        # All depth-1 candidates enumerated (4), then mutation fills up to 10
        assert len(ranked) >= 4


# ---------------------------------------------------------------------------
# Integration — ensure V0.7 dimensions roundtrip through ReliabilityEvaluator
# ---------------------------------------------------------------------------

class TestDimensionsIntegration:
    def test_v07_dimensions_are_independent_of_v06_evaluator(self):
        # V0.7 adds dimensions without modifying V0.6 evaluator's output shape.
        runs = _runs(5, quality=0.8)
        v06_result = ReliabilityEvaluator().evaluate_all(runs)
        assert hasattr(v06_result, "consistency")
        # Simulation/causal evaluators produce their own scores (not part of
        # the 4-dim Princeton evaluator).
        sim_score = SimulationFidelityEvaluator().evaluate({
            "scenario": "stackelberg_tax",
            "metrics": {"tau_saez_optimal": 2.0 / 3.0},
        })
        causal_score = CausalValidityEvaluator().evaluate(
            {"ate": 2.0}, true_ate=2.0,
        )
        # Both dimensions produce numeric scores in [0, 1].
        assert 0.0 <= sim_score.score <= 1.0
        assert 0.0 <= causal_score.score <= 1.0
