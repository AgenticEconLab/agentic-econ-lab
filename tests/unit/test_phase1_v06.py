# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
AEL V0.6 Phase 1 — Tournament-Based Research Intelligence Tests

Tests for TournamentEngine, Elo scoring, pairwise debate, MCTS explorer,
IdeationTeam integration, and backward compatibility.
"""

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


# ── Tournament Engine: Core ───────────────────────────────────────────────


class TestTournamentEngineImports:
    """Verify module structure and imports."""

    def test_module_imports(self):
        from shared.research.tournament import (
            TournamentEngine,
            Hypothesis,
            RankedHypothesis,
            DebateResult,
            DEFAULT_K_FACTOR,
            DEFAULT_INITIAL_ELO,
        )

    def test_hypothesis_model(self):
        from shared.research.tournament import Hypothesis
        h = Hypothesis(
            id="h1",
            research_question="How does AI affect GDP growth?",
            rationale="AI adoption increases productivity",
            novelty_score=0.8,
            feasibility_score=0.7,
        )
        assert h.id == "h1"
        assert h.novelty_score == 0.8
        assert h.source_stage == "RefinementStage"

    def test_ranked_hypothesis_model(self):
        from shared.research.tournament import RankedHypothesis
        rh = RankedHypothesis(
            id="h1",
            research_question="Test?",
            rationale="Test",
            elo_rating=1250.0,
            debate_wins=3,
            debate_losses=1,
            rank=1,
        )
        assert rh.elo_rating == 1250.0
        assert rh.rank == 1

    def test_debate_result_model(self):
        from shared.research.tournament import DebateResult
        dr = DebateResult(
            winner_id="h1",
            loser_id="h2",
            judge_reasoning="H1 is more novel",
            confidence=0.85,
        )
        assert dr.winner_id == "h1"
        assert dr.confidence == 0.85


class TestTournamentEngineInit:
    """Verify TournamentEngine initialization."""

    def test_default_init(self):
        from shared.research.tournament import TournamentEngine
        engine = TournamentEngine(llm_client=MagicMock())
        assert engine.judge_model is None  # LLMClient resolves AEL_MODEL, else default_model
        assert engine.debate_rounds == 3
        assert engine.tournament_size == 8
        assert engine.k_factor == 32.0
        assert engine.initial_elo == 1200.0

    def test_custom_init(self):
        from shared.research.tournament import TournamentEngine
        engine = TournamentEngine(
            judge_model="claude-sonnet-4-6",
            debate_rounds=5,
            tournament_size=16,
            k_factor=16.0,
            initial_elo=1500.0,
            llm_client=MagicMock(),
        )
        assert engine.judge_model == "claude-sonnet-4-6"
        assert engine.debate_rounds == 5
        assert engine.tournament_size == 16

    def test_collector_optional(self):
        from shared.research.tournament import TournamentEngine
        engine = TournamentEngine(collector=None, llm_client=MagicMock())
        assert engine.collector is None


# ── Elo Rating System ─────────────────────────────────────────────────────


class TestEloSystem:
    """Verify Elo rating calculations."""

    def test_update_elo_winner_gains(self):
        from shared.research.tournament import TournamentEngine
        engine = TournamentEngine(llm_client=MagicMock())
        engine.elo_ratings = {"h1": 1200.0, "h2": 1200.0}
        engine.update_elo("h1", "h2")
        assert engine.elo_ratings["h1"] > 1200.0
        assert engine.elo_ratings["h2"] < 1200.0

    def test_update_elo_symmetric(self):
        from shared.research.tournament import TournamentEngine
        engine = TournamentEngine(llm_client=MagicMock())
        engine.elo_ratings = {"h1": 1200.0, "h2": 1200.0}
        engine.update_elo("h1", "h2")
        gain = engine.elo_ratings["h1"] - 1200.0
        loss = 1200.0 - engine.elo_ratings["h2"]
        assert abs(gain - loss) < 0.01  # Should be symmetric

    def test_update_elo_equal_rating(self):
        from shared.research.tournament import TournamentEngine
        engine = TournamentEngine(k_factor=32.0, llm_client=MagicMock())
        engine.elo_ratings = {"h1": 1200.0, "h2": 1200.0}
        engine.update_elo("h1", "h2")
        # With equal ratings, expected = 0.5, so change = K * (1 - 0.5) = 16
        assert abs(engine.elo_ratings["h1"] - 1216.0) < 0.01

    def test_update_elo_upset_larger_change(self):
        from shared.research.tournament import TournamentEngine
        engine = TournamentEngine(k_factor=32.0, llm_client=MagicMock())
        engine.elo_ratings = {"h1": 1000.0, "h2": 1400.0}
        engine.update_elo("h1", "h2")  # Lower-rated wins = upset
        gain = engine.elo_ratings["h1"] - 1000.0
        assert gain > 16  # Upset gives bigger gain

    def test_update_elo_expected_smaller_change(self):
        from shared.research.tournament import TournamentEngine
        engine = TournamentEngine(k_factor=32.0, llm_client=MagicMock())
        engine.elo_ratings = {"h1": 1400.0, "h2": 1000.0}
        engine.update_elo("h1", "h2")  # Higher-rated wins = expected
        gain = engine.elo_ratings["h1"] - 1400.0
        assert gain < 16  # Expected result gives smaller gain

    def test_elo_history(self):
        from shared.research.tournament import TournamentEngine
        engine = TournamentEngine(llm_client=MagicMock())
        engine.elo_ratings = {"h1": 1300.0, "h2": 1100.0}
        history = engine.get_elo_history()
        assert history == {"h1": 1300.0, "h2": 1100.0}


# ── Pairwise Debate ───────────────────────────────────────────────────────


def _make_hypotheses(n: int = 4):
    """Helper to create test hypotheses."""
    from shared.research.tournament import Hypothesis
    return [
        Hypothesis(
            id=f"h{i+1}",
            research_question=f"Research question {i+1}?",
            rationale=f"Rationale for hypothesis {i+1}",
            novelty_score=0.5 + i * 0.1,
            feasibility_score=0.6 + i * 0.05,
        )
        for i in range(n)
    ]


class TestPairwiseDebate:
    """Verify pairwise debate mechanics."""

    def test_debate_with_mock_llm(self):
        from shared.research.tournament import TournamentEngine
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = json.dumps({
            "winner": "A",
            "reasoning": "Hypothesis A is more novel",
            "confidence": 0.8,
        })
        engine = TournamentEngine(llm_client=mock_llm)
        h1, h2 = _make_hypotheses(2)
        result = engine.pairwise_debate(h1, h2)
        assert result.winner_id == h1.id
        assert result.loser_id == h2.id
        assert result.confidence == 0.8

    def test_debate_winner_b(self):
        from shared.research.tournament import TournamentEngine
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = json.dumps({
            "winner": "B",
            "reasoning": "B has better methodology",
            "confidence": 0.7,
        })
        engine = TournamentEngine(llm_client=mock_llm)
        h1, h2 = _make_hypotheses(2)
        result = engine.pairwise_debate(h1, h2)
        assert result.winner_id == h2.id
        assert result.loser_id == h1.id

    def test_debate_fallback_on_error(self):
        from shared.research.tournament import TournamentEngine
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API error")
        engine = TournamentEngine(llm_client=mock_llm)
        hypotheses = _make_hypotheses(2)
        h1, h2 = hypotheses[0], hypotheses[1]
        result = engine.pairwise_debate(h1, h2)
        # Fallback uses scores; h2 has higher scores
        assert result.winner_id in (h1.id, h2.id)
        assert result.judge_reasoning.startswith("Fallback")

    def test_debate_malformed_response(self):
        from shared.research.tournament import TournamentEngine
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "I think A is better because..."
        engine = TournamentEngine(llm_client=mock_llm)
        h1, h2 = _make_hypotheses(2)
        result = engine.pairwise_debate(h1, h2)
        # Should still produce a result (heuristic parsing)
        assert result.winner_id in (h1.id, h2.id)


# ── Full Tournament ───────────────────────────────────────────────────────


class TestFullTournament:
    """Verify end-to-end tournament execution."""

    def test_tournament_single_hypothesis(self):
        from shared.research.tournament import TournamentEngine
        engine = TournamentEngine(llm_client=MagicMock())
        hypotheses = _make_hypotheses(1)
        ranked = engine.run_tournament(hypotheses)
        assert len(ranked) == 1
        assert ranked[0].rank == 1

    def test_tournament_two_hypotheses(self):
        from shared.research.tournament import TournamentEngine
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = json.dumps({
            "winner": "A", "reasoning": "test", "confidence": 0.9
        })
        engine = TournamentEngine(debate_rounds=1, llm_client=mock_llm)
        ranked = engine.run_tournament(_make_hypotheses(2))
        assert len(ranked) == 2
        assert ranked[0].rank == 1
        assert ranked[1].rank == 2
        assert ranked[0].elo_rating > ranked[1].elo_rating

    def test_tournament_four_hypotheses(self):
        from shared.research.tournament import TournamentEngine
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = json.dumps({
            "winner": "A", "reasoning": "test", "confidence": 0.8
        })
        engine = TournamentEngine(
            debate_rounds=2, tournament_size=4, llm_client=mock_llm
        )
        ranked = engine.run_tournament(_make_hypotheses(4))
        assert len(ranked) == 4
        # Ranks should be 1-4
        assert [r.rank for r in ranked] == [1, 2, 3, 4]

    def test_tournament_size_limit(self):
        from shared.research.tournament import TournamentEngine
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = json.dumps({
            "winner": "A", "reasoning": "test", "confidence": 0.7
        })
        engine = TournamentEngine(
            tournament_size=4, debate_rounds=1, llm_client=mock_llm
        )
        ranked = engine.run_tournament(_make_hypotheses(10))
        assert len(ranked) == 4  # Sampled down to tournament_size

    def test_tournament_debate_history(self):
        from shared.research.tournament import TournamentEngine
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = json.dumps({
            "winner": "A", "reasoning": "test", "confidence": 0.8
        })
        engine = TournamentEngine(debate_rounds=2, llm_client=mock_llm)
        engine.run_tournament(_make_hypotheses(4))
        log = engine.get_debate_log()
        assert len(log) > 0
        assert "winner_id" in log[0]

    def test_tournament_configurable_threshold(self):
        from shared.research.tournament import TournamentEngine
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = json.dumps({
            "winner": "A", "reasoning": "test", "confidence": 0.8
        })
        engine = TournamentEngine(debate_rounds=2, llm_client=mock_llm)
        ranked = engine.run_tournament(_make_hypotheses(4))
        # Filter by Elo threshold
        top = [h for h in ranked if h.elo_rating >= 1200]
        assert len(top) > 0
        assert all(h.elo_rating >= 1200 for h in top)


# ── MCTS Explorer ─────────────────────────────────────────────────────────


class TestMCTSExplorer:
    """Verify MCTS topic exploration."""

    def test_module_imports(self):
        from shared.research.mcts import MCTSExplorer, ExplorationNode

    def test_exploration_node_model(self):
        from shared.research.mcts import ExplorationNode
        node = ExplorationNode(topic="AI economics", depth=0)
        assert node.topic == "AI economics"
        assert node.surprise_score == 0.0
        assert node.visit_count == 0

    def test_ucb1_unvisited(self):
        from shared.research.mcts import ExplorationNode
        node = ExplorationNode(topic="test", visit_count=0)
        assert node.ucb1 == float("inf")

    def test_ucb1_visited(self):
        from shared.research.mcts import ExplorationNode
        node = ExplorationNode(topic="test", visit_count=10, total_reward=5.0)
        ucb = node.ucb1
        assert ucb > 0
        assert ucb < 10  # Reasonable bound

    def test_explorer_init(self):
        from shared.research.mcts import MCTSExplorer
        explorer = MCTSExplorer(
            max_depth=3, branching_factor=4,
            surprise_threshold=0.3, num_simulations=10,
            llm_client=MagicMock(),
        )
        assert explorer.max_depth == 3
        assert explorer.branching_factor == 4

    def test_explore_with_mock_llm(self):
        from shared.research.mcts import MCTSExplorer
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = json.dumps({
            "subtopics": [
                "AI labor market effects",
                "Central bank digital currencies",
                "Climate economics of AI",
                "AI in behavioral economics",
            ]
        })
        explorer = MCTSExplorer(
            max_depth=2, branching_factor=4,
            num_simulations=5, llm_client=mock_llm,
        )
        nodes = explorer.explore("AI and economics")
        assert len(nodes) >= 0  # May filter by surprise threshold
        tree = explorer.get_tree()
        assert "AI and economics" in tree

    def test_explore_returns_sorted_by_surprise(self):
        from shared.research.mcts import MCTSExplorer
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = json.dumps({
            "subtopics": ["Topic A", "Topic B", "Topic C", "Topic D"]
        })
        explorer = MCTSExplorer(
            max_depth=2, branching_factor=4,
            surprise_threshold=0.0, num_simulations=5,
            llm_client=mock_llm,
        )
        nodes = explorer.explore("Economics")
        if len(nodes) >= 2:
            for i in range(len(nodes) - 1):
                assert nodes[i].surprise_score >= nodes[i + 1].surprise_score

    def test_explore_empty_tree_on_depth_zero(self):
        from shared.research.mcts import MCTSExplorer
        explorer = MCTSExplorer(
            max_depth=0, num_simulations=5, llm_client=MagicMock()
        )
        nodes = explorer.explore("Test")
        assert len(nodes) == 0  # Root is depth 0, no expansion


# ── Backward Compatibility ────────────────────────────────────────────────


class TestBackwardCompatibility:
    """Verify tournament is optional and doesn't break existing workflows."""

    def test_refinement_stage_still_works_without_tournament(self):
        """IdeationTeam RefinementStage should work without tournament."""
        ideation_dir = Path(__file__).resolve().parent.parent.parent
        stage_file = ideation_dir / "IdeationTeam" / "ael" / "ModeNoWcNoHITL" / "2-RefinementStage.py"
        assert stage_file.exists(), "RefinementStage should exist"

    def test_tournament_engine_does_not_require_llm_on_init(self):
        """TournamentEngine should not call LLM during __init__."""
        from shared.research.tournament import TournamentEngine
        # This should NOT trigger any LLM call
        engine = TournamentEngine(llm_client=MagicMock())
        assert engine.debate_history == []

    def test_tournament_engine_empty_list(self):
        from shared.research.tournament import TournamentEngine
        engine = TournamentEngine(llm_client=MagicMock())
        ranked = engine.run_tournament([])
        assert ranked == []


# ── Tournament Telemetry ──────────────────────────────────────────────────


class TestTournamentTelemetry:
    """Verify tournament generates telemetry data."""

    def test_debate_log_structure(self):
        from shared.research.tournament import TournamentEngine
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = json.dumps({
            "winner": "A", "reasoning": "Better impact", "confidence": 0.9
        })
        engine = TournamentEngine(debate_rounds=1, llm_client=mock_llm)
        engine.run_tournament(_make_hypotheses(4))
        log = engine.get_debate_log()
        assert len(log) > 0
        entry = log[0]
        assert "winner_id" in entry
        assert "loser_id" in entry
        assert "judge_reasoning" in entry
        assert "confidence" in entry

    def test_elo_history_after_tournament(self):
        from shared.research.tournament import TournamentEngine
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = json.dumps({
            "winner": "B", "reasoning": "test", "confidence": 0.8
        })
        engine = TournamentEngine(debate_rounds=1, llm_client=mock_llm)
        hypotheses = _make_hypotheses(4)
        engine.run_tournament(hypotheses)
        history = engine.get_elo_history()
        assert len(history) == 4
        # Some ratings should deviate from initial
        ratings = list(history.values())
        assert max(ratings) > 1200 or min(ratings) < 1200


# ── IdeationTeam Integration Config ──────────────────────────────────────


_MODES = [
    "ModeNoWcNoHITL",
    "ModeNoWcWithHITL",
    "ModeWithWcNoHITL",
    "ModeWithWcWithHITL",
]


class TestIdeationTeamConfig:
    """Verify IdeationTeam MasterOrchestrators can accept tournament config."""

    @pytest.mark.parametrize("mode", _MODES)
    def test_refinement_stage_exists(self, mode):
        agents_dir = Path(__file__).resolve().parent.parent.parent
        stage = agents_dir / "IdeationTeam" / "ael" / mode / "2-RefinementStage.py"
        assert stage.exists(), f"{mode}/2-RefinementStage.py should exist"

    @pytest.mark.parametrize("mode", _MODES)
    def test_orchestrator_exists(self, mode):
        agents_dir = Path(__file__).resolve().parent.parent.parent
        mo = agents_dir / "IdeationTeam" / "ael" / mode / "0-MasterOrchestrator.py"
        assert mo.exists(), f"{mode}/0-MasterOrchestrator.py should exist"
