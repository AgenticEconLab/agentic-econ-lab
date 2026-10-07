# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Tournament Engine — Elo-rated hypothesis ranking via LLM debate (V0.6 Phase 1).

Implements the tournament-based hypothesis ranking pattern used by
Google AI Co-Scientist, Ai2 AutoDiscovery, and Sakana AI-Scientist-v2.

Usage:
    from shared.research.tournament import TournamentEngine, Hypothesis

    hypotheses = [
        Hypothesis(id="h1", research_question="How does AI affect GDP?",
                   rationale="...", novelty_score=0.8, feasibility_score=0.7),
        ...
    ]

    engine = TournamentEngine(
        debate_rounds=3,
        tournament_size=8,
    )
    ranked = engine.run_tournament(hypotheses)
    top = [h for h in ranked if h.elo_rating >= 1200]
"""

import itertools
import math
import random
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Data Models ───────────────────────────────────────────────────────────


class Hypothesis(BaseModel):
    """A research hypothesis/question to be ranked."""

    id: str
    research_question: str
    rationale: str
    novelty_score: float = 0.0
    feasibility_score: float = 0.0
    source_stage: str = "RefinementStage"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DebateResult(BaseModel):
    """Result of a pairwise debate between two hypotheses."""

    winner_id: str
    loser_id: str
    advocate_arguments: Dict[str, str] = Field(default_factory=dict)
    judge_reasoning: str = ""
    confidence: float = 0.5
    round_number: int = 0


class RankedHypothesis(Hypothesis):
    """A hypothesis with tournament ranking information."""

    elo_rating: float = 1200.0
    debate_wins: int = 0
    debate_losses: int = 0
    judge_comments: List[str] = Field(default_factory=list)
    rank: int = 0


# ── Tournament Engine ─────────────────────────────────────────────────────


# Default K-factor for Elo updates
DEFAULT_K_FACTOR = 32.0
DEFAULT_INITIAL_ELO = 1200.0


class TournamentEngine:
    """Elo-rated tournament for hypothesis ranking via LLM debate.

    Args:
        judge_model: Model name for the debate judge (default: AEL_MODEL, else default_model).
        debate_rounds: Number of debate rounds per tournament.
        tournament_size: Max hypotheses to include (random sample if more).
        k_factor: Elo K-factor (higher = more volatile ratings).
        initial_elo: Starting Elo rating for all hypotheses.
        collector: Optional MetricsCollector for observability.
        llm_client: Optional pre-configured LLMClient (for testing).
    """

    def __init__(
        self,
        judge_model: Optional[str] = None,
        debate_rounds: int = 3,
        tournament_size: int = 8,
        k_factor: float = DEFAULT_K_FACTOR,
        initial_elo: float = DEFAULT_INITIAL_ELO,
        collector: Optional[Any] = None,
        llm_client: Optional[Any] = None,
    ):
        self.judge_model = judge_model
        self.debate_rounds = debate_rounds
        self.tournament_size = tournament_size
        self.k_factor = k_factor
        self.initial_elo = initial_elo
        self.collector = collector
        self._llm_client = llm_client
        self.elo_ratings: Dict[str, float] = {}
        self.debate_history: List[DebateResult] = []

    @property
    def llm_client(self):
        """Lazy-init LLM client."""
        if self._llm_client is None:
            from shared.llm import LLMClient
            self._llm_client = LLMClient(
                model=self.judge_model,
                collector=self.collector,
                agent_name="TournamentJudge",
            )
        return self._llm_client

    def run_tournament(self, hypotheses: List[Hypothesis]) -> List[RankedHypothesis]:
        """Run Elo-rated tournament: pairwise debates, judge scoring, ranking.

        Args:
            hypotheses: List of hypotheses to rank.

        Returns:
            Hypotheses sorted by Elo rating (descending).
        """
        if len(hypotheses) <= 1:
            return [
                RankedHypothesis(**h.model_dump(), elo_rating=self.initial_elo, rank=1)
                for h in hypotheses
            ]

        # Sample if more than tournament_size
        participants = list(hypotheses)
        if len(participants) > self.tournament_size:
            participants = random.sample(participants, self.tournament_size)

        # Initialize Elo ratings
        self.elo_ratings = {h.id: self.initial_elo for h in participants}
        self.debate_history = []

        # Generate pairings for each round
        for round_num in range(1, self.debate_rounds + 1):
            pairs = self._generate_pairings(participants)
            for h1, h2 in pairs:
                result = self.pairwise_debate(h1, h2, round_number=round_num)
                self.debate_history.append(result)
                self.update_elo(result.winner_id, result.loser_id)

        # Build ranked results
        ranked = self._build_ranked(participants)
        return ranked

    def pairwise_debate(
        self, h1: Hypothesis, h2: Hypothesis, round_number: int = 0
    ) -> DebateResult:
        """Two LLM advocates argue for/against each hypothesis; judge decides.

        Args:
            h1: First hypothesis.
            h2: Second hypothesis.
            round_number: Current round (for logging).

        Returns:
            DebateResult with winner, loser, and judge reasoning.
        """
        # Build debate prompt
        system_prompt = (
            "You are an impartial research judge evaluating two competing research "
            "hypotheses. Compare them on: (1) novelty and originality, (2) feasibility "
            "and testability, (3) potential impact, (4) methodological clarity. "
            "You MUST pick a winner — no ties allowed. "
            "Respond with JSON: {\"winner\": \"A\" or \"B\", "
            "\"reasoning\": \"...\", \"confidence\": 0.5-1.0}"
        )

        user_prompt = (
            f"**Hypothesis A** (ID: {h1.id}):\n"
            f"Question: {h1.research_question}\n"
            f"Rationale: {h1.rationale}\n"
            f"Novelty: {h1.novelty_score:.2f}, Feasibility: {h1.feasibility_score:.2f}\n\n"
            f"**Hypothesis B** (ID: {h2.id}):\n"
            f"Question: {h2.research_question}\n"
            f"Rationale: {h2.rationale}\n"
            f"Novelty: {h2.novelty_score:.2f}, Feasibility: {h2.feasibility_score:.2f}\n\n"
            f"Which hypothesis is stronger for advancing economics research? "
            f"Pick winner A or B."
        )

        try:
            response = self.llm_client.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])
            result = self._parse_judge_response(response, h1, h2)
        except Exception:
            # Fallback: use scores
            result = self._fallback_judge(h1, h2)

        result.round_number = round_number
        return result

    def update_elo(self, winner_id: str, loser_id: str) -> None:
        """Standard Elo update with K-factor adjustment.

        Args:
            winner_id: ID of the winning hypothesis.
            loser_id: ID of the losing hypothesis.
        """
        r_winner = self.elo_ratings.get(winner_id, self.initial_elo)
        r_loser = self.elo_ratings.get(loser_id, self.initial_elo)

        # Expected scores
        e_winner = 1.0 / (1.0 + math.pow(10, (r_loser - r_winner) / 400.0))
        e_loser = 1.0 - e_winner

        # Update ratings
        self.elo_ratings[winner_id] = r_winner + self.k_factor * (1.0 - e_winner)
        self.elo_ratings[loser_id] = r_loser + self.k_factor * (0.0 - e_loser)

    def get_elo_history(self) -> Dict[str, float]:
        """Return current Elo ratings for all participants."""
        return dict(self.elo_ratings)

    def get_debate_log(self) -> List[Dict[str, Any]]:
        """Return debate history as list of dicts for telemetry."""
        return [d.model_dump() for d in self.debate_history]

    # ── Private Methods ───────────────────────────────────────────────────

    def _generate_pairings(
        self, participants: List[Hypothesis]
    ) -> List[tuple]:
        """Generate pairings for a round (round-robin subset)."""
        ids = [h.id for h in participants]
        id_to_h = {h.id: h for h in participants}

        # Sort by current Elo for Swiss-style pairing
        sorted_ids = sorted(ids, key=lambda x: self.elo_ratings.get(x, self.initial_elo), reverse=True)

        pairs = []
        used = set()
        for i in range(0, len(sorted_ids) - 1, 2):
            a, b = sorted_ids[i], sorted_ids[i + 1]
            if a not in used and b not in used:
                pairs.append((id_to_h[a], id_to_h[b]))
                used.add(a)
                used.add(b)

        return pairs

    def _parse_judge_response(
        self, response: str, h1: Hypothesis, h2: Hypothesis
    ) -> DebateResult:
        """Parse LLM judge response to determine winner."""
        import json as _json
        try:
            from shared.json_repair import safe_json_loads
            data = safe_json_loads(response)
        except Exception:
            data = None

        if data and isinstance(data, dict):
            winner_label = str(data.get("winner", "A")).upper().strip()
            reasoning = data.get("reasoning", "")
            confidence = float(data.get("confidence", 0.5))
        else:
            # Heuristic: look for "A" or "B" in response
            winner_label = "A" if "A" in response.upper()[:50] else "B"
            reasoning = response[:200]
            confidence = 0.5

        winner_id = h1.id if winner_label == "A" else h2.id
        loser_id = h2.id if winner_label == "A" else h1.id

        return DebateResult(
            winner_id=winner_id,
            loser_id=loser_id,
            advocate_arguments={h1.id: h1.rationale, h2.id: h2.rationale},
            judge_reasoning=reasoning,
            confidence=min(max(confidence, 0.0), 1.0),
        )

    def _fallback_judge(self, h1: Hypothesis, h2: Hypothesis) -> DebateResult:
        """Fallback judge using novelty + feasibility scores."""
        s1 = h1.novelty_score + h1.feasibility_score
        s2 = h2.novelty_score + h2.feasibility_score
        if s1 >= s2:
            winner, loser = h1, h2
        else:
            winner, loser = h2, h1
        return DebateResult(
            winner_id=winner.id,
            loser_id=loser.id,
            judge_reasoning="Fallback: scored by novelty + feasibility",
            confidence=0.5,
        )

    def _build_ranked(self, participants: List[Hypothesis]) -> List[RankedHypothesis]:
        """Build ranked hypothesis list from Elo ratings and debate history."""
        # Count wins/losses
        wins: Dict[str, int] = {}
        losses: Dict[str, int] = {}
        comments: Dict[str, List[str]] = {}
        for d in self.debate_history:
            wins[d.winner_id] = wins.get(d.winner_id, 0) + 1
            losses[d.loser_id] = losses.get(d.loser_id, 0) + 1
            if d.judge_reasoning:
                comments.setdefault(d.winner_id, []).append(d.judge_reasoning)

        ranked = []
        for h in participants:
            rh = RankedHypothesis(
                **h.model_dump(),
                elo_rating=self.elo_ratings.get(h.id, self.initial_elo),
                debate_wins=wins.get(h.id, 0),
                debate_losses=losses.get(h.id, 0),
                judge_comments=comments.get(h.id, []),
            )
            ranked.append(rh)

        # Sort by Elo descending
        ranked.sort(key=lambda r: r.elo_rating, reverse=True)

        # Assign ranks
        for i, rh in enumerate(ranked, 1):
            rh.rank = i

        return ranked
