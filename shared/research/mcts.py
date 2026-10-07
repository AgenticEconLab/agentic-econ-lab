# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
MCTS Explorer — Monte Carlo Tree Search for research topic exploration (V0.6 Phase 1).

Inspired by Ai2 AutoDiscovery's MCTS + Bayesian surprise pattern for
discovering non-obvious research directions from seed topics.

Usage:
    from shared.research.mcts import MCTSExplorer, ExplorationNode

    explorer = MCTSExplorer(max_depth=3, branching_factor=4)
    nodes = explorer.explore("AI and monetary policy")
    for node in nodes:
        print(f"{node.topic} (surprise={node.surprise_score:.2f})")
"""

import math
import random
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ExplorationNode(BaseModel):
    """A node in the MCTS topic exploration tree."""

    topic: str
    depth: int = 0
    parent_topic: Optional[str] = None
    surprise_score: float = 0.0
    visit_count: int = 0
    total_reward: float = 0.0
    children: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def ucb1(self) -> float:
        """Upper Confidence Bound for Trees (UCT) score."""
        if self.visit_count == 0:
            return float("inf")
        exploitation = self.total_reward / self.visit_count
        exploration = math.sqrt(2 * math.log(max(self.visit_count, 1)) / self.visit_count)
        return exploitation + exploration


class MCTSExplorer:
    """Monte Carlo Tree Search for research topic exploration.

    Expands a topic tree by generating sub-topics via LLM and scoring
    them by Bayesian surprise (how unexpected/novel a direction is).

    Args:
        max_depth: Maximum tree depth.
        branching_factor: Number of children per node.
        surprise_threshold: Minimum surprise score to keep a node.
        num_simulations: Number of MCTS rollouts.
        collector: Optional MetricsCollector.
        llm_client: Optional pre-configured LLMClient (for testing).
    """

    def __init__(
        self,
        max_depth: int = 3,
        branching_factor: int = 4,
        surprise_threshold: float = 0.3,
        num_simulations: int = 20,
        collector: Optional[Any] = None,
        llm_client: Optional[Any] = None,
    ):
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.surprise_threshold = surprise_threshold
        self.num_simulations = num_simulations
        self.collector = collector
        self._llm_client = llm_client
        self.nodes: Dict[str, ExplorationNode] = {}

    @property
    def llm_client(self):
        """Lazy-init LLM client."""
        if self._llm_client is None:
            from shared.llm import LLMClient
            self._llm_client = LLMClient(
                collector=self.collector,
                agent_name="MCTSExplorer",
            )
        return self._llm_client

    def explore(self, seed_topic: str) -> List[ExplorationNode]:
        """Expand topic tree from seed; return nodes sorted by surprise.

        Args:
            seed_topic: Root topic to explore from.

        Returns:
            List of ExplorationNode sorted by surprise_score (descending).
        """
        root = ExplorationNode(topic=seed_topic, depth=0)
        self.nodes[seed_topic] = root

        for _ in range(self.num_simulations):
            # Selection: find most promising unexpanded node
            node = self._select(root)
            if node.depth < self.max_depth:
                # Expansion: generate children
                children = self._expand(node)
                for child in children:
                    # Simulation: estimate surprise
                    reward = self._simulate(child)
                    # Backpropagation
                    self._backpropagate(child, reward)

        # Return all nodes above surprise threshold, sorted
        result = [
            n for n in self.nodes.values()
            if n.surprise_score >= self.surprise_threshold and n.depth > 0
        ]
        result.sort(key=lambda n: n.surprise_score, reverse=True)
        return result

    def get_tree(self) -> Dict[str, ExplorationNode]:
        """Return the full exploration tree."""
        return dict(self.nodes)

    # ── Private Methods ───────────────────────────────────────────────────

    def _select(self, node: ExplorationNode) -> ExplorationNode:
        """Select the most promising node using UCB1."""
        if not node.children:
            return node
        best_child = None
        best_ucb = -float("inf")
        for child_topic in node.children:
            child = self.nodes.get(child_topic)
            if child is None:
                continue
            ucb = child.ucb1
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child
        if best_child is None:
            return node
        return self._select(best_child)

    def _expand(self, node: ExplorationNode) -> List[ExplorationNode]:
        """Generate child topics via LLM."""
        if node.children:
            return []  # Already expanded

        try:
            response = self.llm_client.invoke([
                {"role": "system", "content": (
                    "You are a research topic explorer. Given a topic, suggest "
                    f"{self.branching_factor} specific, surprising sub-directions "
                    "that are less obvious but potentially impactful for economics research. "
                    "Respond with JSON: {\"subtopics\": [\"topic1\", \"topic2\", ...]}"
                )},
                {"role": "user", "content": f"Explore sub-directions of: {node.topic}"},
            ])
            subtopics = self._parse_subtopics(response)
        except Exception:
            subtopics = [f"{node.topic} — aspect {i+1}" for i in range(self.branching_factor)]

        children = []
        for topic in subtopics[:self.branching_factor]:
            # Deduplicate: skip topics already in the tree
            if topic in self.nodes:
                continue
            child = ExplorationNode(
                topic=topic,
                depth=node.depth + 1,
                parent_topic=node.topic,
            )
            self.nodes[topic] = child
            node.children.append(topic)
            children.append(child)
        return children

    def _simulate(self, node: ExplorationNode) -> float:
        """Estimate surprise/novelty score for a node.

        Uses a simple heuristic based on topic uniqueness.
        In production, this would use Bayesian surprise against a corpus.
        """
        # Simple heuristic: reward depth and topic diversity
        depth_bonus = node.depth / self.max_depth
        # Check overlap with existing topics
        all_topics = " ".join(n.topic for n in self.nodes.values())
        words = set(node.topic.lower().split())
        overlap = sum(1 for w in words if all_topics.lower().count(w) > 1)
        uniqueness = 1.0 - (overlap / max(len(words), 1))
        surprise = 0.5 * depth_bonus + 0.5 * uniqueness
        node.surprise_score = max(node.surprise_score, surprise)
        return surprise

    def _backpropagate(self, node: ExplorationNode, reward: float) -> None:
        """Update visit counts and rewards up the tree."""
        current = node
        while current is not None:
            current.visit_count += 1
            current.total_reward += reward
            if current.parent_topic:
                current = self.nodes.get(current.parent_topic)
            else:
                break

    def _parse_subtopics(self, response: str) -> List[str]:
        """Parse LLM response into list of subtopics."""
        try:
            from shared.json_repair import safe_json_loads
            data = safe_json_loads(response)
            if isinstance(data, dict) and "subtopics" in data:
                return [str(s) for s in data["subtopics"]]
        except Exception:
            pass
        # Fallback: split by newlines
        lines = [l.strip().lstrip("- ").lstrip("0123456789.") for l in response.split("\n") if l.strip()]
        return lines[:self.branching_factor]
