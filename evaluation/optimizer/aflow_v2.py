# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
A²Flow (Self-Adaptive Flow) offline workflow optimizer (V0.7).

Reference: "A²Flow: Self-Adaptive Flow for Automated Workflow Optimization"
(arXiv 2511.20693, AAAI 2026). Supersedes AFlow's hand-coded operators by
extracting operators automatically in 3 stages: observation, abstraction,
selection.

This module is *offline-only*: it proposes candidate workflows and scores
them against a static fitness function supplied by the caller. Runtime
pipelines are never modified — developers use the proposals as a source of
improvement suggestions.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence


@dataclass
class WorkflowOperator:
    name: str
    abstract: str                              # what the operator does in plain English
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowCandidate:
    operators: List[WorkflowOperator]
    fitness: float = 0.0
    notes: str = ""

    @property
    def signature(self) -> tuple:
        return tuple(op.name for op in self.operators)


FitnessFn = Callable[[WorkflowCandidate], float]


class AFlowV2Optimizer:
    """Self-adaptive workflow optimizer.

    Parameters
    ----------
    operators
        Library of discovered operators (rule-based; real A²Flow uses LLM
        abstraction of agent traces).
    fitness_fn
        Scoring function. Signature ``(candidate) -> float``. Higher is better.
    max_depth
        Maximum number of operators per candidate workflow.
    budget
        Maximum candidates to explore.
    seed
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        operators: Sequence[WorkflowOperator],
        fitness_fn: FitnessFn,
        *,
        max_depth: int = 3,
        budget: int = 50,
        seed: int = 0,
    ) -> None:
        if not operators:
            raise ValueError("operators library cannot be empty")
        self.operators = list(operators)
        self.fitness_fn = fitness_fn
        self.max_depth = max_depth
        self.budget = budget
        self._rng = random.Random(seed)

    def optimize(self) -> List[WorkflowCandidate]:
        """Enumerate or sample candidates up to ``budget`` and rank by fitness."""
        explored: List[WorkflowCandidate] = []
        seen: set[tuple] = set()

        # Phase 1 — deterministic enumeration of short workflows
        for depth in range(1, self.max_depth + 1):
            for combo in itertools.product(self.operators, repeat=depth):
                if len(explored) >= self.budget:
                    break
                cand = WorkflowCandidate(operators=list(combo))
                if cand.signature in seen:
                    continue
                seen.add(cand.signature)
                cand.fitness = self.fitness_fn(cand)
                explored.append(cand)
            if len(explored) >= self.budget:
                break

        # Phase 2 — random mutation of the best candidate (if budget remains).
        # Cap attempts to avoid infinite loops when the search space is exhausted.
        if explored and len(explored) < self.budget:
            best = max(explored, key=lambda c: c.fitness)
            max_attempts = self.budget * 5
            attempts = 0
            while len(explored) < self.budget and attempts < max_attempts:
                attempts += 1
                mutated_ops = list(best.operators)
                if mutated_ops:
                    idx = self._rng.randrange(len(mutated_ops))
                    mutated_ops[idx] = self._rng.choice(self.operators)
                cand = WorkflowCandidate(operators=mutated_ops, notes="mutated")
                if cand.signature in seen:
                    continue
                seen.add(cand.signature)
                cand.fitness = self.fitness_fn(cand)
                explored.append(cand)

        explored.sort(key=lambda c: c.fitness, reverse=True)
        return explored

    def best(self) -> Optional[WorkflowCandidate]:
        ranked = self.optimize()
        return ranked[0] if ranked else None
