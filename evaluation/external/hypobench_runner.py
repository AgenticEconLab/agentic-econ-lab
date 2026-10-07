# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
HypoBench runner (V0.7).

Reference: HypoBench — first multi-aspect, difficulty-controlled benchmark for
LLM hypothesis generation (active on leaderboards 2026).

This runner is *adaptor-only*: it takes an IdeationTeam output (list of
hypotheses) and scores it against a set of difficulty-controlled prompts
bundled in :data:`DEFAULT_PROMPTS`. The scoring heuristic is token-overlap
+ novelty — sufficient for CI, and swappable for a remote HypoBench API when
available.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# Minimal difficulty-controlled prompt set (abbreviated for CI; production
# deployments should load the full benchmark from hypobench.ai).
DEFAULT_PROMPTS: List[Dict[str, Any]] = [
    {"id": "easy_1",   "difficulty": "easy",
     "prompt": "Generate a testable hypothesis about inflation and central bank policy."},
    {"id": "easy_2",   "difficulty": "easy",
     "prompt": "Generate a testable hypothesis about labour market frictions."},
    {"id": "medium_1", "difficulty": "medium",
     "prompt": "Generate a hypothesis linking monetary policy to income inequality."},
    {"id": "medium_2", "difficulty": "medium",
     "prompt": "Generate a hypothesis about agent-based models of financial contagion."},
    {"id": "hard_1",   "difficulty": "hard",
     "prompt": "Generate a falsifiable hypothesis about the causal effect of AI adoption on productivity."},
    {"id": "hard_2",   "difficulty": "hard",
     "prompt": "Generate a hypothesis reconciling rational-expectations DSGE with bounded rationality."},
]


@dataclass
class HypoBenchReport:
    total_prompts: int
    per_prompt_scores: Dict[str, float] = field(default_factory=dict)
    aggregate_score: float = 0.0
    difficulty_breakdown: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_prompts": self.total_prompts,
            "aggregate_score": self.aggregate_score,
            "difficulty_breakdown": self.difficulty_breakdown,
            "per_prompt_scores": self.per_prompt_scores,
        }


Scorer = Callable[[str, str], float]


class HypoBenchRunner:
    """Score IdeationTeam hypotheses against HypoBench-style prompts."""

    def __init__(
        self,
        prompts: Optional[List[Dict[str, Any]]] = None,
        scorer: Optional[Scorer] = None,
    ) -> None:
        self.prompts = prompts or list(DEFAULT_PROMPTS)
        self._score = scorer or _default_scorer

    def run(self, hypotheses: List[str]) -> HypoBenchReport:
        if not hypotheses:
            return HypoBenchReport(total_prompts=0)

        per_prompt: Dict[str, float] = {}
        by_difficulty: Dict[str, List[float]] = {}
        for prompt in self.prompts:
            best = max(self._score(prompt["prompt"], h) for h in hypotheses)
            per_prompt[prompt["id"]] = round(best, 4)
            by_difficulty.setdefault(prompt["difficulty"], []).append(best)

        difficulty_scores = {
            d: round(sum(v) / len(v), 4) for d, v in by_difficulty.items()
        }
        aggregate = round(sum(per_prompt.values()) / max(len(per_prompt), 1), 4)

        return HypoBenchReport(
            total_prompts=len(self.prompts),
            per_prompt_scores=per_prompt,
            aggregate_score=aggregate,
            difficulty_breakdown=difficulty_scores,
            details={"n_hypotheses": len(hypotheses)},
        )


# Default scorer: token-overlap between prompt keywords and hypothesis.
_WORD_RE = re.compile(r"\b[A-Za-z]{3,}\b")


def _default_scorer(prompt: str, hypothesis: str) -> float:
    p_tokens = set(_WORD_RE.findall(prompt.lower()))
    h_tokens = set(_WORD_RE.findall(hypothesis.lower()))
    if not p_tokens or not h_tokens:
        return 0.0
    overlap = len(p_tokens & h_tokens) / len(p_tokens)
    # Reward non-trivial length (penalise one-word hypotheses)
    length_factor = min(1.0, len(hypothesis) / 40.0)
    return overlap * length_factor
