# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Learned Model Router — ML-based model selection trained from evaluation data (V0.6 Phase 5).

Implements the RouteLLM pattern: train a lightweight classifier on AEL evaluation
results to predict which model will produce the best output for a given prompt+task.

Falls back to static routing (ModelRouter) when the learned model is unavailable.

Usage:
    from shared.llm_router_learned import LearnedRouter

    router = LearnedRouter()
    model = router.route("Analyze the impact of AI on GDP growth", task_type="reasoning")
"""

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

from shared.llm_router import MODEL_CATALOG, DEFAULT_ROUTES, ModelRouter


# Complexity signals that suggest routing to a stronger model
_COMPLEXITY_KEYWORDS = {
    "high": {"analyze", "evaluate", "compare", "synthesize", "critique", "debate",
             "formulate", "derive", "prove", "complex", "multi-step", "reasoning"},
    "medium": {"summarize", "explain", "describe", "outline", "list", "classify",
               "generate", "create", "design", "model"},
    "low": {"extract", "format", "convert", "parse", "translate", "lookup",
            "count", "filter", "sort", "retrieve"},
}

# Task type to complexity tier default mapping
_TASK_COMPLEXITY = {
    "extraction": "low",
    "formatting": "low",
    "batch": "low",
    "budget_extraction": "low",
    "summarization": "medium",
    "structured_output": "medium",
    "code_generation": "medium",
    "evaluation": "medium",
    "reasoning": "high",
    "reasoning_cot": "high",
    "reasoning_heavy": "high",
    "tool_heavy": "high",
    "calibration": "high",
}

# Model tiers by cost (ascending)
_MODEL_TIERS: Dict[str, List[str]] = {
    "budget": ["gpt-4.1-nano", "qwen3-8b", "mistral-small-3.2", "deepseek-v3.2"],
    "standard": ["gpt-4o-mini", "gpt-4.1-mini", "deepseek-chat", "gemini-2.5-flash"],
    "premium": ["gpt-4.1", "claude-sonnet-4-6", "gemini-2.5-pro", "gpt-5.4"],
    "flagship": ["claude-opus-4-6", "gemini-3.1-pro", "o4-mini"],
}


class LearnedRouter:
    """ML-based model routing trained from AEL evaluation data.

    Uses a lightweight heuristic classifier (keyword + task complexity)
    with optional fine-tuning from evaluation results.

    Args:
        eval_data_path: Path to AEL evaluation results JSON for fine-tuning.
        cost_threshold: Max input cost $/1M tokens for budget-conscious routing.
    """

    def __init__(
        self,
        eval_data_path: Optional[str] = None,
        cost_threshold: float = 5.0,
    ):
        self.cost_threshold = cost_threshold
        self._preference_scores: Dict[str, Dict[str, float]] = {}
        self._static_router = ModelRouter()

        if eval_data_path and os.path.exists(eval_data_path):
            self._load_eval_data(eval_data_path)

    def route(
        self,
        prompt: str,
        task_type: str = "default",
        budget_mode: bool = False,
    ) -> str:
        """Select optimal model based on prompt complexity and task type.

        Args:
            prompt: The prompt text to analyze.
            task_type: Task type hint.
            budget_mode: If True, prefer cheaper models.

        Returns:
            Selected model name.
        """
        complexity = self._estimate_complexity(prompt, task_type)

        # Select tier based on complexity
        if budget_mode or complexity == "low":
            tier = "budget"
        elif complexity == "medium":
            tier = "standard"
        else:
            tier = "premium"

        # If we have learned preferences, use them
        if task_type in self._preference_scores:
            return self._select_from_preferences(task_type, tier)

        # Fall back to tier-based selection
        candidates = _MODEL_TIERS.get(tier, _MODEL_TIERS["standard"])
        for model in candidates:
            if model in MODEL_CATALOG:
                provider = MODEL_CATALOG[model][0]
                if provider in self._static_router.available_providers:
                    return model

        # Ultimate fallback
        return self._static_router.select_model(task_type)

    def _estimate_complexity(self, prompt: str, task_type: str) -> str:
        """Estimate prompt complexity from keywords and task type.

        Args:
            prompt: Prompt text.
            task_type: Task type.

        Returns:
            Complexity level: "low", "medium", or "high".
        """
        # Task type default
        base = _TASK_COMPLEXITY.get(task_type, "medium")

        if not prompt:
            return base

        words = set(prompt.lower().split())

        # Count keyword hits per level
        scores = {}
        for level, keywords in _COMPLEXITY_KEYWORDS.items():
            scores[level] = len(words & keywords)

        # If prompt has clear complexity signals, use them
        max_score = max(scores.values())
        if max_score > 0:
            for level in ["high", "medium", "low"]:
                if scores[level] == max_score:
                    return level

        # Longer prompts tend to be more complex
        if len(prompt) > 500:
            if base == "low":
                return "medium"
            elif base == "medium":
                return "high"

        return base

    def _select_from_preferences(self, task_type: str, tier: str) -> str:
        """Select model from learned preference scores.

        Args:
            task_type: Task type.
            tier: Complexity tier.

        Returns:
            Model name.
        """
        prefs = self._preference_scores.get(task_type, {})
        tier_models = set(_MODEL_TIERS.get(tier, []))

        # Filter to available models in tier
        candidates = []
        for model, score in prefs.items():
            if model in tier_models:
                provider = MODEL_CATALOG.get(model, ("openai",))[0]
                if provider in self._static_router.available_providers:
                    candidates.append((score, model))

        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][1]

        return self._static_router.select_model(task_type)

    def _load_eval_data(self, path: str) -> None:
        """Load and process evaluation results for preference learning.

        Args:
            path: Path to evaluation results JSON.
        """
        try:
            with open(path) as f:
                data = json.load(f)

            # Extract model performance from eval results
            for entry in data if isinstance(data, list) else [data]:
                model = entry.get("model", "")
                task = entry.get("task_type", "default")
                score = entry.get("score", 0.5)

                if model and task:
                    if task not in self._preference_scores:
                        self._preference_scores[task] = {}
                    self._preference_scores[task][model] = score
        except Exception:
            pass  # Graceful degradation to static routing

    def get_complexity_estimate(self, prompt: str, task_type: str = "default") -> str:
        """Public access to complexity estimation (for debugging/logging).

        Args:
            prompt: Prompt text.
            task_type: Task type.

        Returns:
            Complexity level string.
        """
        return self._estimate_complexity(prompt, task_type)


# Per-team budget profiles based on actual execution characteristics
TEAM_BUDGET_PROFILES: Dict[str, Dict[str, Any]] = {
    "IdeationTeam": {
        "max_cost": 1.50,
        "max_tokens": 300_000,
        "stage_timeouts": [180, 240, 120],
        "preferred_tier": "standard",
    },
    "LiteratureTeam": {
        "max_cost": 3.00,
        "max_tokens": 600_000,
        "stage_timeouts": [300, 480, 360],
        "preferred_tier": "standard",
    },
    "DataTeam": {
        "max_cost": 2.00,
        "max_tokens": 400_000,
        "stage_timeouts": [300, 240, 300],
        "preferred_tier": "budget",
    },
    "ModelTeam": {
        "max_cost": 2.50,
        "max_tokens": 500_000,
        "stage_timeouts": [240, 300, 360],
        "preferred_tier": "premium",
    },
}


def get_team_budget(team: str) -> Dict[str, Any]:
    """Get budget profile for a team.

    Args:
        team: Team name (e.g., "IdeationTeam").

    Returns:
        Budget profile dict with max_cost, max_tokens, stage_timeouts.
    """
    return TEAM_BUDGET_PROFILES.get(team, {
        "max_cost": 2.00,
        "max_tokens": 400_000,
        "stage_timeouts": [300, 300, 300],
        "preferred_tier": "standard",
    })
