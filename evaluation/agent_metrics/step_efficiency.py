# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Step Efficiency Scorer — Score execution path efficiency from observability traces.

Measures whether the workflow took an efficient path through its stages,
looking at LLM call counts, token usage, and stage duration patterns.

No additional LLM calls needed — computed purely from observability data.
"""

from typing import Any, Dict, List, Optional


# Expected LLM call ranges per team (min, max for healthy execution)
EXPECTED_LLM_CALLS: Dict[str, Dict[str, tuple]] = {
    "IdeationTeam": {
        "SourcingStage": (2, 15),
        "RefinementStage": (2, 10),
        "IntegrationStage": (1, 8),
    },
    "LiteratureTeam": {
        "LiteratureGatheringStage": (3, 20),
        "GapDetectionStage": (1, 8),
        "SynthesisStage": (1, 8),
    },
    "DataTeam": {
        "DataSourceStage": (1, 10),
        "DataCleaningStage": (1, 8),
        "QualityAssuranceStage": (1, 6),
    },
    "ModelTeam": {
        "TheoryStage": (2, 10),
        "ModelDesignStage": (2, 10),
        "CalibrationStage": (1, 8),
    },
}


class StepEfficiencyScorer:
    """
    Score execution path efficiency from observability traces.

    Metrics:
    - LLM call efficiency: Were calls within expected ranges?
    - Token efficiency: Input/output token ratio (high output = productive)
    - Stage balance: Were stages roughly proportional in effort?
    """

    def __init__(
        self,
        expected_calls: Optional[Dict[str, Dict[str, tuple]]] = None,
    ):
        self.expected_calls = expected_calls or EXPECTED_LLM_CALLS

    def score(
        self,
        execution_log: Dict[str, Any],
        team: str = "",
    ) -> Dict[str, Any]:
        """
        Score step efficiency from an execution log.

        Args:
            execution_log: Parsed execution_log.json with observability data.
            team: Team name (e.g., "IdeationTeam").

        Returns:
            Dict with step_efficiency score (0-1) and details.
        """
        obs = execution_log.get("observability", {})
        llm_calls = obs.get("llm_calls", [])
        stages = execution_log.get("stages", [])

        total_llm_calls = len(llm_calls)

        if total_llm_calls == 0:
            return {
                "step_efficiency": 0.5,
                "details": "No LLM calls recorded",
                "llm_calls_total": 0,
                "call_efficiency": 0.5,
                "token_efficiency": 0.5,
                "stage_balance": 1.0,
            }

        # 1. Call count efficiency
        call_efficiency = self._score_call_count(total_llm_calls, team)

        # 2. Token efficiency (output tokens / total tokens)
        total_input = sum(c.get("input_tokens", 0) for c in llm_calls)
        total_output = sum(c.get("output_tokens", 0) for c in llm_calls)
        total_tokens = total_input + total_output
        token_ratio = total_output / total_tokens if total_tokens > 0 else 0.5
        # Ideal ratio: 15-40% output tokens (too low = wasteful prompts, too high = suspiciously little input)
        if 0.10 <= token_ratio <= 0.50:
            token_efficiency = 1.0
        else:
            token_efficiency = max(0, 1.0 - abs(token_ratio - 0.30) * 2)

        # 3. Stage balance (coefficient of variation of stage durations)
        stage_durations = [s.get("duration_seconds", 0) for s in stages if s.get("duration_seconds", 0) > 0]
        if len(stage_durations) >= 2:
            mean_d = sum(stage_durations) / len(stage_durations)
            variance = sum((d - mean_d) ** 2 for d in stage_durations) / len(stage_durations)
            std_d = variance ** 0.5
            cv = std_d / mean_d if mean_d > 0 else 0
            # CV < 1.0 is balanced; > 2.0 is very unbalanced
            stage_balance = max(0, 1.0 - cv / 2.0)
        else:
            stage_balance = 1.0

        # Combined score: 40% call efficiency + 30% token efficiency + 30% stage balance
        score = (
            0.40 * call_efficiency
            + 0.30 * token_efficiency
            + 0.30 * stage_balance
        )

        return {
            "step_efficiency": round(score, 4),
            "llm_calls_total": total_llm_calls,
            "call_efficiency": round(call_efficiency, 4),
            "token_efficiency": round(token_efficiency, 4),
            "stage_balance": round(stage_balance, 4),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
        }

    def _score_call_count(self, total_calls: int, team: str) -> float:
        """Score based on whether call count is within expected range."""
        team_expected = self.expected_calls.get(team, {})
        if not team_expected:
            return 0.8  # No expectations, assume reasonable

        # Sum expected ranges across stages
        min_total = sum(r[0] for r in team_expected.values())
        max_total = sum(r[1] for r in team_expected.values())

        if min_total <= total_calls <= max_total:
            return 1.0
        elif total_calls < min_total:
            return max(0.3, total_calls / min_total)
        else:
            return max(0.3, max_total / total_calls)
