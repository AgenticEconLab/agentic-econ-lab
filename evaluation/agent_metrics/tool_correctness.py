# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Tool Correctness Scorer — Score tool invocation quality from observability traces.

Analyzes execution logs to measure whether agents called the right tools
for their tasks (e.g., did LiteratureTeam agents use arxiv_search, not web_search?).

No additional LLM calls needed — computed purely from observability data.
"""

from typing import Any, Dict, List, Optional


# Expected tool usage patterns per team/stage
EXPECTED_TOOLS: Dict[str, Dict[str, List[str]]] = {
    "IdeationTeam": {
        "SourcingStage": ["arxiv_search", "web_search", "semantic_scholar_search"],
        "RefinementStage": [],  # LLM-only stage
        "IntegrationStage": [],  # LLM-only stage
    },
    "LiteratureTeam": {
        "LiteratureGatheringStage": ["arxiv_search", "semantic_scholar_search", "web_search"],
        "GapDetectionStage": [],
        "SynthesisStage": [],
    },
    "DataTeam": {
        "DataSourceStage": ["fred_fetch", "web_fetch", "yfinance_fetch"],
        "DataCleaningStage": [],
        "QualityAssuranceStage": [],
    },
    "ModelTeam": {
        "TheoryStage": ["arxiv_search", "web_search"],
        "ModelDesignStage": [],
        "CalibrationStage": [],
    },
}

# Tools that are never appropriate (e.g., data agents shouldn't search literature)
FORBIDDEN_TOOLS: Dict[str, List[str]] = {
    "DataTeam": ["arxiv_search"],
}


class ToolCorrectnessScorer:
    """
    Score tool invocation quality from observability traces.

    Metrics:
    - Tool selection accuracy: Were the right tools called for the task?
    - Forbidden tool avoidance: Were any inappropriate tools invoked?
    - Tool success rate: What fraction of tool calls succeeded?
    """

    def __init__(
        self,
        expected_tools: Optional[Dict[str, Dict[str, List[str]]]] = None,
        forbidden_tools: Optional[Dict[str, List[str]]] = None,
    ):
        self.expected_tools = expected_tools or EXPECTED_TOOLS
        self.forbidden_tools = forbidden_tools or FORBIDDEN_TOOLS

    def score(
        self,
        execution_log: Dict[str, Any],
        team: str = "",
    ) -> Dict[str, Any]:
        """
        Score tool correctness from an execution log.

        Args:
            execution_log: Parsed execution_log.json with observability data.
            team: Team name (e.g., "IdeationTeam").

        Returns:
            Dict with tool_correctness score (0-1) and details.
        """
        obs = execution_log.get("observability", {})
        tool_calls = obs.get("tool_calls", [])

        if not tool_calls:
            # No tool calls — check if tools were expected
            expected = self._get_all_expected(team)
            if expected:
                return {
                    "tool_correctness": 0.5,  # Neutral — tools expected but none called
                    "details": "No tool calls recorded but tools were expected",
                    "tool_calls_total": 0,
                    "correct_tools": 0,
                    "forbidden_tools_used": 0,
                    "tool_success_rate": 1.0,
                }
            return {
                "tool_correctness": 1.0,  # No tools expected, none called
                "details": "No tools expected or called",
                "tool_calls_total": 0,
                "correct_tools": 0,
                "forbidden_tools_used": 0,
                "tool_success_rate": 1.0,
            }

        # Count correct, forbidden, and successful tool calls
        total = len(tool_calls)
        correct = 0
        forbidden = 0
        successful = 0
        expected_set = set(self._get_all_expected(team))
        forbidden_set = set(self.forbidden_tools.get(team, []))

        for call in tool_calls:
            tool_name = call.get("tool", call.get("name", ""))
            status = call.get("status", "success")

            if tool_name in expected_set:
                correct += 1
            if tool_name in forbidden_set:
                forbidden += 1
            if status == "success":
                successful += 1

        # Score components
        selection_score = correct / total if total > 0 else 1.0
        forbidden_penalty = forbidden / total if total > 0 else 0.0
        success_rate = successful / total if total > 0 else 1.0

        # Combined score: 50% selection + 25% no-forbidden + 25% success
        score = (
            0.50 * selection_score
            + 0.25 * (1.0 - forbidden_penalty)
            + 0.25 * success_rate
        )

        return {
            "tool_correctness": round(score, 4),
            "tool_calls_total": total,
            "correct_tools": correct,
            "forbidden_tools_used": forbidden,
            "tool_success_rate": round(success_rate, 4),
            "selection_score": round(selection_score, 4),
        }

    def _get_all_expected(self, team: str) -> List[str]:
        """Get all expected tools for a team across all stages."""
        team_tools = self.expected_tools.get(team, {})
        all_tools = []
        for stage_tools in team_tools.values():
            all_tools.extend(stage_tools)
        return list(set(all_tools))
