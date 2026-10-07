# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Plan Adherence Scorer — Score adherence to expected stage plan.

Verifies that the workflow executed stages in the correct order,
completed all planned stages, and produced expected output types.

No additional LLM calls needed — computed purely from execution log structure.
"""

from typing import Any, Dict, List, Optional


# Expected stage plans per team
EXPECTED_STAGES: Dict[str, List[str]] = {
    "IdeationTeam": ["Sourcing", "Refinement", "Integration"],
    "LiteratureTeam": ["LiteratureGathering", "GapDetection", "Synthesis"],
    "DataTeam": ["DataSource", "DataCleaning", "QualityAssurance"],
    "ModelTeam": ["Theory", "ModelDesign", "Calibration"],
}

# Expected output files per team (at least one of these should exist)
EXPECTED_OUTPUTS: Dict[str, List[str]] = {
    "IdeationTeam": ["finalized_research_questions", "refinement_results"],
    "LiteratureTeam": ["literature_review", "gap_analysis", "bibliography"],
    "DataTeam": ["data_output", "quality_report"],
    "ModelTeam": ["theory_output", "model_design_output", "calibration_output"],
}


class PlanAdherenceScorer:
    """
    Score adherence to expected stage plan.

    Metrics:
    - Stage completion: Did all planned stages execute?
    - Stage ordering: Were stages executed in the correct order?
    - Output coverage: Were expected output types produced?
    """

    def __init__(
        self,
        expected_stages: Optional[Dict[str, List[str]]] = None,
        expected_outputs: Optional[Dict[str, List[str]]] = None,
    ):
        self.expected_stages = expected_stages or EXPECTED_STAGES
        self.expected_outputs = expected_outputs or EXPECTED_OUTPUTS

    def score(
        self,
        execution_log: Dict[str, Any],
        team: str = "",
    ) -> Dict[str, Any]:
        """
        Score plan adherence from an execution log.

        Args:
            execution_log: Parsed execution_log.json.
            team: Team name (e.g., "IdeationTeam").

        Returns:
            Dict with plan_adherence score (0-1) and details.
        """
        stages = execution_log.get("stages", [])
        expected = self.expected_stages.get(team, [])

        if not expected:
            return {
                "plan_adherence": 0.8,
                "details": f"No expected plan defined for {team}",
                "stages_completed": len(stages),
                "stages_expected": 0,
                "ordering_correct": True,
                "output_coverage": 1.0,
            }

        # 1. Stage completion
        executed_names = [s.get("name", "") for s in stages]
        completed_names = [s.get("name", "") for s in stages if s.get("status") == "success"]
        completion_rate = len(completed_names) / len(expected) if expected else 1.0
        completion_rate = min(1.0, completion_rate)

        # 2. Stage ordering
        ordering_correct = self._check_ordering(completed_names, expected)

        # 3. Output coverage
        output_coverage = self._check_outputs(stages, team)

        # Combined score: 50% completion + 25% ordering + 25% output
        ordering_score = 1.0 if ordering_correct else 0.5
        score = (
            0.50 * completion_rate
            + 0.25 * ordering_score
            + 0.25 * output_coverage
        )

        return {
            "plan_adherence": round(score, 4),
            "stages_completed": len(completed_names),
            "stages_expected": len(expected),
            "ordering_correct": ordering_correct,
            "output_coverage": round(output_coverage, 4),
            "completion_rate": round(completion_rate, 4),
        }

    def _check_ordering(self, executed: List[str], expected: List[str]) -> bool:
        """Check if executed stages appear in the expected order."""
        expected_idx = 0
        for name in executed:
            if expected_idx < len(expected) and name == expected[expected_idx]:
                expected_idx += 1
        return expected_idx == len(expected)

    def _check_outputs(self, stages: List[Dict], team: str) -> float:
        """Check what fraction of expected outputs were produced."""
        expected = self.expected_outputs.get(team, [])
        if not expected:
            return 1.0

        # Gather all output files from stages
        all_outputs = []
        for s in stages:
            all_outputs.extend(s.get("output_files", []))
        output_str = " ".join(all_outputs).lower()

        found = sum(1 for e in expected if e.lower() in output_str)
        return found / len(expected) if expected else 1.0
