# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Trajectory Evaluator — Agent-as-Judge for decision sequence quality.

Extracts decision points from execution logs and evaluates the quality
of the decision trajectory, separate from output quality.

Computes:
- Decision coherence (do decisions follow logically from context?)
- Information utilization (does each stage use prior stage outputs?)
- Scope management (is the workflow scope maintained or appropriately adjusted?)
- Stage transition quality (are stage transitions justified?)

Usage:
    from evaluation.trajectory.trajectory_evaluator import TrajectoryEvaluator

    evaluator = TrajectoryEvaluator()
    score = evaluator.evaluate_trajectory(execution_log, outputs)
    print(f"Trajectory quality: {score.overall_score:.3f}")
"""

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field


class DecisionPoint(BaseModel):
    """A single decision point extracted from the execution trajectory."""

    stage_name: str = Field(description="Stage where the decision was made")
    stage_number: int = Field(default=0, description="Stage sequence number")
    decision_type: str = Field(
        default="stage_execution",
        description="Type: stage_execution, tool_selection, output_formatting",
    )
    inputs_available: List[str] = Field(
        default_factory=list, description="Inputs available at this point"
    )
    outputs_produced: List[str] = Field(
        default_factory=list, description="Outputs produced"
    )
    duration_sec: float = Field(default=0.0, description="Duration of this decision")
    success: bool = Field(default=True, description="Whether the decision succeeded")
    error_info: Optional[str] = Field(default=None, description="Error if any")
    item_count: int = Field(default=0, description="Items processed/produced")


class TrajectoryScore(BaseModel):
    """Score for a complete workflow trajectory."""

    team: str
    mode: str
    decision_points: List[DecisionPoint] = Field(default_factory=list)
    decision_coherence: float = Field(
        default=0.0,
        description="Score [0-1]: do decisions follow logically from context?",
    )
    information_utilization: float = Field(
        default=0.0,
        description="Score [0-1]: does each stage use prior stage outputs?",
    )
    scope_management: float = Field(
        default=0.0,
        description="Score [0-1]: is scope maintained throughout?",
    )
    stage_transition_quality: float = Field(
        default=0.0,
        description="Score [0-1]: are stage transitions well-justified?",
    )
    overall_score: float = Field(
        default=0.0,
        description="Weighted average of trajectory sub-scores [0-1]",
    )
    llm_evaluated: bool = Field(
        default=False,
        description="Whether LLM-as-Judge was used (vs structural-only)",
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "team": self.team,
            "mode": self.mode,
            "n_decision_points": len(self.decision_points),
            "decision_coherence": round(self.decision_coherence, 4),
            "information_utilization": round(self.information_utilization, 4),
            "scope_management": round(self.scope_management, 4),
            "stage_transition_quality": round(self.stage_transition_quality, 4),
            "overall_score": round(self.overall_score, 4),
            "llm_evaluated": self.llm_evaluated,
        }


class TrajectoryEvaluator:
    """
    Evaluates the quality of agent decision sequences.

    Works in two modes:
    1. Structural evaluation (no LLM): Computes trajectory quality from
       execution log metadata alone (stage ordering, durations, error handling).
    2. LLM-as-Judge evaluation: Additionally uses an LLM to assess decision
       coherence from actual output content.
    """

    def __init__(
        self,
        llm_invoke_fn: Optional[Callable[[str], str]] = None,
    ):
        """
        Args:
            llm_invoke_fn: Optional function for LLM-as-Judge evaluation.
                Signature: (prompt: str) -> str. If None, only structural
                evaluation is performed (no API cost).
        """
        self._invoke = llm_invoke_fn

    def evaluate_trajectory(
        self,
        execution_log: Dict[str, Any],
        outputs: Optional[Dict[str, Any]] = None,
    ) -> TrajectoryScore:
        """
        Evaluate a workflow trajectory.

        Args:
            execution_log: The execution_log.json dict.
            outputs: Optional workflow output files for content-aware evaluation.

        Returns:
            TrajectoryScore with per-dimension scores.
        """
        team = execution_log.get("team", "")
        mode = execution_log.get("mode", "")

        # Extract decision points from execution log
        decision_points = self._extract_decision_points(execution_log)

        # Structural evaluation (free, no LLM calls)
        coherence = self._structural_coherence(decision_points)
        utilization = self._structural_utilization(decision_points)
        scope = self._structural_scope(decision_points, execution_log)
        transitions = self._structural_transitions(decision_points)

        llm_evaluated = False

        # LLM-as-Judge evaluation (optional, costs API calls)
        if self._invoke is not None and outputs:
            llm_scores = self._llm_evaluate(
                execution_log, outputs, decision_points
            )
            if llm_scores:
                # Blend structural and LLM scores (0.3 structural + 0.7 LLM)
                coherence = 0.3 * coherence + 0.7 * llm_scores.get(
                    "decision_coherence", coherence
                )
                utilization = 0.3 * utilization + 0.7 * llm_scores.get(
                    "information_utilization", utilization
                )
                scope = 0.3 * scope + 0.7 * llm_scores.get(
                    "scope_management", scope
                )
                transitions = 0.3 * transitions + 0.7 * llm_scores.get(
                    "stage_transition_quality", transitions
                )
                llm_evaluated = True

        # Weighted overall score
        overall = (
            0.30 * coherence
            + 0.25 * utilization
            + 0.25 * scope
            + 0.20 * transitions
        )

        return TrajectoryScore(
            team=team,
            mode=mode,
            decision_points=decision_points,
            decision_coherence=coherence,
            information_utilization=utilization,
            scope_management=scope,
            stage_transition_quality=transitions,
            overall_score=overall,
            llm_evaluated=llm_evaluated,
        )

    def _extract_decision_points(
        self, execution_log: Dict[str, Any]
    ) -> List[DecisionPoint]:
        """Extract decision points from execution log stages."""
        stages = execution_log.get("stages", [])
        points = []

        for i, stage in enumerate(stages):
            outputs_produced = stage.get("output_files", []) or []
            duration = stage.get("duration_seconds", 0) or 0

            points.append(
                DecisionPoint(
                    stage_name=stage.get("name", f"Stage_{i + 1}"),
                    stage_number=stage.get("number", i + 1),
                    decision_type="stage_execution",
                    outputs_produced=outputs_produced,
                    duration_sec=duration,
                    success=stage.get("status") == "success",
                    error_info=stage.get("error"),
                    item_count=stage.get("item_count", 0) or 0,
                )
            )

        return points

    def _structural_coherence(self, points: List[DecisionPoint]) -> float:
        """
        Structural decision coherence: are stages in correct order
        and do they all succeed?
        """
        if not points:
            return 0.0

        # Stage ordering
        numbers = [p.stage_number for p in points]
        is_ordered = numbers == sorted(numbers)
        order_score = 1.0 if is_ordered else 0.5

        # Success rate
        successes = sum(1 for p in points if p.success)
        success_rate = successes / len(points)

        return 0.6 * order_score + 0.4 * success_rate

    def _structural_utilization(self, points: List[DecisionPoint]) -> float:
        """
        Information utilization: does each stage produce outputs
        that could inform the next stage?
        """
        if not points:
            return 0.0
        if len(points) < 2:
            return 1.0 if points[0].outputs_produced else 0.5

        # Check if each stage produces outputs (except possibly the last)
        producing = sum(1 for p in points if p.outputs_produced)
        return producing / len(points)

    def _structural_scope(
        self,
        points: List[DecisionPoint],
        execution_log: Dict[str, Any],
    ) -> float:
        """
        Scope management: is the pipeline complete (all expected stages run)?
        """
        if not points:
            return 0.0

        # Check item count progression (should not drop to zero unexpectedly)
        item_counts = [p.item_count for p in points]
        non_zero = sum(1 for c in item_counts if c > 0)
        coverage = non_zero / len(item_counts) if item_counts else 0

        # Check that we have the expected number of stages (typically 3)
        expected = execution_log.get("summary", {}).get("total_stages", len(points))
        actual = len(points)
        completeness = min(1.0, actual / expected) if expected > 0 else 1.0

        return 0.5 * coverage + 0.5 * completeness

    def _structural_transitions(self, points: List[DecisionPoint]) -> float:
        """
        Stage transition quality: smooth transitions between stages.
        Penalizes failures and excessive duration variance.
        """
        if len(points) < 2:
            return 1.0 if points and points[0].success else 0.0

        # All transitions successful
        transition_scores = []
        for i in range(len(points) - 1):
            current = points[i]
            next_pt = points[i + 1]
            # Good transition: current succeeds and produces output
            score = 0.0
            if current.success:
                score += 0.5
            if current.outputs_produced:
                score += 0.3
            if next_pt.success:
                score += 0.2
            transition_scores.append(score)

        return sum(transition_scores) / len(transition_scores) if transition_scores else 0.0

    def _llm_evaluate(
        self,
        execution_log: Dict[str, Any],
        outputs: Dict[str, Any],
        decision_points: List[DecisionPoint],
    ) -> Optional[Dict[str, float]]:
        """Use LLM to evaluate trajectory quality from output content."""
        if self._invoke is None:
            return None

        # Build prompt
        meta = execution_log.get("metadata", {})
        topic = meta.get("research_topic", meta.get("research_question", "Not specified"))

        stage_summary = ""
        for dp in decision_points:
            status = "success" if dp.success else "FAILED"
            outputs_str = ", ".join(dp.outputs_produced[:3]) if dp.outputs_produced else "none"
            stage_summary += (
                f"  {dp.stage_number}. {dp.stage_name} [{status}]: "
                f"{dp.item_count} items, outputs: {outputs_str}\n"
            )

        output_text = ""
        total_chars = 0
        max_chars = 15000
        for filename, content in outputs.items():
            if total_chars >= max_chars:
                break
            text = json.dumps(content, indent=1, default=str) if not isinstance(content, str) else content
            remaining = max_chars - total_chars
            if len(text) > remaining:
                text = text[:remaining] + "..."
            output_text += f"### {filename}\n{text}\n\n"
            total_chars += len(text)

        prompt = f"""You are evaluating the DECISION TRAJECTORY of an agentic research workflow.
Focus on HOW the workflow made decisions, not just the final output quality.

## Context
Research Topic: {topic}
Team: {execution_log.get("team", "")}
Mode: {execution_log.get("mode", "")}

## Execution Trajectory
{stage_summary}

## Workflow Outputs (for cross-stage analysis)
{output_text}

## Evaluation Criteria
Score each dimension on 1-5 scale:

1. **Decision Coherence**: Do sequential stage decisions follow logically from prior context?
2. **Information Utilization**: Does each stage effectively use outputs from prior stages?
3. **Scope Management**: Is the research scope maintained or appropriately adjusted?
4. **Stage Transition Quality**: Are transitions between stages smooth and justified?

## Required JSON Output
{{
  "decision_coherence": <1-5>,
  "information_utilization": <1-5>,
  "scope_management": <1-5>,
  "stage_transition_quality": <1-5>,
  "justification": "<2-3 sentences>"
}}"""

        try:
            response = self._invoke(prompt)
            data = json.loads(response)
            # Normalize 1-5 to 0-1
            return {
                k: (max(1, min(5, data.get(k, 3))) - 1) / 4.0
                for k in [
                    "decision_coherence",
                    "information_utilization",
                    "scope_management",
                    "stage_transition_quality",
                ]
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            return None


def compute_trajectory_output_correlation(
    trajectory_scores: List[float],
    output_quality_scores: List[float],
) -> float:
    """
    Compute Pearson correlation between trajectory and output quality scores.

    Args:
        trajectory_scores: List of trajectory overall_score values.
        output_quality_scores: List of corresponding output quality scores.

    Returns:
        Pearson r in [-1, 1], or 0.0 if computation fails.
    """
    n = len(trajectory_scores)
    if n < 2 or n != len(output_quality_scores):
        return 0.0

    mean_t = sum(trajectory_scores) / n
    mean_o = sum(output_quality_scores) / n

    cov = sum(
        (t - mean_t) * (o - mean_o)
        for t, o in zip(trajectory_scores, output_quality_scores)
    )
    var_t = sum((t - mean_t) ** 2 for t in trajectory_scores)
    var_o = sum((o - mean_o) ** 2 for o in output_quality_scores)

    denom = math.sqrt(var_t * var_o)
    if denom == 0:
        return 0.0

    return cov / denom
