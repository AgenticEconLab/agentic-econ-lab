# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Stratified reliability reporter (V0.7).

Reference: HAL paper (arXiv 2602.16666) — reliability is task-type-dependent;
aggregate scores mislead. V0.6 shipped a single ``ReliabilityResult`` per run
bundle. V0.7 adds per-task-type breakdowns.

Typical task types in AEL: ``ideation``, ``literature_review``,
``data_collection``, ``modeling``, ``simulation``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from evaluation.reliability.evaluator import (
    ReliabilityEvaluator,
    ReliabilityResult,
    RunResult,
)


@dataclass
class StratifiedReliabilityReport:
    task_type: str
    consistency: float
    robustness: float
    predictability: float
    safety: float
    overall: float
    n_runs: int
    details: Dict[str, str] = field(default_factory=dict)


class StratifiedReliabilityEvaluator:
    """Runs the Princeton 4-dim evaluator per task type and emits a report list.

    Input format: a dict mapping task_type -> list[RunResult]. Task types with
    zero runs are skipped; task types with ≥1 run are scored using the V0.6
    :class:`ReliabilityEvaluator`.
    """

    def __init__(
        self,
        base_evaluator: Optional[ReliabilityEvaluator] = None,
    ) -> None:
        self._base = base_evaluator or ReliabilityEvaluator()

    def evaluate(
        self,
        runs_by_task: Dict[str, List[RunResult]],
        paraphrased_by_task: Optional[Dict[str, List[RunResult]]] = None,
    ) -> List[StratifiedReliabilityReport]:
        reports: List[StratifiedReliabilityReport] = []
        paraphrased_by_task = paraphrased_by_task or {}
        for task_type, runs in runs_by_task.items():
            if not runs:
                continue
            para = paraphrased_by_task.get(task_type) or []
            result: ReliabilityResult = self._base.evaluate_all(runs, para)
            reports.append(StratifiedReliabilityReport(
                task_type=task_type,
                consistency=result.consistency,
                robustness=result.robustness,
                predictability=result.predictability,
                safety=result.safety,
                overall=result.overall,
                n_runs=len(runs),
            ))
        return reports

    def to_dict(
        self,
        reports: List[StratifiedReliabilityReport],
    ) -> Dict[str, Dict[str, float]]:
        """Return a serialisable summary dict keyed by task_type."""
        out: Dict[str, Dict[str, float]] = {}
        for r in reports:
            out[r.task_type] = {
                "consistency": r.consistency,
                "robustness": r.robustness,
                "predictability": r.predictability,
                "safety": r.safety,
                "overall": r.overall,
                "n_runs": r.n_runs,
            }
        return out

    def distinct_task_types(
        self,
        reports: List[StratifiedReliabilityReport],
    ) -> int:
        return len(set(r.task_type for r in reports))
