# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
SimulationFidelity evaluation dimension (V0.7).

Compares an ABIDES-Economist / LLM-Economist / MALLES simulation result
against closed-form or published benchmarks for that scenario. Returns a
score in [0, 1] where 1 = perfect match.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class SimulationFidelityScore:
    score: float
    scenario: str
    expected: Dict[str, float]
    observed: Dict[str, float]
    deviations: Dict[str, float] = field(default_factory=dict)
    notes: str = ""


# Closed-form or published benchmark anchors.
BENCHMARKS: Dict[str, Dict[str, float]] = {
    "stackelberg_tax":       {"tau_saez_optimal": 2.0 / 3.0},
    "dsge_monetary":         {"mean_inflation": 0.0, "mean_output_gap": 0.0},
    "dsge_fiscal":           {"fiscal_multiplier": 0.5},  # rough macro consensus
    "consumer_choice":       {"hit_rate": 0.775},  # MALLES target
}


class SimulationFidelityEvaluator:
    """Score a SimulationResult against :data:`BENCHMARKS`."""

    def __init__(self, *, tolerance: float = 0.05) -> None:
        self.tolerance = tolerance

    def evaluate(self, simulation_result: Dict[str, Any]) -> SimulationFidelityScore:
        scenario = simulation_result.get("scenario", "")
        metrics = simulation_result.get("metrics", {})
        expected = BENCHMARKS.get(scenario, {})
        deviations: Dict[str, float] = {}
        hits: int = 0

        for key, target in expected.items():
            obs = metrics.get(key)
            if obs is None:
                deviations[key] = float("inf")
                continue
            abs_dev = abs(float(obs) - float(target))
            # Normalise by |target| or 1 if target is near zero.
            scale = max(abs(float(target)), 1.0)
            deviations[key] = abs_dev / scale
            if deviations[key] <= self.tolerance:
                hits += 1

        if not expected:
            score = 0.0
            notes = f"No benchmark for scenario {scenario!r}"
        else:
            score = hits / len(expected)
            notes = f"{hits}/{len(expected)} metrics within tolerance={self.tolerance}"

        return SimulationFidelityScore(
            score=round(score, 4),
            scenario=scenario,
            expected=expected,
            observed={k: metrics.get(k, float("nan")) for k in expected},
            deviations=deviations,
            notes=notes,
        )
