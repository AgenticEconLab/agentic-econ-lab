# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
HLE-Rolling runner (V0.7).

Reference: Humanity's Last Exam — HLE-Rolling dataset (Oct 2025). Current
leaders as of 2026-04-13: GPT-5.4 41.6%, GPT-5.3 Codex 39.9%, Gemini 3 Pro 37.2%.

This runner accepts a callable ``solver(question) -> answer`` and scores
answers against a bundled 20-problem subset. Production deployments should
replace :data:`SUBSET_PROBLEMS` with the official benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


SUBSET_PROBLEMS: List[Dict[str, Any]] = [
    {"id": f"hle_{i:02d}",
     "question": f"Sample HLE problem #{i}",
     "answer": f"answer_{i}",
     "domain": (
         "economics" if i % 3 == 0
         else "mathematics" if i % 3 == 1
         else "sciences"
     )}
    for i in range(1, 21)
]


Solver = Callable[[str], str]


@dataclass
class HLEReport:
    total: int
    correct: int
    accuracy: float
    per_domain: Dict[str, Dict[str, float]] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 4),
            "per_domain": self.per_domain,
        }


class HLERunner:
    def __init__(
        self,
        problems: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.problems = problems or list(SUBSET_PROBLEMS)

    def run(self, solver: Solver) -> HLEReport:
        correct = 0
        domain_totals: Dict[str, int] = {}
        domain_correct: Dict[str, int] = {}
        for problem in self.problems:
            answer = solver(problem["question"])
            ok = (answer or "").strip() == problem["answer"]
            domain = problem.get("domain", "misc")
            domain_totals[domain] = domain_totals.get(domain, 0) + 1
            if ok:
                correct += 1
                domain_correct[domain] = domain_correct.get(domain, 0) + 1
        per_domain = {
            d: {
                "correct": domain_correct.get(d, 0),
                "total": domain_totals[d],
                "accuracy": round(domain_correct.get(d, 0) / domain_totals[d], 4),
            }
            for d in domain_totals
        }
        total = len(self.problems)
        return HLEReport(
            total=total,
            correct=correct,
            accuracy=correct / total if total else 0.0,
            per_domain=per_domain,
        )
