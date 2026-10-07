# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Judge-reliability bait check.

Before trusting Tier-2 scores, verify the judge actually:
  1. DISCRIMINATES — scores a known-GOOD output well above a known-BAD one
     (good_mean - bad_mean >= margin), and
  2. is RELIABLE — low score variance across repeats (max per-dimension stdev <= tol).

Run with a served (local) judge:  python -m evaluation.audits.judge_reliability vllm/<served>
or import run_judge_reliability(judge_model). No commercial cloud needed.
"""
from __future__ import annotations

import statistics
import sys
from typing import Dict, List

# A coherent, well-structured, cited economics literature-review excerpt (known GOOD).
_GOOD_REVIEW = """\
# Literature Review: Monetary Policy Transmission under Household Heterogeneity

## Synthesis
A substantial body of work establishes that the transmission of monetary policy operates
unevenly across households. Kaplan, Moll, and Violante (2018) formalize the HANK framework,
showing that indirect (general-equilibrium) effects dominate the direct intertemporal-
substitution channel emphasized by representative-agent models. Auclert (2019) decomposes
transmission into interest-rate-exposure, earnings-heterogeneity, and Fisher channels,
providing sufficient statistics that subsequent empirical work (e.g., Cloyne, Ferreira, and
Surico, 2020) has estimated using administrative data.

## Identified gaps
1. Most HANK estimates rely on pre-2015 data and do not capture the post-pandemic regime of
   simultaneously high inflation and rapid rate hikes; the stability of the estimated channels
   across regimes is untested.
2. The interaction between household balance-sheet composition and the Fisher channel remains
   under-identified outside the United States.

## Methods note
Estimates combine calibrated HANK models with local-projection IV using high-frequency
monetary surprises. Robustness to the surprise-construction window is reported.
"""

_GOOD_BIB = """\
Kaplan, G., Moll, B., & Violante, G. (2018). Monetary Policy According to HANK. AER, 108(3).
Auclert, A. (2019). Monetary Policy and the Redistribution Channel. AER, 109(6).
Cloyne, J., Ferreira, C., & Surico, P. (2020). Monetary Policy when Households have Debt. ReStud, 87(1).
"""

# Incoherent / empty / ungrounded output (known BAD).
_BAD_REVIEW = """\
Economics is a topic. There are many papers about money and things. Interest rates can go up or
down. Some researchers did studies. The results were results. More research is needed. In
conclusion, the economy is important and we reviewed it. Papers: various authors, many years.
"""


def run_judge_reliability(judge_model: str, repeats: int = 2, margin: float = 0.2,
                          tol: float = 0.2) -> bool:
    """Return True iff the judge discriminates (good-bad>=margin) AND is reliable
    (max per-dimension stdev across repeats <= tol)."""
    from evaluation.scoring.llm_evaluator import LLMEvaluator, LLM_EVALUATED_DIMENSIONS

    ev = LLMEvaluator(model=judge_model, temperature=0.0)
    print(f"Judge: {judge_model}  ·  dims: {list(LLM_EVALUATED_DIMENSIONS)}  ·  repeats: {repeats}")
    baits = {
        "GOOD": {"literature_review.txt": _GOOD_REVIEW, "bibliography.txt": _GOOD_BIB},
        "BAD": {"literature_review.txt": _BAD_REVIEW},
    }
    per_label: Dict[str, List[Dict[str, float]]] = {}
    for label, outputs in baits.items():
        runs: List[Dict[str, float]] = []
        for i in range(repeats):
            scores: Dict[str, float] = {}
            for dim in LLM_EVALUATED_DIMENSIONS:
                try:
                    scores[dim] = ev.evaluate_dimension(dim, "LiteratureTeam", "bait", outputs, {}).overall_score
                except Exception as e:
                    print(f"  {label} run{i} {dim}: ERROR {type(e).__name__}: {str(e)[:80]}")
            runs.append(scores)
            print(f"  {label} run{i}: mean={statistics.mean(scores.values()):.3f}" if scores else f"  {label} run{i}: no scores")
        per_label[label] = runs

    def all_vals(label):
        return [v for run in per_label[label] for v in run.values()]
    good_m = statistics.mean(all_vals("GOOD")) if all_vals("GOOD") else 0.0
    bad_m = statistics.mean(all_vals("BAD")) if all_vals("BAD") else 1.0
    # reliability: largest per-dimension stdev across repeats (GOOD bait)
    max_std = 0.0
    if repeats > 1:
        dims = per_label["GOOD"][0].keys()
        for d in dims:
            vals = [r[d] for r in per_label["GOOD"] if d in r]
            if len(vals) > 1:
                max_std = max(max_std, statistics.pstdev(vals))

    discriminates = (good_m - bad_m) >= margin
    reliable = max_std <= tol
    print(f"\nGOOD mean={good_m:.3f}  BAD mean={bad_m:.3f}  margin={good_m - bad_m:+.3f} (need >= {margin})")
    print(f"max per-dim stdev (GOOD) = {max_std:.3f} (need <= {tol})")
    verdict = discriminates and reliable
    print(f"JUDGE-RELIABILITY: {'PASS' if verdict else 'FAIL'}  "
          f"(discriminates={discriminates}, reliable={reliable})")
    return verdict


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "vllm/mistral-small-3.2-24b-fp8"
    sys.exit(0 if run_judge_reliability(model) else 1)
