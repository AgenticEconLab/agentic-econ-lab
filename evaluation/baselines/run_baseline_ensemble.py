#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Run multi-model ensemble Tier-2 evaluation on single-prompt baseline outputs.

Applies the same 2×2 ensemble evaluation (GPT-4o-mini × 2 + Claude Haiku 4.5 × 2)
to baseline outputs so that workflow-vs-baseline comparison uses a consistent
evaluation methodology.

Usage:
    python run_baseline_ensemble.py
    python run_baseline_ensemble.py --repeats 3
    python run_baseline_ensemble.py --models "gpt-4o-mini,claude-haiku-4-5-20251001"
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def main():
    parser = argparse.ArgumentParser(
        description="Run ensemble Tier-2 evaluation on baseline outputs"
    )
    parser.add_argument(
        "--repeats", type=int, default=2,
        help="Number of evaluation passes per model (default: 2)"
    )
    parser.add_argument(
        "--models", type=str, default="vllm/mistral-small-3.2-24b-fp8,vllm/gemma-3-27b-it-fp8",
        help="Comma-separated evaluator models (open-weight vLLM by default; commercial allowed)"
    )
    parser.add_argument(
        "--single-pass", action="store_true", default=False,
        help="Run single-pass evaluation (no ensemble) for backward compatibility"
    )
    args = parser.parse_args()

    # Set up paths
    baselines_dir = Path(__file__).parent
    outputs_dir = baselines_dir / "outputs"
    agents_dir = baselines_dir.parent.parent  # repository root
    # Ensure imports
    for p in [str(agents_dir), str(agents_dir / "evaluation")]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from evaluation.scoring.llm_evaluator import (
        LLMEvaluator,
        average_ensemble_scores,
        compute_ensemble_metadata,
    )
    from shared.observability import MetricsCollector

    # Create observability collector for evaluation LLM calls
    eval_collector = MetricsCollector()

    teams = ["IdeationTeam", "LiteratureTeam", "ModelTeam", "DataTeam"]
    mode = "SinglePromptBaseline"
    model_names = [m.strip() for m in args.models.split(",")]

    if args.single_pass:
        # Single-pass mode (backward compatible)
        print("=" * 60)
        print("BASELINE TIER-2 EVALUATION (single pass)")
        print("=" * 60)

        evaluator = LLMEvaluator(model="vllm/mistral-small-3.2-24b-fp8", temperature=0.0, collector=eval_collector)
        baseline_scores = {}

        for team in teams:
            print(f"\n  Evaluating {team} baseline...")
            try:
                scores = evaluator.evaluate_workflow(team, mode, outputs_dir)
                baseline_scores[team] = {
                    dim: s.overall_score for dim, s in scores.items()
                }
                for dim, s in scores.items():
                    print(f"    {dim}: {s.overall_score:.4f}")
            except Exception as e:
                print(f"    ERROR: {e}")
                baseline_scores[team] = {}

        # Save
        out_path = outputs_dir / "baseline_tier2_scores.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(baseline_scores, f, indent=2)
        print(f"\nSaved to: {out_path}")
        _print_observability_summary(eval_collector)
        return

    # Ensemble mode
    repeats = args.repeats
    total_passes = len(model_names) * repeats

    print("=" * 60)
    print(f"BASELINE TIER-2 ENSEMBLE ({len(model_names)} models × {repeats} repeats)")
    print(f"Models: {', '.join(model_names)}")
    print("=" * 60)

    # Instantiate evaluators
    evaluators = {}
    for model_name in model_names:
        try:
            evaluators[model_name] = LLMEvaluator(
                model=model_name, temperature=0.0, collector=eval_collector
            )
            print(f"  Initialized: {model_name}")
        except Exception as e:
            print(f"  WARNING: Could not initialize {model_name}: {e}")

    if not evaluators:
        print("ERROR: No evaluators initialized.")
        return

    baseline_scores = {}
    all_ensemble_meta = {}

    for team in teams:
        print(f"\n  Evaluating {team} baseline — {total_passes} passes")

        model_passes: Dict[str, list] = {m: [] for m in evaluators}
        all_passes_flat = []

        for model_name, evaluator in evaluators.items():
            try:
                passes = evaluator.evaluate_workflow_ensemble(
                    team, mode, outputs_dir, repeats=repeats
                )
                model_passes[model_name] = passes
                all_passes_flat.extend(passes)
            except Exception as e:
                print(f"    ERROR [{model_name}]: {e}")

        if not all_passes_flat:
            print(f"    No passes succeeded for {team}")
            baseline_scores[team] = {}
            continue

        # Average across all passes
        averaged = average_ensemble_scores(all_passes_flat, team, mode)
        baseline_scores[team] = {
            dim: round(s.overall_score, 4) for dim, s in averaged.items()
        }

        # Compute ensemble metadata
        all_ensemble_meta[team] = compute_ensemble_metadata(
            model_passes, repeats_per_model=repeats
        )

        for dim, s in averaged.items():
            print(f"    {dim}: {s.overall_score:.4f} (ensemble)")

    # Save averaged scores (same format as before for compatibility)
    out_path = outputs_dir / "baseline_tier2_scores.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(baseline_scores, f, indent=2)
    print(f"\nBaseline scores saved to: {out_path}")

    # Save detailed ensemble results
    ensemble_detail = {
        "generated_at": datetime.now().isoformat(),
        "evaluation_method": "ensemble",
        "models": model_names,
        "repeats_per_model": repeats,
        "total_passes_per_team": total_passes,
        "scores": baseline_scores,
        "ensemble_metadata": {
            team: meta.to_summary_dict()
            for team, meta in all_ensemble_meta.items()
        },
        "evaluation_observability": eval_collector.get_summary(),
    }
    detail_path = outputs_dir / "baseline_tier2_ensemble.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(ensemble_detail, f, indent=2)
    print(f"Ensemble detail saved to: {detail_path}")

    # Print ensemble statistics
    print("\n" + "=" * 60)
    print("ENSEMBLE STATISTICS")
    print("=" * 60)
    for team, meta in all_ensemble_meta.items():
        print(f"\n  {team}:")
        for model_name in meta.models:
            stds = meta.within_model_std.get(model_name, {})
            if stds:
                avg_std = sum(stds.values()) / len(stds)
                print(f"    {model_name} within-std: avg={avg_std:.4f}")
        if meta.between_model_gap:
            print(f"    Between-model gap: "
                  f"{', '.join(f'{d}={v:+.3f}' for d, v in meta.between_model_gap.items())}")

    _print_observability_summary(eval_collector)


def _print_observability_summary(collector) -> None:
    """Print observability summary to console."""
    obs = collector.get_summary()
    llm_stats = obs.get("llm", {})
    if llm_stats.get("total_calls", 0) > 0:
        print("\n" + "=" * 60)
        print("EVALUATION OBSERVABILITY SUMMARY")
        print("=" * 60)
        print(f"  Total LLM calls: {llm_stats['total_calls']}")
        print(f"  Total tokens:    {llm_stats['total_tokens']}")
        print(f"  Total cost:      ${llm_stats['total_cost_usd']:.4f}")
        print(f"  Avg latency:     {llm_stats['avg_latency_seconds']:.2f}s")
        print(f"  Errors:          {llm_stats['error_count']}")


if __name__ == "__main__":
    main()
