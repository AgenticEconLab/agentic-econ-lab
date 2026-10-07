#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Run two-tier evaluation on AEL workflow execution logs.

This script:
1. Tier 1: Calculates structural metrics from execution_log.json (all 10 dimensions)
2. Tier 2: Runs LLM-as-reviewer content quality assessment (4 dimensions)
3. Combines both tiers into final scores
4. Optional: Consensus, trajectory, adversarial, benchmark evaluation
5. Generates output for the paper tables and figures
"""

import json
import os
import re
import sys
import math
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


def load_execution_log(team: str, mode: str, base_path: Path) -> Optional[Dict[str, Any]]:
    """Load execution log for a team/mode."""
    log_path = base_path / team / "ael" / mode / "execution_log.json"
    if log_path.exists():
        with open(log_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


EFFICIENCY_REFERENCE_SECONDS_PER_TEAM = 3600.0


def calculate_tier1_metrics(log: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate Tier 1 (structural) evaluation metrics from execution log."""
    metrics = {}

    # Basic info
    total_duration = log.get("total_duration_seconds", 0)
    stages = log.get("stages", [])
    errors = log.get("errors", [])

    # === QUALITY CLUSTER (Structural) ===

    item_counts = [s.get("item_count", 0) or 0 for s in stages]
    successful_stages = sum(1 for s in stages if s.get("status") == "success")
    total_stages = len(stages)
    stages_with_outputs = sum(1 for s in stages if s.get("output_files"))

    stage_completion = successful_stages / total_stages if total_stages > 0 else 0
    output_coverage = stages_with_outputs / total_stages if total_stages > 0 else 0
    error_count = len(errors)
    if error_count == 0:
        error_penalty = 1.0
    elif stage_completion == 1.0:
        error_penalty = 0.7
    else:
        error_penalty = max(0, 1 - (error_count / max(total_stages, 1)))

    metrics["reliability"] = stage_completion * output_coverage * error_penalty
    metrics["error_rate"] = error_count
    metrics["stage_completion_rate"] = stage_completion
    metrics["correctness"] = stage_completion * (1 if error_count == 0 else 0.5)

    stage_numbers = [s.get("number", i) for i, s in enumerate(stages)]
    is_valid_order = stage_numbers == sorted(stage_numbers)
    metrics["stage_ordering"] = 1.0 if is_valid_order else 0.0
    metrics["soundness"] = 1.0 if is_valid_order else 0.0

    # === OPERATIONAL CLUSTER ===

    total_items = sum(item_counts)
    metrics["execution_time"] = total_duration
    metrics["time_per_item"] = total_duration / total_items if total_items > 0 else 0
    # A linear max(0, 1 - T/3600) would score every run longer than an hour as 0 (a 4-5 h
    # chained run always 0). Hyperbolic normalization against a
    # reference duration: 0.5 at T = T_ref, never 0, monotone decreasing. T_ref = one hour per
    # team (a chained pipeline log counts each team-stage); recorded next to the score.
    n_teams = total_stages if log.get("team") == "Pipeline" else 1
    ref = float(EFFICIENCY_REFERENCE_SECONDS_PER_TEAM) * max(1, n_teams)
    metrics["efficiency_reference_seconds"] = ref
    metrics["efficiency"] = ref / (ref + total_duration) if total_duration > 0 else 1.0

    metrics["throughput"] = (total_items / (total_duration / 60)) if total_duration > 0 else 0
    metrics["scalability"] = min(1.0, metrics["throughput"] / 10)

    if error_count > 0:
        recovered = sum(1 for e in errors if e.get("recovered", False))
        metrics["error_recovery_rate"] = recovered / error_count
    else:
        metrics["error_recovery_rate"] = 1.0
    metrics["robustness"] = metrics["error_recovery_rate"]

    # === EPISTEMIC CLUSTER (Structural) ===

    metrics["documentation_coverage"] = stages_with_outputs / total_stages if total_stages > 0 else 0
    metrics["transparency"] = metrics["documentation_coverage"]

    provenance_fields = ["execution_id", "team", "mode", "started_at", "completed_at"]
    present = sum(1 for f in provenance_fields if log.get(f))
    metrics["provenance_completeness"] = present / len(provenance_fields)
    metrics["traceability"] = metrics["provenance_completeness"]

    has_metadata = bool(log.get("metadata"))
    metrics["config_completeness"] = 1.0 if has_metadata else 0.5
    metrics["reproducibility"] = metrics["config_completeness"]

    # === INNOVATION CLUSTER (Structural) ===

    if total_items > 0 and len(item_counts) > 1:
        probs = [c / total_items for c in item_counts if c > 0]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        max_entropy = math.log2(len(probs)) if len(probs) > 1 else 1.0
        metrics["output_diversity"] = entropy / max_entropy if max_entropy > 0 else 0
    else:
        metrics["output_diversity"] = 0.5

    metrics["coverage_breadth"] = min(1.0, total_items / 50)
    metrics["innovation_potential"] = (metrics["output_diversity"] + metrics["coverage_breadth"]) / 2

    # === DOMAIN CLUSTER (Structural baselines) ===

    # Decision quality: structural proxy from stage progression and error handling
    stage_progression = 1.0 if (successful_stages == total_stages and is_valid_order) else 0.5
    error_handling = metrics["error_recovery_rate"]
    metrics["decision_quality"] = (stage_progression + error_handling) / 2

    # Economic rigor: structural proxy (content quality assessed by Tier 2)
    metrics["economic_rigor"] = stage_completion * output_coverage

    # === V0.7 ECONOMICS CLUSTER (Tier 1 structural) ===
    # Defaults preserve the V0.6 numeric shape when V0.7 artefacts are absent.
    metrics["simulation_fidelity"] = _tier1_simulation_fidelity(log)
    metrics["causal_validity"] = _tier1_causal_validity(log)

    return metrics


def _tier1_simulation_fidelity(log: Dict[str, Any]) -> Optional[float]:
    """V0.7 structural score: run SimulationFidelityEvaluator on any
    simulation_output embedded in the execution log. Returns None ("not evaluated") when no
    simulation was run or the evaluator failed — never a neutral 0.5 that would read as a
    measured score.
    """
    sim = log.get("simulation_result") or log.get("simulation") or {}
    if not isinstance(sim, dict) or not sim:
        return None
    try:
        from evaluation.dimensions import SimulationFidelityEvaluator
        score = SimulationFidelityEvaluator().evaluate(sim).score
        return None if score is None else float(score)
    except Exception:
        return None


def _tier1_causal_validity(log: Dict[str, Any]) -> Optional[float]:
    """V0.7 structural score: run CausalValidityEvaluator on any
    causal_estimate embedded in the execution log. Returns None ("not evaluated")
    when no estimate or no reference (truth/prior) is available.
    """
    estimate = log.get("causal_estimate")
    if not isinstance(estimate, dict) or estimate.get("ate") is None:
        return None                     # absent, or a refusal record
    try:
        from evaluation.dimensions import CausalValidityEvaluator
        true_ate = estimate.get("true_ate")
        prior_mean = estimate.get("prior_mean")
        score = CausalValidityEvaluator().evaluate(
            estimate,
            true_ate=float(true_ate) if true_ate is not None else None,
            prior_mean=float(prior_mean) if prior_mean is not None else None,
        ).score
        return None if score is None else float(score)
    except Exception:
        return None


def _ensure_imports(base_path: Path):
    """Ensure evaluation imports are available."""
    agents_dir = str(base_path)
    if agents_dir not in sys.path:
        sys.path.insert(0, agents_dir)
    eval_dir = str(base_path / "evaluation")
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)


def run_tier2_evaluation(
    teams_list: List[tuple],
    base_path: Path,
    results: Dict[str, Dict],
    collector: Optional[Any] = None,
    model: str = "vllm/mistral-small-3.2-24b-fp8",
) -> Dict[str, Dict]:
    """Run Tier 2 LLM-as-reviewer evaluation for all configurations.

    `model` is the judge; default is the local Mistral judge (no commercial cloud).
    For an ``ollama/<served-name>`` judge, serve it on vLLM port 11434 first.
    """
    _ensure_imports(base_path)
    from evaluation.scoring.llm_evaluator import LLMEvaluator

    evaluator = LLMEvaluator(model=model, temperature=0.0, collector=collector)
    print(f"  Judge model: {model}")
    tier2_results = {}

    print("\n" + "=" * 70)
    print("TIER 2: LLM-AS-REVIEWER CONTENT QUALITY ASSESSMENT")
    print("=" * 70)

    for team, mode in teams_list:
        key = f"{team}_{mode}"
        if key not in results:
            continue

        print(f"\n  Evaluating {team} ({mode})...")
        try:
            llm_scores = evaluator.evaluate_workflow(team, mode, base_path)
            tier2_results[key] = llm_scores
        except Exception as e:
            print(f"    ERROR: {e}")
            tier2_results[key] = {}

    return tier2_results


def run_tier2_ensemble_evaluation(
    teams_list: List[tuple],
    base_path: Path,
    results: Dict[str, Dict],
    model_names: List[str],
    repeats: int = 2,
    collector: Optional[Any] = None,
) -> tuple:
    """Run Tier 2 evaluation as a multi-model ensemble.

    For each configuration, runs each model `repeats` times and averages
    all passes to produce stable ensemble scores.

    Args:
        teams_list: List of (team, mode) tuples.
        base_path: Base path to agent directories.
        results: Current results dict (for filtering valid configs).
        model_names: List of evaluator model identifiers.
        repeats: Number of passes per model.
        collector: Optional MetricsCollector for observability tracking.

    Returns:
        Tuple of (tier2_averaged_results, ensemble_metadata_per_config).
    """
    _ensure_imports(base_path)
    from evaluation.scoring.llm_evaluator import (
        LLMEvaluator,
        average_ensemble_scores,
        compute_ensemble_metadata,
    )

    # Instantiate one evaluator per model
    evaluators = {}
    for model_name in model_names:
        try:
            evaluators[model_name] = LLMEvaluator(
                model=model_name, temperature=0.0, collector=collector
            )
            print(f"  Initialized evaluator: {model_name}")
        except Exception as e:
            print(f"  WARNING: Could not initialize {model_name}: {e}")

    if not evaluators:
        print("  ERROR: No evaluators initialized. Aborting ensemble.")
        return {}, {}

    total_models = len(evaluators)
    total_passes = total_models * repeats

    print("\n" + "=" * 70)
    print(f"TIER 2: MULTI-MODEL ENSEMBLE ({total_models} models × {repeats} repeats = {total_passes} passes)")
    print(f"Models: {', '.join(evaluators.keys())}")
    print("=" * 70)

    tier2_averaged = {}
    ensemble_meta = {}

    for team, mode in teams_list:
        key = f"{team}_{mode}"
        if key not in results:
            continue

        print(f"\n  Evaluating {team} ({mode}) — {total_passes} passes total")

        # Collect passes per model
        model_passes: Dict[str, list] = {m: [] for m in evaluators}
        all_passes_flat = []

        for model_name, evaluator in evaluators.items():
            try:
                passes = evaluator.evaluate_workflow_ensemble(
                    team, mode, base_path, repeats=repeats
                )
                model_passes[model_name] = passes
                all_passes_flat.extend(passes)
            except Exception as e:
                print(f"    ERROR [{model_name}]: {e}")

        if not all_passes_flat:
            print(f"    No passes succeeded for {key}")
            tier2_averaged[key] = {}
            continue

        # Average across all passes
        tier2_averaged[key] = average_ensemble_scores(
            all_passes_flat, team, mode
        )

        # Compute ensemble statistics
        ensemble_meta[key] = compute_ensemble_metadata(
            model_passes, repeats_per_model=repeats
        )

        # Print summary
        for dim, score in tier2_averaged[key].items():
            print(f"    {dim}: {score.overall_score:.3f} (ensemble avg)")

    return tier2_averaged, ensemble_meta


def _load_workflow_outputs(team: str, mode: str, base_path: Path) -> Dict[str, Any]:
    """Load workflow output files for a team/mode (for adversarial/benchmark/trajectory)."""
    from evaluation.scoring.llm_evaluator import TEAM_OUTPUT_FILES, TEAM_OUTPUT_FALLBACKS

    output_dir = base_path / team / "ael" / mode
    outputs: Dict[str, Any] = {}

    for filename in TEAM_OUTPUT_FILES.get(team, []):
        filepath = output_dir / filename
        if filepath.exists():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    outputs[filename] = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

    if len(outputs) < 2:
        for filename in TEAM_OUTPUT_FALLBACKS.get(team, []):
            filepath = output_dir / filename
            if filepath.exists() and filename not in outputs:
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        outputs[filename] = json.load(f)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass

    return outputs


def run_consensus_evaluation(
    teams_list: List[tuple],
    base_path: Path,
    results: Dict[str, Dict],
    model_names: Optional[List[str]] = None,
    collector: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run ConsensusEngine evaluation for all configurations."""
    _ensure_imports(base_path)
    from evaluation.consensus.consensus_engine import ConsensusEngine, CalibrationMethod

    engine = ConsensusEngine(
        model_names=model_names,
        calibration=CalibrationMethod.MEAN_SHIFT,
    )

    print("\n" + "=" * 70)
    print("CONSENSUS EVALUATION (Multi-Model + Calibration)")
    print(f"Models: {', '.join(engine.model_names)}")
    print("=" * 70)

    consensus_results = {}
    for team, mode in teams_list:
        key = f"{team}_{mode}"
        if key not in results:
            continue

        print(f"\n  Evaluating {team} ({mode})...")
        try:
            cr = engine.evaluate_with_consensus(
                team, mode, base_path, collector=collector,
            )
            consensus_results[key] = cr
            print(f"    Fleiss κ: {cr.fleiss_kappa:.3f}")
            for dim, score in cr.consensus_scores.items():
                print(f"    {dim}: {score:.3f}")
        except Exception as e:
            print(f"    ERROR: {e}")

    return consensus_results


def run_trajectory_evaluation(
    teams_list: List[tuple],
    base_path: Path,
    results: Dict[str, Dict],
    use_llm: bool = False,
    collector: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run TrajectoryEvaluator for all configurations."""
    _ensure_imports(base_path)
    from evaluation.trajectory.trajectory_evaluator import TrajectoryEvaluator

    llm_fn = None
    if use_llm:
        from shared.llm import LLMClient
        client = LLMClient(collector=collector)
        llm_fn = lambda prompt: client.invoke(prompt, agent="TrajectoryEvaluator")

    evaluator = TrajectoryEvaluator(llm_invoke_fn=llm_fn)

    print("\n" + "=" * 70)
    print(f"TRAJECTORY EVALUATION ({'LLM-as-Judge' if use_llm else 'Structural only'})")
    print("=" * 70)

    trajectory_results = {}
    for team, mode in teams_list:
        key = f"{team}_{mode}"
        if key not in results:
            continue

        log = results[key].get("log")
        if not log:
            continue

        outputs = _load_workflow_outputs(team, mode, base_path) if use_llm else None

        print(f"\n  Evaluating {team} ({mode})...")
        try:
            score = evaluator.evaluate_trajectory(log, outputs)
            trajectory_results[key] = score
            print(f"    Overall: {score.overall_score:.3f} "
                  f"(coh={score.decision_coherence:.3f}, util={score.information_utilization:.3f}, "
                  f"scope={score.scope_management:.3f}, trans={score.stage_transition_quality:.3f})")
        except Exception as e:
            print(f"    ERROR: {e}")

    return trajectory_results


def run_adversarial_evaluation(
    teams_list: List[tuple],
    base_path: Path,
    results: Dict[str, Dict],
    collector: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run AdversarialReviewer for all configurations."""
    _ensure_imports(base_path)
    from evaluation.consensus.adversarial_reviewer import AdversarialReviewer
    from shared.llm import LLMClient

    client = LLMClient(collector=collector)
    llm_fn = lambda prompt: client.invoke(prompt, agent="AdversarialReviewer")
    reviewer = AdversarialReviewer(
        llm_invoke_fn=llm_fn,
        model=os.environ.get("AEL_JUDGE_MODEL", "vllm/mistral-small-3.2-24b-fp8"),
    )

    print("\n" + "=" * 70)
    print("ADVERSARIAL REVIEW (Critic Agent)")
    print("=" * 70)

    adversarial_results = {}
    for team, mode in teams_list:
        key = f"{team}_{mode}"
        if key not in results:
            continue

        outputs = _load_workflow_outputs(team, mode, base_path)
        log = results[key].get("log")

        if not outputs:
            print(f"\n  {team} ({mode}): No output files found, skipping")
            continue

        print(f"\n  Reviewing {team} ({mode})...")
        try:
            report = reviewer.review(team, mode, outputs, log)
            adversarial_results[key] = report
            print(f"    Issues: {report.total_issues} "
                  f"(critical={report.critical_count}, major={report.major_count}, "
                  f"minor={report.minor_count}) | Penalty: {report.penalty:.3f}")
        except Exception as e:
            print(f"    ERROR: {e}")

    return adversarial_results


def run_benchmark_evaluation(
    teams_list: List[tuple],
    base_path: Path,
    results: Dict[str, Dict],
) -> Dict[str, Any]:
    """Run BenchmarkComparator for all configurations."""
    _ensure_imports(base_path)
    from evaluation.benchmarks.econ_benchmarks import BenchmarkComparator, get_team_benchmark

    comparator = BenchmarkComparator()

    print("\n" + "=" * 70)
    print("BENCHMARK COMPARISON (Reference-Based Quality)")
    print("=" * 70)

    benchmark_results = {}
    for team, mode in teams_list:
        key = f"{team}_{mode}"
        if key not in results:
            continue

        benchmark = get_team_benchmark(team)
        if not benchmark:
            continue

        outputs = _load_workflow_outputs(team, mode, base_path)
        if not outputs:
            print(f"\n  {team} ({mode}): No output files found, skipping")
            continue

        print(f"\n  Comparing {team} ({mode})...")
        try:
            br = comparator.compare(outputs, benchmark)
            benchmark_results[key] = br
            print(f"    Relative quality: {br.relative_quality:.3f}")
            for feat, matched in br.feature_matches.items():
                print(f"      {'✓' if matched else '✗'} {feat}")
        except Exception as e:
            print(f"    ERROR: {e}")

    return benchmark_results


def combine_scores(
    tier1: Dict[str, float],
    tier2: Dict[str, Any],
    tier1_weight: float = 0.0,
    tier2_weight: float = 1.0,
) -> Dict[str, float]:
    """Combine Tier-1 (structural / execution-integrity) and Tier-2 (LLM content) scores.

    For CONTENT-judged dimensions the Tier-1 structural proxy is saturated (~1.0 for any
    clean run), so blending it (the old 0.3/0.7 default) only added a uniform +0.30 floor
    that inflated and flattened the quality signal. Default is now Tier-2-authoritative for
    content dims (tier1_weight=0.0); structural-only dims (reliability, efficiency, robustness,
    traceability, reproducibility, scalability, simulation_fidelity, causal_validity) still
    carry Tier-1 = execution integrity. Pass tier1_weight>0 to restore the blend.
    """
    combined = {}
    tier2_dims = {"correctness", "soundness", "innovation_potential", "transparency", "decision_quality", "economic_rigor"}

    dimensions = [
        "reliability", "correctness", "soundness",
        "efficiency", "scalability", "robustness",
        "transparency", "traceability", "reproducibility",
        "innovation_potential",
        "decision_quality", "economic_rigor",
        # V0.7 — 14-dimension evaluation
        "simulation_fidelity", "causal_validity",
    ]

    for dim in dimensions:
        t1 = tier1.get(dim)          # None = not evaluated; kept as None, never 0/0.5
        if dim in tier2_dims and dim in tier2 and hasattr(tier2[dim], "overall_score"):
            t2 = tier2[dim].overall_score
            combined[dim] = round((tier1_weight * t1 if t1 is not None else 0.0)
                                  + tier2_weight * t2, 4)
        else:
            combined[dim] = t1

    return combined


def main():
    """Run two-tier evaluation and generate report."""
    parser = argparse.ArgumentParser(description="Run AEL two-tier evaluation")
    parser.add_argument(
        "--tier2", action="store_true", default=False,
        help="Run Tier 2 LLM-as-reviewer evaluation (judge set by --judge-model)"
    )
    parser.add_argument(
        "--judge-model", type=str,
        default=os.environ.get("AEL_JUDGE_MODEL", "vllm/mistral-small-3.2-24b-fp8"),
        help="Tier-2 judge model. 3-layer plan: HPC vLLM / local Ollama both use "
             "ollama/<served-name> (-> localhost:11434); commercial (gpt-4o-mini, claude-*) "
             "is cloud fallback. Default: local Mistral judge (env AEL_JUDGE_MODEL)."
    )
    parser.add_argument(
        "--tier1-weight", type=float, default=0.0,
        help="Weight for the Tier-1 structural proxy in CONTENT dims (default: 0.0 — content "
             "is Tier-2-authoritative; the structural proxy saturates and would inflate). >0 blends."
    )
    parser.add_argument(
        "--tier2-weight", type=float, default=1.0,
        help="Weight for Tier-2 content scores in content dims (default: 1.0)"
    )
    parser.add_argument(
        "--tier2-ensemble", action="store_true", default=False,
        help="Run Tier 2 evaluation as a multi-model ensemble. Default = the two local "
             "judges (Mistral + Gemma); each must be served on vLLM. Averaged scores + metadata."
    )
    parser.add_argument(
        "--ensemble-repeats", type=int, default=2,
        help="Number of evaluation passes per model in ensemble mode (default: 2)"
    )
    parser.add_argument(
        "--ensemble-models", type=str,
        default="vllm/mistral-small-3.2-24b-fp8,vllm/gemma-3-27b-it-fp8",
        help="Comma-separated evaluator models for ensemble mode (default: local judges; "
             "commercial gpt-4o-mini/claude-* allowed as cloud fallback)."
    )
    parser.add_argument(
        "--consensus", action="store_true", default=False,
        help="Run multi-model consensus evaluation with score calibration"
    )
    parser.add_argument(
        "--trajectory", action="store_true", default=False,
        help="Run trajectory evaluation (decision sequence quality)"
    )
    parser.add_argument(
        "--adversarial", action="store_true", default=False,
        help="Run adversarial review (critic agent for weakness detection)"
    )
    parser.add_argument(
        "--benchmark", action="store_true", default=False,
        help="Run benchmark comparison against reference exemplars"
    )
    parser.add_argument(
        "--multirun-dir", type=str, default=None,
        help="Path to multirun output directory (from multirun_orchestrator.py). "
             "Aggregates scores across runs and writes evaluation_results_multirun.json"
    )
    parser.add_argument(
        "--agent-metrics", action="store_true", default=False,
        help="Run agent-level metrics (tool correctness, step efficiency, "
             "plan adherence, argument correctness)"
    )
    parser.add_argument(
        "--lightweight-judge", action="store_true", default=False,
        help="Use lightweight judge model for consensus evaluation"
    )
    args = parser.parse_args()

    # Handle multirun aggregation mode
    if args.multirun_dir:
        tier1_ok = run_multirun_aggregation(args.multirun_dir)   # tier-1 (structural)
        if args.tier2:
            # Observability collector: unlike the single-run path below, this branch
            # returns early, so without this the --multirun-dir path never captured
            # per-call cost/token/latency data.
            multirun_base = Path(__file__).parent.parent
            _ensure_imports(multirun_base)
            from shared.observability import MetricsCollector
            eval_collector = MetricsCollector()
            t2 = run_multirun_tier2(args.multirun_dir, args.judge_model, collector=eval_collector)   # tier-2 (LLM judge)

            obs_summary = eval_collector.get_summary()
            llm_stats = obs_summary.get("llm", {})
            if llm_stats.get("total_calls", 0) > 0:
                print("\n" + "=" * 70)
                print("EVALUATION OBSERVABILITY SUMMARY (multirun)")
                print("=" * 70)
                print(f"  Total LLM calls: {llm_stats['total_calls']}")
                print(f"  Total tokens:    {llm_stats['total_tokens']}")
                print(f"  Total cost:      ${llm_stats['total_cost_usd']:.4f}")
                print(f"  Avg latency:     {llm_stats['avg_latency_seconds']:.2f}s")
                print(f"  Errors:          {llm_stats['error_count']}")

                judge_slug = re.sub(r"[^A-Za-z0-9.-]+", "_", args.judge_model.split("/")[-1]) or "judge"
                obs_path = Path(args.multirun_dir) / f"evaluation_observability_multirun_{judge_slug}.json"
                try:
                    with open(obs_path, "w", encoding="utf-8") as f:
                        json.dump({"judge_model": args.judge_model, **obs_summary}, f, indent=2)
                    print(f"\nObservability summary -> {obs_path}")
                except Exception as e:
                    print(f"(could not write {obs_path}: {e})")
            # Judge failures or an empty result make the command fail, so batch jobs report it.
            n_err = getattr(run_multirun_tier2, "last_errors", 0)
            if n_err or not t2:
                print(f"Tier-2 evaluation incomplete: {n_err} error(s), "
                      f"{len(t2 or {})} configuration(s) scored", file=sys.stderr)
                sys.exit(1)
        if not tier1_ok:
            print("Tier-1 aggregation found no run data", file=sys.stderr)
            sys.exit(1)
        return

    is_ensemble = args.tier2_ensemble
    base_path = Path(__file__).parent.parent

    # Create observability collector for evaluation LLM calls
    needs_collector = args.tier2 or is_ensemble or args.consensus or args.trajectory or args.adversarial
    eval_collector = None
    if needs_collector:
        _ensure_imports(base_path)
        from shared.observability import MetricsCollector
        eval_collector = MetricsCollector()

    # All 15 configurations
    teams = [
        ("IdeationTeam", "ModeNoWcNoHITL"),
        ("IdeationTeam", "ModeNoWcWithHITL"),
        ("IdeationTeam", "ModeWithWcNoHITL"),
        ("IdeationTeam", "ModeWithWcWithHITL"),
        ("LiteratureTeam", "ModeNoWcNoHITL"),
        ("LiteratureTeam", "ModeNoWcWithHITL"),
        ("LiteratureTeam", "ModeWithWcNoHITL"),
        ("LiteratureTeam", "ModeWithWcWithHITL"),
        ("ModelTeam", "ModeNoWcNoHITL"),
        ("ModelTeam", "ModeNoWcWithHITL"),
        ("ModelTeam", "ModeWithWcNoHITL"),
        ("ModelTeam", "ModeWithWcWithHITL"),
        ("DataTeam", "ModeOpenSourceAPI"),
        ("DataTeam", "ModePremiumSubscribed"),
        ("DataTeam", "ModeUserUploaded"),
    ]

    results = {}

    # =====================================================================
    # TIER 1: Structural Metrics
    # =====================================================================
    print("=" * 70)
    print("TIER 1: STRUCTURAL EVALUATION (from execution logs)")
    print("=" * 70)
    print(f"Generated at: {datetime.now().isoformat()}")

    for team, mode in teams:
        log = load_execution_log(team, mode, base_path)
        key = f"{team}_{mode}"
        if log:
            metrics = calculate_tier1_metrics(log)
            results[key] = {
                "team": team,
                "mode": mode,
                "log": log,
                "tier1": metrics,
                "tier2": {},
                "combined": {},
            }
            print(f"\n  {team} ({mode})")
            print(f"    Duration: {log.get('total_duration_seconds', 0):.1f}s | "
                  f"Stages: {log.get('summary', {}).get('successful_stages', 0)}/"
                  f"{log.get('summary', {}).get('total_stages', 0)} | "
                  f"Items: {log.get('summary', {}).get('total_items', 0)} | "
                  f"Errors: {log.get('summary', {}).get('total_errors', 0)}")
        else:
            print(f"\n  {team} ({mode}): No execution log found")

    # =====================================================================
    # TIER 2: LLM-as-Reviewer (optional)
    # =====================================================================
    tier2_results = {}
    ensemble_metadata = {}

    if args.tier2_ensemble:
        # Ensemble mode: multi-model × multi-pass
        model_names = [m.strip() for m in args.ensemble_models.split(",")]
        tier2_results, ensemble_metadata = run_tier2_ensemble_evaluation(
            teams, base_path, results,
            model_names=model_names,
            repeats=args.ensemble_repeats,
            collector=eval_collector,
        )
        for key, scores in tier2_results.items():
            if key in results:
                results[key]["tier2"] = scores
    elif args.tier2:
        tier2_results = run_tier2_evaluation(teams, base_path, results, collector=eval_collector, model=args.judge_model)
        for key, scores in tier2_results.items():
            if key in results:
                results[key]["tier2"] = scores

    # =====================================================================
    # PHASE 5: EXTENDED EVALUATIONS (consensus, trajectory, adversarial, benchmark)
    # =====================================================================
    consensus_results = {}
    trajectory_results = {}
    adversarial_results = {}
    benchmark_results = {}

    if args.consensus:
        model_names = [m.strip() for m in args.ensemble_models.split(",")]
        consensus_results = run_consensus_evaluation(
            teams, base_path, results,
            model_names=model_names,
            collector=eval_collector,
        )

    if args.trajectory:
        use_llm = args.tier2 or is_ensemble  # Use LLM only if Tier 2 is also enabled
        trajectory_results = run_trajectory_evaluation(
            teams, base_path, results,
            use_llm=use_llm,
            collector=eval_collector,
        )

    if args.adversarial:
        adversarial_results = run_adversarial_evaluation(
            teams, base_path, results,
            collector=eval_collector,
        )

    if args.benchmark:
        benchmark_results = run_benchmark_evaluation(
            teams, base_path, results,
        )

    # =====================================================================
    # AGENT-LEVEL METRICS (optional)
    # =====================================================================
    agent_metrics_results = {}
    if args.agent_metrics:
        _ensure_imports(base_path)
        from evaluation.agent_metrics.tool_correctness import ToolCorrectnessScorer
        from evaluation.agent_metrics.step_efficiency import StepEfficiencyScorer
        from evaluation.agent_metrics.plan_adherence import PlanAdherenceScorer
        from evaluation.agent_metrics.argument_correctness import ArgumentCorrectnessScorer

        tc_scorer = ToolCorrectnessScorer()
        se_scorer = StepEfficiencyScorer()
        pa_scorer = PlanAdherenceScorer()
        ac_scorer = ArgumentCorrectnessScorer()

        print("\n" + "=" * 70)
        print("AGENT-LEVEL METRICS")
        print("=" * 70)

        for key in results:
            team = results[key]["team"]
            log = results[key].get("log", {})
            tc = tc_scorer.score(log, team=team)
            se = se_scorer.score(log, team=team)
            pa = pa_scorer.score(log, team=team)
            ac = ac_scorer.score(log, team=team)

            agent_metrics_results[key] = {
                "tool_correctness": tc,
                "step_efficiency": se,
                "plan_adherence": pa,
                "argument_correctness": ac,
            }

            print(f"\n  {team} ({results[key]['mode']})")
            print(f"    Tool Correctness: {tc['tool_correctness']:.3f} | "
                  f"Step Efficiency: {se['step_efficiency']:.3f} | "
                  f"Plan Adherence: {pa['plan_adherence']:.3f} | "
                  f"Arg Correctness: {ac['argument_correctness']:.3f}")

    # =====================================================================
    # COMBINE SCORES
    # =====================================================================
    for key in results:
        t1 = results[key]["tier1"]
        t2 = results[key].get("tier2", {})
        results[key]["combined"] = combine_scores(
            t1, t2, args.tier1_weight, args.tier2_weight
        )

    # =====================================================================
    # OUTPUT: Summary Tables
    # =====================================================================
    dimensions = [
        "reliability", "correctness", "soundness",
        "efficiency", "scalability", "robustness",
        "transparency", "traceability", "reproducibility",
        "innovation_potential",
        "decision_quality", "economic_rigor",
        # V0.7 — 14-dimension evaluation
        "simulation_fidelity", "causal_validity",
    ]
    tier2_dims = {"correctness", "soundness", "innovation_potential", "transparency", "decision_quality", "economic_rigor"}

    print("\n" + "=" * 70)
    if is_ensemble:
        print("TWO-TIER ENSEMBLE EVALUATION SCORES")
    elif args.tier2:
        print("TWO-TIER EVALUATION SCORES")
    else:
        print("TIER 1 STRUCTURAL SCORES (run with --tier2 for content quality)")
    print("=" * 70)

    # Print header
    header_dims = ["Rel", "Cor", "Snd", "Eff", "Scl", "Rob", "Trn", "Trc", "Rep", "Inn", "Dec", "Eco"]
    print(f"\n{'Config':<40}", end="")
    for h in header_dims:
        print(f" {h:>6}", end="")
    print()
    print("-" * 105)

    score_key = "combined" if (args.tier2 or is_ensemble) else "tier1"
    radar_data = {}

    for team, mode in teams:
        key = f"{team}_{mode}"
        if key in results:
            short_key = f"{team[:4]}_{mode.replace('Mode', '')}"
            print(f"{short_key:<40}", end="")
            scores = results[key][score_key]
            for dim in dimensions:
                s = scores.get(dim)
                print(f" {s:>6.3f}" if isinstance(s, (int, float)) else f" {'n/a':>6}", end="")
            print()
            # radar: dimensions not evaluated for this configuration are left out
            radar_data[key] = [scores.get(dim) if isinstance(scores.get(dim), (int, float))
                               else 0.0 for dim in dimensions]

    # =====================================================================
    # TIER 2 DETAIL (if available)
    # =====================================================================
    if args.tier2 or is_ensemble:
        print("\n" + "=" * 70)
        print("TIER 2 DETAIL: LLM-AS-REVIEWER SUB-CRITERIA SCORES")
        if is_ensemble:
            print("(Ensemble-averaged scores)")
        print("=" * 70)

        for team, mode in teams:
            key = f"{team}_{mode}"
            if key in results and results[key].get("tier2"):
                print(f"\n  {team} ({mode})")
                for dim, score_obj in results[key]["tier2"].items():
                    if hasattr(score_obj, "sub_criteria"):
                        print(f"    {dim}: {score_obj.overall_score:.3f} "
                              f"(raw avg: {score_obj.raw_average:.2f}/5)")
                        for sc in score_obj.sub_criteria:
                            print(f"      - {sc.criterion_name}: {sc.score}/5 — {sc.justification[:80]}")

    # =====================================================================
    # ENSEMBLE STATISTICS (if applicable)
    # =====================================================================
    if is_ensemble and ensemble_metadata:
        print("\n" + "=" * 70)
        print("ENSEMBLE STATISTICS")
        print("=" * 70)

        # Within-model std summary
        for key_name in sorted(ensemble_metadata.keys()):
            meta = ensemble_metadata[key_name]
            print(f"\n  {key_name}:")
            for model_name in meta.models:
                stds = meta.within_model_std.get(model_name, {})
                if stds:
                    avg_std = sum(stds.values()) / len(stds)
                    print(f"    {model_name} within-std: avg={avg_std:.4f} "
                          f"({', '.join(f'{d}={v:.3f}' for d, v in stds.items())})")
            if meta.between_model_gap:
                print(f"    Between-model gap: "
                      f"{', '.join(f'{d}={v:+.3f}' for d, v in meta.between_model_gap.items())}")

    # =====================================================================
    # SAVE RESULTS
    # =====================================================================
    output_path = Path(__file__).parent / "evaluation_results_two_tier.json"

    eval_method = "two_tier_ensemble" if is_ensemble else ("two_tier" if args.tier2 else "tier1_only")

    save_data = {
        "generated_at": datetime.now().isoformat(),
        "evaluation_method": eval_method,
        "tier1_weight": args.tier1_weight,
        "tier2_weight": args.tier2_weight,
        "total_configurations": len(teams),
        "configurations_with_logs": len(results),
        "dimensions": dimensions,
        "tier2_dimensions": list(tier2_dims),
        "results": {},
        "radar_data": radar_data,
        "extended_evaluations": {
            "consensus": args.consensus,
            "trajectory": args.trajectory,
            "adversarial": args.adversarial,
            "benchmark": args.benchmark,
        },
    }

    # Add ensemble metadata at top level if applicable
    if is_ensemble and ensemble_metadata:
        # Pick any config's metadata for global info
        any_meta = next(iter(ensemble_metadata.values()), None)
        if any_meta:
            save_data["ensemble_config"] = {
                "models": any_meta.models,
                "repeats_per_model": any_meta.repeats_per_model,
                "total_passes_per_config": any_meta.total_passes,
            }

    for key in results:
        entry = {
            "team": results[key]["team"],
            "mode": results[key]["mode"],
            "tier1": results[key]["tier1"],
            "combined": results[key]["combined"],
            "summary": results[key]["log"].get("summary", {}),
        }
        # Add tier2 detail if available
        if results[key].get("tier2"):
            entry["tier2"] = {}
            for dim, score_obj in results[key]["tier2"].items():
                if hasattr(score_obj, "overall_score"):
                    entry["tier2"][dim] = {
                        "overall_score": score_obj.overall_score,
                        "raw_average": score_obj.raw_average,
                        "sub_criteria": [
                            {
                                "criterion_name": sc.criterion_name,
                                "score": sc.score,
                                "justification": sc.justification,
                            }
                            for sc in score_obj.sub_criteria
                        ],
                    }
        # Add ensemble metadata per config if available
        if is_ensemble and key in ensemble_metadata:
            entry["tier2_ensemble"] = ensemble_metadata[key].to_summary_dict()
        # Add Phase 5 extended evaluation results
        if key in consensus_results:
            entry["consensus"] = consensus_results[key].to_dict()
        if key in trajectory_results:
            ts = trajectory_results[key]
            entry["trajectory"] = {
                "overall_score": round(ts.overall_score, 4),
                "decision_coherence": round(ts.decision_coherence, 4),
                "information_utilization": round(ts.information_utilization, 4),
                "scope_management": round(ts.scope_management, 4),
                "stage_transition_quality": round(ts.stage_transition_quality, 4),
                "llm_evaluated": ts.llm_evaluated,
                "decision_points": len(ts.decision_points),
            }
        if key in adversarial_results:
            entry["adversarial"] = adversarial_results[key].to_dict()
        if key in benchmark_results:
            entry["benchmark"] = benchmark_results[key].to_dict()
        if key in agent_metrics_results:
            entry["agent_metrics"] = agent_metrics_results[key]
        save_data["results"][key] = entry

    # Add evaluation observability summary if collector was used
    if eval_collector is not None:
        obs_summary = eval_collector.get_summary()
        save_data["evaluation_observability"] = obs_summary

        # Print observability summary
        llm_stats = obs_summary.get("llm", {})
        if llm_stats.get("total_calls", 0) > 0:
            print("\n" + "=" * 70)
            print("EVALUATION OBSERVABILITY SUMMARY")
            print("=" * 70)
            print(f"  Total LLM calls: {llm_stats['total_calls']}")
            print(f"  Total tokens:    {llm_stats['total_tokens']}")
            print(f"  Total cost:      ${llm_stats['total_cost_usd']:.4f}")
            print(f"  Avg latency:     {llm_stats['avg_latency_seconds']:.2f}s")
            print(f"  Errors:          {llm_stats['error_count']}")

            by_stage = obs_summary.get("by_stage", {})
            if by_stage:
                print(f"\n  Per-dimension breakdown:")
                for stage_name, stage_data in sorted(by_stage.items()):
                    s_llm = stage_data.get("llm", {})
                    if s_llm.get("total_calls", 0) > 0:
                        print(f"    {stage_name}: {s_llm['total_calls']} calls, "
                              f"${s_llm['total_cost_usd']:.4f}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2)

    print(f"\n\nResults saved to: {output_path}")

    # Also save Tier-1-only results for backward compatibility
    t1_output_path = Path(__file__).parent / "evaluation_results.json"
    with open(t1_output_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "total_modes": len(teams),
            "modes_with_logs": len(results),
            "results": {
                key: {
                    "team": results[key]["team"],
                    "mode": results[key]["mode"],
                    "metrics": results[key]["tier1"],
                    "summary": results[key]["log"].get("summary", {}),
                }
                for key in results
            },
            "radar_data": {
                key: [results[key]["tier1"].get(dim, 0) for dim in dimensions]
                for key in results
            },
            "dimensions": dimensions,
        }, f, indent=2)

    # =====================================================================
    # LATEX OUTPUT
    # =====================================================================
    print("\n" + "=" * 70)
    print("LATEX TABLE VALUES")
    print("=" * 70)

    for team, mode in teams:
        key = f"{team}_{mode}"
        if key in results:
            scores = results[key][score_key]
            t2_info = results[key].get("tier2", {})
            print(f"% {team} - {mode}")
            for dim in dimensions:
                s = scores.get(dim, 0)
                tier_label = "T2" if (dim in tier2_dims and dim in t2_info) else "T1"
                s_txt = f"{s:.3f}" if s is not None else "n/a"
                print(f"%   {dim}: {s_txt} [{tier_label}]")
            print()

    # Nothing evaluated, or a requested Tier-2 evaluation incomplete, makes the command fail.
    if not results:
        print("No execution log was found; nothing was evaluated.", file=sys.stderr)
        sys.exit(1)
    if args.tier2 or args.tier2_ensemble:
        from evaluation.scoring.rubrics import LLM_EVALUATED_DIMENSIONS
        incomplete = [k for k in results
                      if any(d not in (tier2_results.get(k) or {}) for d in LLM_EVALUATED_DIMENSIONS)]
        if incomplete:
            print(f"Tier-2 evaluation incomplete for: {', '.join(incomplete)}", file=sys.stderr)
            sys.exit(1)


# IdeationTeam key deliverable + its fallback name. A run is only scored if a
# FRESH one (written by THIS run) is present — see _stale_rep_reason.
_IDEATION_FINALIZED_FILES = (
    "finalized_research_questions_automated.json",
    "finalized_research_questions.json",
)


def _stale_rep_reason(team: str, run_dir: Path) -> Optional[str]:
    """Freshness/provenance guard for a single multirun rep.

    Returns a human-readable reason string when this rep must NOT be scored
    because its key deliverable was not produced by this run, else ``None``.

    Crashed IdeationTeam HITL reps (Integration stage failure) leave no fresh
    ``finalized_research_questions*.json``; the shared mode-dir or a sibling run
    can then supply a STALE/leftover file that the judge would happily score.
    We require the finalized file to (a) exist in the run dir and (b) be no older
    than the run's own ``started_at`` (recorded in ``execution_log.json``), i.e.
    it was actually written during this run. Only IdeationTeam is guarded; other
    teams have no finalized_research_questions deliverable and are untouched.
    """
    if team != "IdeationTeam":
        return None

    # Run start time: prefer the execution_log's recorded started_at (robust to
    # shutil.copy2 mtime preservation); fall back to the log file's own mtime.
    log_path = run_dir / "execution_log.json"
    started_ts: Optional[float] = None
    if log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                _log = json.load(f)
            _ts = _log.get("started_at")
            if _ts:
                started_ts = datetime.fromisoformat(_ts).timestamp()
        except Exception:
            started_ts = None
        if started_ts is None:
            try:
                started_ts = log_path.stat().st_mtime
            except OSError:
                started_ts = None

    finalized = next(
        (run_dir / n for n in _IDEATION_FINALIZED_FILES if (run_dir / n).exists()),
        None,
    )
    if finalized is None:
        return "no finalized_research_questions*.json produced this run (failed rep)"
    if started_ts is not None:
        try:
            mtime = finalized.stat().st_mtime
        except OSError:
            mtime = 0.0
        # 2s slack mirrors the runner's stale-copy guard (start_ts - 2).
        if mtime < started_ts - 2:
            return (
                f"stale {finalized.name} predates run start "
                f"(mtime {mtime:.0f} < started_at {started_ts:.0f}) — leftover/sibling file"
            )
    return None


def run_multirun_tier2(multirun_dir: str, judge_model: str, collector: Optional[Any] = None) -> Dict[str, Dict]:
    """Tier-2 LLM-as-reviewer over a MULTIRUN tree (``team/mode/run_NNN/``).

    Evaluates every rep with the (local, by default) judge and averages per config.
    Writes ``tier2_multirun.json`` into the multirun dir. Serve the judge first
    (``ollama/<served-name>`` -> vLLM on port 11434).

    Reps whose key deliverable is stale/missing (a crashed rep that did not write
    a fresh ``finalized_research_questions*.json``) are SKIPPED, not scored on a
    leftover/sibling file — see ``_stale_rep_reason``.
    """
    import statistics
    base = Path(multirun_dir)
    _ensure_imports(base)
    from evaluation.scoring.llm_evaluator import LLMEvaluator

    evaluator = LLMEvaluator(model=judge_model, temperature=0.0, collector=collector)
    print("\n" + "=" * 70)
    print(f"TIER 2 (multirun) — judge: {judge_model}")
    print("=" * 70)
    out: Dict[str, Dict] = {}
    from evaluation.scoring.rubrics import LLM_EVALUATED_DIMENSIONS
    n_errors = 0
    for team_dir in sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith("_")):
        team = team_dir.name
        for mode_dir in sorted(p for p in team_dir.iterdir() if p.is_dir()):
            mode = mode_dir.name
            run_dirs = sorted(mode_dir.glob("run_*"))
            if not run_dirs:
                continue
            per_run = []
            skipped = 0
            for rd in run_dirs:
                stale = _stale_rep_reason(team, rd)
                if stale:
                    skipped += 1
                    print(f"  {team}/{mode}/{rd.name}: SKIP (not scored) — {stale}")
                    continue
                try:
                    scores = evaluator.evaluate_workflow(team, mode, base, output_dir=rd)
                    if scores:
                        per_run.append(scores)
                        missing = [d for d in LLM_EVALUATED_DIMENSIONS if d not in scores]
                        if missing:
                            n_errors += 1
                            print(f"  {team}/{mode}/{rd.name}: ERROR dimensions not scored: {missing}")
                    else:
                        n_errors += 1
                        print(f"  {team}/{mode}/{rd.name}: ERROR no scores returned")
                except Exception as e:
                    n_errors += 1
                    print(f"  {team}/{mode}/{rd.name}: ERROR {e}")
            if not per_run:
                if skipped:
                    print(f"  {team}/{mode}: 0 scored reps ({skipped} skipped stale/failed)")
                continue
            # union over repetitions, so a partial first repetition does not drop dimensions
            dims = list(dict.fromkeys(d for r in per_run for d in r))
            agg = {}
            for d in dims:
                vals = [r[d].overall_score for r in per_run if d in r]
                if vals:
                    agg[d] = {
                        "mean": round(statistics.mean(vals), 4),
                        "stdev": round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0,
                        "n": len(vals),
                    }
            out[f"{team}_{mode}"] = agg
            _skip_note = f" ({skipped} skipped stale/failed)" if skipped else ""
            print(f"  {team}/{mode}: {len(per_run)} reps{_skip_note} · " +
                  ", ".join(f"{d}={agg[d]['mean']:.2f}" for d in dims if d in agg))
    # Two judges writing one filename would clobber each other —
    # the second job would silently erase the first judge's scores. Suffix by judge.
    judge_slug = re.sub(r"[^A-Za-z0-9.-]+", "_", judge_model.split("/")[-1]) or "judge"
    out_path = base / f"tier2_multirun_{judge_slug}.json"
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"judge_model": judge_model, "configs": out}, f, indent=2)
        print(f"\nTier-2 multirun scores -> {out_path}")
    except Exception as e:
        n_errors += 1
        print(f"(could not write {out_path}: {e})")
    run_multirun_tier2.last_errors = n_errors
    return out


def run_multirun_aggregation(multirun_dir: str) -> bool:
    """
    Aggregate scores from a multirun output directory.

    Reads execution_log.json from each run_NNN/ subdirectory,
    computes Tier-1 metrics, aggregates with CIs, and writes
    evaluation_results_multirun.json.
    """
    from evaluation.analysis.statistical_aggregator import StatisticalAggregator

    agg = StatisticalAggregator()
    multirun_path = Path(multirun_dir)

    print("=" * 70)
    print("MULTI-RUN AGGREGATION")
    print("=" * 70)
    print(f"Source: {multirun_path}")

    aggregated, effects = agg.aggregate_multirun_dir(multirun_path)

    if not aggregated:
        print("  No run data found. Check directory structure.")
        return False

    # Print summary
    dimensions = [
        "reliability", "correctness", "soundness",
        "efficiency", "scalability", "robustness",
        "transparency", "traceability", "reproducibility",
        "innovation_potential",
        "decision_quality", "economic_rigor",
        # V0.7 — 14-dimension evaluation
        "simulation_fidelity", "causal_validity",
    ]
    dim_headers = ["Rel", "Cor", "Snd", "Eff", "Scl", "Rob", "Trn", "Trc", "Rep", "Inn", "SimF", "CausV"]

    print(f"\n{'Config':<35} {'n':>3}", end="")
    for h in dim_headers:
        print(f" {h:>10}", end="")
    print()
    print("-" * 145)

    for config_key, agg_scores in sorted(aggregated.items()):
        short = config_key.replace("Team_Mode", " ").replace("Team_", " ")
        print(f"{short:<35} {agg_scores.n_runs:>3}", end="")
        for dim in dimensions:
            stats = agg_scores.dimensions.get(dim)
            if stats and stats.n > 1:
                ci_half = (stats.ci_upper - stats.ci_lower) / 2
                print(f" {stats.mean:.2f}±{ci_half:.2f}", end="")
            elif stats:
                print(f" {stats.mean:.3f}   ", end="")
            else:
                print("       --  ", end="")
        print()

    # Save results
    output_path = Path(__file__).parent / "evaluation_results_multirun.json"
    save_data = {
        "generated_at": datetime.now().isoformat(),
        "source_dir": str(multirun_path),
        "aggregated": {k: v.to_dict() for k, v in aggregated.items()},
        "factorial_effects": {k: v.to_dict() for k, v in effects.items()},
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nMulti-run results saved to: {output_path}")

    # Generate and print LaTeX tables
    print("\n" + "=" * 70)
    print("LATEX: MULTI-RUN SUMMARY TABLE")
    print("=" * 70)
    print(agg.generate_summary_table(aggregated))

    if effects:
        print("\n" + "=" * 70)
        print("LATEX: FACTORIAL EFFECTS TABLE")
        print("=" * 70)
        # Group effects by team for cleaner output
        print(agg.generate_effects_table(effects))
    return True


if __name__ == "__main__":
    main()
