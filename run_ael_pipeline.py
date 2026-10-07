# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Top-level entry point for the cross-team research pipeline.

Runs all configured teams (Ideation → Literature → Model → Data) end-to-end
with artifact passing, budget control, and message bus notifications.

Usage:
    python run_ael_pipeline.py --topic "AI in economics"
    python run_ael_pipeline.py --topic "Fiscal policy" --config pipeline/configs/ideation_only.yaml
    python run_ael_pipeline.py --topic "Monetary policy" --mode ModeNoWcNoHITL
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

# Ensure codes is on sys.path
agents_dir = str(Path(__file__).resolve().parent)
if agents_dir not in sys.path:
    sys.path.insert(0, agents_dir)

from dotenv import load_dotenv

from pipeline.pipeline_config import load_pipeline_config, PipelineConfig, StageConfig, BudgetConfig
from pipeline.research_pipeline import ResearchPipelineOrchestrator
from pipeline.team_runners import (
    run_ideation_team,
    run_literature_team,
    run_model_team,
    run_data_team,
    run_estimation_team,
    run_reporting_team,
    run_code_team,
)
from shared.model_config import load_model_config
from shared.reliability.checkpoint import CheckpointManager
from shared.tools.register_all import register_all_tools


# Built-in fallback if ael_config.yaml is missing or has no pipeline_modes block.
# The BEST configuration is the default: the flagship
# WithWcWithHITL for the four original teams — WithHITL is headless-safe because the
# LLM-economist committee resolves every checkpoint (the pipeline defaults AEL_HITL_MODE
# to llm_economist) — and NoWcWithHITL for the new §4.5-§4.7 teams, which have no Wc axis.
_DEFAULT_PIPELINE_MODES = {
    "IdeationTeam": "ModeWithWcWithHITL",
    "LiteratureTeam": "ModeWithWcWithHITL",
    "ModelTeam": "ModeWithWcWithHITL",
    # DataTeam is organized by data source, not the 2x2 matrix; its HITL checkpoints are
    # gated by the resolved HITL mode (committee in pipeline runs), not the mode name.
    "DataTeam": "open_source_api",
    "EstimationTeam": "ModeNoWcWithHITL",
    "ReportingTeam": "ModeNoWcWithHITL",
    "CodeTeam": "ModeNoWcWithHITL",
}


def _resolve_pipeline_modes(
    mode: Optional[str] = None,
    data_mode: Optional[str] = None,
) -> Dict[str, str]:
    """Resolve per-team modes from ael_config.yaml, applying CLI overrides.

    Priority (highest to lowest):
      1. Explicit ``mode`` arg (applied to Ideation/Literature/Model)
         and ``data_mode`` arg (applied to DataTeam).
      2. ``pipeline_modes:`` block in ``ael_config.yaml``.
      3. Built-in defaults (all WithWcWithHITL + open_source_api).
    """
    cfg = load_model_config() or {}
    yaml_modes = cfg.get("pipeline_modes") or {}

    resolved = dict(_DEFAULT_PIPELINE_MODES)
    for team, value in yaml_modes.items():
        if team in resolved and isinstance(value, str) and value:
            resolved[team] = value

    # --mode overrides only the Wc-axis teams (matches the CLI help text); the new
    # §4.5-§4.7 teams keep their NoWcWithHITL default unless set via pipeline_modes.
    if mode:
        for team in ("IdeationTeam", "LiteratureTeam", "ModelTeam"):
            resolved[team] = mode
    if data_mode:
        resolved["DataTeam"] = data_mode

    return resolved


def build_default_config(
    mode: Optional[str] = None,
    data_mode: Optional[str] = None,
) -> PipelineConfig:
    """Build the default 4-team pipeline config programmatically.

    Per-team modes come from ``ael_config.yaml`` (``pipeline_modes:``); the
    optional ``mode`` and ``data_mode`` arguments override them when provided.
    """
    modes = _resolve_pipeline_modes(mode=mode, data_mode=data_mode)

    return PipelineConfig(
        name="full_research_pipeline",
        mode=modes["IdeationTeam"],
        stages=[
            StageConfig(
                team="IdeationTeam",
                mode=modes["IdeationTeam"],
                inputs=["research_topic"],
                outputs=["research_questions"],
                depends_on=[],
            ),
            StageConfig(
                team="LiteratureTeam",
                mode=modes["LiteratureTeam"],
                inputs=["research_questions"],
                outputs=["literature_review", "knowledge_graph", "gap_analysis"],
                depends_on=["IdeationTeam"],
            ),
            StageConfig(
                team="ModelTeam",
                mode=modes["ModelTeam"],
                inputs=["research_questions", "literature_review"],
                outputs=["model_specification"],
                depends_on=["IdeationTeam", "LiteratureTeam"],
            ),
            StageConfig(
                team="DataTeam",
                mode=modes["DataTeam"],
                # model_design + data_requirements_spec carry the data requirements
                inputs=["research_questions", "model_specification", "model_design",
                        "data_requirements_spec"],
                outputs=["validated_dataset", "data_source"],
                depends_on=["ModelTeam"],
            ),
            StageConfig(
                team="EstimationTeam",
                mode=modes["EstimationTeam"],
                # executable_model / model_design are optional (CodeTeam may be off);
                # depends_on CodeTeam only orders it first, a disabled CodeTeam is skipped.
                inputs=["research_questions", "model_specification", "data_source",
                        "executable_model", "model_design"],
                outputs=["estimation_results"],
                depends_on=["DataTeam", "CodeTeam"],
            ),
            # Optional node: enabled via AEL_CODE_TEAM_ENABLED=1.
            StageConfig(
                team="CodeTeam",
                mode=modes["CodeTeam"],
                inputs=["model_design", "model_specification"],
                outputs=["executable_model"],
                depends_on=["ModelTeam"],
                enabled=os.environ.get("AEL_CODE_TEAM_ENABLED", "").lower() in ("1", "true", "yes"),
            ),
            StageConfig(
                team="ReportingTeam",
                mode=modes["ReportingTeam"],
                inputs=["research_questions", "literature_review", "model_specification",
                        "data_source", "estimation_results"],
                outputs=["research_report"],
                depends_on=["EstimationTeam"],
            ),
        ],
        budget=BudgetConfig(
            max_total_cost_usd=5.0,
            max_per_team_cost_usd=2.0,
            # In a full run the 4 core teams alone used 627k tokens (5 finals, 344-paper
            # corpus, 3-model committee), so the 4-team-era 500k ceiling killed the
            # deferred teams. On local vLLM tokens are a runaway guard, not a cost — size
            # for the 7-team committee pipeline with headroom.
            max_total_tokens=2_000_000,
        ),
        output_dir="pipeline_output",
        # Escape hatch for resumes: the loop's decisions are not checkpointed, so a resumed
        # run would repeat them. AEL_FEASIBILITY_LOOP=0 skips the loop when its artifacts
        # already exist in the run directory.
        feasibility_loop=os.environ.get("AEL_FEASIBILITY_LOOP", "1").lower()
        not in ("0", "false", "no"),
    )


# Default topic: the seed topic of the runs reported in the paper. `--topic auto` instead runs
# field-balanced direction discovery (shared/tools/direction_scout.py).
_DEV_STAGE_DEFAULT_TOPIC = "Agent-based and heterogeneous-agent modeling in macroeconomics and monetary policy"


def main():
    """Parse arguments and run the pipeline."""
    load_dotenv()
    register_all_tools()

    parser = argparse.ArgumentParser(
        description="Run the AEL research pipeline end-to-end."
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=_DEV_STAGE_DEFAULT_TOPIC,
        help=(
            "Research topic to investigate (default: the seed topic of the runs reported in "
            "the paper). Pass 'auto' for field-balanced direction discovery: a slate across "
            "the JEL top-level fields from live signals, ranked by the LLM-Economist committee."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to pipeline YAML config (default: the built-in seven-team config; CodeTeam is enabled by AEL_CODE_TEAM_ENABLED=1).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help=(
            "Override the per-team execution mode for Ideation/Literature/Model. "
            "When omitted, modes are read from ael_config.yaml (pipeline_modes:)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: pipeline_output).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from the last checkpoint if available.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override LLM model for all teams/stages (e.g., ollama/llama-4-scout).",
    )
    parser.add_argument(
        "--data-mode",
        type=str,
        default=None,
        choices=["open_source_api", "premium_subscribed", "user_uploaded"],
        help=(
            "Override the DataTeam workflow type. When omitted, value is read "
            "from ael_config.yaml (pipeline_modes.DataTeam)."
        ),
    )
    parser.add_argument(
        "--hitl-mode",
        type=str,
        default=None,
        choices=["interactive", "auto", "llm_economist"],
        help=(
            "How to resolve the feasibility-review HITL checkpoint in a full run. "
            "Default for a full pipeline run is 'llm_economist' (the reasoned 3-model "
            "committee) — a headless autonomous run has no human at the console. Use "
            "'auto' for a cheap infrastructure smoke-test (documented-limitation default, "
            "no model re-runs), or 'interactive' for a real human. Explicit AEL_HITL_MODE "
            "/ AUTO_HITL_MODE env vars take precedence."
        ),
    )
    args = parser.parse_args()

    # Global model override via CLI
    if args.model:
        os.environ["AEL_MODEL"] = args.model

    # Resolve the feasibility-review HITL mode for this full run. Precedence:
    #   --hitl-mode  >  an already-set AEL_HITL_MODE / AUTO_HITL_MODE  >  default.
    # The default for a *full pipeline run* is the LLM-Economist committee, NOT the
    # global 'interactive' default: there is no human at a headless autonomous run, so
    # 'interactive' would merely block each prompt until it times out. (Standalone team
    # runs keep the interactive default — this override is scoped to the pipeline.)
    if args.hitl_mode:
        os.environ["AEL_HITL_MODE"] = args.hitl_mode
    elif "AEL_HITL_MODE" not in os.environ and os.getenv("AUTO_HITL_MODE", "").lower() not in ("true", "1", "yes"):
        os.environ["AEL_HITL_MODE"] = "llm_economist"
    print(f"[pipeline] HITL resolution mode: {os.environ.get('AEL_HITL_MODE', 'auto (AUTO_HITL_MODE)')}")

    # Fair direction discovery: with --topic auto, the run's direction
    # is CHOSEN from a field-balanced slate by the committee — never imposed by a hardcoded
    # seed. The full slate + ballots are registered as a provenance artifact.
    direction_provenance = None
    if str(args.topic).strip().lower() == "auto":
        from shared.tools.direction_scout import choose_direction, discover_directions
        print("[direction-scout] discovering a field-balanced slate (JEL top-level fields)...")
        slate = discover_directions()
        print(f"[direction-scout] slate: {len(slate['candidates'])} candidate(s), "
              f"grounded={slate['grounded']}")
        for c in slate["candidates"]:
            print(f"  [{c['field_code']}] {c['direction'][:110]}")
        decision = choose_direction(slate)
        if not decision.get("chosen"):
            print(f"[direction-scout] FAILED: {decision.get('error')} — refusing to fall "
                  "back to a hardcoded seed; rerun with an explicit --topic.")
            return 1
        args.topic = decision["chosen"]["direction"]
        direction_provenance = {"slate": slate, "decision": decision}
        print(f"[direction-scout] chosen ({decision['method']}): "
              f"[{decision['chosen']['field_code']}] {args.topic}")
        for b in decision.get("ballots", []):
            print(f"    ballot {b.get('lens')}/{b.get('model', '')[:30]}: "
                  f"ranking={b.get('ranking')} {str(b.get('rationale', ''))[:90]}")

    # Load or build config
    if args.config:
        config = load_pipeline_config(args.config)
    else:
        config = build_default_config(mode=args.mode, data_mode=args.data_mode)

    if args.output_dir:
        config.output_dir = args.output_dir

    # Initialize checkpoint manager (before pipeline so we can check resume)
    checkpoint_dir = os.path.join(config.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_mgr = CheckpointManager(checkpoint_dir=checkpoint_dir)

    # Check for resume — pipeline.run() interprets resume_from as the
    # team to START at, so we map "last completed" to "next in order"
    # and seed the artifact_store with completed teams' outputs.
    resume_from = None
    resume_skip_teams = None
    initial_artifacts: Dict[str, Any] = {}
    if direction_provenance:
        initial_artifacts["direction_provenance"] = direction_provenance
    if args.resume and checkpoint_mgr.can_resume():
        execution_order = config.get_execution_order()
        completed_team_outputs = {}
        # Walk in chronological order; keep the latest non-empty outputs
        # (the post-pipeline save writes a checkpoint with no `outputs`
        # key, which we want to skip).
        for cp in checkpoint_mgr.list_checkpoints():
            if cp.stage_name not in execution_order:
                continue
            outputs = (cp.data or {}).get("outputs") or {}
            if outputs:
                completed_team_outputs[cp.stage_name] = outputs

        if completed_team_outputs:
            # Last completed team in execution order
            completed_in_order = [t for t in execution_order if t in completed_team_outputs]
            last_completed = completed_in_order[-1]
            last_idx = execution_order.index(last_completed)
            resume_skip_teams = set(completed_in_order)   # skip EXACTLY the checkpointed teams
            if last_idx + 1 < len(execution_order):
                resume_from = execution_order[last_idx + 1]
                print(f"  Resuming: skipping {', '.join(completed_in_order)}; starting at {resume_from}")
            else:
                print(f"  Resuming: all teams already completed in prior run ({', '.join(completed_in_order)}); nothing to do")

            # Flatten skipped teams' outputs into initial_artifacts so
            # downstream teams can read them via upstream_artifacts.
            for outputs in completed_team_outputs.values():
                if isinstance(outputs, dict):
                    initial_artifacts.update(outputs)

    # Per-team checkpoint callback
    def on_team_done(team_name, team_result):
        if team_result.success:
            checkpoint_mgr.save(
                stage_name=team_name,
                data={"outputs": team_result.outputs},
                metadata={"duration": team_result.duration_sec},
            )

    # Create orchestrator with checkpoint callback
    pipeline = ResearchPipelineOrchestrator(config, on_team_complete=on_team_done)

    # Register team runners
    pipeline.register_team_runner("IdeationTeam", run_ideation_team)
    pipeline.register_team_runner("LiteratureTeam", run_literature_team)
    pipeline.register_team_runner("ModelTeam", run_model_team)
    pipeline.register_team_runner("DataTeam", run_data_team)
    pipeline.register_team_runner("EstimationTeam", run_estimation_team)
    pipeline.register_team_runner("ReportingTeam", run_reporting_team)
    pipeline.register_team_runner("CodeTeam", run_code_team)

    # Print header
    print("=" * 70)
    print("  AEL Research Pipeline")
    print("=" * 70)
    print(f"  Topic:    {args.topic}")
    print(f"  Config:   {args.config or 'default (4-team)'}")
    stage_modes = {s.team: s.mode for s in config.stages}
    if len(set(stage_modes.values())) == 1:
        print(f"  Mode:     {next(iter(stage_modes.values()))}")
    else:
        print(f"  Modes:")
        for team in config.get_execution_order():
            print(f"            {team:16s} {stage_modes.get(team, '?')}")
    print(f"  Teams:    {' → '.join(config.get_execution_order())}")
    print(f"  Budget:   ${config.budget.max_total_cost_usd:.2f}")
    print(f"  Output:   {config.output_dir}")
    print(f"  Run ID:   {pipeline.pipeline_run_id}")
    if resume_from:
        print(f"  Resume:   from {resume_from}")
    print("=" * 70)
    print()

    # Run pipeline
    start = time.time()
    result = pipeline.run(
        research_topic=args.topic,
        initial_artifacts=initial_artifacts or None,
        resume_from=resume_from,
        skip_teams=resume_skip_teams if resume_from else None,
    )
    elapsed = time.time() - start

    # Save checkpoint after pipeline completes
    if result.teams_completed:
        last_team = result.teams_completed[-1]
        checkpoint_mgr.save(
            stage_name=last_team,
            data={"teams_completed": result.teams_completed, "success": result.success},
            metadata={"duration": elapsed, "run_id": pipeline.pipeline_run_id},
        )

    # Print results
    print()
    print("=" * 70)
    if result.success:
        print("  PIPELINE COMPLETE")
    else:
        print("  PIPELINE FAILED")
    print("=" * 70)
    print(f"  Duration:   {elapsed:.1f}s")
    print(f"  Completed:  {', '.join(result.teams_completed) or 'none'}")
    if result.teams_failed:
        print(f"  Failed:     {', '.join(result.teams_failed)}")
    print(f"  Artifacts:  {len(result.artifact_names)}")
    print(f"  Messages:   {pipeline.message_bus.message_count()}")

    # Show per-team results
    for team_name, team_result in result.team_results.items():
        status = "OK" if team_result.success else "FAIL"
        print(f"  [{status}] {team_name}: {team_result.duration_sec:.1f}s", end="")
        if team_result.error:
            print(f" -- {team_result.error[:60]}")
        else:
            print(f" -- {len(team_result.outputs)} artifact(s)")

    print("=" * 70)

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
