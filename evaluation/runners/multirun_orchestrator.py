# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Multi-run orchestrator for repeated workflow execution.

Runs the same workflow configuration N times with isolated output directories,
producing a multirun_manifest.json for downstream statistical analysis.

Usage:
    python -m evaluation.runners.multirun_orchestrator \
        --team IdeationTeam \
        --mode ModeNoWcNoHITL \
        --runs 10 \
        --output-dir ./multirun_results

    # Run all modes for a team
    python -m evaluation.runners.multirun_orchestrator \
        --team IdeationTeam \
        --all-modes \
        --runs 5

    # Dry run to verify paths
    python -m evaluation.runners.multirun_orchestrator \
        --team IdeationTeam \
        --mode ModeNoWcNoHITL \
        --runs 3 \
        --dry-run
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .ael_runner import AELRunner
from .base import RunConfig


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_multirun(
    team: str,
    mode: str,
    n_runs: int,
    output_dir: Path,
    base_path: Optional[Path] = None,
    timeout_seconds: int = 3600,
    inputs: Optional[dict] = None,
    dry_run: bool = False,
) -> dict:
    """
    Execute a workflow N times with isolated outputs.

    Args:
        team: Team name (IdeationTeam, LiteratureTeam, ModelTeam, DataTeam)
        mode: Mode name (e.g., ModeNoWcNoHITL)
        n_runs: Number of repetitions
        output_dir: Base output directory
        base_path: Path to the repository root
        timeout_seconds: Timeout per run
        inputs: Override default inputs
        dry_run: If True, validate only without executing

    Returns:
        Dictionary with manifest data
    """
    if base_path is None:
        # Default: the repository root, located relative to this module
        base_path = Path(__file__).resolve().parent.parent.parent

    runner = AELRunner(base_path=base_path)

    # Validate team/mode
    if team not in runner.SUPPORTED_TEAMS:
        logger.error(f"Unsupported team: {team}. Supported: {runner.SUPPORTED_TEAMS}")
        sys.exit(1)

    supported_modes = runner.get_supported_modes(team)
    if mode not in supported_modes:
        logger.error(f"Unsupported mode: {mode}. Supported: {supported_modes}")
        sys.exit(1)

    # Use default inputs if not provided
    if inputs is None:
        team_config = runner.get_team_config(team)
        inputs = team_config.get("default_inputs", {})

    # Build output directory structure: output_dir/team/mode/
    run_output_dir = output_dir / team / mode

    config = RunConfig(
        team=team,
        mode=mode,
        inputs=inputs,
        output_dir=run_output_dir,
        timeout_seconds=timeout_seconds,
        capture_logs=True,
    )

    if dry_run:
        logger.info(f"[DRY RUN] Validating {team}/{mode}...")
        dry_result = runner.dry_run(config)
        logger.info(f"  Valid: {dry_result['valid']}")
        logger.info(f"  Workflow path: {dry_result['workflow_path']}")
        logger.info(f"  Entry point: {dry_result['entry_point']}")
        logger.info(f"  Expected stages: {dry_result['expected_stages']}")
        if dry_result['errors']:
            for err in dry_result['errors']:
                logger.error(f"  Error: {err}")
        return dry_result

    logger.info(f"Starting multi-run: {team}/{mode} x {n_runs}")
    logger.info(f"Output directory: {run_output_dir}")
    logger.info(f"Inputs: {json.dumps(inputs)}")
    logger.info(f"Timeout per run: {timeout_seconds}s")
    logger.info("=" * 60)

    results = runner.run_repeated(
        config=config,
        n_runs=n_runs,
        output_base_dir=run_output_dir,
    )

    # Print summary
    successful = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)
    durations = [r.duration_seconds for r in results if r.duration_seconds]
    avg_duration = sum(durations) / len(durations) if durations else 0

    logger.info("=" * 60)
    logger.info(f"Multi-run complete: {team}/{mode}")
    logger.info(f"  Successful: {successful}/{n_runs}")
    logger.info(f"  Failed: {failed}/{n_runs}")
    logger.info(f"  Avg duration: {avg_duration:.1f}s")
    logger.info(f"  Manifest: {run_output_dir / 'multirun_manifest.json'}")

    # Read and return manifest
    manifest_path = run_output_dir / "multirun_manifest.json"
    manifest: dict = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    manifest["n_failed_runs"] = failed
    return manifest


def run_all_modes(
    team: str,
    n_runs: int,
    output_dir: Path,
    base_path: Optional[Path] = None,
    timeout_seconds: int = 3600,
    dry_run: bool = False,
) -> List[dict]:
    """Run multi-run for all modes of a given team."""
    if base_path is None:
        base_path = Path(__file__).resolve().parent.parent.parent

    runner = AELRunner(base_path=base_path)
    modes = runner.get_supported_modes(team)

    results = []
    for mode in modes:
        logger.info(f"\n{'#' * 60}")
        logger.info(f"# {team} / {mode}")
        logger.info(f"{'#' * 60}")

        manifest = run_multirun(
            team=team,
            mode=mode,
            n_runs=n_runs,
            output_dir=output_dir,
            base_path=base_path,
            timeout_seconds=timeout_seconds,
            dry_run=dry_run,
        )
        results.append(manifest)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Multi-run orchestrator for repeated workflow execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run IdeationTeam ModeNoWcNoHITL 10 times
  python -m evaluation.runners.multirun_orchestrator \\
      --team IdeationTeam --mode ModeNoWcNoHITL --runs 10

  # Run all modes for LiteratureTeam 5 times each
  python -m evaluation.runners.multirun_orchestrator \\
      --team LiteratureTeam --all-modes --runs 5

  # Dry run to check paths
  python -m evaluation.runners.multirun_orchestrator \\
      --team IdeationTeam --mode ModeNoWcNoHITL --runs 3 --dry-run
        """,
    )
    parser.add_argument(
        "--team", required=True,
        choices=["IdeationTeam", "LiteratureTeam", "ModelTeam", "DataTeam"],
        help="Team to run",
    )
    parser.add_argument(
        "--mode",
        help="Specific mode to run (e.g., ModeNoWcNoHITL)",
    )
    parser.add_argument(
        "--all-modes", action="store_true",
        help="Run all modes for the specified team",
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Number of repetitions (default: 3)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./multirun_results"),
        help="Base output directory (default: ./multirun_results)",
    )
    parser.add_argument(
        "--base-path", type=Path, default=None,
        help="Path to the repository root (auto-detected if omitted)",
    )
    parser.add_argument(
        "--timeout", type=int, default=3600,
        help="Timeout per run in seconds (default: 3600)",
    )
    parser.add_argument(
        "--inputs", type=str, default=None,
        help='JSON string of inputs (e.g., \'{"research_topic":"AI economics"}\')',
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate configuration without executing",
    )

    args = parser.parse_args()

    if not args.mode and not args.all_modes:
        parser.error("Either --mode or --all-modes is required")

    inputs = json.loads(args.inputs) if args.inputs else None

    if args.all_modes:
        manifests = run_all_modes(
            team=args.team,
            n_runs=args.runs,
            output_dir=args.output_dir,
            base_path=args.base_path,
            timeout_seconds=args.timeout,
            dry_run=args.dry_run,
        )
    else:
        manifests = [run_multirun(
            team=args.team,
            mode=args.mode,
            n_runs=args.runs,
            output_dir=args.output_dir,
            base_path=args.base_path,
            timeout_seconds=args.timeout,
            inputs=inputs,
            dry_run=args.dry_run,
        )]

    # A failed repetition makes the command fail, so batch jobs report it.
    n_failed = sum((m or {}).get("n_failed_runs", 0) for m in manifests)
    if n_failed:
        logger.error(f"{n_failed} run(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
