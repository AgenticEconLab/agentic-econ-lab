# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Master Orchestrator for Automated Model Development (No Human-in-the-Loop, No FireCrawl)
This script runs all three ModelTeam stages sequentially in a fully automated pipeline.

Pipeline:
1. Theory Stage: Develop theoretical frameworks from research questions and literature
2. Model Design Stage: Design computational models based on theoretical frameworks
3. Calibration Stage: Define calibration strategy and parameter identification

Input: Research questions + Literature review from LiteratureTeam
Output: Complete model specification with theoretical framework, design, and calibration strategy
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Add parent directories to path for shared imports
current_dir = Path(__file__).resolve().parent
agents_dir = current_dir.parent.parent.parent  # repository root
if str(agents_dir) not in sys.path:
    sys.path.insert(0, str(agents_dir))

from shared.instrumentation import WorkflowLogger
from shared.observability import MetricsCollector
from shared.console_ui import ConsoleUI
from shared.auto_input import auto_input, get_default
from shared.guardrails.budget_controller import BudgetController, BudgetExceededError
from shared.guardrails.schema_validator import SchemaValidator
from shared.memory.memory_manager import MemoryManager
from shared.tools.register_all import register_all_tools
from shared.llm_router import ModelRouter
from shared.model_config import load_model_config, stage_model
from shared.reliability.state_guard import StateGuard


# ============================================================================
# Environment Configuration
# ============================================================================

def get_env_path():
    """Find and return the path to the .env file in the repository root (the directory holding ael_config.yaml and run_ael_pipeline.py)."""
    current_dir = Path(__file__).resolve().parent
    while not ((current_dir / "ael_config.yaml").exists() and (current_dir / "run_ael_pipeline.py").exists()) and current_dir.parent != current_dir:
        current_dir = current_dir.parent
    if (current_dir / "ael_config.yaml").exists() and (current_dir / "run_ael_pipeline.py").exists():
        env_path = current_dir / ".env"
        if env_path.exists():
            return str(env_path)
    return None

env_path = get_env_path()
if env_path:
    load_dotenv(env_path)
else:
    load_dotenv()

# Set quiet mode for stage imports
os.environ["AGENT_QUIET_MODE"] = "true"
register_all_tools()


def main(cli_topic=None):
    """Main orchestrator that runs all three ModelTeam stages automatically."""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    output_dir = Path(script_dir)

    # Initialize Console UI
    ui = ConsoleUI(
        team="ModelTeam",
        mode="ModeNoWcNoHITL",
        total_stages=3
    )

    # Initialize workflow logger
    logger = WorkflowLogger(
        team="ModelTeam",
        mode="ModeNoWcNoHITL",
        output_dir=output_dir,
        framework="ael",
        quiet=True
    )

    # Initialize observability collector
    collector = MetricsCollector()
    logger.set_metrics_collector(collector)

    # Initialize budget controller (cost circuit-breaker)
    budget = BudgetController(max_cost_usd=5.0, max_tokens=500_000)

    # Load per-stage model configuration from ael_config.yaml.
    # Prefer a mode-local ael_config.yaml (sets V0.7 feature flags such as
    # falsifier_enabled) and export it so every downstream load_model_config()
    # / is_enabled() call (incl. the Falsifier hook) resolves the same file.
    _local_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ael_config.yaml")
    if os.path.exists(_local_cfg) and not os.environ.get("AEL_MODEL_CONFIG"):
        os.environ["AEL_MODEL_CONFIG"] = _local_cfg
    model_config = load_model_config()

    # Initialize state guard for inter-stage hash verification
    state_guard = StateGuard()

    # Stage timeout configuration (seconds per stage)
    stage_timeouts = {
        "TheoryStage": 300,
        "ModelDesignStage": 300,
        "CalibrationStage": 300,
    }

    # Initialize memory manager (cross-run persistence)
    memory_dir = str(output_dir / "memory")
    memory = MemoryManager(run_id=logger.execution_id, memory_dir=memory_dir)
    memory.set_context("team", "ModelTeam")
    memory.set_context("mode", "ModeNoWcNoHITL")

    # Track results
    workflow_success = True
    total_errors = 0
    frameworks = []
    models = []
    calibration_strategies = []
    questions = []

    # Check for required input files
    if cli_topic:
        research_questions_file = None
    else:
        research_questions_file = auto_input(
            "Enter path to research questions JSON file (or press Enter for default): ",
            default=get_default("model_research_questions_path")
        ).strip()
    if not cli_topic and (not research_questions_file or not os.path.exists(research_questions_file)):
        # Prefer ModelTeam's own committed research-questions input (stable, controlled eval
        # input); fall back to the IdeationTeam finalized questions for a live pipeline.
        default_paths = [
            "research_questions.json",
            "../../IdeationTeam/ael/ModeNoWcNoHITL/finalized_research_questions_automated.json",
            "../../../IdeationTeam/ael/ModeNoWcNoHITL/finalized_research_questions_automated.json",
        ]
        for path in default_paths:
            if os.path.exists(path):
                research_questions_file = path
                break

        if not research_questions_file or not os.path.exists(research_questions_file):
            ui.error("No research questions file found. Please provide a valid path.")
            sys.exit(1)

    literature_file = auto_input(
        "Enter path to literature review file (or press Enter to skip): ",
        default=get_default("questions_file_path")
    ).strip()
    if not literature_file:
        default_lit_paths = [
            "../../LiteratureTeam/ael/ModeNoWcNoHITL/literature_review.txt",
            "../../../LiteratureTeam/ael/ModeNoWcNoHITL/literature_review.txt",
            "literature_review.txt"
        ]
        for path in default_lit_paths:
            if os.path.exists(path):
                literature_file = path
                break

    # Recall similar past runs
    similar_runs = memory.recall_similar_runs(cli_topic or os.path.basename(research_questions_file or ""))
    if similar_runs:
        ui.info(f"Found {len(similar_runs)} similar past run(s) in memory")

    # Start execution tracking
    logger.start_execution(metadata={
        "research_questions_file": research_questions_file or "cli_topic",
        "literature_file": literature_file if literature_file else "Not provided"
    })

    # Print workflow header
    ui.workflow_header(
        topic=cli_topic or os.path.basename(research_questions_file or ""),
        execution_id=logger.execution_id,
        literature=os.path.basename(literature_file) if literature_file else "None"
    )

    # Show loaded API keys
    ui.show_api_keys()

    # ========== STAGE 1: THEORY DEVELOPMENT ==========
    ui.stage_start(1, "Theoretical Framework Development")
    logger.start_stage("Theory", 1)
    theory_file = "theory_output.json"

    try:
        from importlib import import_module
        stage1 = import_module('1-TheoryStage')

        with stage_model(model_config, "ModelTeam", "TheoryStage") as model:
            ui.info(f"Stage model: {model}")
            orchestrator1 = stage1.TheoryStageOrchestrator(collector=collector)

        # Load research questions
        if cli_topic:
            questions = [{"question": cli_topic, "priority_rank": 1, "priority_score": 0.95}]
        else:
            with open(research_questions_file, 'r', encoding='utf-8') as f:
                rq_data = json.load(f)

            if isinstance(rq_data, dict):
                questions = rq_data.get('questions', rq_data.get('final_questions', []))
            else:
                questions = rq_data

        # Load literature if available
        literature_data = None
        if literature_file and os.path.exists(literature_file):
            if literature_file.endswith('.json'):
                with open(literature_file, 'r', encoding='utf-8') as f:
                    literature_data = json.load(f)
            elif literature_file.endswith('.txt'):
                with open(literature_file, 'r', encoding='utf-8') as f:
                    literature_text = f.read()
                    literature_data = {"review_text": literature_text}

        # Run theory stage
        theory_output = orchestrator1.run_theory_pipeline(
            research_questions=questions,
            literature_batch=literature_data or {}
        )

        orchestrator1.save_theory_output(theory_file)
        frameworks = theory_output.theoretical_frameworks if theory_output else []

        # Count granular items for scalability
        theory_item_count = 0
        for fw in frameworks:
            theory_item_count += len(getattr(fw, 'assumptions', []))
            theory_item_count += len(getattr(fw, 'conceptual_components', []))
            theory_item_count += len(getattr(fw, 'mathematical_formulations', []))
            theory_item_count += len(getattr(fw, 'testable_predictions', []))
        theory_item_count = max(theory_item_count, len(frameworks))

        ui.agent_result("Theorist", items=theory_item_count, message=f"components across {len(frameworks)} frameworks")

        ui.stage_complete(1, status="success", items=theory_item_count,
                          duration=logger.get_stage_duration("Theory"), output_file=theory_file)
        logger.end_stage("Theory", status="success", item_count=theory_item_count,
                         output_files=[theory_file])
        state_guard.sign_output(theory_output.model_dump() if hasattr(theory_output, 'model_dump') else theory_output, stage_name="TheoryStage")

        # Schema validation after Stage 1
        try:
            if theory_output:
                SchemaValidator.validate_or_raise("ModelTeam", "TheoryStage", theory_output.model_dump())
        except (ValueError, Exception) as e:
            ui.warning(f"Schema validation: {e}")
            logger.log_error(error_type="SchemaValidation", message=str(e), recovered=True)

        # V0.7: DASES Falsifier pass — pair with Theorist output
        try:
            from shared.econ import run_falsifier_pass, attach_falsifier_reports
            with open(theory_file, "r", encoding="utf-8") as _f:
                _theory_data = json.load(_f)
            _falsifier_reports = run_falsifier_pass(_theory_data, collector=collector)
            if _falsifier_reports:
                attach_falsifier_reports(theory_file, _falsifier_reports)
                ui.info(f"Falsifier: {len(_falsifier_reports)} claim(s) evaluated")
        except Exception as _e:
            ui.warning(f"Falsifier hook skipped: {_e}")

        # Budget check after Stage 1
        budget_status = budget.check(collector, current_team="ModelTeam")
        for w in budget_status.warnings:
            ui.warning(w)

    except BudgetExceededError as e:
        ui.error(f"Budget exceeded after Stage 1: {e}")
        logger.log_error(error_type="BudgetExceededError", message=str(e), recovered=False)
        logger.end_stage("Theory", status="error", error_message=str(e))
        logger.end_execution(success=False)
        logger.save_execution_log()
        sys.exit(1)

    except Exception as e:
        ui.error(f"Stage 1 failed: {e}")
        logger.log_error(error_type=type(e).__name__, message=str(e), recovered=False)
        logger.end_stage("Theory", status="error", error_message=str(e))
        workflow_success = False
        total_errors += 1
        ui.workflow_complete(success=False, stages_completed=0, total_errors=total_errors)
        logger.end_execution(success=False)
        logger.save_execution_log()
        sys.exit(1)

    # ========== STAGE 2: MODEL DESIGN ==========
    ui.stage_start(2, "Computational Model Design")
    logger.start_stage("ModelDesign", 2)
    design_file = "model_design_output.json"

    try:
        stage2 = import_module('2-ModelDesignStage')
        with stage_model(model_config, "ModelTeam", "ModelDesignStage") as model:
            ui.info(f"Stage model: {model}")
            orchestrator2 = stage2.ModelDesignOrchestrator(collector=collector)

        with open(theory_file, 'r', encoding='utf-8') as f:
            theory_data = json.load(f)

        design_output = orchestrator2.run_design_pipeline(
            theory_output=theory_data
        )

        orchestrator2.save_design_output(design_file)
        models = design_output.formal_models if design_output else []

        # Count granular items for scalability
        design_item_count = 0
        for m in models:
            design_item_count += len(getattr(m, 'variables', []))
            design_item_count += len(getattr(m, 'parameters', []))
            design_item_count += len(getattr(m, 'equations', []))
            design_item_count += len(getattr(m, 'constraints', []))
        design_item_count = max(design_item_count, len(models))

        ui.agent_result("ModelDesigner", items=design_item_count, message=f"components across {len(models)} models")

        ui.stage_complete(2, status="success", items=design_item_count,
                          duration=logger.get_stage_duration("ModelDesign"), output_file=design_file)
        logger.end_stage("ModelDesign", status="success", item_count=design_item_count,
                         output_files=[design_file])

        # Schema validation after Stage 2
        try:
            if design_output:
                SchemaValidator.validate_or_raise("ModelTeam", "ModelDesignStage", design_output.model_dump())
        except (ValueError, Exception) as e:
            ui.warning(f"Schema validation: {e}")
            logger.log_error(error_type="SchemaValidation", message=str(e), recovered=True)

        # Budget check after Stage 2
        budget_status = budget.check(collector, current_team="ModelTeam")
        for w in budget_status.warnings:
            ui.warning(w)

    except BudgetExceededError as e:
        ui.error(f"Budget exceeded after Stage 2: {e}")
        logger.log_error(error_type="BudgetExceededError", message=str(e), recovered=False)
        logger.end_stage("ModelDesign", status="error", error_message=str(e))
        logger.end_execution(success=False)
        logger.save_execution_log()
        sys.exit(1)

    except Exception as e:
        ui.error(f"Stage 2 failed: {e}")
        logger.log_error(error_type=type(e).__name__, message=str(e), recovered=False)
        logger.end_stage("ModelDesign", status="error", error_message=str(e))
        workflow_success = False
        total_errors += 1
        ui.workflow_complete(success=False, stages_completed=1, total_errors=total_errors)
        logger.end_execution(success=False)
        logger.save_execution_log()
        sys.exit(1)

    # ========== STAGE 3: CALIBRATION STRATEGY ==========
    ui.stage_start(3, "Calibration & Parameter Identification")
    logger.start_stage("Calibration", 3)
    calibration_file = "calibration_output.json"

    try:
        stage3 = import_module('3-CalibrationStage')
        with stage_model(model_config, "ModelTeam", "CalibrationStage") as model:
            ui.info(f"Stage model: {model}")
            orchestrator3 = stage3.CalibrationOrchestrator(collector=collector)

        with open(design_file, 'r', encoding='utf-8') as f:
            design_data = json.load(f)

        calibration_output = orchestrator3.run_calibration_pipeline(
            model_design_output=design_data
        )

        orchestrator3.save_calibration_output(calibration_file)
        calibration_strategies = calibration_output.calibrated_models if calibration_output else []

        # Count granular items for scalability
        calib_item_count = 0
        for c in calibration_strategies:
            calib_item_count += len(getattr(c, 'empirical_targets', []))
            calib_item_count += len(getattr(c, 'calibrated_parameters', []))
            calib_item_count += len(getattr(c, 'model_moments', []))
            calib_item_count += len(getattr(c, 'robustness_checks', []))
        calib_item_count = max(calib_item_count, len(calibration_strategies))

        ui.agent_result("Calibrator", items=calib_item_count, message=f"components across {len(calibration_strategies)} strategies")

        ui.stage_complete(3, status="success", items=calib_item_count,
                          duration=logger.get_stage_duration("Calibration"), output_file=calibration_file)
        logger.end_stage("Calibration", status="success", item_count=calib_item_count,
                         output_files=[calibration_file])

        # Schema validation after Stage 3
        try:
            if calibration_output:
                SchemaValidator.validate_or_raise("ModelTeam", "CalibrationStage", calibration_output.model_dump())
        except (ValueError, Exception) as e:
            ui.warning(f"Schema validation: {e}")
            logger.log_error(error_type="SchemaValidation", message=str(e), recovered=True)

        # V0.7: CausalFM estimator + optional SimulationStage
        try:
            from shared.econ import (
                run_causal_fm_pass, attach_causal_estimate,
                maybe_run_simulation_stage,
            )
            with open(calibration_file, "r", encoding="utf-8") as _f:
                _calib_data = json.load(_f)
            _causal_estimate = run_causal_fm_pass(_calib_data, collector=collector)
            if _causal_estimate:
                attach_causal_estimate(calibration_file, _causal_estimate)
                ui.info(f"CausalFM: method={_causal_estimate.get('method')}, ate={_causal_estimate.get('ate')}")
            _sim_payload = maybe_run_simulation_stage(
                calibration_file, str(output_dir), mode="ModeNoWcNoHITL", collector=collector,
            )
            if _sim_payload and "simulation_result" in _sim_payload:
                ui.info(f"Simulation: scenario={_sim_payload['simulation_result'].get('scenario')}")
        except Exception as _e:
            ui.warning(f"Post-calibration hooks skipped: {_e}")

    except Exception as e:
        ui.error(f"Stage 3 failed: {e}")
        logger.log_error(error_type=type(e).__name__, message=str(e), recovered=False)
        logger.end_stage("Calibration", status="error", error_message=str(e))
        workflow_success = False
        total_errors += 1
        ui.workflow_complete(success=False, stages_completed=2, total_errors=total_errors)
        logger.end_execution(success=False)
        logger.save_execution_log()
        sys.exit(1)

    # ========== WORKFLOW COMPLETE ==========
    # Record trajectory in episodic memory
    from shared.memory.episodic import RunTrajectory
    trajectory = RunTrajectory(
        run_id=logger.execution_id,
        research_topic=cli_topic or os.path.basename(research_questions_file or ""),
        teams_completed=["ModelTeam"] if workflow_success else [],
        teams_failed=[] if workflow_success else ["ModelTeam"],
        success=workflow_success,
        mode="ModeNoWcNoHITL",
        artifact_names=[theory_file, design_file, calibration_file] if workflow_success else [],
    )
    memory.record_trajectory(trajectory)

    if workflow_success:
        memory.remember_fact(
            f"Built {len(models)} model(s) with {len(calibration_strategies)} calibration strategies",
            category="modeling", source="ModelTeam/ModeNoWcNoHITL",
        )

    memory.close()

    logger.end_execution(success=workflow_success)
    log_path = logger.save_execution_log()

    ui.workflow_complete(
        success=workflow_success, stages_completed=3, total_errors=total_errors,
        output_files=[theory_file, design_file, calibration_file],
        summary={
            "Research Questions": len(questions),
            "Frameworks": len(frameworks),
            "Models": len(models),
            "Calibrated": len(calibration_strategies)
        }
    )

    ui.observability_summary(collector)
    from shared.telemetry import export_telemetry
    export_telemetry(collector, str(output_dir), team="ModelTeam", mode="ModeNoWcNoHITL")

    ui.info(f"Execution log: {log_path}")

    # Propagate failure to the caller (multirun/runner) so a crashed/aborted stage is
    # NOT silently reported as success and its stale outputs scored. (V0.7 eval finding;
    # matches IdeationTeam's sys.exit(1) pattern.)
    if not workflow_success:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AEL ModelTeam — ModeNoWcNoHITL")
    parser.add_argument("--topic", type=str, default=None, help="Research topic (skips interactive prompt)")
    parser.add_argument("--model", type=str, default=None, help="Override LLM model for all stages")
    args = parser.parse_args()
    if args.model:
        os.environ["AEL_MODEL"] = args.model
    main(cli_topic=args.topic)
