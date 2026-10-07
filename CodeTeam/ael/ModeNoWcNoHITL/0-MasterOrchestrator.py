# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Master Orchestrator for Code Implementation & Debugging (No Web-crawl, WITH Human-in-the-Loop)
Runs all three CodeTeam stages sequentially.

Pipeline:
1. Code Generation Stage: modules DERIVED from the parsed equation system (no LLM code)
2. Validation Stage: executed check battery (compiles, finite, steady state, sensitivity)
3. Experimentation Stage: comparative statics around the calibrated point

Input:  model_design_output.json (+ calibration_output.json)
Output: generation_output.json, validation_output.json, experimentation_output.json,
        generated_models/*.py
"""

import argparse
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

current_dir = Path(__file__).resolve().parent
agents_dir = current_dir.parent.parent.parent  # repository root
if str(agents_dir) not in sys.path:
    sys.path.insert(0, str(agents_dir))

from shared.instrumentation import WorkflowLogger
from shared.observability import MetricsCollector
from shared.console_ui import ConsoleUI
from shared.auto_input import auto_input, get_default
from shared.model_config import load_model_config, stage_model
from shared.tools.register_all import register_all_tools


def get_env_path():
    d = Path(__file__).resolve().parent
    while not ((d / "ael_config.yaml").exists() and (d / "run_ael_pipeline.py").exists()) and d.parent != d:
        d = d.parent
    if (d / "ael_config.yaml").exists() and (d / "run_ael_pipeline.py").exists():
        p = d / ".env"
        if p.exists():
            return str(p)
    return None


env_path = get_env_path()
if env_path:
    load_dotenv(env_path)
else:
    load_dotenv()

os.environ["AGENT_QUIET_MODE"] = "true"
register_all_tools()

MODE = "ModeNoWcNoHITL"
HITL = "WithHITL" in MODE


def _load_json(path):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def main(design_path=None, calibration_path=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    output_dir = Path(script_dir)

    ui = ConsoleUI(team="CodeTeam", mode=MODE, total_stages=3)
    logger = WorkflowLogger(team="CodeTeam", mode=MODE, output_dir=output_dir,
                            framework="ael", quiet=True)
    collector = MetricsCollector()
    logger.set_metrics_collector(collector)
    model_config = load_model_config()

    design_path = (design_path or os.environ.get("AEL_CODE_MODEL_DESIGN")
                   or auto_input("Path to model_design_output.json: ",
                                 default=get_default("code_model_design_path")).strip()
                   or "../../ModelTeam/ael/ModeNoWcNoHITL/model_design_output.json")
    calibration_path = (calibration_path or os.environ.get("AEL_CODE_CALIBRATION")
                        or "../../ModelTeam/ael/ModeNoWcNoHITL/calibration_output.json")
    design = _load_json(design_path)
    calibration = _load_json(calibration_path)
    if not design:
        ui.error(f"No model design at '{design_path}' — nothing to generate from.")
        sys.exit(1)

    logger.start_execution(metadata={"model_design_file": design_path,
                                     "calibration_file": calibration_path, "hitl": HITL})
    ui.workflow_header(topic=os.path.basename(design_path), execution_id=logger.execution_id)

    workflow_success = True
    total_errors = 0
    from importlib import import_module

    # ========== STAGE 1: CODE GENERATION ==========
    ui.stage_start(1, "Derived Code Generation (sympy -> module)")
    logger.start_stage("CodeGeneration", 1)
    generation_file = "generation_output.json"
    try:
        stage1 = import_module("1-CodeGenerationStage")
        with stage_model(model_config, "CodeTeam", "CodeGenerationStage") as model:
            ui.info(f"Stage model: {model}")
            orch1 = stage1.CodeGenerationOrchestrator(collector=collector)
        results1 = orch1.run_generation_pipeline(design, calibration,
                                                 modules_dir=str(output_dir / "generated_models"))
        orch1.save_generation_output(generation_file)
        verdicts = [r.verdict for r in results1]
        if HITL:
            review = auto_input(
                "Review of the generated module interfaces (recorded): ",
                default=get_default("code_generation_review"),
                context={"stage": "CodeGenerationStage",
                         "verdicts": verdicts,
                         "reasons": [r.reason for r in results1]},
            ).strip()
            if review:
                logger.log_error(error_type="HITLReview", message=review[:400], recovered=True)
        ui.agent_result("Coder", items=len(results1), message=f"verdicts: {verdicts}")
        ui.stage_complete(1, status="success", items=len(results1),
                          duration=logger.get_stage_duration("CodeGeneration"),
                          output_file=generation_file)
        logger.end_stage("CodeGeneration", status="success", item_count=len(results1),
                         output_files=[generation_file])
    except Exception as e:
        ui.error(f"Stage 1 failed: {e}")
        logger.log_error(error_type=type(e).__name__, message=str(e), recovered=False)
        logger.end_stage("CodeGeneration", status="error", error_message=str(e))
        logger.end_execution(success=False)
        logger.save_execution_log()
        ui.workflow_complete(success=False, stages_completed=0, total_errors=1)
        sys.exit(1)

    # ========== STAGE 2: VALIDATION ==========
    ui.stage_start(2, "Executed Validation Battery")
    logger.start_stage("Validation", 2)
    validation_file = "validation_output.json"
    try:
        stage2 = import_module("2-ValidationStage")
        with stage_model(model_config, "CodeTeam", "ValidationStage") as model:
            ui.info(f"Stage model: {model}")
            orch2 = stage2.CodeValidationOrchestrator(collector=collector)
        results2 = orch2.run_validation_pipeline(_load_json(generation_file),
                                                 workdir=str(output_dir / "generated_models"))
        orch2.save_validation_output(validation_file)
        ui.agent_result("Debugger", items=len(results2),
                        message=f"verdicts: {[v.verdict for v in results2]}")
        ui.stage_complete(2, status="success", items=len(results2),
                          duration=logger.get_stage_duration("Validation"),
                          output_file=validation_file)
        logger.end_stage("Validation", status="success", item_count=len(results2),
                         output_files=[validation_file])
    except Exception as e:
        ui.error(f"Stage 2 failed: {e}")
        logger.log_error(error_type=type(e).__name__, message=str(e), recovered=False)
        logger.end_stage("Validation", status="error", error_message=str(e))
        logger.end_execution(success=False)
        logger.save_execution_log()
        ui.workflow_complete(success=False, stages_completed=1, total_errors=1)
        sys.exit(1)

    # ========== STAGE 3: EXPERIMENTATION ==========
    ui.stage_start(3, "Comparative Statics")
    logger.start_stage("Experimentation", 3)
    experimentation_file = "experimentation_output.json"
    version_manifest_file = "code_version_manifest.json"
    try:
        stage3 = import_module("3-ExperimentationStage")
        with stage_model(model_config, "CodeTeam", "ExperimentationStage") as model:
            ui.info(f"Stage model: {model}")
            orch3 = stage3.ExperimentationOrchestrator(collector=collector)
        results3 = orch3.run_experimentation_pipeline(
            _load_json(generation_file), _load_json(validation_file),
            workdir=str(output_dir / "generated_models"))
        orch3.save_experimentation_output(experimentation_file)
        orch3.save_version_manifest(version_manifest_file)
        n_sweeps = sum(len(e.statics) for e in results3)
        ui.agent_result("BatchRunner", items=max(n_sweeps, 1),
                        message=f"verdicts: {[e.verdict for e in results3]}")
        n_versioned = len(orch3.version_manifest.get("model_versions", []))
        ui.agent_result("VersionManager", items=n_versioned, message=version_manifest_file)
        ui.agent_result("Optimizer", items=orch3.performance_profile.get("models_profiled", 0),
                        message=f"total sweep time {orch3.performance_profile.get('total_sweep_time_sec', 0)}s")
        ui.agent_result("DocuAgent", items=1 if orch3.technical_documentation else 0,
                        message="technical documentation")
        ui.stage_complete(3, status="success", items=max(n_sweeps, 1),
                          duration=logger.get_stage_duration("Experimentation"),
                          output_file=experimentation_file)
        logger.end_stage("Experimentation", status="success", item_count=max(n_sweeps, 1),
                         output_files=[experimentation_file, version_manifest_file])
    except Exception as e:
        ui.error(f"Stage 3 failed: {e}")
        logger.log_error(error_type=type(e).__name__, message=str(e), recovered=False)
        logger.end_stage("Experimentation", status="error", error_message=str(e))
        workflow_success = False
        total_errors += 1

    logger.end_execution(success=workflow_success)
    log_path = logger.save_execution_log()
    ui.workflow_complete(
        success=workflow_success, stages_completed=3 if workflow_success else 2,
        total_errors=total_errors,
        output_files=[generation_file, validation_file, experimentation_file, version_manifest_file],
    )
    ui.observability_summary(collector)
    from shared.telemetry import export_telemetry
    export_telemetry(collector, str(output_dir), team="CodeTeam", mode=MODE)
    ui.info(f"Execution log: {log_path}")
    if not workflow_success:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"AEL CodeTeam — {MODE}")
    parser.add_argument("--design", type=str, default=None,
                        help="Path to ModelTeam model_design_output.json")
    parser.add_argument("--calibration", type=str, default=None,
                        help="Path to ModelTeam calibration_output.json")
    parser.add_argument("--model", type=str, default=None,
                        help="Override LLM model for all stages")
    args = parser.parse_args()
    if args.model:
        os.environ["AEL_MODEL"] = args.model
    main(design_path=args.design, calibration_path=args.calibration)
