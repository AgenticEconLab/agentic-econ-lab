# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Master Orchestrator for Empirical Estimation & Validation (No Web-crawl, WITH Human-in-the-Loop)
Runs all three EstimationTeam stages with review checkpoints.

Pipeline:
1. Estimation Stage: LLM proposes an EstimationSpec
   -> HITL checkpoint BEFORE estimation: reviewer (human, or the LLM-economist committee in
      pipeline auto mode) sees the proposed spec, the series and the observation budget;
      ONE bounded revision, re-reviewed; then the deterministic harness estimates it
2. Validation & Diagnostics Stage: statsmodels battery; severe failures downgrade the verdict
3. Inference & Robustness Stage: hypothesis tests + stability sweeps
   -> final review checkpoint (recorded with the outputs)

Input:  model specification (ModelTeam) + validated dataset artifact (DataTeam)
Output: estimation_output.json, validation_output.json, inference_output.json
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
from shared.guardrails.budget_controller import BudgetController, BudgetExceededError
from shared.model_config import load_model_config, stage_model
from shared.reliability.state_guard import StateGuard
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

MODE = "ModeNoWcWithHITL"

_APPROVAL_TOKENS = ("approve", "approved", "ok", "looks good", "lgtm", "no concerns", "proceed")


def _is_approval(feedback: str) -> bool:
    text = (feedback or "").strip().lower()
    if not text:
        return True
    return any(text.startswith(tok) or text == tok for tok in _APPROVAL_TOKENS)


def _resolve_input(cli_value, env_var, prompt, default_key, default_paths):
    """CLI arg > env var > auto_input > first existing default path."""
    if cli_value and os.path.exists(cli_value):
        return cli_value
    env_value = os.environ.get(env_var, "")
    if env_value and os.path.exists(env_value):
        return env_value
    answer = auto_input(prompt, default=get_default(default_key)).strip()
    if answer and os.path.exists(answer):
        return answer
    for p in default_paths:
        if os.path.exists(p):
            return p
    return None


def main(model_spec_path=None, data_path=None, research_question=""):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    output_dir = Path(script_dir)

    ui = ConsoleUI(team="EstimationTeam", mode=MODE, total_stages=3)
    logger = WorkflowLogger(team="EstimationTeam", mode=MODE, output_dir=output_dir,
                            framework="ael", quiet=True)
    collector = MetricsCollector()
    logger.set_metrics_collector(collector)
    budget = BudgetController(max_cost_usd=5.0, max_tokens=500_000)

    _local_cfg = os.path.join(script_dir, "ael_config.yaml")
    if os.path.exists(_local_cfg) and not os.environ.get("AEL_MODEL_CONFIG"):
        os.environ["AEL_MODEL_CONFIG"] = _local_cfg
    model_config = load_model_config()
    state_guard = StateGuard()

    model_spec_path = _resolve_input(
        model_spec_path, "AEL_ESTIM_MODEL_SPEC",
        "Enter path to model specification JSON (or press Enter for default): ",
        "estimation_model_spec_path",
        [
            "../../ModelTeam/ael/ModeNoWcWithHITL/calibration_output.json",
            "../../ModelTeam/ael/ModeNoWcWithHITL/model_design_output.json",
            "../../ModelTeam/ael/ModeNoWcNoHITL/model_design_output.json",
        ])
    data_path = _resolve_input(
        data_path, "AEL_ESTIM_DATA",
        "Enter path to DataTeam output JSON (or press Enter for default): ",
        "estimation_data_path",
        [
            "../../DataTeam/ael/open_source_api/api_source_output.json",
            "../../../DataTeam/ael/open_source_api/api_source_output.json",
        ])
    if not model_spec_path or not data_path:
        ui.error("Missing inputs: need a model specification JSON and a DataTeam output JSON "
                 "(CLI --model-spec/--data, env AEL_ESTIM_MODEL_SPEC/AEL_ESTIM_DATA, or defaults).")
        sys.exit(1)

    with open(model_spec_path, "r", encoding="utf-8") as f:
        model_spec_data = json.load(f)
    with open(data_path, "r", encoding="utf-8") as f:
        data_artifact = json.load(f)

    logger.start_execution(metadata={
        "model_spec_file": model_spec_path,
        "data_file": data_path,
        "research_question": research_question or "(from model spec)",
        "hitl": True,
    })
    ui.workflow_header(topic=research_question or os.path.basename(model_spec_path),
                       execution_id=logger.execution_id)
    ui.show_api_keys()

    workflow_success = True
    total_errors = 0
    from importlib import import_module

    # ========== STAGE 1: ESTIMATION (+ HITL revision round) ==========
    ui.stage_start(1, "Empirical Estimation (deterministic harness)")
    logger.start_stage("Estimation", 1)
    estimation_file = "estimation_output.json"
    try:
        stage1 = import_module("1-EstimationStage")
        with stage_model(model_config, "EstimationTeam", "EstimationStage") as model:
            ui.info(f"Stage model: {model}")
            orch1 = stage1.EstimationOrchestrator(collector=collector,
                                                  output_dir=str(output_dir))

        # HITL checkpoint BEFORE estimation: the reviewer sees the proposed spec,
        # the available series and the observation budget — no coefficients or fit. A
        # rejection triggers ONE revision, which is reviewed again; a rejected revision
        # leaves the original proposal in place with the objections recorded.
        def _review(ctx, round_number):
            feedback = orch1.collect_human_feedback(round_number=round_number, context=ctx)
            approved = _is_approval(feedback)
            if not approved:
                ui.info(f"Reviewer objections in round {round_number}")
            return approved, feedback

        out1 = orch1.run_reviewed_pipeline(
            model_spec_data, data_artifact, review=_review,
            research_question=research_question)
        orch1.save_estimation_output(estimation_file)

        n_items = max(len(out1.outcome.coefficients), 1)
        ui.agent_result("Estimator", items=n_items,
                        message=f"verdict={out1.outcome.verdict}"
                                + (f" ({out1.outcome.reason})" if out1.outcome.reason else ""))
        ui.stage_complete(1, status="success", items=n_items,
                          duration=logger.get_stage_duration("Estimation"),
                          output_file=estimation_file)
        logger.end_stage("Estimation", status="success", item_count=n_items,
                         output_files=[estimation_file])
        state_guard.sign_output(out1.model_dump(), stage_name="EstimationStage")
        for w in budget.check(collector, current_team="EstimationTeam").warnings:
            ui.warning(w)
    except BudgetExceededError as e:
        ui.error(f"Budget exceeded after Stage 1: {e}")
        logger.log_error(error_type="BudgetExceededError", message=str(e), recovered=False)
        logger.end_stage("Estimation", status="error", error_message=str(e))
        logger.end_execution(success=False)
        logger.save_execution_log()
        sys.exit(1)
    except Exception as e:
        ui.error(f"Stage 1 failed: {e}")
        logger.log_error(error_type=type(e).__name__, message=str(e), recovered=False)
        logger.end_stage("Estimation", status="error", error_message=str(e))
        logger.end_execution(success=False)
        logger.save_execution_log()
        ui.workflow_complete(success=False, stages_completed=0, total_errors=1)
        sys.exit(1)

    # ========== STAGE 2: VALIDATION & DIAGNOSTICS ==========
    ui.stage_start(2, "Validation & Diagnostics (statsmodels battery)")
    logger.start_stage("ValidationDiagnostics", 2)
    validation_file = "validation_output.json"
    try:
        stage2 = import_module("2-ValidationDiagnosticsStage")
        with stage_model(model_config, "EstimationTeam", "ValidationDiagnosticsStage") as model:
            ui.info(f"Stage model: {model}")
            orch2 = stage2.ValidationOrchestrator(collector=collector)
        with open(estimation_file, "r", encoding="utf-8") as f:
            estimation_data = json.load(f)
        out2 = orch2.run_validation_pipeline(estimation_data)
        orch2.save_validation_output(validation_file)
        n_tests = len(out2.diagnostics.results)
        ui.agent_result("Validator", items=n_tests,
                        message=f"battery={out2.diagnostics.overall}, "
                                f"verdict={out2.outcome.verdict}")
        ui.stage_complete(2, status="success", items=n_tests,
                          duration=logger.get_stage_duration("ValidationDiagnostics"),
                          output_file=validation_file)
        logger.end_stage("ValidationDiagnostics", status="success", item_count=n_tests,
                         output_files=[validation_file])
        for w in budget.check(collector, current_team="EstimationTeam").warnings:
            ui.warning(w)
    except Exception as e:
        ui.error(f"Stage 2 failed: {e}")
        logger.log_error(error_type=type(e).__name__, message=str(e), recovered=False)
        logger.end_stage("ValidationDiagnostics", status="error", error_message=str(e))
        logger.end_execution(success=False)
        logger.save_execution_log()
        ui.workflow_complete(success=False, stages_completed=1, total_errors=1)
        sys.exit(1)

    # ========== STAGE 3: INFERENCE & ROBUSTNESS (+ final review) ==========
    ui.stage_start(3, "Hypothesis Testing & Robustness")
    logger.start_stage("InferenceRobustness", 3)
    inference_file = "inference_output.json"
    try:
        stage3 = import_module("3-InferenceRobustnessStage")
        with stage_model(model_config, "EstimationTeam", "InferenceRobustnessStage") as model:
            ui.info(f"Stage model: {model}")
            orch3 = stage3.InferenceOrchestrator(collector=collector)
        with open(validation_file, "r", encoding="utf-8") as f:
            validation_data = json.load(f)
        out3 = orch3.run_inference_pipeline(validation_data,
                                            research_question=research_question)

        # Final review checkpoint: recorded alongside the outputs (no re-run — the
        # revision authority sits at the specification checkpoint after Stage 1).
        final_context = {
            "stage": "InferenceRobustnessStage",
            "summary": out3.summary,
            "hypotheses": [h.model_dump() for h in out3.inference.hypotheses],
            "stability_score": out3.inference.stability_score,
            "final_verdict": out3.outcome.verdict,
        }
        final_review = auto_input(
            "Final review of the estimation results (recorded with the outputs): ",
            default=get_default("estimation_final_review"), context=final_context,
        ).strip()
        if final_review:
            out3 = out3.model_copy(update={
                "metadata": {**out3.metadata, "final_review": final_review}})
            orch3.output = out3
        orch3.save_inference_output(inference_file)

        n_checks = len(out3.inference.hypotheses) + len(out3.inference.robustness)
        ui.agent_result("HypothesisTester", items=max(n_checks, 1), message=out3.summary[:80])
        ui.agent_result("Optimizer", items=len(out3.performance_profile.stage_timing_sec),
                        message=f"total {out3.performance_profile.total_sec}s "
                                f"(det {out3.performance_profile.deterministic_sec}s / "
                                f"llm {out3.performance_profile.llm_sec}s)")
        ui.stage_complete(3, status="success", items=max(n_checks, 1),
                          duration=logger.get_stage_duration("InferenceRobustness"),
                          output_file=inference_file)
        logger.end_stage("InferenceRobustness", status="success", item_count=max(n_checks, 1),
                         output_files=[inference_file])
    except Exception as e:
        ui.error(f"Stage 3 failed: {e}")
        logger.log_error(error_type=type(e).__name__, message=str(e), recovered=False)
        logger.end_stage("InferenceRobustness", status="error", error_message=str(e))
        workflow_success = False
        total_errors += 1

    logger.end_execution(success=workflow_success)
    log_path = logger.save_execution_log()
    ui.workflow_complete(
        success=workflow_success, stages_completed=3 if workflow_success else 2,
        total_errors=total_errors,
        output_files=[estimation_file, validation_file, inference_file],
    )
    ui.observability_summary(collector)
    from shared.telemetry import export_telemetry
    export_telemetry(collector, str(output_dir), team="EstimationTeam", mode=MODE)
    ui.info(f"Execution log: {log_path}")
    if not workflow_success:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"AEL EstimationTeam — {MODE}")
    parser.add_argument("--model-spec", type=str, default=None,
                        help="Path to ModelTeam specification JSON")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to DataTeam output JSON (retrieved_data provenance)")
    parser.add_argument("--question", type=str, default="",
                        help="Research question the estimation should address")
    parser.add_argument("--model", type=str, default=None,
                        help="Override LLM model for all stages")
    args = parser.parse_args()
    if args.model:
        os.environ["AEL_MODEL"] = args.model
    main(model_spec_path=args.model_spec, data_path=args.data,
         research_question=args.question)
