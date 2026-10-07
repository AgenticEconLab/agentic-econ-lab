# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Master Orchestrator for Interpretation & Reporting (No Web-crawl, WITH Human-in-the-Loop)
Runs all three ReportingTeam stages sequentially.

Pipeline:
1. Interpretation Stage: deterministic magnitudes + figures from the estimation artifact
2. Drafting Stage: deterministic report assembly + LLM narrative blocks
3. Quality Stage: number-consistency check; verdict annotated on the report

Input:  estimation inference_output.json (+ optional upstream artifact JSONs)
Output: interpretation_output.json, drafting_output.json, quality_output.json,
        research_report.md, figures/
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

MODE = "ModeNoWcWithHITL"
HITL = "WithHITL" in MODE


def _load_json(path):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def main(estimation_path=None, artifacts_dir=None, research_question=""):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    output_dir = Path(script_dir)

    ui = ConsoleUI(team="ReportingTeam", mode=MODE, total_stages=3)
    logger = WorkflowLogger(team="ReportingTeam", mode=MODE, output_dir=output_dir,
                            framework="ael", quiet=True)
    collector = MetricsCollector()
    logger.set_metrics_collector(collector)
    model_config = load_model_config()

    estimation_path = (estimation_path or os.environ.get("AEL_REPORT_ESTIMATION")
                       or auto_input("Path to estimation inference_output.json: ",
                                     default=get_default("reporting_estimation_path")).strip()
                       or "../../EstimationTeam/ael/ModeNoWcNoHITL/inference_output.json")
    estimation_results = _load_json(estimation_path)
    if not estimation_results:
        ui.error(f"No estimation results at '{estimation_path}' — nothing to report on.")
        sys.exit(1)

    # Optional upstream artifacts (pipeline artifacts dir or individual files)
    artifacts_dir = artifacts_dir or os.environ.get("AEL_REPORT_ARTIFACTS", "")
    def _artifact(name):
        if artifacts_dir:
            data = _load_json(os.path.join(artifacts_dir, f"{name}.json"))
            return data.get("data", data) if isinstance(data, dict) else {}
        return {}
    research_questions = _artifact("research_questions")
    literature_review = _artifact("literature_review")
    model_specification = _artifact("model_specification")
    data_source = _artifact("data_source")
    feasibility_report = _load_json(os.path.join(artifacts_dir, "..", "feasibility_report.json")) \
        if artifacts_dir else {}
    literature_batch = _load_json(os.path.join(artifacts_dir, "..", "LiteratureTeam",
                                               "literature_batch.json")) if artifacts_dir else {}

    logger.start_execution(metadata={"estimation_file": estimation_path,
                                     "artifacts_dir": artifacts_dir or "(none)",
                                     "hitl": HITL})
    ui.workflow_header(topic=research_question or os.path.basename(estimation_path),
                       execution_id=logger.execution_id)

    workflow_success = True
    total_errors = 0
    from importlib import import_module

    # ========== STAGE 1: INTERPRETATION ==========
    ui.stage_start(1, "Economic Magnitudes & Figures (deterministic)")
    logger.start_stage("Interpretation", 1)
    interpretation_file = "interpretation_output.json"
    try:
        stage1 = import_module("1-InterpretationStage")
        with stage_model(model_config, "ReportingTeam", "InterpretationStage") as model:
            ui.info(f"Stage model: {model}")
            orch1 = stage1.InterpretationOrchestrator(collector=collector)
        out1 = orch1.run_interpretation_pipeline(
            estimation_results, research_question=research_question,
            figures_dir=str(output_dir / "figures"))
        orch1.save_interpretation_output(interpretation_file)
        n1 = max(len(out1.interpretation.effects), 1)
        ui.agent_result("ResultsInterpreter", items=n1,
                        message=f"{len(out1.interpretation.figure_files)} figure(s)")
        ui.stage_complete(1, status="success", items=n1,
                          duration=logger.get_stage_duration("Interpretation"),
                          output_file=interpretation_file)
        logger.end_stage("Interpretation", status="success", item_count=n1,
                         output_files=[interpretation_file])
    except Exception as e:
        ui.error(f"Stage 1 failed: {e}")
        logger.log_error(error_type=type(e).__name__, message=str(e), recovered=False)
        logger.end_stage("Interpretation", status="error", error_message=str(e))
        logger.end_execution(success=False)
        logger.save_execution_log()
        ui.workflow_complete(success=False, stages_completed=0, total_errors=1)
        sys.exit(1)

    # ========== STAGE 2: DRAFTING ==========
    ui.stage_start(2, "Report Drafting (deterministic assembly + narrative)")
    logger.start_stage("Drafting", 2)
    drafting_file = "drafting_output.json"
    report_file = "research_report.md"
    try:
        stage2 = import_module("2-DraftingStage")
        with stage_model(model_config, "ReportingTeam", "DraftingStage") as model:
            ui.info(f"Stage model: {model}")
            orch2 = stage2.DraftingOrchestrator(collector=collector,
                                                output_dir=str(output_dir))
        interpretation_output = _load_json(interpretation_file)
        out2 = orch2.run_drafting_pipeline(
            research_questions, literature_review, model_specification, data_source,
            estimation_results, interpretation_output,
            feasibility_report=feasibility_report or None,
            report_file=report_file, literature_batch=literature_batch)

        if HITL:
            # A rejected draft gets ONE revision, which is reviewed again; if the
            # revision is rejected too, it is published with the objections in Limitations.
            def _draft_review(out, round_number):
                ctx = {"stage": "DraftingStage",
                       "report_excerpt": out.report_markdown[:2500],
                       "narratives": out.drafting.narratives}
                fb = orch2.collect_human_feedback(round_number=round_number, context=ctx)
                approved = not fb or any(fb.lower().startswith(t) for t in
                                         ("approve", "ok", "looks good", "lgtm"))
                return approved, fb

            approved, fb = _draft_review(out2, 1)
            if not approved:
                ui.info("Reviewer requested changes — one revision round")
                out2 = orch2.run_drafting_pipeline(
                    research_questions, literature_review, model_specification, data_source,
                    estimation_results, interpretation_output,
                    feasibility_report=feasibility_report or None,
                    report_file=report_file, feedback=fb, literature_batch=literature_batch)
                approved2, fb2 = _draft_review(out2, 2)
                if not approved2:
                    ui.info("Revision not approved — published with the objections recorded")
                    out2 = orch2.publish_with_objections(
                        [f"objections to the first draft: {fb}",
                         f"objections to the revised draft: {fb2}"])

        orch2.save_drafting_output(drafting_file)
        n2 = max(out2.drafting.section_count, 1)
        ui.agent_result("Reporter", items=n2, message=f"report -> {report_file}")
        ui.stage_complete(2, status="success", items=n2,
                          duration=logger.get_stage_duration("Drafting"),
                          output_file=report_file)
        logger.end_stage("Drafting", status="success", item_count=n2,
                         output_files=[drafting_file, report_file])
    except Exception as e:
        ui.error(f"Stage 2 failed: {e}")
        logger.log_error(error_type=type(e).__name__, message=str(e), recovered=False)
        logger.end_stage("Drafting", status="error", error_message=str(e))
        logger.end_execution(success=False)
        logger.save_execution_log()
        ui.workflow_complete(success=False, stages_completed=1, total_errors=1)
        sys.exit(1)

    # ========== STAGE 3: QUALITY ==========
    ui.stage_start(3, "Number-Consistency Check")
    logger.start_stage("Quality", 3)
    quality_file = "quality_output.json"
    formatted_report_file = "research_report_formatted.md"
    try:
        stage3 = import_module("3-QualityStage")
        with stage_model(model_config, "ReportingTeam", "QualityStage") as model:
            ui.info(f"Stage model: {model}")
            orch3 = stage3.QualityOrchestrator(collector=collector)
        drafting_output = _load_json(drafting_file)
        interpretation_output = _load_json(interpretation_file)
        out3 = orch3.run_quality_pipeline(
            drafting_output,
            source_artifacts={"research_questions": research_questions,
                              "literature_review": literature_review,
                              "model_specification": model_specification,
                              "data_source": data_source,
                              "estimation_results": estimation_results,
                              "interpretation": interpretation_output,
                              "literature_batch": literature_batch},
            research_question=research_question, data_source=data_source,
            literature_batch=literature_batch, formatted_report_file=formatted_report_file)
        orch3.save_quality_output(quality_file)
        c = out3.quality.consistency
        ui.agent_result("Proofreader", items=c.total_numbers,
                        message=f"{c.verified}/{c.total_numbers} verified ({c.verdict})")
        ui.agent_result("JournalAdvisor", items=len(out3.quality.journal_shortlist),
                        message=", ".join(m.name for m in out3.quality.journal_shortlist) or
                                "no shortlist match")
        ui.agent_result("Formatter", items=1, message=formatted_report_file)
        ui.stage_complete(3, status="success", items=c.total_numbers,
                          duration=logger.get_stage_duration("Quality"),
                          output_file=quality_file)
        logger.end_stage("Quality", status="success", item_count=c.total_numbers,
                         output_files=[quality_file, formatted_report_file])
    except Exception as e:
        ui.error(f"Stage 3 failed: {e}")
        logger.log_error(error_type=type(e).__name__, message=str(e), recovered=False)
        logger.end_stage("Quality", status="error", error_message=str(e))
        workflow_success = False
        total_errors += 1

    logger.end_execution(success=workflow_success)
    log_path = logger.save_execution_log()
    ui.workflow_complete(
        success=workflow_success, stages_completed=3 if workflow_success else 2,
        total_errors=total_errors,
        output_files=[interpretation_file, drafting_file, quality_file, report_file,
                     formatted_report_file],
    )
    ui.observability_summary(collector)
    from shared.telemetry import export_telemetry
    export_telemetry(collector, str(output_dir), team="ReportingTeam", mode=MODE)
    ui.info(f"Execution log: {log_path}")
    if not workflow_success:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"AEL ReportingTeam — {MODE}")
    parser.add_argument("--estimation", type=str, default=None,
                        help="Path to EstimationTeam inference_output.json")
    parser.add_argument("--artifacts-dir", type=str, default=None,
                        help="Pipeline artifacts/ directory for upstream context")
    parser.add_argument("--question", type=str, default="",
                        help="Research question for the report header")
    parser.add_argument("--model", type=str, default=None,
                        help="Override LLM model for all stages")
    args = parser.parse_args()
    if args.model:
        os.environ["AEL_MODEL"] = args.model
    main(estimation_path=args.estimation, artifacts_dir=args.artifacts_dir,
         research_question=args.question)
