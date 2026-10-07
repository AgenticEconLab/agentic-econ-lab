# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Master Orchestrator for Research Question Generation with Human-in-the-Loop (No FireCrawl)
This script runs all three stages sequentially with human feedback between rounds.

Pipeline:
1. Sourcing Stage: Gather literature from research keywords (2 rounds with feedback)
2. Refinement Stage: Generate research concepts and questions (2 rounds with feedback)
3. Integration Stage: Contextualize and prioritize final questions (2 rounds with feedback)

Input: Research keywords or basic research ideas
Output: Finalized, prioritized research questions
Human Interaction: Feedback collection after each round in each stage
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

# Load environment variables
load_dotenv()

# Set quiet mode for stage imports
os.environ["AGENT_QUIET_MODE"] = "true"
register_all_tools()


def main(cli_topic=None):
    """Main orchestrator that runs all three stages with human-in-the-loop."""

    # Change to script directory to ensure outputs are saved there
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    output_dir = Path(script_dir)

    # Initialize Console UI
    ui = ConsoleUI(
        team="IdeationTeam",
        mode="ModeNoWcWithHITL",
        total_stages=3
    )

    # ========== CONFIGURATION ==========
    if cli_topic:
        research_topic = cli_topic
    else:
        research_topic = auto_input(
            "Enter your research topic or keywords: ",
            default=get_default("research_topic")
        ).strip()
    if not research_topic:
        research_topic = get_default("research_topic")

    # Initialize workflow logger
    logger = WorkflowLogger(
        team="IdeationTeam",
        mode="ModeNoWcWithHITL",
        output_dir=output_dir,
        framework="ael",
        quiet=True
    )

    # Initialize metrics collector
    collector = MetricsCollector()
    logger.set_metrics_collector(collector)

    # Initialize budget controller (cost circuit-breaker)
    budget = BudgetController(max_cost_usd=5.0, max_tokens=500_000)

    # Load per-stage model configuration from ael_config.yaml
    model_config = load_model_config()

    # Initialize state guard for inter-stage hash verification
    state_guard = StateGuard()

    # Stage timeout configuration (seconds per stage)
    stage_timeouts = {
        "SourcingStage": 300,
        "RefinementStage": 300,
        "IntegrationStage": 300,
    }

    # Initialize memory manager (cross-run persistence)
    memory_dir = str(output_dir / "memory")
    memory = MemoryManager(run_id=logger.execution_id, memory_dir=memory_dir)
    memory.set_context("research_topic", research_topic)
    memory.set_context("team", "IdeationTeam")
    memory.set_context("mode", "ModeNoWcWithHITL")

    # Recall similar past runs
    similar_runs = memory.recall_similar_runs(research_topic)
    if similar_runs:
        ui.info(f"Found {len(similar_runs)} similar past run(s) in memory")

    # Start execution tracking
    logger.start_execution(metadata={"research_topic": research_topic})

    # Print workflow header
    ui.workflow_header(
        topic=research_topic,
        execution_id=logger.execution_id,
        mode_features="HITL (2 rounds per stage)"
    )

    # Show loaded API keys
    ui.show_api_keys()

    # Track workflow success
    workflow_success = True
    total_errors = 0
    literature_count = 0
    question_count = 0
    final_count = 0

    # ========== STAGE 1: LITERATURE SOURCING (2 ROUNDS) ==========
    ui.stage_start(1, "Literature Sourcing", description="2 rounds with human feedback")
    logger.start_stage("SourcingStage", 1)
    literature_file = "literature_results_all_rounds.csv"

    try:
        from importlib import import_module
        stage1 = import_module('1-SourcingStage')

        with stage_model(model_config, "IdeationTeam", "SourcingStage") as model:
            ui.info(f"Stage model: {model}")
            orchestrator1 = stage1.MultiAgentOrchestrator(quiet=True, collector=collector)

        # ROUND 1
        ui.info("Round 1: Initial search...")
        round1_results = orchestrator1.run_search_round(
            research_topic=research_topic,
            round_number=1,
            feedback=None,
            max_results_per_agent=8
        )
        ui.agent_result("Round 1", items=len(round1_results), message="Initial search complete")

        # Persist Round 1 so reviewers can open the full list, and build a
        # top-N preview for the HITL checkpoint.
        orchestrator1.save_results(literature_file, round_number=1)
        round1_lit_path = os.path.abspath(f"round1_{literature_file}")
        _ranked = sorted(
            round1_results,
            key=lambda r: getattr(r, "relevance_score", None) or 0,
            reverse=True,
        )
        _preview_items = [
            {
                "rank": i + 1,
                "title": getattr(r, "title", ""),
                "source": getattr(r, "source", ""),
                "year": getattr(r, "year", "") or "",
                "cites": getattr(r, "citation_count", None) or 0,
            }
            for i, r in enumerate(_ranked[:10])
        ]
        ui.hitl_checkpoint(
            1, "Literature Review",
            summary={"Papers found": len(round1_results)},
            preview={
                "title": f"Top {len(_preview_items)} of {len(round1_results)} papers (by relevance):",
                "items": _preview_items,
                "columns": [
                    ("rank", "#", 3),
                    ("title", "Title", 60),
                    ("source", "Source", 12),
                    ("year", "Year", 5),
                    ("cites", "Cites", 6),
                ],
                "full_contents_path": round1_lit_path,
                "full_contents_label": "Full ranked list",
            },
        )
        feedback1 = orchestrator1.collect_human_feedback(
            round_number=1, context={"papers_under_review": _preview_items}
        )
        ui.hitl_response(feedback1.get("feedback", "approved") if isinstance(feedback1, dict) else str(feedback1)[:50])

        # ROUND 2
        ui.info("Round 2: Refined search with feedback...")
        round2_results = orchestrator1.run_search_round(
            research_topic=research_topic,
            round_number=2,
            feedback=feedback1,
            max_results_per_agent=8
        )
        ui.agent_result("Round 2", items=len(round2_results), message="Refined search complete")

        orchestrator1.save_results(literature_file)
        literature_count = len(orchestrator1.all_results)

        ui.stage_complete(
            1, status="success",
            items=literature_count,
            duration=logger.get_stage_duration("SourcingStage"),
            output_file=literature_file
        )
        logger.end_stage("SourcingStage", status="success",
                         item_count=literature_count,
                         output_files=[literature_file])

        # Schema validation after Stage 1 (CSV output — best-effort)
        try:
            stage1_data = {
                "literature_items": [r.model_dump() if hasattr(r, 'model_dump') else r for r in orchestrator1.all_results],
                "search_queries": [],
                "metadata": {"stage": "SourcingStage", "item_count": literature_count}
            }
            SchemaValidator.validate_or_raise("IdeationTeam", "SourcingStage", stage1_data)
        except (ValueError, Exception) as e:
            ui.warning(f"Schema validation: {e}")
            logger.log_error(error_type="SchemaValidation", message=str(e), recovered=True)
        state_guard.sign_output(stage1_data, stage_name="SourcingStage")

        # Budget check after Stage 1
        budget_status = budget.check(collector, current_team="IdeationTeam")
        for w in budget_status.warnings:
            ui.warning(w)

    except BudgetExceededError as e:
        ui.error(f"Budget exceeded after Stage 1: {e}")
        logger.log_error(error_type="BudgetExceededError", message=str(e), recovered=False)
        logger.end_stage("SourcingStage", status="error", error_message=str(e))
        logger.end_execution(success=False)
        logger.save_execution_log()
        return

    except Exception as e:
        ui.error(f"Stage 1 failed: {e}")
        logger.log_error(error_type=type(e).__name__, message=str(e), recovered=False)
        logger.end_stage("SourcingStage", status="error", error_message=str(e))
        workflow_success = False
        total_errors += 1
        ui.workflow_complete(success=False, stages_completed=0, total_errors=total_errors)
        logger.end_execution(success=False)
        logger.save_execution_log()
        return

    # ========== STAGE 2: REFINEMENT (2 ROUNDS) ==========
    ui.stage_start(2, "Research Question Refinement", description="2 rounds with human feedback")
    logger.start_stage("RefinementStage", 2)
    refinement_file = "refinement_results_all_rounds.json"

    try:
        stage2 = import_module('2-RefinementStage')
        with stage_model(model_config, "IdeationTeam", "RefinementStage") as model:
            ui.info(f"Stage model: {model}")
            orchestrator2 = stage2.RefinementOrchestrator(quiet=True, collector=collector)
        literature_df = orchestrator2.load_literature(literature_file)

        # ROUND 1
        ui.info("Round 1: Initial concept generation...")
        concepts1, questions1 = orchestrator2.run_refinement_round(
            literature_df=literature_df,
            round_number=1, feedback=None,
            num_concepts=8, num_questions=6
        )
        ui.agent_result("Ideator R1", items=len(concepts1), message=f"{len(concepts1)} concepts generated")
        ui.agent_result("Refiner R1", items=len(questions1), message=f"{len(questions1)} questions formulated")

        # Persist Round 1 refinement so reviewers can open the full detail,
        # and show concept titles + question stems in the checkpoint preview.
        orchestrator2.save_results(refinement_file, round_number=1)
        round1_ref_path = os.path.abspath(f"round1_{refinement_file}")
        _concept_rows = [
            {
                "rank": i + 1,
                "title": getattr(c, "concept_title", ""),
                "novelty": round(getattr(c, "novelty_score", None) or 0, 2),
            }
            for i, c in enumerate(concepts1)
        ]
        _question_rows = [
            {
                "rank": i + 1,
                "question": getattr(q, "question", ""),
                "feasibility": round(getattr(q, "feasibility_score", None) or 0, 2),
            }
            for i, q in enumerate(questions1)
        ]
        ui.hitl_checkpoint(
            2, "Concept & Question Review",
            summary={"Concepts": len(concepts1), "Questions": len(questions1)},
            preview={
                "sections": [
                    {
                        "subtitle": "Concepts",
                        "items": _concept_rows,
                        "columns": [
                            ("rank", "#", 3),
                            ("title", "Concept Title", 65),
                            ("novelty", "Novelty", 7),
                        ],
                    },
                    {
                        "subtitle": "Questions",
                        "items": _question_rows,
                        "columns": [
                            ("rank", "#", 3),
                            ("question", "Research Question", 65),
                            ("feasibility", "Feasible", 8),
                        ],
                    },
                ],
                "full_contents_path": round1_ref_path,
                "full_contents_label": "Full concepts + questions",
            },
        )
        feedback2 = orchestrator2.collect_human_feedback(
            round_number=1, context={"concepts": _concept_rows, "questions": _question_rows}
        )
        ui.hitl_response(feedback2.get("feedback", "approved") if isinstance(feedback2, dict) else str(feedback2)[:50])

        # ROUND 2
        ui.info("Round 2: Refined generation with feedback...")
        concepts2, questions2 = orchestrator2.run_refinement_round(
            literature_df=literature_df,
            round_number=2, feedback=feedback2,
            num_concepts=8, num_questions=6
        )
        ui.agent_result("Ideator R2", items=len(concepts2), message=f"{len(concepts2)} concepts refined")
        ui.agent_result("Refiner R2", items=len(questions2), message=f"{len(questions2)} questions refined")

        orchestrator2.save_results(refinement_file)
        question_count = len(orchestrator2.all_questions)

        ui.stage_complete(
            2, status="success",
            items=question_count,
            duration=logger.get_stage_duration("RefinementStage"),
            output_file=refinement_file
        )
        logger.end_stage("RefinementStage", status="success",
                         item_count=question_count,
                         output_files=[refinement_file])

        # Schema validation after Stage 2
        try:
            with open(refinement_file, 'r', encoding='utf-8') as f:
                SchemaValidator.validate_or_raise("IdeationTeam", "RefinementStage", json.load(f))
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
            ui.warning(f"Schema validation: {e}")
            logger.log_error(error_type="SchemaValidation", message=str(e), recovered=True)

        # Budget check after Stage 2
        budget_status = budget.check(collector, current_team="IdeationTeam")
        for w in budget_status.warnings:
            ui.warning(w)

    except BudgetExceededError as e:
        ui.error(f"Budget exceeded after Stage 2: {e}")
        logger.log_error(error_type="BudgetExceededError", message=str(e), recovered=False)
        logger.end_stage("RefinementStage", status="error", error_message=str(e))
        logger.end_execution(success=False)
        logger.save_execution_log()
        return

    except Exception as e:
        ui.error(f"Stage 2 failed: {e}")
        logger.log_error(error_type=type(e).__name__, message=str(e), recovered=False)
        logger.end_stage("RefinementStage", status="error", error_message=str(e))
        workflow_success = False
        total_errors += 1
        ui.workflow_complete(success=False, stages_completed=1, total_errors=total_errors)
        logger.end_execution(success=False)
        logger.save_execution_log()
        return

    # ========== STAGE 3: INTEGRATION (2 ROUNDS) ==========
    ui.stage_start(3, "Question Integration & Prioritization", description="2 rounds with human feedback")
    logger.start_stage("IntegrationStage", 3)
    final_file = "finalized_research_questions.json"

    try:
        stage3 = import_module('3-IntegrationStage')
        with stage_model(model_config, "IdeationTeam", "IntegrationStage") as model:
            ui.info(f"Stage model: {model}")
            orchestrator3 = stage3.IntegrationOrchestrator(quiet=True, collector=collector)
            orchestrator3.research_topic = research_topic  # seed topic for the topic check
        initial_questions = orchestrator3.load_refinement_results(refinement_file)

        # ROUND 1
        ui.info("Round 1: Initial integration...")
        contextualized1, prioritized1 = orchestrator3.run_integration_round(
            questions=initial_questions,
            round_number=1, feedback=None,
            max_final_questions=5
        )
        ui.agent_result("Contextualizer R1", items=len(contextualized1))
        ui.agent_result("Finalizer R1", items=len(prioritized1))

        # Persist Round 1 integration so reviewers can open the full detail,
        # and show the prioritized questions with scores in the checkpoint.
        orchestrator3.save_results(final_file, round_number=1)
        round1_int_path = os.path.abspath(f"round1_{final_file}")
        _prioritized_rows = [
            {
                "rank": getattr(q, "priority_rank", i + 1),
                "question": getattr(q, "question", ""),
                "score": round(getattr(q, "priority_score", None) or 0, 2),
            }
            for i, q in enumerate(prioritized1)
        ]
        ui.hitl_checkpoint(
            3, "Prioritized Questions Review",
            summary={"Prioritized": len(prioritized1)},
            preview={
                "title": f"Prioritized research questions (Round 1):",
                "items": _prioritized_rows,
                "columns": [
                    ("rank", "#", 3),
                    ("question", "Research Question", 68),
                    ("score", "Score", 5),
                ],
                "full_contents_path": round1_int_path,
                "full_contents_label": "Full integration detail (incl. frameworks & rationale)",
            },
        )
        feedback3 = orchestrator3.collect_integration_feedback(
            round_number=1, num_questions=len(prioritized1),
            context={"prioritized_questions": _prioritized_rows}
        )
        ui.hitl_response("feedback collected")

        # ROUND 2
        questions_for_round2 = orchestrator3.convert_prioritized_to_research_questions(prioritized1)
        ui.info("Round 2: Refined integration with feedback...")
        contextualized2, prioritized2 = orchestrator3.run_integration_round(
            questions=questions_for_round2,
            round_number=2, feedback=feedback3,
            max_final_questions=5
        )
        ui.agent_result("Contextualizer R2", items=len(contextualized2))
        ui.agent_result("Finalizer R2", items=len(prioritized2))

        orchestrator3.save_final_questions(final_file)
        final_count = len(prioritized2)

        ui.stage_complete(
            3, status="success",
            items=final_count,
            duration=logger.get_stage_duration("IntegrationStage"),
            output_file=final_file
        )
        logger.end_stage("IntegrationStage", status="success",
                         item_count=final_count,
                         output_files=[final_file])

        # Schema validation after Stage 3
        try:
            with open(final_file, 'r', encoding='utf-8') as f:
                SchemaValidator.validate_or_raise("IdeationTeam", "IntegrationStage", json.load(f))
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
            ui.warning(f"Schema validation: {e}")
            logger.log_error(error_type="SchemaValidation", message=str(e), recovered=True)

    except Exception as e:
        ui.error(f"Stage 3 failed: {e}")
        logger.log_error(error_type=type(e).__name__, message=str(e), recovered=False)
        logger.end_stage("IntegrationStage", status="error", error_message=str(e))
        workflow_success = False
        total_errors += 1
        ui.workflow_complete(success=False, stages_completed=2, total_errors=total_errors)
        logger.end_execution(success=False)
        logger.save_execution_log()
        return


    # V0.7: CitationVerifier post-processor over IdeationTeam rationales
    try:
        import json as _json
        from shared.verification import (
            run_citation_verifier_pass, extract_ideation_texts, attach_citation_audit,
        )
        _final_path = final_file if 'final_file' in dir() else None
        if _final_path:
            try:
                with open(_final_path, "r", encoding="utf-8") as _f:
                    _final_data = _json.load(_f)
            except Exception:
                _final_data = {}
            _texts = extract_ideation_texts(_final_data)
            _audit = run_citation_verifier_pass(_texts, collector=collector)
            if _audit:
                attach_citation_audit(_final_path, _audit)
                ui.info(
                    f"CitationVerifier: {_audit['overall_hallucinated']}/{_audit['overall_total']} "
                    f"hallucinated ({_audit['overall_hallucination_rate']:.1%})"
                )
    except Exception as _e:
        ui.warning(f"CitationVerifier hook skipped: {_e}")

    # ========== WORKFLOW COMPLETE ==========
    # Record trajectory in episodic memory
    from shared.memory.episodic import RunTrajectory
    trajectory = RunTrajectory(
        run_id=logger.execution_id,
        research_topic=research_topic,
        teams_completed=["IdeationTeam"] if workflow_success else [],
        teams_failed=[] if workflow_success else ["IdeationTeam"],
        success=workflow_success,
        mode="ModeNoWcWithHITL",
        artifact_names=[final_file] if workflow_success else [],
    )
    memory.record_trajectory(trajectory)

    # Remember key facts for future runs
    if workflow_success and final_count > 0:
        memory.remember_fact(
            f"Generated {final_count} research questions on '{research_topic}'",
            category="ideation", source="IdeationTeam/ModeNoWcWithHITL",
        )

    memory.close()

    logger.end_execution(success=workflow_success)
    log_path = logger.save_execution_log()

    ui.workflow_complete(
        success=workflow_success,
        stages_completed=3,
        total_errors=total_errors,
        output_files=[final_file],
        summary={
            "Literature Papers": literature_count,
            "Research Questions": question_count,
            "Final Questions": final_count
        }
    )

    ui.observability_summary(collector)
    from shared.telemetry import export_telemetry
    export_telemetry(collector, str(output_dir), team="IdeationTeam", mode="ModeNoWcWithHITL")

    if 'prioritized2' in dir():
        ui.research_summary(
            title="FINALIZED RESEARCH QUESTIONS",
            items=[{"question": q.question, "priority_score": q.priority_score} for q in prioritized2],
            item_key="question", score_key="priority_score"
        )

    ui.info(f"Execution log: {log_path}")

    # Propagate failure to the caller (multirun/runner) so a crashed stage is NOT
    # silently reported as success and its stale outputs scored.
    if not workflow_success:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AEL IdeationTeam — ModeNoWcWithHITL")
    parser.add_argument("--topic", type=str, default=None, help="Research topic (skips interactive prompt)")
    parser.add_argument("--model", type=str, default=None, help="Override LLM model for all stages")
    args = parser.parse_args()
    if args.model:
        os.environ["AEL_MODEL"] = args.model
    main(cli_topic=args.topic)
