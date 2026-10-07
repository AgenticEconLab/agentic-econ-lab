# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Master Orchestrator for Literature Team Pipeline (WITH FireCrawl AND Human-in-the-Loop)
This script runs all three stages with FireCrawl web scraping and human-in-the-loop checkpoints.

Pipeline:
1. Literature Gathering Stage: Retrieve and analyze literature (with FireCrawl enrichment)
2. Gap Detection Stage: Identify research gaps and construct knowledge graph
3. Synthesis Stage: Generate literature review and research plan

Input: Research questions (JSON from IdeationTeam or manual input)
Output: Literature review, research plan, and bibliography
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
from shared.rag.document_store import DocumentStore, Document
from shared.rag.hybrid_retriever import HybridRetriever
from shared.guardrails.grounding_checker import GroundingChecker, Citation
from shared.tools.register_all import register_all_tools
from shared.llm_router import ModelRouter
from shared.model_config import load_model_config, stage_model
from shared.reliability.state_guard import StateGuard

# Load environment variables from the repository root .env
def get_env_path():
    """Find the .env file in the repository root (the directory holding ael_config.yaml and run_ael_pipeline.py)."""
    current = os.path.dirname(os.path.abspath(__file__))
    while current != os.path.dirname(current):
        if (os.path.exists(os.path.join(current, "ael_config.yaml")) and os.path.exists(os.path.join(current, "run_ael_pipeline.py"))):
            env_path = os.path.join(current, ".env")
            if os.path.exists(env_path):
                return env_path
        current = os.path.dirname(current)
    return None

env_path = get_env_path()
if env_path:
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

# Set quiet mode for stage imports
os.environ["AGENT_QUIET_MODE"] = "true"
register_all_tools()


def main(cli_topic=None):
    """Main orchestrator with FireCrawl and HITL checkpoints."""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    output_dir = Path(script_dir)

    # Initialize Console UI
    ui = ConsoleUI(
        team="LiteratureTeam",
        mode="ModeWithWcWithHITL",
        total_stages=3
    )

    # Initialize workflow logger
    logger = WorkflowLogger(
        team="LiteratureTeam",
        mode="ModeWithWcWithHITL",
        output_dir=output_dir,
        framework="ael",
        quiet=True
    )

    workflow_success = True
    total_errors = 0

    # Load research questions
    if cli_topic:
        research_questions = [{
            "question": cli_topic,
            "priority_rank": 1,
            "priority_score": 0.95,
            "theoretical_framework": "User-specified topic",
            "methodology": ["Literature review"]
        }]
        questions_file = None
    else:
        questions_file = auto_input(
            "Enter path to research questions JSON (or press Enter to create sample): ",
            default=get_default("questions_file_path")
        ).strip()

    if not cli_topic and questions_file and os.path.exists(questions_file):
        try:
            with open(questions_file, 'r', encoding='utf-8') as f:
                questions_data = json.load(f)

            if isinstance(questions_data, list):
                research_questions = questions_data
            elif isinstance(questions_data, dict):
                if 'questions' in questions_data:
                    research_questions = questions_data['questions']
                elif 'prioritized' in questions_data:
                    research_questions = questions_data['prioritized']
                else:
                    research_questions = [questions_data]
            else:
                raise ValueError("Unexpected JSON format")
        except Exception:
            research_questions = create_sample_questions()
    elif not cli_topic:
        research_questions = create_sample_questions()

    # Initialize metrics collector for observability
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
        "LiteratureGatheringStage": 300,
        "GapDetectionStage": 300,
        "SynthesisStage": 300,
    }

    # Initialize memory manager (cross-run persistence)
    memory_dir = str(output_dir / "memory")
    memory = MemoryManager(run_id=logger.execution_id, memory_dir=memory_dir)
    memory.set_context("team", "LiteratureTeam")
    memory.set_context("mode", "ModeWithWcWithHITL")

    # Initialize document store and retriever for literature recall
    doc_store_dir = str(output_dir / "memory" / "documents")
    doc_store = DocumentStore(store_dir=doc_store_dir)
    retriever = HybridRetriever(document_store=doc_store)

    # Start execution tracking
    logger.start_execution(metadata={
        "num_research_questions": len(research_questions),
        "firecrawl_enabled": True
    })

    # Print workflow header
    firecrawl_status = "enabled" if os.getenv("FIRECRAWL_API_KEY") else "disabled"
    ui.workflow_header(
        topic=f"{len(research_questions)} research questions",
        execution_id=logger.execution_id,
        firecrawl=firecrawl_status,
        mode_features="FireCrawl + HITL (iterative refinement)"
    )

    # Show loaded API keys
    ui.show_api_keys()

    # Pipeline state
    literature_batch = None
    batch_file = "literature_batch.json"
    csv_file = "literature_items.csv"
    max_papers_per_question = 15

    gap_analysis = None
    gap_file = "gap_analysis_results.json"
    gaps_csv = "research_gaps.csv"
    graph_file = "knowledge_graph.json"

    synthesis_result = None
    synthesis_file = "synthesis_results.json"
    review_file = "literature_review.txt"
    plan_file = "research_plan.txt"
    bib_file = "bibliography.txt"

    current_stage = 1
    pipeline_complete = False

    # Recall prior papers from document store
    research_topic_text = " ".join(
        q.get("question", str(q)) if isinstance(q, dict) else str(q)
        for q in research_questions[:3]
    )
    memory.set_context("research_topic", research_topic_text[:200])
    prior_papers = retriever.search(research_topic_text, top_k=20)
    if prior_papers:
        ui.info(f"Recalled {len(prior_papers)} papers from prior runs")

    # Recall similar past runs
    similar_runs = memory.recall_similar_runs(research_topic_text)
    if similar_runs:
        ui.info(f"Found {len(similar_runs)} similar past run(s) in memory")

    while not pipeline_complete:
        # ========== STAGE 1: LITERATURE GATHERING ==========
        if current_stage == 1:
            ui.stage_start(1, "Literature Gathering", description="FireCrawl + HITL checkpoint")
            logger.start_stage("LiteratureGatheringStage", 1)

            try:
                from importlib import import_module
                stage1 = import_module('1-LiteratureGatheringStage')

                formatted_questions = []
                for q in research_questions:
                    if isinstance(q, dict):
                        formatted_questions.append(stage1.ResearchQuestion(**q))
                    else:
                        formatted_questions.append(stage1.ResearchQuestion(
                            question=str(q), priority_rank=1, priority_score=0.8
                        ))

                with stage_model(model_config, "LiteratureTeam", "LiteratureGatheringStage") as model:
                    ui.info(f"Stage model: {model}")
                    orchestrator1 = stage1.LiteratureGatheringOrchestrator(collector=collector)
                literature_batch = orchestrator1.run_gathering_pipeline(
                    research_questions=formatted_questions,
                    max_papers_per_question=max_papers_per_question
                )

                orchestrator1.save_literature_batch(batch_file)
                orchestrator1.save_literature_csv(csv_file)

                ui.agent_result("TopicCrawler", items=len(literature_batch.literature_items))
                ui.agent_result("TrendTracker", items=len(literature_batch.trend_analyses))
                ui.agent_result("CiteKeeper", items=len(literature_batch.citations))
                ui.agent_result("InsightSummarizer", items=len(literature_batch.insights))

                # Build preview: top-10 items sorted by relevance_score (falls back to citation_count).
                _ranked = sorted(
                    list(literature_batch.literature_items),
                    key=lambda r: (
                        getattr(r, "relevance_score", None) or 0,
                        getattr(r, "citation_count", None) or 0,
                    ),
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
                _lit_csv_path = os.path.abspath(csv_file)

                # HITL Checkpoint 1
                ui.hitl_checkpoint(
                    1, "Literature Review",
                    summary={
                        "Literature Items": len(literature_batch.literature_items),
                        "Max papers/question": max_papers_per_question,
                    },
                    preview={
                        "title": f"Top {len(_preview_items)} of {len(literature_batch.literature_items)} gathered papers:",
                        "items": _preview_items,
                        "columns": [
                            ("rank", "#", 3),
                            ("title", "Title", 60),
                            ("source", "Source", 12),
                            ("year", "Year", 5),
                            ("cites", "Cites", 6),
                        ],
                        "full_contents_path": _lit_csv_path,
                        "full_contents_label": "Full literature list",
                    },
                )

                proceed = auto_input(
                    "Approve and proceed to Stage 2? (yes/no/refine): ",
                    default=get_default("approve_stage")
                ).strip().lower()

                if proceed in ['yes', 'y']:
                    ui.hitl_response("approved")
                    logger.end_stage("LiteratureGatheringStage", status="success",
                                     item_count=len(literature_batch.literature_items),
                                     output_files=[batch_file, csv_file])
                    state_guard.sign_output(literature_batch.model_dump() if hasattr(literature_batch, 'model_dump') else literature_batch, stage_name="LiteratureGatheringStage")

                    # Schema validation after Stage 1
                    try:
                        if literature_batch:
                            SchemaValidator.validate_or_raise("LiteratureTeam", "LiteratureGatheringStage", {"literature_batch": literature_batch.model_dump(), "metadata": {}})
                    except (ValueError, Exception) as e:
                        ui.warning(f"Schema validation: {e}")
                        logger.log_error(error_type="SchemaValidation", message=str(e), recovered=True)

                    ui.stage_complete(1, status="success", items=len(literature_batch.literature_items),
                                      duration=logger.get_stage_duration("LiteratureGatheringStage"),
                                      output_file=batch_file)

                    # Budget check after Stage 1
                    budget_status = budget.check(collector, current_team="LiteratureTeam")
                    for w in budget_status.warnings:
                        ui.warning(w)

                    # Populate document store with newly gathered literature
                    if literature_batch and hasattr(literature_batch, 'literature_items'):
                        new_docs = []
                        for item in literature_batch.literature_items:
                            doc = Document(
                                doc_id=getattr(item, 'url', '') or getattr(item, 'title', 'unknown'),
                                title=getattr(item, 'title', ''),
                                abstract=getattr(item, 'abstract', ''),
                                authors=getattr(item, 'authors', []),
                                year=getattr(item, 'year', None),
                                source=getattr(item, 'source', ''),
                                url=getattr(item, 'url', ''),
                            )
                            new_docs.append(doc)
                        if new_docs:
                            added = doc_store.add_documents(new_docs, compute_embeddings=False)
                            ui.info(f"Added {added} papers to document store (total: {doc_store.count()})")

                    current_stage = 2
                elif proceed in ['refine', 'r']:
                    ui.hitl_response("refining search parameters")
                    choice = auto_input(
                        "Adjust: 1=More papers, 2=Fewer papers, 3=Add question, 4=Re-run, 5=Exit: ",
                        default=get_default("refine_choice")
                    ).strip()

                    if choice in ['1', '2']:
                        new_max = auto_input(
                            f"New max papers per question (current: {max_papers_per_question}): ",
                            default=get_default("new_max_papers")
                        ).strip()
                        try:
                            max_papers_per_question = int(new_max)
                            ui.info(f"Updated to {max_papers_per_question} papers per question")
                        except ValueError:
                            ui.info("Invalid number. Keeping current value.")
                    elif choice == '3':
                        additional_q = auto_input(
                            "Enter additional research question: ",
                            default=get_default("additional_question")
                        ).strip()
                        if additional_q:
                            formatted_questions.append(stage1.ResearchQuestion(
                                question=additional_q,
                                priority_rank=len(formatted_questions) + 1,
                                priority_score=0.8
                            ))
                            research_questions.append(additional_q)
                            ui.info(f"Added question: {additional_q[:50]}...")
                    elif choice == '5':
                        ui.info("Pipeline cancelled by user")
                        return True  # deliberate user cancel is not a stage failure
                    ui.info("Re-running Stage 1...")
                else:
                    retry = auto_input(
                        "Refine search? (yes/no): ",
                        default=get_default("retry_choice")
                    ).strip().lower()
                    if retry not in ['yes', 'y']:
                        ui.info("Pipeline stopped by user")
                        return True  # deliberate user stop is not a stage failure

            except BudgetExceededError as e:
                ui.error(f"Budget exceeded after Stage 1: {e}")
                logger.log_error(error_type="BudgetExceededError", message=str(e), recovered=False)
                logger.end_stage("LiteratureGatheringStage", status="error", error_message=str(e))
                logger.end_execution(success=False)
                logger.save_execution_log()
                return

            except Exception as e:
                ui.error(f"Stage 1 failed: {e}")
                logger.log_error(error_type=type(e).__name__, message=str(e), recovered=False)
                retry = auto_input("Try again? (yes/no): ", default=get_default("retry_choice")).strip().lower()
                if retry not in ['yes', 'y']:
                    logger.end_stage("LiteratureGatheringStage", status="error", error_message=str(e))
                    workflow_success = False
                    total_errors += 1
                    logger.end_execution(success=False)
                    logger.save_execution_log()
                    ui.workflow_complete(success=False, stages_completed=0, total_errors=total_errors)
                    return

        # ========== STAGE 2: GAP DETECTION ==========
        elif current_stage == 2:
            ui.stage_start(2, "Gap Detection & Knowledge Graph", description="With HITL checkpoint")
            logger.start_stage("GapDetectionStage", 2)

            try:
                stage2 = import_module('2-GapDetectionStage')

                with open(batch_file, 'r', encoding='utf-8') as f:
                    literature_batch_dict = json.load(f)

                with stage_model(model_config, "LiteratureTeam", "GapDetectionStage") as model:
                    ui.info(f"Stage model: {model}")
                    orchestrator2 = stage2.GapDetectionOrchestrator(collector=collector)
                gap_analysis = orchestrator2.run_gap_detection_pipeline(literature_batch_dict)

                orchestrator2.save_gap_analysis(gap_file)
                orchestrator2.save_gaps_csv(gaps_csv)
                orchestrator2.save_graph_json(graph_file)

                ui.agent_result("PaperDecomposer", items=len(gap_analysis.paper_structures))
                ui.agent_result("GapFinder", items=len(gap_analysis.research_gaps))
                ui.agent_result("KnowledgeWeaver", items=len(gap_analysis.knowledge_graph.nodes),
                                message=f"{len(gap_analysis.knowledge_graph.edges)} edges")

                # Build preview: gaps sorted by severity (highest first).
                _severity_order = {"high": 3, "medium": 2, "low": 1}
                _ranked_gaps = sorted(
                    list(gap_analysis.research_gaps),
                    key=lambda g: _severity_order.get(str(getattr(g, "severity", "")).lower(), 0),
                    reverse=True,
                )
                _gap_rows = [
                    {
                        "rank": i + 1,
                        "gap_type": getattr(g, "gap_type", ""),
                        "title": getattr(g, "gap_title", ""),
                        "severity": getattr(g, "severity", ""),
                    }
                    for i, g in enumerate(_ranked_gaps[:10])
                ]
                _gap_path = os.path.abspath(gap_file)

                # HITL Checkpoint 2
                ui.hitl_checkpoint(
                    2, "Gap Analysis Review",
                    summary={
                        "Gaps Found": len(gap_analysis.research_gaps),
                        "Graph Nodes": len(gap_analysis.knowledge_graph.nodes),
                    },
                    preview={
                        "title": f"Top {len(_gap_rows)} of {len(gap_analysis.research_gaps)} research gaps (by severity):",
                        "items": _gap_rows,
                        "columns": [
                            ("rank", "#", 3),
                            ("gap_type", "Type", 18),
                            ("title", "Gap Title", 55),
                            ("severity", "Severity", 8),
                        ],
                        "full_contents_path": _gap_path,
                        "full_contents_label": "Full gap analysis (incl. knowledge graph)",
                    },
                )

                proceed = auto_input(
                    "Approve and proceed to Stage 3? (yes/no): ",
                    default=get_default("approve_stage")
                ).strip().lower()

                if proceed in ['yes', 'y']:
                    ui.hitl_response("approved")
                    logger.end_stage("GapDetectionStage", status="success",
                                     item_count=len(gap_analysis.research_gaps),
                                     output_files=[gap_file, gaps_csv, graph_file])

                    # Schema validation after Stage 2
                    try:
                        if gap_analysis:
                            SchemaValidator.validate_or_raise("LiteratureTeam", "GapDetectionStage", {"gap_analysis": gap_analysis.model_dump(), "metadata": {}})
                    except (ValueError, Exception) as e:
                        ui.warning(f"Schema validation: {e}")
                        logger.log_error(error_type="SchemaValidation", message=str(e), recovered=True)

                    ui.stage_complete(2, status="success", items=len(gap_analysis.research_gaps),
                                      duration=logger.get_stage_duration("GapDetectionStage"),
                                      output_file=gap_file)

                    # Budget check after Stage 2
                    budget_status = budget.check(collector, current_team="LiteratureTeam")
                    for w in budget_status.warnings:
                        ui.warning(w)

                    current_stage = 3
                else:
                    ui.hitl_response("not approved - going back")
                    feedback = auto_input(
                        "1=Go back to Stage 1, 2=Exit: ",
                        default=get_default("go_back_choice")
                    ).strip()
                    if feedback == '1':
                        ui.info("Returning to Stage 1...")
                        current_stage = 1
                    else:
                        ui.info("Pipeline stopped by user")
                        return True  # deliberate user stop is not a stage failure

            except BudgetExceededError as e:
                ui.error(f"Budget exceeded after Stage 2: {e}")
                logger.log_error(error_type="BudgetExceededError", message=str(e), recovered=False)
                logger.end_stage("GapDetectionStage", status="error", error_message=str(e))
                logger.end_execution(success=False)
                logger.save_execution_log()
                return

            except Exception as e:
                ui.error(f"Stage 2 failed: {e}")
                logger.log_error(error_type=type(e).__name__, message=str(e), recovered=False)
                retry = auto_input("Try again? (yes/no): ", default=get_default("retry_choice")).strip().lower()
                if retry not in ['yes', 'y']:
                    logger.end_stage("GapDetectionStage", status="error", error_message=str(e))
                    workflow_success = False
                    total_errors += 1
                    logger.end_execution(success=False)
                    logger.save_execution_log()
                    ui.workflow_complete(success=False, stages_completed=1, total_errors=total_errors)
                    return

        # ========== STAGE 3: SYNTHESIS ==========
        elif current_stage == 3:
            ui.stage_start(3, "Literature Review & Research Plan Synthesis", description="With HITL checkpoint")
            logger.start_stage("SynthesisStage", 3)

            try:
                stage3 = import_module('3-SynthesisStage')

                with open(gap_file, 'r', encoding='utf-8') as f:
                    gap_analysis_dict = json.load(f)

                with stage_model(model_config, "LiteratureTeam", "SynthesisStage") as model:
                    ui.info(f"Stage model: {model}")
                    orchestrator3 = stage3.SynthesisOrchestrator(collector=collector)
                synthesis_result = orchestrator3.run_synthesis_pipeline(
            gap_analysis_dict, orchestrator3.load_literature_metadata(batch_file))

                orchestrator3.save_synthesis_result(synthesis_file)
                orchestrator3.save_literature_review(review_file)
                orchestrator3.save_research_plan(plan_file)
                orchestrator3.save_bibliography(bib_file)
                orchestrator3.save_consolidated_review(os.path.join(script_dir, "consolidated_review.json"))

                ui.agent_result("KnowledgeWeaver", items=len(synthesis_result.literature_review.sections),
                                message="review sections")
                ui.agent_result("PlanArchitect", items=len(synthesis_result.research_plan.research_objectives),
                                message="objectives")
                ui.agent_result("CiteKeeper", items=synthesis_result.bibliography.total_references,
                                message="references")

                # Build preview: review section titles + research objectives side by side.
                _section_rows = [
                    {
                        "rank": i + 1,
                        "section": getattr(s, "section_title", "") or getattr(s, "title", ""),
                    }
                    for i, s in enumerate(synthesis_result.literature_review.sections)
                ]
                _objective_rows = [
                    {
                        "rank": i + 1,
                        "objective": getattr(o, "objective", "") or str(o),
                    }
                    for i, o in enumerate(synthesis_result.research_plan.research_objectives)
                ]
                _review_path = os.path.abspath(review_file)

                # HITL Checkpoint 3
                ui.hitl_checkpoint(
                    3, "Final Synthesis Review",
                    summary={
                        "Review Sections": len(synthesis_result.literature_review.sections),
                        "Objectives": len(synthesis_result.research_plan.research_objectives),
                        "References": synthesis_result.bibliography.total_references,
                    },
                    preview={
                        "sections": [
                            {
                                "subtitle": "Literature review sections",
                                "items": _section_rows,
                                "columns": [
                                    ("rank", "#", 3),
                                    ("section", "Section Title", 72),
                                ],
                            },
                            {
                                "subtitle": "Research objectives",
                                "items": _objective_rows,
                                "columns": [
                                    ("rank", "#", 3),
                                    ("objective", "Objective", 72),
                                ],
                            },
                        ],
                        "full_contents_path": _review_path,
                        "full_contents_label": "Full literature review (also: research_plan.txt, bibliography.txt)",
                    },
                )

                proceed = auto_input(
                    "Approve final outputs? (yes/no): ",
                    default=get_default("approve_stage")
                ).strip().lower()

                if proceed in ['yes', 'y']:
                    ui.hitl_response("approved")
                    logger.end_stage("SynthesisStage", status="success",
                                     item_count=len(synthesis_result.literature_review.sections),
                                     output_files=[synthesis_file, review_file, plan_file, bib_file])

                    # Schema validation after Stage 3
                    try:
                        if synthesis_result:
                            SchemaValidator.validate_or_raise("LiteratureTeam", "SynthesisStage", {"synthesis": synthesis_result.model_dump(), "metadata": {}})
                    except (ValueError, Exception) as e:
                        ui.warning(f"Schema validation: {e}")
                        logger.log_error(error_type="SchemaValidation", message=str(e), recovered=True)

                    # Grounding check on synthesis citations
                    if synthesis_result and hasattr(synthesis_result, 'bibliography'):
                        checker = GroundingChecker(document_store=doc_store)
                        citations_to_check = []
                        refs = getattr(synthesis_result.bibliography, 'formatted_references', [])
                        for ref in refs[:20]:  # Check up to 20 citations
                            citations_to_check.append(Citation(
                                title=getattr(ref, 'title', getattr(ref, 'paper_title', '')),
                                authors=getattr(ref, 'authors', []),
                                year=getattr(ref, 'year', None),
                            ))
                        if citations_to_check:
                            grounding_report = checker.check_citations(citations_to_check)
                            ui.info(f"Citation grounding: {grounding_report.verified}/{grounding_report.total} verified")
                            if grounding_report.hallucinated > 0:
                                ui.warning(f"Hallucinated citations: {grounding_report.hallucinated}")
                                logger.log_error(error_type="HallucinatedCitations",
                                                 message=f"{grounding_report.hallucinated} hallucinated citations",
                                                 recovered=True)

                    ui.stage_complete(3, status="success",
                                      items=len(synthesis_result.literature_review.sections),
                                      duration=logger.get_stage_duration("SynthesisStage"),
                                      output_file=synthesis_file)
                    pipeline_complete = True
                else:
                    ui.hitl_response("not approved")
                    feedback = auto_input(
                        "1=Go back to Stage 2, 2=Go back to Stage 1, 3=Exit: ",
                        default="3"
                    ).strip()
                    if feedback == '1':
                        ui.info("Returning to Stage 2...")
                        current_stage = 2
                    elif feedback == '2':
                        ui.info("Returning to Stage 1...")
                        current_stage = 1
                    else:
                        ui.info("Pipeline stopped by user")
                        return True  # deliberate user stop is not a stage failure

            except Exception as e:
                ui.error(f"Stage 3 failed: {e}")
                logger.log_error(error_type=type(e).__name__, message=str(e), recovered=False)
                retry = auto_input("Try again? (yes/no): ", default=get_default("retry_choice")).strip().lower()
                if retry not in ['yes', 'y']:
                    logger.end_stage("SynthesisStage", status="error", error_message=str(e))
                    workflow_success = False
                    total_errors += 1
                    logger.end_execution(success=False)
                    logger.save_execution_log()
                    ui.workflow_complete(success=False, stages_completed=2, total_errors=total_errors)
                    return


    # V0.7: CitationVerifier post-processor over synthesis text
    try:
        import json as _json
        from shared.verification import (
            run_citation_verifier_pass, extract_review_texts, attach_citation_audit,
        )
        _syn_path = synthesis_file if 'synthesis_file' in dir() else None
        if _syn_path:
            with open(_syn_path, "r", encoding="utf-8") as _f:
                _syn_data = _json.load(_f)
            _texts = extract_review_texts(_syn_data)
            # extract_review_texts only reads top-level string keys, but synthesis
            # nests the review prose under 'literature_review' — pull those blobs in
            # so the auditor scans the actual text rather than nothing.
            _lr = _syn_data.get("literature_review") if isinstance(_syn_data, dict) else None
            if isinstance(_lr, dict):
                for _k in ("abstract", "introduction", "synthesis"):
                    _v = _lr.get(_k)
                    if isinstance(_v, str) and _v.strip():
                        _texts.append(_v)
                for _s in (_lr.get("sections") or []):
                    if isinstance(_s, dict):
                        _sc = _s.get("section_content") or _s.get("content")
                        if isinstance(_sc, str) and _sc.strip():
                            _texts.append(_sc)
            _audit = run_citation_verifier_pass(_texts, collector=collector)
            if _audit:
                if _audit.get("overall_total", 0) == 0:
                    # The review cites papers by full title (no [key]/(Author, Year)
                    # markers), so there is nothing to verify. Report N/A rather than
                    # a misleading 0.0% "clean" rate that looks like a passed audit.
                    _audit["overall_hallucination_rate"] = None
                    _audit["status"] = "N/A"
                    _audit["note"] = ("No parseable in-text citation markers "
                                      "([key] or (Author, Year)) found in the review body; "
                                      "hallucination rate not computable.")
                    attach_citation_audit(_syn_path, _audit)
                    ui.info("CitationVerifier: N/A (no parseable in-text citation markers to audit)")
                else:
                    attach_citation_audit(_syn_path, _audit)
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
        research_topic=research_topic_text[:200] if 'research_topic_text' in dir() else "literature review",
        teams_completed=["LiteratureTeam"] if workflow_success else [],
        teams_failed=[] if workflow_success else ["LiteratureTeam"],
        success=workflow_success,
        mode="ModeWithWcWithHITL",
        artifact_names=[batch_file, gap_file, synthesis_file] if workflow_success else [],
    )
    memory.record_trajectory(trajectory)

    if workflow_success and literature_batch:
        memory.remember_fact(
            f"Gathered {len(literature_batch.literature_items)} papers, found {len(gap_analysis.research_gaps)} gaps",
            category="literature", source="LiteratureTeam/ModeWithWcWithHITL",
        )

    doc_store.close()
    memory.close()

    logger.end_execution(success=workflow_success)
    log_path = logger.save_execution_log()

    ui.workflow_complete(
        success=workflow_success, stages_completed=3, total_errors=total_errors,
        output_files=[review_file, plan_file, bib_file],
        summary={
            "Literature Items": len(literature_batch.literature_items),
            "Research Gaps": len(gap_analysis.research_gaps),
            "Review Sections": len(synthesis_result.literature_review.sections),
            "Research Objectives": len(synthesis_result.research_plan.research_objectives)
        }
    )

    ui.observability_summary(collector)
    from shared.telemetry import export_telemetry
    export_telemetry(collector, str(output_dir), team="LiteratureTeam", mode="ModeWithWcWithHITL")

    if gap_analysis.research_gaps:
        ui.research_summary(
            title="TOP RESEARCH GAPS",
            items=[{"gap": f"[{g.gap_type.upper()}] {g.gap_title}", "severity": g.severity}
                   for g in gap_analysis.research_gaps[:5]],
            item_key="gap", score_key="severity"
        )

    ui.info(f"Execution log: {log_path}")

    # Surface the stage outcome to the caller (runner / CI). __main__ turns a falsy
    # return into a non-zero exit code, so a failed stage is never masked as a clean
    # exit 0.
    return workflow_success


def create_sample_questions():
    """Create sample research questions for demonstration."""
    return [
        {
            "question": "How do agent-based models improve monetary policy analysis compared to traditional DSGE models?",
            "priority_rank": 1,
            "priority_score": 0.95,
            "theoretical_framework": "Agent-based computational economics",
            "methodology": ["Agent-based modeling", "Comparative analysis", "Simulation"]
        },
        {
            "question": "What are the computational challenges in scaling agent-based macroeconomic models to realistic population sizes?",
            "priority_rank": 2,
            "priority_score": 0.88,
            "theoretical_framework": "Computational economics",
            "methodology": ["High-performance computing", "Parallel processing", "Algorithmic optimization"]
        },
        {
            "question": "How can heterogeneous agent models capture the distributional effects of monetary policy?",
            "priority_rank": 3,
            "priority_score": 0.82,
            "theoretical_framework": "Heterogeneous agent macroeconomics",
            "methodology": ["HANK models", "Distributional analysis", "Empirical validation"]
        }
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AEL LiteratureTeam — ModeWithWcWithHITL")
    parser.add_argument("--topic", type=str, default=None, help="Research topic (skips interactive prompt)")
    parser.add_argument("--model", type=str, default=None, help="Override LLM model for all stages")
    args = parser.parse_args()
    if args.model:
        os.environ["AEL_MODEL"] = args.model
    # Non-zero exit on any stage failure so a failed run is never masked as success.
    sys.exit(0 if main(cli_topic=args.topic) else 1)
