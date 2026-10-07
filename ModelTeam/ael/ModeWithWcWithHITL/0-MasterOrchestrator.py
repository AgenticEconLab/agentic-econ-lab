# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Master Orchestrator for Model Development with FireCrawl and Human-in-the-Loop
This script runs all three ModelTeam stages with FireCrawl web scraping and human validation.

Pipeline:
1. Theory Stage: Develop theoretical frameworks (with FireCrawl methodology references)
   -> HITL Checkpoint: Validate theoretical framework before proceeding
2. Model Design Stage: Design computational models (with FireCrawl library documentation)
   -> HITL Checkpoint: Review model design before calibration
3. Calibration Stage: Define calibration strategy (with FireCrawl implementation examples)
   -> HITL Checkpoint: Assess calibration results before final output

FireCrawl Integration:
- Library documentation scraping for implementation guidance
- Methodology references for theoretical foundations
- Implementation examples for calibration techniques

Input: Research questions + Literature review from LiteratureTeam
Output: Complete model specification with theoretical framework, design, and calibration strategy
"""

import os
import sys
import json
import argparse
import requests
from datetime import datetime
from typing import Dict, List
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


# ============================================================================
# Web-crawl Integration (open stack)
# ============================================================================
#
# Migrated off DIRECT Firecrawl onto the open web-crawl/search stack:
#   * SEARCH (query -> URLs):  shared.tools.news_search.search_news
#                              (SearXNG -> DDGS -> Tavily -> Brave; open-first)
#   * CRAWL  (URL  -> content): shared.tools.webcrawl_tool.WebCrawlClient
#                              (trafilatura -> crawl4ai -> tavily -> firecrawl;
#                               Firecrawl is ONLY the last fallback inside the
#                               chain — never a direct api.firecrawl.dev call).
#
# ``WebSearchClient`` keeps the exact method names + string-output shape the
# orchestrator's stages expect, so downstream context-injection is unchanged.

from shared.tools.news_search import search_news, active_search_provider
from shared.tools.webcrawl_tool import WebCrawlClient


class WebSearchClient:
    """Open-stack drop-in for the old FireCrawlClient.

    Search via ``search_news`` (open-first), then optionally enrich the top
    hits with full article text via the open ``WebCrawlClient`` crawl chain.
    No direct Firecrawl call: Firecrawl survives only as the last link of the
    WebCrawlClient fallback chain.
    """

    def __init__(self):
        self._wc = WebCrawlClient(agent="ModelTeamWebCrawler")
        # Always "enabled": the open chain (trafilatura/DDGS) needs no API key.
        self.enabled = True

    def active_provider(self) -> str:
        """Active web-crawl provider for status/logging (e.g. 'trafilatura')."""
        try:
            return self._wc.active_provider() or "trafilatura"
        except Exception:
            return "trafilatura"

    def active_search_provider(self) -> str:
        """Active search provider for status/logging (e.g. 'searxng'/'ddgs')."""
        try:
            return active_search_provider()
        except Exception:
            return "none"

    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search the open web; enrich top results with crawled full text."""
        try:
            hits = search_news(query, max_results=max_results, categories="general")
        except Exception:
            hits = []
        if not hits:
            return []

        results = []
        for item in hits[:max_results]:
            url = item.get("url", "")
            content = item.get("content", "") or ""
            # Enrich with full article text via the open crawl chain.
            if url:
                try:
                    res = self._wc.scrape(url)
                    if res.success and res.markdown:
                        content = res.markdown
                except Exception:
                    pass
            results.append({
                "title": item.get("title", "Web Content"),
                "url": url,
                "content": content[:2000],
                "source": "WebCrawl"
            })
        return results

    def search_library_docs(self, library_name: str, topic: str = "") -> str:
        """Search for library documentation."""
        query = f"{library_name} documentation {topic} Python".strip()
        results = self.search(query, max_results=3)

        if not results:
            return ""

        doc_summary = f"\n--- WebCrawl: Documentation for {library_name} ---\n"
        for i, result in enumerate(results, 1):
            doc_summary += f"\n{i}. {result['title']}\n"
            doc_summary += f"   URL: {result['url']}\n"
            doc_summary += f"   {result['content'][:500]}...\n"

        return doc_summary

    def search_methodology_refs(self, methodology: str) -> str:
        """Search for methodology references."""
        query = f"{methodology} economic modeling methodology reference"
        results = self.search(query, max_results=3)

        if not results:
            return ""

        ref_summary = f"\n--- WebCrawl: Methodology References for {methodology} ---\n"
        for i, result in enumerate(results, 1):
            ref_summary += f"\n{i}. {result['title']}\n"
            ref_summary += f"   URL: {result['url']}\n"
            ref_summary += f"   {result['content'][:500]}...\n"

        return ref_summary

    def search_implementation_examples(self, method_name: str, language: str = "Python") -> str:
        """Search for implementation examples."""
        query = f"{method_name} {language} implementation example calibration"
        results = self.search(query, max_results=3)

        if not results:
            return ""

        examples_summary = f"\n--- WebCrawl: Implementation Examples for {method_name} ---\n"
        for i, result in enumerate(results, 1):
            examples_summary += f"\n{i}. {result['title']}\n"
            examples_summary += f"   URL: {result['url']}\n"
            examples_summary += f"   {result['content'][:600]}...\n"

        return examples_summary

    def search_calibration_data(self, model_type: str, data_source: str = "") -> str:
        """Search for calibration data sources and parameter values."""
        query = f"{model_type} calibration parameters {data_source} economics"
        results = self.search(query, max_results=3)

        if not results:
            return ""

        data_summary = f"\n--- WebCrawl: Calibration Data for {model_type} ---\n"
        for i, result in enumerate(results, 1):
            data_summary += f"\n{i}. {result['title']}\n"
            data_summary += f"   URL: {result['url']}\n"
            data_summary += f"   {result['content'][:500]}...\n"

        return data_summary


# Global web-crawl client (open stack; replaces the direct-Firecrawl client)
firecrawl_client = WebSearchClient()


# ============================================================================
# HITL Helper Functions
# ============================================================================

def human_checkpoint(ui, stage_name, items, item_type="items", preview=None):
    """
    Human-in-the-loop checkpoint for validating stage outputs.

    Args:
        preview: Optional content-preview dict forwarded to
            ``ui.hitl_checkpoint`` so reviewers see a summarized table of
            ``items`` before the decision prompt. See ConsoleUI.hitl_checkpoint
            for the accepted shape.

    Returns:
        tuple: (approved: bool, feedback: str or None)
    """
    ui.hitl_checkpoint(0, stage_name, summary={item_type: len(items)}, preview=preview)

    while True:
        choice = auto_input(
            f"\nYour decision for {stage_name} [A/R/S/Q]: ",
            default=get_default("hitl_decision")
        ).strip().upper()

        if choice == 'A':
            ui.hitl_response("approved")
            return True, None

        elif choice == 'R':
            feedback = auto_input(
                "Feedback: ",
                default=get_default("feedback_text")
            ).strip()
            if not feedback:
                feedback = "No feedback provided (auto-mode)"
            ui.hitl_response(f"rejected: {feedback[:50]}")
            return False, feedback

        elif choice == 'S':
            confirm = auto_input(
                "Are you sure you want to skip validation? [y/N]: ",
                default=get_default("skip_confirm")
            ).strip().lower()
            if confirm == 'y':
                ui.hitl_response("skipped")
                return True, "SKIPPED"
            continue

        elif choice == 'Q':
            ui.hitl_response("quit")
            return False, "USER_QUIT"


def main(cli_topic=None):
    """Main orchestrator with FireCrawl and human-in-the-loop checkpoints."""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    output_dir = Path(script_dir)

    # Initialize Console UI
    ui = ConsoleUI(
        team="ModelTeam",
        mode="ModeWithWcWithHITL",
        total_stages=3
    )

    # Initialize workflow logger
    logger = WorkflowLogger(
        team="ModelTeam",
        mode="ModeWithWcWithHITL",
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
    memory.set_context("mode", "ModeWithWcWithHITL")

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
    webcrawl_provider = firecrawl_client.active_provider()
    search_provider = firecrawl_client.active_search_provider()
    firecrawl_status = f"{webcrawl_provider} (open stack)"
    logger.start_execution(metadata={
        "research_questions_file": research_questions_file or "cli_topic",
        "literature_file": literature_file if literature_file else "Not provided",
        "webcrawl_provider": webcrawl_provider,
        "search_provider": search_provider,
    })

    # Print workflow header
    ui.workflow_header(
        topic=cli_topic or os.path.basename(research_questions_file or ""),
        execution_id=logger.execution_id,
        firecrawl=firecrawl_status,
        literature=os.path.basename(literature_file) if literature_file else "None",
        mode_features="Open web-crawl + HITL checkpoints"
    )

    # Show loaded API keys
    ui.show_api_keys()

    # ========== STAGE 1: THEORY DEVELOPMENT (with FireCrawl) ==========
    ui.stage_start(1, "Theoretical Framework Development", description="FireCrawl methodology refs + HITL")
    logger.start_stage("TheoryStage", 1)
    theory_file = "theory_output.json"

    try:
        from importlib import import_module
        stage1 = import_module('1-TheoryStage')

        with stage_model(model_config, "ModelTeam", "TheoryStage") as model:
            ui.info(f"Stage model: {model}")
            orchestrator1 = stage1.TheoryStageOrchestrator(collector=collector)

        if cli_topic:
            questions = [{"question": cli_topic, "priority_rank": 1, "priority_score": 0.95}]
        else:
            with open(research_questions_file, 'r', encoding='utf-8') as f:
                rq_data = json.load(f)

            if isinstance(rq_data, dict):
                questions = rq_data.get('questions', rq_data.get('final_questions', []))
            else:
                questions = rq_data

        literature_data = None
        if literature_file and os.path.exists(literature_file):
            if literature_file.endswith('.json'):
                with open(literature_file, 'r', encoding='utf-8') as f:
                    literature_data = json.load(f)
            elif literature_file.endswith('.txt'):
                with open(literature_file, 'r', encoding='utf-8') as f:
                    literature_text = f.read()
                    literature_data = {"review_text": literature_text}

        # FireCrawl enrichment: Search for methodology references
        firecrawl_context = ""
        if firecrawl_client.enabled:
            ui.info("WebCrawl: Searching for methodology references...")
            methodology_keywords = ["DSGE", "agent-based", "heterogeneous agent", "OLG",
                                   "general equilibrium", "game theory", "mechanism design"]
            for keyword in methodology_keywords:
                for q in questions[:3]:
                    q_text = q.get('question', str(q)) if isinstance(q, dict) else str(q)
                    if keyword.lower() in q_text.lower():
                        refs = firecrawl_client.search_methodology_refs(keyword)
                        if refs:
                            firecrawl_context += refs
                        break

        if firecrawl_context and hasattr(orchestrator1, 'set_external_context'):
            orchestrator1.set_external_context(firecrawl_context)

        theory_output = orchestrator1.run_theory_pipeline(
            research_questions=questions,
            literature_batch=literature_data or {}
        )

        orchestrator1.save_theory_output(theory_file)
        frameworks = theory_output.theoretical_frameworks if theory_output else []

        # Save FireCrawl references
        if firecrawl_context:
            fc_refs_file = "webcrawl_methodology_refs.txt"
            with open(fc_refs_file, 'w', encoding='utf-8') as f:
                f.write(f"WebCrawl Methodology References\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*60 + "\n\n")
                f.write(firecrawl_context)

        ui.agent_result("Theorist", items=len(frameworks), message="frameworks developed")

    except BudgetExceededError as e:
        ui.error(f"Budget exceeded after Stage 1: {e}")
        logger.log_error(error_type="BudgetExceededError", message=str(e), recovered=False)
        logger.end_stage("TheoryStage", status="error", error_message=str(e))
        logger.end_execution(success=False)
        logger.save_execution_log()
        sys.exit(1)

    except Exception as e:
        ui.error(f"Stage 1 failed: {e}")
        logger.log_error(error_type=type(e).__name__, message=str(e), recovered=False)
        logger.end_stage("TheoryStage", status="error", error_message=str(e))
        workflow_success = False
        total_errors += 1
        ui.workflow_complete(success=False, stages_completed=0, total_errors=total_errors)
        logger.end_execution(success=False)
        logger.save_execution_log()
        sys.exit(1)

    # HITL Checkpoint 1: Validate Theory
    _fw_rows = [
        {
            "rank": i + 1,
            "title": getattr(fw, "framework_title", "") or getattr(fw, "title", ""),
            "assumptions": len(getattr(fw, "assumptions", []) or []),
            "predictions": len(getattr(fw, "testable_predictions", []) or []),
        }
        for i, fw in enumerate(frameworks)
    ]
    _theory_path = os.path.abspath(theory_file)
    approved, feedback = human_checkpoint(
        ui, "Theory Validation", frameworks, "Frameworks",
        preview={
            "title": f"Theoretical frameworks ({len(frameworks)}):",
            "items": _fw_rows,
            "columns": [
                ("rank", "#", 3),
                ("title", "Framework Title", 62),
                ("assumptions", "#Assump", 8),
                ("predictions", "#Preds", 7),
            ],
            "full_contents_path": _theory_path,
            "full_contents_label": "Full theory output (incl. assumptions, equations, predictions)",
        },
    )

    if not approved:
        if feedback and feedback != "USER_QUIT":
            feedback_file = f"theory_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(feedback_file, 'w', encoding='utf-8') as f:
                f.write(f"Theory Stage Feedback\nDate: {datetime.now().isoformat()}\n\n{feedback}")
        logger.end_stage("TheoryStage", status="error", error_message="HITL not approved")
        workflow_success = False
        total_errors += 1
        ui.workflow_complete(success=False, stages_completed=0, total_errors=total_errors)
        logger.end_execution(success=False)
        logger.save_execution_log()
        sys.exit(1)

    # Count granular items for scalability
    theory_item_count = 0
    for fw in frameworks:
        theory_item_count += len(getattr(fw, 'assumptions', []))
        theory_item_count += len(getattr(fw, 'conceptual_components', []))
        theory_item_count += len(getattr(fw, 'mathematical_formulations', []))
        theory_item_count += len(getattr(fw, 'testable_predictions', []))
    theory_item_count = max(theory_item_count, len(frameworks))

    ui.stage_complete(1, status="success", items=theory_item_count,
                      duration=logger.get_stage_duration("TheoryStage"), output_file=theory_file)
    logger.end_stage("TheoryStage", status="success", item_count=theory_item_count,
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

    # ========== STAGE 2: MODEL DESIGN (with FireCrawl) ==========
    ui.stage_start(2, "Computational Model Design", description="FireCrawl library docs + HITL")
    logger.start_stage("ModelDesignStage", 2)
    design_file = "model_design_output.json"

    try:
        stage2 = import_module('2-ModelDesignStage')
        with stage_model(model_config, "ModelTeam", "ModelDesignStage") as model:
            ui.info(f"Stage model: {model}")
            orchestrator2 = stage2.ModelDesignOrchestrator(collector=collector)

        with open(theory_file, 'r', encoding='utf-8') as f:
            theory_data = json.load(f)

        # FireCrawl enrichment: Search for library documentation
        firecrawl_docs = ""
        if firecrawl_client.enabled:
            ui.info("WebCrawl: Searching for library documentation...")
            libraries = [
                ("numpy", "numerical computing arrays"),
                ("scipy", "optimization solving"),
                ("numba", "JIT compilation performance"),
                ("dynare", "DSGE solution"),
                ("mesa", "agent-based modeling"),
            ]
            for lib_name, topic in libraries[:3]:
                docs = firecrawl_client.search_library_docs(lib_name, topic)
                if docs:
                    firecrawl_docs += docs

        if firecrawl_docs and hasattr(orchestrator2, 'set_external_context'):
            orchestrator2.set_external_context(firecrawl_docs)

        design_output = orchestrator2.run_design_pipeline(
            theory_output=theory_data
        )

        orchestrator2.save_design_output(design_file)
        models = design_output.formal_models if design_output else []

        # Save FireCrawl documentation
        if firecrawl_docs:
            fc_docs_file = "webcrawl_library_docs.txt"
            with open(fc_docs_file, 'w', encoding='utf-8') as f:
                f.write(f"WebCrawl Library Documentation\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*60 + "\n\n")
                f.write(firecrawl_docs)

        ui.agent_result("ModelDesigner", items=len(models), message="models designed")

    except BudgetExceededError as e:
        ui.error(f"Budget exceeded after Stage 2: {e}")
        logger.log_error(error_type="BudgetExceededError", message=str(e), recovered=False)
        logger.end_stage("ModelDesignStage", status="error", error_message=str(e))
        logger.end_execution(success=False)
        logger.save_execution_log()
        sys.exit(1)

    except Exception as e:
        ui.error(f"Stage 2 failed: {e}")
        logger.log_error(error_type=type(e).__name__, message=str(e), recovered=False)
        logger.end_stage("ModelDesignStage", status="error", error_message=str(e))
        workflow_success = False
        total_errors += 1
        ui.workflow_complete(success=False, stages_completed=1, total_errors=total_errors)
        logger.end_execution(success=False)
        logger.save_execution_log()
        sys.exit(1)

    # HITL Checkpoint 2: Review Model Design
    _model_rows = [
        {
            "rank": i + 1,
            "name": getattr(m, "model_name", "") or getattr(m, "name", f"model_{i + 1}"),
            "variables": len(getattr(m, "variables", []) or []),
            "params": len(getattr(m, "parameters", []) or []),
            "equations": len(getattr(m, "equations", []) or []),
        }
        for i, m in enumerate(models)
    ]
    _design_path = os.path.abspath(design_file)
    approved, feedback = human_checkpoint(
        ui, "Model Design Review", models, "Models",
        preview={
            "title": f"Formal models ({len(models)}):",
            "items": _model_rows,
            "columns": [
                ("rank", "#", 3),
                ("name", "Model", 40),
                ("variables", "#Vars", 6),
                ("params", "#Params", 8),
                ("equations", "#Eqs", 5),
            ],
            "full_contents_path": _design_path,
            "full_contents_label": "Full model design (incl. equations, constraints)",
        },
    )

    if not approved:
        if feedback and feedback != "USER_QUIT":
            feedback_file = f"model_design_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(feedback_file, 'w', encoding='utf-8') as f:
                f.write(f"Model Design Feedback\nDate: {datetime.now().isoformat()}\n\n{feedback}")
        logger.end_stage("ModelDesignStage", status="error", error_message="HITL not approved")
        workflow_success = False
        total_errors += 1
        ui.workflow_complete(success=False, stages_completed=1, total_errors=total_errors)
        logger.end_execution(success=False)
        logger.save_execution_log()
        sys.exit(1)

    # Count granular items for scalability
    design_item_count = 0
    for m in models:
        design_item_count += len(getattr(m, 'variables', []))
        design_item_count += len(getattr(m, 'parameters', []))
        design_item_count += len(getattr(m, 'equations', []))
        design_item_count += len(getattr(m, 'constraints', []))
    design_item_count = max(design_item_count, len(models))

    ui.stage_complete(2, status="success", items=design_item_count,
                      duration=logger.get_stage_duration("ModelDesignStage"), output_file=design_file)
    logger.end_stage("ModelDesignStage", status="success", item_count=design_item_count,
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

    # ========== STAGE 3: CALIBRATION (with FireCrawl) ==========
    ui.stage_start(3, "Calibration & Parameter Identification", description="FireCrawl examples + HITL")
    logger.start_stage("CalibrationStage", 3)
    calibration_file = "calibration_output.json"

    try:
        stage3 = import_module('3-CalibrationStage')
        with stage_model(model_config, "ModelTeam", "CalibrationStage") as model:
            ui.info(f"Stage model: {model}")
            orchestrator3 = stage3.CalibrationOrchestrator(collector=collector)

        with open(design_file, 'r', encoding='utf-8') as f:
            design_data = json.load(f)

        # FireCrawl enrichment: Implementation examples and calibration data
        firecrawl_calibration = ""
        if firecrawl_client.enabled:
            ui.info("WebCrawl: Searching for calibration resources...")

            calibration_methods = [
                "simulated method of moments",
                "maximum likelihood estimation",
                "Bayesian estimation DSGE",
                "GMM estimation economics",
            ]
            for method in calibration_methods[:2]:
                examples = firecrawl_client.search_implementation_examples(method)
                if examples:
                    firecrawl_calibration += examples

            data_sources = [
                ("macroeconomic", "FRED data"),
                ("labor market", "BLS statistics"),
            ]
            for model_type, source in data_sources[:1]:
                data_refs = firecrawl_client.search_calibration_data(model_type, source)
                if data_refs:
                    firecrawl_calibration += data_refs

        if firecrawl_calibration and hasattr(orchestrator3, 'set_external_context'):
            orchestrator3.set_external_context(firecrawl_calibration)

        calibration_output = orchestrator3.run_calibration_pipeline(
            model_design_output=design_data
        )

        orchestrator3.save_calibration_output(calibration_file)
        calibration_strategies = calibration_output.calibrated_models if calibration_output else []

        # Save FireCrawl calibration resources
        if firecrawl_calibration:
            fc_calib_file = "webcrawl_calibration_resources.txt"
            with open(fc_calib_file, 'w', encoding='utf-8') as f:
                f.write(f"WebCrawl Calibration Resources\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*60 + "\n\n")
                f.write(firecrawl_calibration)

        ui.agent_result("Calibrator", items=len(calibration_strategies), message="strategies developed")

    except Exception as e:
        ui.error(f"Stage 3 failed: {e}")
        logger.log_error(error_type=type(e).__name__, message=str(e), recovered=False)
        logger.end_stage("CalibrationStage", status="error", error_message=str(e))
        workflow_success = False
        total_errors += 1
        ui.workflow_complete(success=False, stages_completed=2, total_errors=total_errors)
        logger.end_execution(success=False)
        logger.save_execution_log()
        sys.exit(1)

    # HITL Checkpoint 3: Assess Calibration
    _calib_rows = [
        {
            "rank": i + 1,
            "name": getattr(cs, "model_name", "") or getattr(cs, "name", f"strategy_{i + 1}"),
            "params": len(getattr(cs, "calibrated_parameters", []) or []),
            "targets": len(getattr(cs, "empirical_targets", []) or []),
            "robustness": len(getattr(cs, "robustness_checks", []) or []),
        }
        for i, cs in enumerate(calibration_strategies)
    ]
    _calib_path = os.path.abspath(calibration_file)
    approved, feedback = human_checkpoint(
        ui, "Calibration Assessment", calibration_strategies, "Strategies",
        preview={
            "title": f"Calibration strategies ({len(calibration_strategies)}):",
            "items": _calib_rows,
            "columns": [
                ("rank", "#", 3),
                ("name", "Strategy / Model", 40),
                ("params", "#Params", 8),
                ("targets", "#Targets", 9),
                ("robustness", "#RobustChk", 11),
            ],
            "full_contents_path": _calib_path,
            "full_contents_label": "Full calibration (incl. parameters, moments, robustness)",
        },
    )

    if not approved:
        if feedback and feedback != "USER_QUIT":
            feedback_file = f"calibration_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(feedback_file, 'w', encoding='utf-8') as f:
                f.write(f"Calibration Feedback\nDate: {datetime.now().isoformat()}\n\n{feedback}")
        logger.end_stage("CalibrationStage", status="error", error_message="HITL not approved")
        workflow_success = False
        total_errors += 1
        ui.workflow_complete(success=False, stages_completed=2, total_errors=total_errors)
        logger.end_execution(success=False)
        logger.save_execution_log()
        sys.exit(1)

    # Count granular items for scalability
    calib_item_count = 0
    for c in calibration_strategies:
        calib_item_count += len(getattr(c, 'empirical_targets', []))
        calib_item_count += len(getattr(c, 'calibrated_parameters', []))
        calib_item_count += len(getattr(c, 'model_moments', []))
        calib_item_count += len(getattr(c, 'robustness_checks', []))
    calib_item_count = max(calib_item_count, len(calibration_strategies))

    ui.stage_complete(3, status="success", items=calib_item_count,
                      duration=logger.get_stage_duration("CalibrationStage"), output_file=calibration_file)
    logger.end_stage("CalibrationStage", status="success", item_count=calib_item_count,
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
            calibration_file, str(output_dir), mode="ModeWithWcWithHITL", collector=collector,
        )
        if _sim_payload and "simulation_result" in _sim_payload:
            ui.info(f"Simulation: scenario={_sim_payload['simulation_result'].get('scenario')}")
    except Exception as _e:
        ui.warning(f"Post-calibration hooks skipped: {_e}")

    # ========== WORKFLOW COMPLETE ==========
    # Record trajectory in episodic memory
    from shared.memory.episodic import RunTrajectory
    trajectory = RunTrajectory(
        run_id=logger.execution_id,
        research_topic=cli_topic or os.path.basename(research_questions_file or ""),
        teams_completed=["ModelTeam"] if workflow_success else [],
        teams_failed=[] if workflow_success else ["ModelTeam"],
        success=workflow_success,
        mode="ModeWithWcWithHITL",
        artifact_names=[theory_file, design_file, calibration_file] if workflow_success else [],
    )
    memory.record_trajectory(trajectory)

    if workflow_success:
        memory.remember_fact(
            f"Built {len(models)} model(s) with {len(calibration_strategies)} calibration strategies",
            category="modeling", source="ModelTeam/ModeWithWcWithHITL",
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
            "Calibrated": len(calibration_strategies),
            "WebCrawl": firecrawl_status
        }
    )

    ui.observability_summary(collector)
    from shared.telemetry import export_telemetry
    export_telemetry(collector, str(output_dir), team="ModelTeam", mode="ModeWithWcWithHITL")

    ui.info(f"Execution log: {log_path}")

    # Propagate failure to the caller (multirun/runner) so a crashed/aborted stage is
    # NOT silently reported as success and its stale outputs scored. (V0.7 eval finding;
    # matches IdeationTeam's sys.exit(1) pattern.)
    if not workflow_success:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AEL ModelTeam — ModeWithWcWithHITL")
    parser.add_argument("--topic", type=str, default=None, help="Research topic (skips interactive prompt)")
    parser.add_argument("--model", type=str, default=None, help="Override LLM model for all stages")
    args = parser.parse_args()
    if args.model:
        os.environ["AEL_MODEL"] = args.model
    main(cli_topic=args.topic)
