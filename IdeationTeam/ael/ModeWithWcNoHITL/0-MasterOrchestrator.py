# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Master Orchestrator for Automated Research Question Generation with Firecrawl (No Human-in-the-Loop)
This script runs all three stages sequentially in a fully automated pipeline with Firecrawl web scraping.

Pipeline:
1. Sourcing Stage: Gather literature from APIs + Firecrawl web scraping (single automated round)
2. Refinement Stage: Generate research concepts and questions (single automated round)
3. Integration Stage: Contextualize and prioritize final questions (single automated round)

Input: Research keywords or basic research ideas (or auto-fetch trending topics)
Output: Finalized, prioritized research questions
Firecrawl: TrendSurfer and CorpusScout use Firecrawl for web scraping

Modes:
- Manual Mode: User provides research topic
- Auto Mode: Automatically fetch trending economic topics from the web (--auto flag)
"""

import os
import sys
import json
import argparse
import requests
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


def fetch_trending_topics() -> str:
    """Discover trending economic research topics — FULLY OPEN path:
      1. SearXNG news search (open; commercial Tavily/Brave only as fallback)
      2. trafilatura crawl of the top articles (open; crawl4ai/commercial fallback)
      3. local LLM summary via LLMClient (honors AEL_MODEL -> vLLM; no cloud gpt-4o-mini)
    Falls back to static topics if the open path is unavailable.
    """
    try:
        from shared.tools.news_search import search_news, active_search_provider
        from shared.tools.webcrawl_tool import WebCrawlClient
        from shared.llm import LLMClient

        results = search_news("economic research trends latest news", max_results=6)
        if not results:
            print("[trending] no news search results; using fallback topics")
            return _get_fallback_topics()
        print(f"[trending] {len(results)} news hits via '{active_search_provider()}'; crawling top articles...")

        # crawl full article text via the open web-crawl chain (trafilatura -> crawl4ai -> ...)
        wc = WebCrawlClient(agent="TrendFetcher")
        articles = []
        for r in results[:5]:
            text = r.get("content") or ""
            try:
                res = wc.scrape(r["url"])
                if res.success and res.markdown:
                    text = res.markdown[:2000]
            except Exception:
                pass
            snippet = f"{r.get('title', '')}\n{text}".strip()
            if snippet:
                articles.append(snippet)
        combined = "\n\n".join(articles) or "\n".join(r.get("title", "") for r in results)

        # summarize with the LOCAL LLM (LLMClient honors AEL_MODEL; no commercial cloud call)
        llm = LLMClient(temperature=0.3, agent_name="TrendFetcher")
        topic = llm.invoke([
            {"role": "system", "content": "You identify emerging economics research topics from recent news. Reply with ONE concise, comprehensive research-topic statement."},
            {"role": "user", "content": f"Recent economic news:\n\n{combined[:6000]}\n\nReturn a single research topic statement."},
        ])
        topic = (topic or "").strip()
        return topic or _get_fallback_topics()
    except Exception as e:
        print(f"[trending] open path failed ({type(e).__name__}: {e}); using fallback topics")
        return _get_fallback_topics()


def _fetch_with_tavily() -> str:
    """Fallback to Tavily Search API, then Brave."""
    try:
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            return _fetch_with_brave()
        response = requests.post(
            "https://api.tavily.com/search",
            headers={"Content-Type": "application/json"},
            json={"api_key": tavily_api_key, "query": "economic research trends 2025", "max_results": 5},
            timeout=15,
        )
        if response.status_code == 200:
            results = response.json().get("results", [])
            topics = [r.get("title", "") for r in results[:5]]
            return "; ".join(topics) if topics else _fetch_with_brave()
        return _fetch_with_brave()
    except Exception:
        return _fetch_with_brave()


def _fetch_with_brave() -> str:
    """Fallback to Brave Search API."""
    try:
        brave_api_key = os.getenv("BRAVE_API_KEY")
        if not brave_api_key:
            return _get_fallback_topics()
        headers = {"Accept": "application/json", "X-Subscription-Token": brave_api_key}
        params = {"q": "economic research trends 2025", "count": 5}
        response = requests.get("https://api.search.brave.com/res/v1/web/search",
                                headers=headers, params=params, timeout=15)
        if response.status_code == 200:
            results = response.json().get("web", {}).get("results", [])
            topics = [r.get("title", "") for r in results[:5]]
            return "; ".join(topics) if topics else _get_fallback_topics()
        return _get_fallback_topics()
    except Exception:
        return _get_fallback_topics()


def _get_fallback_topics() -> str:
    """Return default trending topics when APIs are unavailable."""
    return (
        "Impact of AI and automation on labor markets and wage inequality; "
        "Central bank digital currencies and monetary policy effectiveness; "
        "Climate change economics and green transition financing"
    )


# Clear cached modules
for module_name in ['1-SourcingStage', '2-RefinementStage', '3-IntegrationStage']:
    if module_name in sys.modules:
        del sys.modules[module_name]


def main(auto_mode=False, cli_topic=None):
    """Main orchestrator with Firecrawl, no HITL."""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    output_dir = Path(script_dir)

    # Initialize Console UI
    ui = ConsoleUI(
        team="IdeationTeam",
        mode="ModeWithWcNoHITL",
        total_stages=3
    )

    # ========== CONFIGURATION ==========
    if cli_topic:
        research_topic = cli_topic
    elif auto_mode:
        ui.info("AUTO MODE: Fetching trending economic topics from the web...")
        research_topic = fetch_trending_topics()
    else:
        research_topic = auto_input(
            "Enter your research topic or keywords: ",
            default=get_default("research_topic")
        ).strip()
        if not research_topic:
            research_topic = "Agent-based modeling in macroeconomics and monetary policy"

    # Initialize workflow logger
    logger = WorkflowLogger(
        team="IdeationTeam",
        mode="ModeWithWcNoHITL",
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
    memory.set_context("mode", "ModeWithWcNoHITL")

    # Recall similar past runs
    similar_runs = memory.recall_similar_runs(research_topic)
    if similar_runs:
        ui.info(f"Found {len(similar_runs)} similar past run(s) in memory")

    # Start execution tracking
    logger.start_execution(metadata={
        "research_topic": research_topic,
        "firecrawl_enabled": bool(os.getenv("FIRECRAWL_API_KEY")),
        "auto_mode": auto_mode
    })

    # Print workflow header
    firecrawl_status = "enabled" if os.getenv("FIRECRAWL_API_KEY") else "disabled"
    ui.workflow_header(
        topic=research_topic,
        execution_id=logger.execution_id,
        firecrawl=firecrawl_status,
        auto_mode=str(auto_mode)
    )

    # Show loaded API keys
    ui.show_api_keys()

    # Track results
    workflow_success = True
    total_errors = 0
    literature_count = 0
    refined_count = 0
    final_questions = []

    # ========== STAGE 1: LITERATURE SOURCING ==========
    ui.stage_start(1, "Literature Sourcing", description="Firecrawl-enhanced web scraping")
    logger.start_stage("Sourcing", 1)
    literature_file = "literature_results_automated.csv"

    try:
        from importlib import import_module
        stage1 = import_module('1-SourcingStage')

        with stage_model(model_config, "IdeationTeam", "SourcingStage") as model:
            ui.info(f"Stage model: {model}")
            orchestrator1 = stage1.MultiAgentOrchestrator(quiet=True, collector=collector)
        literature_results = orchestrator1.run_automated_search(
            research_topic=research_topic,
            max_results_per_agent=8
        )
        orchestrator1.save_results(literature_file)

        agent_stats = orchestrator1.get_agent_stats() if hasattr(orchestrator1, 'get_agent_stats') else {}
        for agent_name, stats in agent_stats.items():
            ui.agent_result(agent_name, items=stats.get('items', 0), errors=stats.get('errors', 0))

        literature_count = len(literature_results)
        stage_errors = sum(s.get('errors', 0) for s in agent_stats.values())
        total_errors += stage_errors

        ui.stage_complete(1, status="success", items=literature_count,
                          duration=logger.get_stage_duration("Sourcing"), output_file=literature_file)
        logger.end_stage("Sourcing", status="success", item_count=literature_count,
                         output_files=[literature_file])

        # Schema validation after Stage 1 (CSV output — best-effort)
        try:
            stage1_data = {
                "literature_items": [r.model_dump() if hasattr(r, 'model_dump') else r for r in literature_results],
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
        logger.end_stage("Sourcing", status="error", error_message=str(e))
        logger.end_execution(success=False)
        logger.save_execution_log()
        return

    except Exception as e:
        ui.error(f"Stage 1 failed: {e}")
        logger.log_error(error_type=type(e).__name__, message=str(e), recovered=False)
        logger.end_stage("Sourcing", status="error", error_message=str(e))
        workflow_success = False
        total_errors += 1
        ui.workflow_complete(success=False, stages_completed=0, total_errors=total_errors)
        logger.end_execution(success=False)
        logger.save_execution_log()
        return

    # ========== STAGE 2: REFINEMENT ==========
    ui.stage_start(2, "Research Question Refinement")
    logger.start_stage("Refinement", 2)
    refinement_file = "refinement_results_automated.json"

    try:
        stage2 = import_module('2-RefinementStage')
        with stage_model(model_config, "IdeationTeam", "RefinementStage") as model:
            ui.info(f"Stage model: {model}")
            orchestrator2 = stage2.RefinementOrchestrator(quiet=True, collector=collector)
        literature_df = orchestrator2.load_literature(literature_file)

        refined_questions = orchestrator2.run_automated_refinement(
            literature_df=literature_df, num_concepts=10, num_questions=8
        )
        orchestrator2.save_results(refinement_file)

        agent_stats = orchestrator2.get_agent_stats() if hasattr(orchestrator2, 'get_agent_stats') else {}
        if agent_stats:
            for agent_name, stats in agent_stats.items():
                ui.agent_result(agent_name, items=stats.get('items', 0), message=stats.get('message'))
        else:
            ui.agent_result("Ideator", items=len(orchestrator2.all_concepts) if hasattr(orchestrator2, 'all_concepts') else 0)
            ui.agent_result("Refiner", items=len(refined_questions))

        refined_count = len(refined_questions)

        ui.stage_complete(2, status="success", items=refined_count,
                          duration=logger.get_stage_duration("Refinement"), output_file=refinement_file)
        logger.end_stage("Refinement", status="success", item_count=refined_count,
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
        logger.end_stage("Refinement", status="error", error_message=str(e))
        logger.end_execution(success=False)
        logger.save_execution_log()
        return

    except Exception as e:
        ui.error(f"Stage 2 failed: {e}")
        logger.log_error(error_type=type(e).__name__, message=str(e), recovered=False)
        logger.end_stage("Refinement", status="error", error_message=str(e))
        workflow_success = False
        total_errors += 1
        ui.workflow_complete(success=False, stages_completed=1, total_errors=total_errors)
        logger.end_execution(success=False)
        logger.save_execution_log()
        return

    # ========== STAGE 3: INTEGRATION ==========
    ui.stage_start(3, "Question Integration & Prioritization")
    logger.start_stage("Integration", 3)
    final_file = "finalized_research_questions_automated.json"
    final_txt_file = "finalized_research_questions_automated.txt"

    try:
        stage3 = import_module('3-IntegrationStage')
        with stage_model(model_config, "IdeationTeam", "IntegrationStage") as model:
            ui.info(f"Stage model: {model}")
            orchestrator3 = stage3.IntegrationOrchestrator(quiet=True, collector=collector)
            orchestrator3.research_topic = research_topic  # seed topic for the topic check
        initial_questions = orchestrator3.load_refinement_results(refinement_file)

        final_questions = orchestrator3.run_automated_integration(
            questions=initial_questions, max_final_questions=5
        )
        orchestrator3.save_final_questions(final_file)

        ui.agent_result("Contextualizer", items=len(initial_questions))
        ui.agent_result("Finalizer", items=len(final_questions))

        ui.stage_complete(3, status="success", items=len(final_questions),
                          duration=logger.get_stage_duration("Integration"), output_file=final_file)
        logger.end_stage("Integration", status="success", item_count=len(final_questions),
                         output_files=[final_file, final_txt_file])

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
        logger.end_stage("Integration", status="error", error_message=str(e))
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
        mode="ModeWithWcNoHITL",
        artifact_names=[final_file, final_txt_file] if workflow_success else [],
    )
    memory.record_trajectory(trajectory)

    # Remember key facts for future runs
    if workflow_success and final_questions:
        memory.remember_fact(
            f"Generated {len(final_questions)} research questions on '{research_topic}'",
            category="ideation", source="IdeationTeam/ModeWithWcNoHITL",
        )

    memory.close()

    logger.end_execution(success=workflow_success)
    log_path = logger.save_execution_log()

    ui.workflow_complete(
        success=workflow_success, stages_completed=3, total_errors=total_errors,
        output_files=[final_file, final_txt_file],
        summary={"Literature Papers": literature_count, "Refined Questions": refined_count,
                 "Final Questions": len(final_questions)}
    )

    ui.observability_summary(collector)
    from shared.telemetry import export_telemetry
    export_telemetry(collector, str(output_dir), team="IdeationTeam", mode="ModeWithWcNoHITL")

    if final_questions:
        ui.research_summary(
            title="FINALIZED RESEARCH QUESTIONS",
            items=[{"question": q.question, "priority_score": q.priority_score} for q in final_questions],
            item_key="question", score_key="priority_score"
        )

    ui.info(f"Execution log: {log_path}")

    # Propagate failure to the caller (multirun/runner) so a crashed stage is NOT
    # silently reported as success and its stale outputs scored.
    if not workflow_success:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AEL IdeationTeam — ModeWithWcNoHITL")
    parser.add_argument('--auto', action='store_true', help='Auto mode: fetch trending topics')
    parser.add_argument("--topic", type=str, default=None, help="Research topic (skips interactive prompt)")
    parser.add_argument("--model", type=str, default=None, help="Override LLM model for all stages")
    args = parser.parse_args()
    if args.model:
        os.environ["AEL_MODEL"] = args.model
    main(auto_mode=args.auto, cli_topic=args.topic)
