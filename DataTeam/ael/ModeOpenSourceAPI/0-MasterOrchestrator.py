# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Master Orchestrator for Open Source API Data Pipeline
Retrieves data from open-source APIs (FRED, Yahoo Finance, World Bank, OECD).

Pipeline:
1. Data Source Stage: Discover and acquire data from open-source APIs
2. Data Cleaning Stage: Clean, align temporally, and integrate multi-source data
3. Quality Assurance Stage: Validate, document, and generate report/codebook

Input: Research question or data requirements
Output: High-quality, validated, and documented datasets with codebook
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
from shared.guardrails.pii_detector import PIIDetector
from shared.guardrails.schema_validator import SchemaValidator
from shared.memory.memory_manager import MemoryManager
from shared.tools.register_all import register_all_tools
from shared.llm_router import ModelRouter
from shared.model_config import load_model_config, stage_model
from shared.reliability.state_guard import StateGuard
from shared.protocols.mcp_fred_config import get_fred_mcp_client, FRED_MCP_ENABLED


# ============================================================================
# Environment Configuration
# ============================================================================

def get_env_path():
    """Find .env file, checking current dir then walking up to the repository root."""
    # Check current directory and parent directories first (team-level .env)
    current_dir = Path(__file__).resolve().parent
    for _ in range(5):
        env_path = current_dir / ".env"
        if env_path.exists():
            return str(env_path)
        if (current_dir / "ael_config.yaml").exists() and (current_dir / "run_ael_pipeline.py").exists():
            break
        current_dir = current_dir.parent
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
# Available Open Source APIs
# ============================================================================

AVAILABLE_APIS = {
    "FRED": {
        "name": "Federal Reserve Economic Data",
        "url": "https://fred.stlouisfed.org/",
        "requires_key": True,
        "key_env_var": "FRED_API_KEY",
        "description": "Comprehensive economic time series from Federal Reserve Bank of St. Louis"
    },
    "Yahoo Finance": {
        "name": "Yahoo Finance",
        "url": "https://finance.yahoo.com/",
        "requires_key": False,
        "key_env_var": None,
        "description": "Stock prices, financial data, market indices"
    },
    "World Bank": {
        "name": "World Bank Open Data",
        "url": "https://data.worldbank.org/",
        "requires_key": False,
        "key_env_var": None,
        "description": "Development indicators, global economic data"
    },
    "OECD": {
        "name": "OECD Data",
        "url": "https://data.oecd.org/",
        "requires_key": False,
        "key_env_var": None,
        "description": "Economic statistics from OECD member countries"
    }
}


def check_api_availability():
    """Check which APIs are available based on environment configuration."""
    available = []
    for api_name, api_info in AVAILABLE_APIS.items():
        if api_info["requires_key"]:
            key = os.getenv(api_info["key_env_var"])
            if key:
                available.append(api_name)
        else:
            available.append(api_name)
    return available


def main(cli_topic=None):
    """Main orchestrator for open-source API data pipeline."""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    output_dir = Path(script_dir)

    # Initialize Console UI
    ui = ConsoleUI(
        team="DataTeam",
        mode="ModeOpenSourceAPI",
        total_stages=3
    )

    # Initialize workflow logger
    logger = WorkflowLogger(
        team="DataTeam",
        mode="ModeOpenSourceAPI",
        output_dir=output_dir,
        framework="ael",
        quiet=True
    )

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
        "DataSourceStage": 300,
        "DataCleaningStage": 300,
        "QualityAssuranceStage": 300,
    }

    # Initialize FRED MCP client (optional, env-controlled)
    fred_mcp_client = get_fred_mcp_client()
    if fred_mcp_client:
        ui.info("FRED MCP client initialized for economic data access")

    # Initialize PII detector for data ingestion screening
    pii_detector = PIIDetector()

    # Initialize memory manager (cross-run persistence)
    memory_dir = str(output_dir / "memory")
    memory = MemoryManager(run_id=logger.execution_id, memory_dir=memory_dir)
    memory.set_context("team", "DataTeam")
    memory.set_context("mode", "ModeOpenSourceAPI")

    # Track results
    workflow_success = True
    total_errors = 0
    discovered_sources = []
    retrieved_data = []
    aligned_datasets = []
    integrated_datasets = []
    documented_datasets = []
    requirements = []

    # Check API availability
    available_apis = check_api_availability()

    # Get research question
    if cli_topic:
        research_question = cli_topic
    else:
        research_question = auto_input(
            "Enter your research question: ",
            default=get_default("research_question")
        ).strip()

    if not research_question:
        research_question = "What is the relationship between GDP growth, unemployment, and inflation in the US economy from 1990 to 2023?"

    memory.set_context("research_question", research_question)

    # Recall similar past runs
    similar_runs = memory.recall_similar_runs(research_question)
    if similar_runs:
        ui.info(f"Found {len(similar_runs)} similar past run(s) in memory")

    # Check for data requirements file
    data_requirements_file = auto_input(
        "Enter path to data requirements JSON (optional): ",
        default=""
    ).strip()

    if not data_requirements_file:
        example_requirements = [
            {
                "requirement_id": "REQ1",
                "variable_name": "GDP",
                "description": "Real GDP data for economic analysis",
                "frequency": "quarterly",
                "time_period": "1990-2023",
                "geographic_coverage": "US",
                "unit_of_measurement": "Billions of chained 2012 dollars",
                "priority": "High",
                "suggested_sources": ["FRED", "BEA"]
            },
            {
                "requirement_id": "REQ2",
                "variable_name": "Unemployment Rate",
                "description": "Civilian unemployment rate",
                "frequency": "monthly",
                "time_period": "1990-2023",
                "geographic_coverage": "US",
                "unit_of_measurement": "Percent",
                "priority": "High",
                "suggested_sources": ["FRED", "BLS"]
            },
            {
                "requirement_id": "REQ3",
                "variable_name": "Inflation Rate",
                "description": "Consumer Price Index inflation rate",
                "frequency": "monthly",
                "time_period": "1990-2023",
                "geographic_coverage": "US",
                "unit_of_measurement": "Percent change",
                "priority": "Medium",
                "suggested_sources": ["FRED", "BLS"]
            }
        ]
        data_requirements_file = "api_data_requirements.json"
        with open(data_requirements_file, 'w', encoding='utf-8') as f:
            json.dump({
                "research_question": research_question,
                "data_requirements": example_requirements,
                "available_apis": available_apis
            }, f, indent=2)

    # Start execution tracking
    logger.start_execution(metadata={
        "research_question": research_question,
        "data_requirements_file": data_requirements_file,
        "available_apis": available_apis
    })

    # Print workflow header
    ui.workflow_header(
        topic=research_question[:60],
        execution_id=logger.execution_id,
        apis=", ".join(available_apis)
    )

    # Show loaded API keys
    ui.show_api_keys()

    # ========== STAGE 1: API SOURCE DISCOVERY & DATA RETRIEVAL ==========
    ui.stage_start(1, "API Source Discovery & Data Retrieval")
    logger.start_stage("DataSource", 1)
    source_file = "api_source_output.json"

    try:
        from importlib import import_module
        stage1 = import_module('1-DataSourceStage')

        with stage_model(model_config, "DataTeam", "DataSourceStage") as model:
            ui.info(f"Stage model: {model}")
            orchestrator1 = stage1.DataSourceOrchestrator(collector=collector)

        with open(data_requirements_file, 'r', encoding='utf-8') as f:
            req_data = json.load(f)

        if isinstance(req_data, dict):
            requirements_list = req_data.get('data_requirements', req_data.get('requirements', []))
            research_q = req_data.get('research_question', research_question)
        else:
            requirements_list = req_data
            research_q = research_question

        requirements = []
        for req in requirements_list:
            if isinstance(req, dict):
                req_obj = stage1.DataRequirement(**req)
            else:
                req_obj = req
            requirements.append(req_obj)

        source_output = orchestrator1.run_source_pipeline(
            research_question=research_q,
            data_requirements=requirements,
            available_apis=available_apis,
            enable_hitl=False
        )

        orchestrator1.save_source_output(source_file)

        discovered_sources = source_output.discovered_sources if source_output else []
        retrieved_data = source_output.retrieved_data if source_output else []

        ui.agent_result("SourceDiscovery", items=len(discovered_sources), message="API sources found")
        ui.agent_result("DataRetrieval", items=len(retrieved_data), message="series retrieved")

        ui.stage_complete(1, status="success", items=len(retrieved_data),
                          duration=logger.get_stage_duration("DataSource"), output_file=source_file)
        logger.end_stage("DataSource", status="success",
                         item_count=len(retrieved_data), output_files=[source_file])
        state_guard.sign_output(source_output.model_dump() if hasattr(source_output, 'model_dump') else source_output, stage_name="DataSourceStage")

        # Schema validation after Stage 1
        try:
            if source_output:
                SchemaValidator.validate_or_raise("DataTeam", "DataSourceStage", source_output.model_dump())
        except (ValueError, Exception) as e:
            ui.warning(f"Schema validation: {e}")
            logger.log_error(error_type="SchemaValidation", message=str(e), recovered=True)

        # PII scan on retrieved data
        if source_output:
            pii_report = pii_detector.scan_dict(source_output.model_dump())
            if pii_report.has_pii:
                ui.warning(f"PII detected in data output: {sorted(pii_report.pii_types_found)}")
                logger.log_error(error_type="PIIDetected",
                                 message=f"PII types found: {sorted(pii_report.pii_types_found)}",
                                 recovered=True)

        # Budget check after Stage 1
        budget_status = budget.check(collector, current_team="DataTeam")
        for w in budget_status.warnings:
            ui.warning(w)

    except BudgetExceededError as e:
        ui.error(f"Budget exceeded after Stage 1: {e}")
        logger.log_error(error_type="BudgetExceededError", message=str(e), recovered=False)
        logger.end_stage("DataSource", status="error", error_message=str(e))
        logger.end_execution(success=False)
        logger.save_execution_log()
        sys.exit(1)  # non-zero exit so the runner does not mask this as success

    except Exception as e:
        ui.error(f"Stage 1 encountered error: {e}")
        logger.log_error(error_type=type(e).__name__, message=str(e), recovered=True)
        total_errors += 1

        # Attempt recovery: retry with reduced scope
        try:
            from importlib import import_module as reimport
            stage1_recovery = reimport('1-DataSourceStage')

            ui.info("Attempting recovery with reduced data requirements...")
            recovery_orchestrator = stage1_recovery.DataSourceOrchestrator(collector=collector)

            # Use only first requirement for recovery
            recovery_reqs = []
            for req in requirements[:1]:
                if isinstance(req, dict):
                    recovery_reqs.append(stage1_recovery.DataRequirement(**req))
                else:
                    recovery_reqs.append(req)

            source_output = recovery_orchestrator.run_source_pipeline(
                research_question=research_q,
                data_requirements=recovery_reqs,
                available_apis=available_apis,
                enable_hitl=False
            )

            recovery_orchestrator.save_source_output(source_file)

            discovered_sources = source_output.discovered_sources if source_output else []
            retrieved_data = source_output.retrieved_data if source_output else []

            ui.info(f"Recovery successful: {len(retrieved_data)} series retrieved")
            logger.end_stage("DataSource", status="success",
                             item_count=len(retrieved_data), output_files=[source_file])
        except Exception as recovery_error:
            ui.error(f"Recovery also failed: {recovery_error}")
            logger.log_error(error_type=type(recovery_error).__name__, message=str(recovery_error), recovered=False)
            logger.end_stage("DataSource", status="error", error_message=str(e))
            workflow_success = False
            total_errors += 1
            ui.workflow_complete(success=False, stages_completed=0, total_errors=total_errors)
            logger.end_execution(success=False)
            logger.save_execution_log()
            sys.exit(1)  # non-zero exit so the runner does not mask this as success

    # ========== STAGE 2: TEMPORAL ALIGNMENT & INTEGRATION ==========
    ui.stage_start(2, "Temporal Alignment & Multi-Source Integration")
    logger.start_stage("DataCleaning", 2)
    cleaning_file = "api_cleaning_output.json"

    try:
        stage2 = import_module('2-DataCleaningStage')
        with stage_model(model_config, "DataTeam", "DataCleaningStage") as model:
            ui.info(f"Stage model: {model}")
            orchestrator2 = stage2.DataCleaningOrchestrator(collector=collector)

        with open(source_file, 'r', encoding='utf-8') as f:
            source_data = json.load(f)

        cleaning_output = orchestrator2.run_cleaning_pipeline(
            data_source_output=source_data,
            research_question=research_q,
            enable_hitl=False
        )

        orchestrator2.save_cleaning_output(cleaning_file)

        aligned_datasets = cleaning_output.aligned_datasets if cleaning_output else []
        integrated_datasets = cleaning_output.integrated_datasets if cleaning_output else []

        ui.agent_result("TemporalAligner", items=len(aligned_datasets), message="datasets aligned")
        ui.agent_result("DataIntegrator", items=len(integrated_datasets), message="integrated sets")

        ui.stage_complete(2, status="success", items=len(integrated_datasets),
                          duration=logger.get_stage_duration("DataCleaning"), output_file=cleaning_file)
        logger.end_stage("DataCleaning", status="success",
                         item_count=len(integrated_datasets), output_files=[cleaning_file])

        # Schema validation after Stage 2
        try:
            if cleaning_output:
                SchemaValidator.validate_or_raise("DataTeam", "DataCleaningStage", cleaning_output.model_dump())
        except (ValueError, Exception) as e:
            ui.warning(f"Schema validation: {e}")
            logger.log_error(error_type="SchemaValidation", message=str(e), recovered=True)

        # Budget check after Stage 2
        budget_status = budget.check(collector, current_team="DataTeam")
        for w in budget_status.warnings:
            ui.warning(w)

    except BudgetExceededError as e:
        ui.error(f"Budget exceeded after Stage 2: {e}")
        logger.log_error(error_type="BudgetExceededError", message=str(e), recovered=False)
        logger.end_stage("DataCleaning", status="error", error_message=str(e))
        logger.end_execution(success=False)
        logger.save_execution_log()
        sys.exit(1)  # non-zero exit so the runner does not mask this as success

    except Exception as e:
        ui.error(f"Stage 2 failed: {e}")
        logger.log_error(error_type=type(e).__name__, message=str(e), recovered=False)
        logger.end_stage("DataCleaning", status="error", error_message=str(e))
        workflow_success = False
        total_errors += 1
        ui.workflow_complete(success=False, stages_completed=1, total_errors=total_errors)
        logger.end_execution(success=False)
        logger.save_execution_log()
        sys.exit(1)  # non-zero exit so the runner does not mask this as success

    # ========== STAGE 3: REPORT & CODEBOOK GENERATION ==========
    ui.stage_start(3, "Report & Codebook Generation")
    logger.start_stage("QualityAssurance", 3)
    qa_file = "api_qa_output.json"
    report_file = f"api_data_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.md"

    try:
        stage3 = import_module('3-QualityAssuranceStage')
        with stage_model(model_config, "DataTeam", "QualityAssuranceStage") as model:
            ui.info(f"Stage model: {model}")
            orchestrator3 = stage3.QualityAssuranceOrchestrator(collector=collector)

        with open(cleaning_file, 'r', encoding='utf-8') as f:
            cleaning_data = json.load(f)

        qa_output = orchestrator3.run_qa_pipeline(
            data_cleaning_output=cleaning_data,
            research_question=research_q,
            enable_hitl=False
        )

        orchestrator3.save_qa_output(qa_file)
        orchestrator3.save_report_codebook(report_file)
        manifest_file = "replication_manifest.json"
        orchestrator3.save_replication_manifest(manifest_file)

        documented_datasets = qa_output.documented_datasets if qa_output else []

        ui.agent_result("QualityValidator", items=len(documented_datasets), message="documented")
        ui.agent_result("ReportGenerator", items=1, message=report_file)
        ui.agent_result("Archivist", items=1, message=manifest_file)

        ui.stage_complete(3, status="success", items=len(documented_datasets),
                          duration=logger.get_stage_duration("QualityAssurance"), output_file=qa_file)
        logger.end_stage("QualityAssurance", status="success",
                         item_count=len(documented_datasets), output_files=[qa_file, report_file, manifest_file])

        # Schema validation after Stage 3
        try:
            if qa_output:
                SchemaValidator.validate_or_raise("DataTeam", "QualityAssuranceStage", qa_output.model_dump())
        except (ValueError, Exception) as e:
            ui.warning(f"Schema validation: {e}")
            logger.log_error(error_type="SchemaValidation", message=str(e), recovered=True)

    except Exception as e:
        ui.error(f"Stage 3 failed: {e}")
        logger.log_error(error_type=type(e).__name__, message=str(e), recovered=False)
        logger.end_stage("QualityAssurance", status="error", error_message=str(e))
        workflow_success = False
        total_errors += 1
        ui.workflow_complete(success=False, stages_completed=2, total_errors=total_errors)
        logger.end_execution(success=False)
        logger.save_execution_log()
        sys.exit(1)  # non-zero exit so the runner does not mask this as success

    # ========== WORKFLOW COMPLETE ==========

    # Record trajectory in episodic memory
    from shared.memory.episodic import RunTrajectory
    trajectory = RunTrajectory(
        run_id=logger.execution_id,
        research_topic=research_question,
        teams_completed=["DataTeam"] if workflow_success else [],
        teams_failed=[] if workflow_success else ["DataTeam"],
        success=workflow_success,
        mode="ModeOpenSourceAPI",
        artifact_names=[source_file, cleaning_file, qa_file] if workflow_success else [],
    )
    memory.record_trajectory(trajectory)

    if workflow_success:
        memory.remember_fact(
            f"Retrieved {len(retrieved_data)} data series for '{research_question[:60]}'",
            category="data_collection", source="DataTeam/ModeOpenSourceAPI",
        )

    memory.close()

    logger.end_execution(success=workflow_success)
    log_path = logger.save_execution_log()

    ui.workflow_complete(
        success=workflow_success, stages_completed=3, total_errors=total_errors,
        output_files=[source_file, cleaning_file, qa_file, report_file, manifest_file],
        summary={
            "API Sources": len(discovered_sources),
            "Series Retrieved": len(retrieved_data),
            "Aligned Datasets": len(aligned_datasets),
            "Integrated Datasets": len(integrated_datasets),
            "Documented Datasets": len(documented_datasets)
        }
    )

    ui.observability_summary(collector)
    from shared.telemetry import export_telemetry
    export_telemetry(collector, str(output_dir), team="DataTeam", mode="ModeOpenSourceAPI")

    ui.info(f"Execution log: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AEL DataTeam — ModeOpenSourceAPI")
    parser.add_argument("--topic", type=str, default=None, help="Research topic (skips interactive prompt)")
    parser.add_argument("--model", type=str, default=None, help="Override LLM model for all stages")
    args = parser.parse_args()
    if args.model:
        os.environ["AEL_MODEL"] = args.model
    main(cli_topic=args.topic)
