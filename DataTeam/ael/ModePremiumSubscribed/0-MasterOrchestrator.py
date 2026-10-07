# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Master Orchestrator for Premium Subscribed Data Pipeline

This workflow retrieves data from premium data sources (Bloomberg, Refinitiv, WRDS) with:
- Credential validation
- Cost estimation
- Query optimization
- Data retrieval
- Quality assessment
- License compliance verification
- Data standardization
- Report and codebook generation

5 HITL Checkpoints:
1. Budget Review (after cost estimation)
2. Query Review (after query optimization)
3. Data Quality Review (after data retrieval)
4. Compliance Review (after license compliance check)
5. Final Approval (after report generation)

Pipeline:
1. Data Source Stage: Credential validation and cost estimation
2. Data Cleaning Stage: Query optimization, retrieval, quality assessment, and compliance
3. Quality Assurance Stage: Data standardization and report generation

Input: Research question requiring premium data
Output: Unified dataset with comprehensive documentation and compliance notes
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
    """Find and return the path to the .env file, checking local ael directory first."""
    current_dir = Path(__file__).resolve().parent

    # First check the ael directory (parent of ModePremiumSubscribed)
    ael_dir = current_dir.parent
    local_env = ael_dir / ".env"
    if local_env.exists():
        return str(local_env)

    # Traverse up to the repository root (the directory holding ael_config.yaml and run_ael_pipeline.py)
    search_dir = current_dir
    while not ((search_dir / "ael_config.yaml").exists() and (search_dir / "run_ael_pipeline.py").exists()) and search_dir.parent != search_dir:
        search_dir = search_dir.parent

    if (search_dir / "ael_config.yaml").exists() and (search_dir / "run_ael_pipeline.py").exists():
        env_path = search_dir / ".env"
        if env_path.exists():
            return str(env_path)

    # Fallback to default
    return None


# Load environment variables from the repository root .env
env_path = get_env_path()
if env_path:
    load_dotenv(env_path)
    print(f"Loaded environment from: {env_path}")
else:
    load_dotenv()
    print("Using default .env loading")

# Set quiet mode for stage imports
os.environ["AGENT_QUIET_MODE"] = "true"
register_all_tools()


# ============================================================================
# Premium Data Sources Configuration
# ============================================================================

PREMIUM_SOURCES = {
    "Bloomberg": {
        "name": "Bloomberg",
        "description": "Bloomberg Professional terminal and API access",
        "data_types": ["Equity prices", "Fixed income", "Derivatives", "Economic data", "News"],
        "api_type": "BLPAPI",
        "env_key": "BLOOMBERG_API_KEY",
        "cost_model": "Per-terminal license + API calls",
        "typical_cost": "$24,000+/year per terminal"
    },
    "Refinitiv": {
        "name": "Refinitiv (LSEG)",
        "description": "Refinitiv Eikon and DataScope access",
        "data_types": ["Equity prices", "FX rates", "Commodities", "ESG data", "News"],
        "api_type": "Eikon API / DataScope",
        "env_key": "REFINITIV_EIKON_API_KEY",
        "cost_model": "Subscription + data extraction fees",
        "typical_cost": "$15,000+/year"
    },
    "WRDS": {
        "name": "WRDS (Wharton Research Data Services)",
        "description": "Academic research data platform",
        "data_types": ["CRSP", "Compustat", "IBES", "TAQ", "OptionMetrics"],
        "api_type": "PostgreSQL / Web Query",
        "env_keys": ["WRDS_USERNAME", "WRDS_PASSWORD"],
        "cost_model": "Institutional subscription",
        "typical_cost": "Varies by institution"
    }
}


def check_premium_credentials():
    """Check availability of premium data source credentials.

    Delegates to Stage 1's ``detect_premium_credentials()`` so a bare API-key
    string is NOT treated as a usable terminal (anti-fabrication). On HPC / keyless
    hosts every vendor resolves to "Not configured" and the pipeline reports an
    honest N/A. Falls back to a conservative all-unavailable status if Stage 1
    cannot be imported (never auto-asserts availability)."""
    try:
        from importlib import import_module
        stage1 = import_module('1-DataSourceStage')
        return stage1.detect_premium_credentials()
    except Exception as e:
        print(f"[WARNING] Could not import credential verifier ({e}); "
              f"defaulting all premium sources to Not configured.")
        return {
            v: {"available": False, "status": "Not configured",
                "note": "Credential verifier unavailable; treated as no terminal"}
            for v in ("Bloomberg", "Refinitiv", "WRDS")
        }


def _write_premium_na_outputs(research_question, na_status, cleaning_file, qa_file, report_file):
    """Write explicit, schema-recognizable N/A artifacts for Stages 2 & 3 when no real
    premium terminal exists. NEVER fabricates records or grades — these files exist so
    the run is HONESTLY DISTINGUISHABLE from a real success and excluded from scoring."""
    note = ("ModePremiumSubscribed requires a live Refinitiv Eikon / Bloomberg / WRDS "
            "terminal. None is verifiably reachable on this host, so NO premium data was "
            "retrieved. This is an HONEST N/A result (not a real success) and must be "
            "EXCLUDED from scored results.")
    ts = datetime.now().isoformat()

    na_cleaning = {
        "research_question": research_question,
        "status": na_status,
        "data_available": False,
        "note": note,
        "query_optimization": {
            "optimized_queries": [], "total_estimated_cost": 0.0,
            "cost_savings": 0.0, "optimization_techniques": [],
        },
        "data_retrieval": {
            "retrieved_datasets": [], "total_records": 0, "total_actual_cost": 0.0,
            "retrieval_success_rate": 0.0, "failed_queries": [], "data_simulated": False,
        },
        "quality_assessment": {
            "overall_score": 0.0, "grade": "N/A",
            "completeness_score": 0.0, "accuracy_score": 0.0,
            "consistency_score": 0.0, "timeliness_score": 0.0,
            "vendor_checks": [], "cross_vendor_validation": {}, "issues": [note],
        },
        "license_compliance": {
            "compliance_status": "N/A", "risk_level": "N/A", "restrictions": [],
            "required_citations": {}, "required_disclaimers": [],
            "usage_recommendations": [], "compliance_risks": [],
        },
        "metadata": {"timestamp": ts, "pipeline_stage": "data_cleaning",
                     "status": na_status, "data_available": False, "skipped": True},
    }

    na_qa = {
        "research_question": research_question,
        "status": na_status,
        "data_available": False,
        "note": note,
        "documented_dataset": {
            "dataset_id": f"PREMIUM_NA_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "dataset_name": "Premium Financial Dataset (N/A — no terminal)",
            "standardization": {
                "identifier_mappings": [], "field_mappings": [], "vendors_merged": [],
                "overlap_handling": "N/A", "transformations_applied": [],
                "final_record_count": 0, "final_field_count": 0, "unified_fields": [],
            },
            "codebook": {
                "title": "Premium Data Codebook (N/A)", "version": "0.0.0",
                "created_date": datetime.now().strftime('%Y-%m-%d'), "variables": [],
                "vendor_attribution": {}, "license_compliance_notes": [note],
                "required_citations": {}, "usage_restrictions": [],
            },
            "report": {
                "title": "Premium Data Acquisition Report (N/A)",
                "research_question": research_question,
                "executive_summary": note,
                "vendors_section": "N/A", "credential_section": "N/A",
                "cost_section": "N/A", "query_section": "N/A",
                "retrieval_section": "N/A", "quality_section": "N/A",
                "compliance_section": "N/A", "standardization_section": "N/A",
                "recommendations": [
                    "Re-run on a host with a live Refinitiv Eikon / Bloomberg / WRDS terminal."],
            },
            "quality_score": 0.0,
            "certification_level": "N/A",
            "compliance_status": "N/A",
            "total_cost": 0.0,
            "ready_for_analysis": False,
        },
        "metadata": {"timestamp": ts, "pipeline_stage": "quality_assurance",
                     "status": na_status, "data_available": False, "skipped": True},
    }

    with open(cleaning_file, 'w', encoding='utf-8') as f:
        json.dump(na_cleaning, f, indent=2)
    with open(qa_file, 'w', encoding='utf-8') as f:
        json.dump(na_qa, f, indent=2)
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# Premium Data Acquisition Report (N/A)\n\n"
                f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"**Research Question**: {research_question}\n\n"
                f"**Status**: {na_status}\n\n"
                f"{note}\n\n"
                f"No vendor data was retrieved; quality/certification are N/A. "
                f"This result is excluded from scored evaluation.\n")


def main(cli_topic=None):
    """Main orchestrator that runs all three DataTeam stages for premium data."""

    # Change to script directory to ensure outputs are saved there
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    output_dir = Path(script_dir)

    # Initialize Console UI
    ui = ConsoleUI(
        team="DataTeam",
        mode="ModePremiumSubscribed",
        total_stages=3
    )

    # Initialize workflow logger for evaluation metrics capture
    logger = WorkflowLogger(
        team="DataTeam",
        mode="ModePremiumSubscribed",
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
    memory.set_context("mode", "ModePremiumSubscribed")

    # Track workflow success
    workflow_success = True

    # ========== CONFIGURATION ==========
    print("\n" + "="*70)
    print("PREMIUM SUBSCRIBED DATA PIPELINE")
    print("Mode: HITL Checkpoints")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # Show loaded API keys
    ui.show_api_keys()
    
    # Display warning
    print("\n⚠️  IMPORTANT: This workflow requires valid premium data subscriptions.")
    print("   Ensure you have necessary credentials and institutional access.")
    print("   Data retrieval may incur significant costs.")
    print("="*70)
    
    # Check credentials
    print("\nChecking premium data source credentials...")
    credentials = check_premium_credentials()
    
    accessible_sources = []
    for source, status in credentials.items():
        status_icon = "✓" if status["available"] else "✗"
        print(f"  {status_icon} {source}: {status['status']}")
        if status["available"]:
            accessible_sources.append(source)
    
    if not accessible_sources:
        print("\n[WARNING] No premium data source credentials found.")
        print("Please configure at least one of the following in your .env file:")
        print("  - BLOOMBERG_API_KEY")
        print("  - REFINITIV_EIKON_API_KEY")
        print("  - WRDS_USERNAME and WRDS_PASSWORD")
        print("\nProceeding with simulation mode...")
    
    print("="*70 + "\n")
    
    # Get research question from user
    if cli_topic:
        research_question = cli_topic
    else:
        print("Enter your research question requiring premium data:")
        research_question = auto_input(
            "> ",
            default="Analyze the relationship between institutional ownership and stock returns using CRSP and Compustat data."
        ).strip()

    if not research_question:
        research_question = "Analyze the relationship between institutional ownership and stock returns using CRSP and Compustat data."
        print(f"Using example research question: {research_question}")

    memory.set_context("research_question", research_question)

    # Recall similar past runs
    similar_runs = memory.recall_similar_runs(research_question)
    if similar_runs:
        ui.info(f"Found {len(similar_runs)} similar past run(s) in memory")

    # Get budget limit
    print("\nEnter your budget limit in USD (or press Enter for no limit):")
    budget_input = auto_input("> ", default="").strip()
    budget_limit = float(budget_input) if budget_input else None
    
    print("\n" + "="*70)
    print("INPUT CONFIGURATION")
    print("="*70)
    print(f"Research Question: {research_question}")
    print(f"Budget Limit: ${budget_limit:,.2f}" if budget_limit else "Budget Limit: No limit")
    print(f"Accessible Sources: {', '.join(accessible_sources) if accessible_sources else 'None (simulation mode)'}")
    print("="*70 + "\n")

    # Start execution tracking
    logger.start_execution(metadata={
        "research_question": research_question,
        "budget_limit": budget_limit,
        "accessible_sources": accessible_sources
    })

    # Initialize tracking variables
    credential_validation = {}
    cost_estimation = {}
    quality_assessment = {}
    license_compliance = {}

    # ========== STAGE 1: CREDENTIAL VALIDATION & COST ESTIMATION ==========
    print("\n" + "="*70)
    print("STAGE 1: CREDENTIAL VALIDATION & COST ESTIMATION")
    print("Includes HITL Checkpoint 1: Budget Review")
    print("="*70)

    logger.start_stage("DataSource", 1)
    source_file = "premium_data_source_output.json"

    try:
        # Import and run Stage 1
        from importlib import import_module
        stage1 = import_module('1-DataSourceStage')

        # Run data source stage
        with stage_model(model_config, "DataTeam", "DataSourceStage") as model:
            ui.info(f"Stage model: {model}")
            orchestrator1 = stage1.DataSourceOrchestrator(collector=collector)

        # Run credential validation and cost estimation
        source_output = orchestrator1.run_source_pipeline(
            research_question=research_question,
            budget_limit=budget_limit,
            credentials_status=credentials
        )

        # Save results
        orchestrator1.save_source_output(source_file)

        credential_validation = source_output.credential_validation if source_output else {}
        cost_estimation = source_output.cost_estimation if source_output else {}

        print(f"\n[Stage 1] Complete: Credentials validated, costs estimated")
        print(f"[Stage 1] Results saved to: {source_file}")

        logger.end_stage(
            "DataSource",
            status="success",
            item_count=len(accessible_sources),
            output_files=[source_file]
        )
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
        print(f"\n[ERROR] Stage 1 failed: {e}")
        import traceback
        traceback.print_exc()
        print("Please ensure 1-DataSourceStage.py is properly configured.")
        logger.log_error(error_type=type(e).__name__, message=str(e), recovered=False)
        logger.end_stage("DataSource", status="error", error_message=str(e))
        workflow_success = False
        logger.end_execution(success=False)
        logger.save_execution_log()
        sys.exit(1)  # non-zero exit so the runner does not mask this as success

    # ========== HONEST N/A SHORT-CIRCUIT ==========
    # When Stage 1 verified NO real premium terminal (the case on HPC / keyless
    # hosts: a bare REFINITIV_EIKON_API_KEY is NOT a terminal), STOP here. Running
    # Stages 2-3 would FABRICATE a Grade-A pipeline over synthetic records and
    # falsely certify it. Emit explicit N/A artifacts (data_available=false, grade
    # "N/A", certification "N/A") that are trivially distinguishable from a real
    # success, so the config is excluded from scored results.
    if source_output is not None and not getattr(source_output, "data_available", False):
        na_status = getattr(source_output, "status", "N/A (no premium terminal)")
        cleaning_file = "premium_data_cleaning_output.json"
        qa_file = "premium_data_qa_output.json"
        report_file = f"premium_data_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.md"
        ui.warning(f"DataTeam/ModePremiumSubscribed: {na_status} — "
                   f"skipping retrieval/QA; NO data to grade (honest N/A).")
        try:
            _write_premium_na_outputs(research_question, na_status,
                                      cleaning_file, qa_file, report_file)
            print(f"[N/A] Wrote honest N/A artifacts: {cleaning_file}, {qa_file}, {report_file}")
        except Exception as e:
            print(f"[N/A] Failed to write N/A artifacts: {e}")
            logger.log_error(error_type="NAOutputWrite", message=str(e), recovered=True)

        # Record Stages 2 & 3 as explicitly skipped (NOT success) for traceability.
        logger.start_stage("DataCleaning", 2)
        logger.end_stage("DataCleaning", status="skipped", item_count=0,
                         output_files=[cleaning_file])
        logger.start_stage("QualityAssurance", 3)
        logger.end_stage("QualityAssurance", status="skipped", item_count=0,
                         output_files=[qa_file, report_file])

        print("\n" + "="*70)
        print("PREMIUM SUBSCRIBED DATA PIPELINE — N/A (NO TERMINAL)")
        print("="*70)
        print(f"Status: {na_status}")
        print("No vendor data retrieved; quality/certification reported as N/A.")
        print("This run is EXCLUDED from scored results (not a real success).")
        print("="*70)

        # The run COMPLETED correctly as N/A (it is not a crash). Record + exit clean.
        try:
            from shared.memory.episodic import RunTrajectory
            memory.record_trajectory(RunTrajectory(
                run_id=logger.execution_id,
                research_topic=research_question,
                teams_completed=[],
                teams_failed=[],
                success=True,
                mode="ModePremiumSubscribed",
                artifact_names=[source_file, cleaning_file, qa_file],
            ))
            memory.close()
        except Exception as e:
            print(f"[N/A] Memory record skipped: {e}")

        logger.end_execution(success=True)
        log_path = logger.save_execution_log()
        print(f"\nExecution log saved to: {log_path}")
        try:
            ui.observability_summary(collector)
            from shared.telemetry import export_telemetry
            export_telemetry(collector, str(output_dir), team="DataTeam",
                             mode="ModePremiumSubscribed")
        except Exception:
            pass
        return

    # ========== STAGE 2: QUERY, RETRIEVAL, QUALITY & COMPLIANCE ==========
    print("\n" + "="*70)
    print("STAGE 2: QUERY OPTIMIZATION, RETRIEVAL, QUALITY & COMPLIANCE")
    print("Includes HITL Checkpoints 2-4: Query, Quality, Compliance Review")
    print("="*70)

    logger.start_stage("DataCleaning", 2)
    cleaning_file = "premium_data_cleaning_output.json"

    try:
        # Import and run Stage 2
        stage2 = import_module('2-DataCleaningStage')

        # Run data cleaning stage
        with stage_model(model_config, "DataTeam", "DataCleaningStage") as model:
            ui.info(f"Stage model: {model}")
            orchestrator2 = stage2.DataCleaningOrchestrator(collector=collector)

        # Load source output
        with open(source_file, 'r', encoding='utf-8') as f:
            source_data = json.load(f)

        # Run query optimization, retrieval, quality, and compliance
        cleaning_output = orchestrator2.run_cleaning_pipeline(
            data_source_output=source_data,
            research_question=research_question
        )

        # Save results
        orchestrator2.save_cleaning_output(cleaning_file)

        query_optimization = cleaning_output.query_optimization if cleaning_output else {}
        data_retrieval = cleaning_output.data_retrieval if cleaning_output else {}
        quality_assessment = cleaning_output.quality_assessment if cleaning_output else {}
        license_compliance = cleaning_output.license_compliance if cleaning_output else {}

        print(f"\n[Stage 2] Complete: Queries optimized, data retrieved, quality assessed, compliance verified")
        print(f"[Stage 2] Results saved to: {cleaning_file}")

        # Check for exported data files
        exported_data_files = getattr(orchestrator2, 'exported_data_files', [])
        if exported_data_files:
            for data_file in exported_data_files:
                print(f"[Stage 2] Raw data exported to: {data_file}")

        logger.end_stage(
            "DataCleaning",
            status="success",
            item_count=1,
            output_files=[cleaning_file] + exported_data_files
        )

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
        print(f"\n[ERROR] Stage 2 failed: {e}")
        import traceback
        traceback.print_exc()
        print("Please ensure 2-DataCleaningStage.py is properly configured.")
        logger.log_error(error_type=type(e).__name__, message=str(e), recovered=False)
        logger.end_stage("DataCleaning", status="error", error_message=str(e))
        workflow_success = False
        logger.end_execution(success=False)
        logger.save_execution_log()
        sys.exit(1)  # non-zero exit so the runner does not mask this as success
    
    # ========== STAGE 3: STANDARDIZATION & REPORT GENERATION ==========
    print("\n" + "="*70)
    print("STAGE 3: DATA STANDARDIZATION & REPORT GENERATION")
    print("Includes HITL Checkpoint 5: Final Approval")
    print("="*70)

    logger.start_stage("QualityAssurance", 3)
    qa_file = "premium_data_qa_output.json"
    report_file = f"premium_data_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.md"

    try:
        # Import and run Stage 3
        stage3 = import_module('3-QualityAssuranceStage')

        # Run quality assurance and documentation stage
        with stage_model(model_config, "DataTeam", "QualityAssuranceStage") as model:
            ui.info(f"Stage model: {model}")
            orchestrator3 = stage3.QualityAssuranceOrchestrator(collector=collector)

        # Load cleaning output
        with open(cleaning_file, 'r', encoding='utf-8') as f:
            cleaning_data = json.load(f)

        # Run standardization and report generation
        qa_output = orchestrator3.run_qa_pipeline(
            data_cleaning_output=cleaning_data,
            research_question=research_question
        )

        # Save final results
        orchestrator3.save_qa_output(qa_file)

        # Generate report and codebook
        orchestrator3.save_report_codebook(report_file)

        # Archivist's replication manifest
        manifest_file = "replication_manifest.json"
        orchestrator3.save_replication_manifest(manifest_file)

        documented_dataset = qa_output.documented_dataset if qa_output else {}

        print(f"\n[Stage 3] Complete: Data standardized, report and codebook generated")
        print(f"[Stage 3] Results saved to: {qa_file}")
        print(f"[Stage 3] Report saved to: {report_file}")
        print(f"[Stage 3] Replication manifest saved to: {manifest_file}")

        logger.end_stage(
            "QualityAssurance",
            status="success",
            item_count=1,
            output_files=[qa_file, report_file, manifest_file]
        )

        # Schema validation after Stage 3
        try:
            if qa_output:
                SchemaValidator.validate_or_raise("DataTeam", "QualityAssuranceStage", qa_output.model_dump())
        except (ValueError, Exception) as e:
            ui.warning(f"Schema validation: {e}")
            logger.log_error(error_type="SchemaValidation", message=str(e), recovered=True)

    except Exception as e:
        print(f"\n[ERROR] Stage 3 failed: {e}")
        import traceback
        traceback.print_exc()
        print("Please ensure 3-QualityAssuranceStage.py is properly configured.")
        logger.log_error(error_type=type(e).__name__, message=str(e), recovered=False)
        logger.end_stage("QualityAssurance", status="error", error_message=str(e))
        workflow_success = False
        logger.end_execution(success=False)
        logger.save_execution_log()
        sys.exit(1)  # non-zero exit so the runner does not mask this as success
    
    # ========== FINAL SUMMARY ==========
    print("\n" + "="*70)
    print("PREMIUM SUBSCRIBED DATA PIPELINE COMPLETE")
    print("="*70)
    print(f"Research Question: {research_question[:60]}..." if len(research_question) > 60 else f"Research Question: {research_question}")
    
    if cost_estimation:
        print(f"\nCost Summary:")
        total_est = getattr(cost_estimation, 'total_estimated_cost', 0) if hasattr(cost_estimation, 'total_estimated_cost') else cost_estimation.get('total_estimated_cost', 0) if isinstance(cost_estimation, dict) else 0
        total_act = getattr(cost_estimation, 'total_actual_cost', 0) if hasattr(cost_estimation, 'total_actual_cost') else cost_estimation.get('total_actual_cost', 0) if isinstance(cost_estimation, dict) else 0
        print(f"  Estimated Total: ${total_est:,.2f}")
        print(f"  Actual Total: ${total_act:,.2f}")

    if quality_assessment:
        print(f"\nQuality Assessment:")
        overall = getattr(quality_assessment, 'overall_score', 'N/A') if hasattr(quality_assessment, 'overall_score') else quality_assessment.get('overall_score', 'N/A') if isinstance(quality_assessment, dict) else 'N/A'
        grade = getattr(quality_assessment, 'grade', 'N/A') if hasattr(quality_assessment, 'grade') else quality_assessment.get('grade', 'N/A') if isinstance(quality_assessment, dict) else 'N/A'
        print(f"  Overall Score: {overall}")
        print(f"  Grade: {grade}")

    if license_compliance:
        print(f"\nLicense Compliance:")
        status = getattr(license_compliance, 'compliance_status', 'N/A') if hasattr(license_compliance, 'compliance_status') else license_compliance.get('compliance_status', 'N/A') if isinstance(license_compliance, dict) else 'N/A'
        risk = getattr(license_compliance, 'risk_level', 'N/A') if hasattr(license_compliance, 'risk_level') else license_compliance.get('risk_level', 'N/A') if isinstance(license_compliance, dict) else 'N/A'
        print(f"  Status: {status}")
        print(f"  Risk Level: {risk}")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # ========== OUTPUT SUMMARY ==========
    print("\n" + "="*70)
    print("OUTPUT FILES SUMMARY")
    print("="*70)
    print(f"1. Source Output: {source_file}")
    print(f"2. Cleaning Output: {cleaning_file}")
    print(f"3. QA Output: {qa_file}")
    print(f"4. Report & Codebook: {report_file}")
    print("="*70)
    
    print("\n⚠️  IMPORTANT: This report contains premium data with license restrictions.")
    print("   Ensure compliance with all vendor license terms.")
    print("   Proper attribution required in all publications.")
    
    print(f"\n✅ Premium Subscribed Data Pipeline completed successfully!")
    print(f"📁 All results saved in: {os.getcwd()}")
    print(f"📊 Report and Codebook: {report_file}")
    print("="*70)

    # Record trajectory in episodic memory
    from shared.memory.episodic import RunTrajectory
    trajectory = RunTrajectory(
        run_id=logger.execution_id,
        research_topic=research_question,
        teams_completed=["DataTeam"] if workflow_success else [],
        teams_failed=[] if workflow_success else ["DataTeam"],
        success=workflow_success,
        mode="ModePremiumSubscribed",
        artifact_names=[source_file, cleaning_file, qa_file] if workflow_success else [],
    )
    memory.record_trajectory(trajectory)

    if workflow_success:
        memory.remember_fact(
            f"Retrieved premium data for '{research_question[:60]}'",
            category="data_collection", source="DataTeam/ModePremiumSubscribed",
        )

    memory.close()

    # End execution and save log
    logger.end_execution(success=workflow_success)
    log_path = logger.save_execution_log()
    print(f"\nExecution log saved to: {log_path}")

    ui.observability_summary(collector)
    from shared.telemetry import export_telemetry
    export_telemetry(collector, str(output_dir), team="DataTeam", mode="ModePremiumSubscribed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AEL DataTeam — ModePremiumSubscribed")
    parser.add_argument("--topic", type=str, default=None, help="Research topic (skips interactive prompt)")
    parser.add_argument("--model", type=str, default=None, help="Override LLM model for all stages")
    args = parser.parse_args()
    if args.model:
        os.environ["AEL_MODEL"] = args.model
    main(cli_topic=args.topic)
