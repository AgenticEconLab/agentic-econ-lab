# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Master Orchestrator for User-Uploaded Data Pipeline
Processes user-uploaded files (CSV, Excel, Stata, SPSS, SAS).

Pipeline:
1. Data Source Stage: File parsing and quality assessment
2. Data Cleaning Stage: Privacy screening, structure inference, and transformation
3. Quality Assurance Stage: Report and codebook generation

Input: User-uploaded file path and optional research question
Output: Transformed dataset with comprehensive documentation
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
# Supported File Formats
# ============================================================================

SUPPORTED_FORMATS = {
    ".csv": {"name": "CSV (Comma-Separated Values)", "parser": "pandas.read_csv"},
    ".xlsx": {"name": "Excel (XLSX)", "parser": "pandas.read_excel"},
    ".xls": {"name": "Excel (XLS)", "parser": "pandas.read_excel"},
    ".dta": {"name": "Stata", "parser": "pandas.read_stata"},
    ".sav": {"name": "SPSS", "parser": "pandas.read_spss"},
    ".sas7bdat": {"name": "SAS", "parser": "pandas.read_sas"},
    ".json": {"name": "JSON", "parser": "pandas.read_json"},
    ".parquet": {"name": "Parquet", "parser": "pandas.read_parquet"}
}


def check_file_format(file_path: str) -> dict:
    """Check if file format is supported and return format info."""
    if not os.path.exists(file_path):
        return {"supported": False, "error": "File not found"}
    ext = os.path.splitext(file_path)[1].lower()
    if ext in SUPPORTED_FORMATS:
        return {"supported": True, "extension": ext, "format_info": SUPPORTED_FORMATS[ext]}
    else:
        return {"supported": False, "extension": ext, "error": f"Unsupported format: {ext}",
                "supported_formats": list(SUPPORTED_FORMATS.keys())}


def main(cli_topic=None):
    """Main orchestrator for user-uploaded data pipeline."""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    output_dir = Path(script_dir)

    # Initialize Console UI
    ui = ConsoleUI(
        team="DataTeam",
        mode="ModeUserUploaded",
        total_stages=3
    )

    # Initialize workflow logger
    logger = WorkflowLogger(
        team="DataTeam",
        mode="ModeUserUploaded",
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
    memory.set_context("mode", "ModeUserUploaded")

    workflow_success = True
    total_errors = 0

    # Get file path
    file_path = auto_input("Enter the path to your data file: ", default="").strip()

    if not file_path:
        file_path = "example_data.csv"
        if not os.path.exists(file_path):
            import pandas as pd
            example_data = {
                "id": range(1, 101),
                "name": [f"Person_{i}" for i in range(1, 101)],
                "age": [25 + (i % 40) for i in range(100)],
                "income": [30000 + (i * 500) for i in range(100)],
                "education_years": [12 + (i % 10) for i in range(100)],
                "employed": [i % 2 == 0 for i in range(100)],
                "region": ["North", "South", "East", "West"] * 25,
                "year": [2020 + (i % 4) for i in range(100)]
            }
            df = pd.DataFrame(example_data)
            df.to_csv(file_path, index=False)

    # Check file format
    format_check = check_file_format(file_path)
    if not format_check["supported"]:
        ui.error(f"Unsupported file format: {format_check.get('error', 'Unknown')}")
        return

    # Get research question
    if cli_topic:
        research_question = cli_topic
    else:
        research_question = auto_input("Enter your research question (optional): ", default="").strip()
    if not research_question:
        research_question = "Analyze the relationship between education, employment, and income."

    memory.set_context("research_question", research_question)

    # Recall similar past runs
    similar_runs = memory.recall_similar_runs(research_question)
    if similar_runs:
        ui.info(f"Found {len(similar_runs)} similar past run(s) in memory")

    # Start execution tracking
    logger.start_execution(metadata={
        "file_path": file_path,
        "file_format": format_check['format_info']['name'],
        "research_question": research_question
    })

    # Print workflow header
    ui.workflow_header(
        topic=research_question[:60],
        execution_id=logger.execution_id,
        file=os.path.basename(file_path),
        format=format_check['format_info']['name']
    )

    # Show loaded API keys
    ui.show_api_keys()

    # Track results
    file_info = {}
    quality_assessment = {}

    # ========== STAGE 1: FILE PARSING & QUALITY ASSESSMENT ==========
    ui.stage_start(1, "File Parsing & Quality Assessment")
    logger.start_stage("DataSource", 1)
    source_file = "user_data_source_output.json"

    try:
        from importlib import import_module
        stage1 = import_module('1-DataSourceStage')

        with stage_model(model_config, "DataTeam", "DataSourceStage") as model:
            ui.info(f"Stage model: {model}")
            orchestrator1 = stage1.DataSourceOrchestrator(collector=collector)
        source_output = orchestrator1.run_source_pipeline(
            file_path=file_path,
            research_question=research_question
        )

        orchestrator1.save_source_output(source_file)

        file_info = source_output.file_info.model_dump() if source_output else {}
        quality_assessment = source_output.quality_assessment.model_dump() if source_output else {}

        ui.agent_result("FileParser", items=file_info.get('num_columns', 0),
                        message=f"{file_info.get('num_rows', 0)} rows")
        ui.agent_result("QualityAssessor", items=1,
                        message=f"score: {quality_assessment.get('overall_score', 'N/A')}")

        ui.stage_complete(1, status="success", items=1,
                          duration=logger.get_stage_duration("DataSource"), output_file=source_file)
        logger.end_stage("DataSource", status="success", item_count=1, output_files=[source_file])
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
        ui.error(f"Stage 1 failed: {e}")
        logger.log_error(error_type=type(e).__name__, message=str(e), recovered=False)
        logger.end_stage("DataSource", status="error", error_message=str(e))
        workflow_success = False
        total_errors += 1
        ui.workflow_complete(success=False, stages_completed=0, total_errors=total_errors)
        logger.end_execution(success=False)
        logger.save_execution_log()
        sys.exit(1)  # non-zero exit so the runner does not mask this as success

    # ========== STAGE 2: PRIVACY, STRUCTURE & TRANSFORMATION ==========
    ui.stage_start(2, "Privacy Screening & Data Transformation")
    logger.start_stage("DataCleaning", 2)
    cleaning_file = "user_data_cleaning_output.json"

    try:
        stage2 = import_module('2-DataCleaningStage')
        with stage_model(model_config, "DataTeam", "DataCleaningStage") as model:
            ui.info(f"Stage model: {model}")
            orchestrator2 = stage2.DataCleaningOrchestrator(collector=collector)

        with open(source_file, 'r', encoding='utf-8') as f:
            source_data = json.load(f)

        cleaning_output = orchestrator2.run_cleaning_pipeline(
            data_source_output=source_data,
            research_question=research_question
        )

        orchestrator2.save_cleaning_output(cleaning_file)

        privacy_results = cleaning_output.privacy_screening.model_dump() if cleaning_output else {}
        structure_results = cleaning_output.structure_inference.model_dump() if cleaning_output else {}

        ui.agent_result("PrivacyScreener", items=1,
                        message=f"risk: {privacy_results.get('risk_level', 'N/A')}")
        ui.agent_result("StructureInferrer", items=1,
                        message=f"type: {structure_results.get('structure_type', 'N/A')}")
        ui.agent_result("DataTransformer", items=1, message="transformed")

        ui.stage_complete(2, status="success", items=1,
                          duration=logger.get_stage_duration("DataCleaning"), output_file=cleaning_file)
        logger.end_stage("DataCleaning", status="success", item_count=1, output_files=[cleaning_file])

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
    qa_file = "user_data_qa_output.json"
    report_file = f"user_data_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.md"

    try:
        stage3 = import_module('3-QualityAssuranceStage')
        with stage_model(model_config, "DataTeam", "QualityAssuranceStage") as model:
            ui.info(f"Stage model: {model}")
            orchestrator3 = stage3.QualityAssuranceOrchestrator(collector=collector)

        with open(cleaning_file, 'r', encoding='utf-8') as f:
            cleaning_data = json.load(f)

        qa_output = orchestrator3.run_qa_pipeline(
            data_cleaning_output=cleaning_data,
            research_question=research_question,
            file_path=file_path
        )

        orchestrator3.save_qa_output(qa_file)
        orchestrator3.save_report_codebook(report_file)
        manifest_file = "replication_manifest.json"
        orchestrator3.save_replication_manifest(manifest_file)

        ui.agent_result("QualityValidator", items=1, message="validated")
        ui.agent_result("ReportGenerator", items=1, message=report_file)
        ui.agent_result("Archivist", items=1, message=manifest_file)

        ui.stage_complete(3, status="success", items=1,
                          duration=logger.get_stage_duration("QualityAssurance"), output_file=qa_file)
        logger.end_stage("QualityAssurance", status="success",
                         item_count=1, output_files=[qa_file, report_file, manifest_file])

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
        mode="ModeUserUploaded",
        artifact_names=[source_file, cleaning_file, qa_file] if workflow_success else [],
    )
    memory.record_trajectory(trajectory)

    if workflow_success:
        memory.remember_fact(
            f"Processed user-uploaded file '{os.path.basename(file_path)}' for '{research_question[:60]}'",
            category="data_collection", source="DataTeam/ModeUserUploaded",
        )

    memory.close()

    logger.end_execution(success=workflow_success)
    log_path = logger.save_execution_log()

    ui.workflow_complete(
        success=workflow_success, stages_completed=3, total_errors=total_errors,
        output_files=[source_file, cleaning_file, qa_file, report_file, manifest_file],
        summary={
            "File": os.path.basename(file_path),
            "Rows": file_info.get('num_rows', 'N/A'),
            "Columns": file_info.get('num_columns', 'N/A'),
            "Quality Score": quality_assessment.get('overall_score', 'N/A')
        }
    )

    ui.observability_summary(collector)
    from shared.telemetry import export_telemetry
    export_telemetry(collector, str(output_dir), team="DataTeam", mode="ModeUserUploaded")

    ui.info(f"Execution log: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AEL DataTeam — ModeUserUploaded")
    parser.add_argument("--topic", type=str, default=None, help="Research topic (skips interactive prompt)")
    parser.add_argument("--model", type=str, default=None, help="Override LLM model for all stages")
    args = parser.parse_args()
    if args.model:
        os.environ["AEL_MODEL"] = args.model
    main(cli_topic=args.topic)
