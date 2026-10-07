# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Workflow instrumentation for evaluation metrics capture.

This module provides the WorkflowLogger class that captures execution metrics
needed for the 10-dimension evaluation framework.

Usage:
    from shared.instrumentation import WorkflowLogger

    logger = WorkflowLogger(
        team="IdeationTeam",
        mode="ModeNoWcNoHITL",
        output_dir=Path(".")
    )

    logger.start_execution()

    logger.start_stage("Sourcing", 1)
    # ... run stage 1 ...
    logger.end_stage("Sourcing", status="success", item_count=25, output_files=["results.csv"])

    logger.start_stage("Refinement", 2)
    # ... run stage 2 ...
    logger.end_stage("Refinement", status="success", item_count=10)

    logger.start_stage("Integration", 3)
    # ... run stage 3 ...
    logger.end_stage("Integration", status="success", item_count=5)

    log_path = logger.save_execution_log()
"""

import json
import uuid
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class StageLog:
    """Log entry for a single stage execution."""

    def __init__(self, name: str, number: int, run_number: Optional[int] = None):
        self.name = name
        self.number = number
        self.run_number = run_number
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.duration_seconds: Optional[float] = None
        self.status: str = "pending"  # pending, running, success, error, partial
        self.item_count: Optional[int] = None
        self.output_files: List[str] = []
        self.error_message: Optional[str] = None

    def start(self):
        """Mark stage as started."""
        self.started_at = datetime.now()
        self.status = "running"

    def complete(
        self,
        status: str = "success",
        item_count: Optional[int] = None,
        output_files: Optional[List[str]] = None,
        error_message: Optional[str] = None,
    ):
        """Mark stage as completed."""
        self.completed_at = datetime.now()
        self.status = status
        self.item_count = item_count
        if output_files:
            self.output_files = output_files
        if error_message:
            self.error_message = error_message

        if self.started_at and self.completed_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = {
            "name": self.name,
            "number": self.number,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "status": self.status,
            "item_count": self.item_count,
            "output_files": self.output_files,
            "error_message": self.error_message,
        }
        if self.run_number is not None:
            d["run_number"] = self.run_number
        return d


class ErrorLog:
    """Log entry for an error/exception."""

    def __init__(
        self,
        error_type: str,
        message: str,
        stage: Optional[str] = None,
        recovered: bool = False,
        traceback_str: Optional[str] = None,
    ):
        self.timestamp = datetime.now()
        self.error_type = error_type
        self.message = message
        self.stage = stage
        self.recovered = recovered
        self.traceback_str = traceback_str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "error_type": self.error_type,
            "message": self.message,
            "stage": self.stage,
            "recovered": self.recovered,
            "traceback": self.traceback_str,
        }


class WorkflowLogger:
    """
    Captures execution metrics for workflow evaluation.

    This logger should be used in Master Orchestrator scripts to capture:
    - Overall execution timing
    - Per-stage timing and status
    - Error/exception information
    - Output file tracking

    The generated execution_log.json can be parsed by the evaluation framework
    to calculate metrics across all 10 evaluation dimensions.
    """

    def __init__(
        self,
        team: str,
        mode: str,
        output_dir: Optional[Path] = None,
        framework: str = "ael",
        quiet: bool = False,
        run_number: Optional[int] = None,
        total_runs: Optional[int] = None,
    ):
        """
        Initialize workflow logger.

        Args:
            team: Team name (IdeationTeam, LiteratureTeam, ModelTeam, DataTeam)
            mode: Mode name (ModeNoWcNoHITL, ModeOpenSourceAPI, etc.)
            output_dir: Directory to save execution log (default: current dir)
            framework: Framework identifier (default: ael)
            quiet: If True, suppress verbose console output (use with ConsoleUI)
            run_number: Run index (1-based) when executing as part of a multi-run batch
            total_runs: Total number of runs in the batch
        """
        self.execution_id = str(uuid.uuid4())[:8]
        self.team = team
        self.mode = mode
        self.framework = framework
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.quiet = quiet
        self.run_number = run_number
        self.total_runs = total_runs

        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.total_duration_seconds: Optional[float] = None

        self.stages: Dict[str, StageLog] = {}
        self.current_stage: Optional[str] = None
        self.errors: List[ErrorLog] = []

        self.success: bool = False
        self.metadata: Dict[str, Any] = {}

        # Observability: optional MetricsCollector for LLM/tool tracking
        self.metrics_collector = None

    def set_metrics_collector(self, collector):
        """Attach a MetricsCollector for LLM/tool observability tracking."""
        self.metrics_collector = collector

    def start_execution(self, metadata: Optional[Dict[str, Any]] = None):
        """
        Mark workflow execution start.

        Args:
            metadata: Optional metadata to include in log (inputs, config, etc.)
        """
        self.started_at = datetime.now()
        if metadata:
            self.metadata = metadata

        if not self.quiet:
            print(f"\n{'=' * 60}")
            print(f"WORKFLOW EXECUTION STARTED")
            print(f"{'=' * 60}")
            print(f"Execution ID: {self.execution_id}")
            print(f"Team: {self.team}")
            print(f"Mode: {self.mode}")
            print(f"Start Time: {self.started_at.isoformat()}")
            print(f"{'=' * 60}\n")

        return self.execution_id

    def start_stage(self, stage_name: str, stage_number: int):
        """
        Mark a stage as started.

        Args:
            stage_name: Name of the stage (e.g., "Sourcing", "Refinement")
            stage_number: Stage number (1, 2, 3)
        """
        stage_log = StageLog(stage_name, stage_number, run_number=self.run_number)
        stage_log.start()
        self.stages[stage_name] = stage_log
        self.current_stage = stage_name

        # Update observability context
        if self.metrics_collector is not None:
            self.metrics_collector.set_context(stage=stage_name)

        if not self.quiet:
            print(f"\n--- Stage {stage_number}: {stage_name} STARTED ---")
            print(f"Time: {stage_log.started_at.isoformat()}")

    def end_stage(
        self,
        stage_name: str,
        status: str = "success",
        item_count: Optional[int] = None,
        output_files: Optional[List[str]] = None,
        error_message: Optional[str] = None,
    ):
        """
        Mark a stage as completed.

        Args:
            stage_name: Name of the stage
            status: Completion status (success, error, partial)
            item_count: Number of items produced (e.g., research questions, papers)
            output_files: List of output file names generated
            error_message: Error message if status is error
        """
        if stage_name not in self.stages:
            # Stage wasn't started, create it now
            self.stages[stage_name] = StageLog(stage_name, len(self.stages) + 1)

        stage_log = self.stages[stage_name]
        stage_log.complete(
            status=status,
            item_count=item_count,
            output_files=output_files,
            error_message=error_message,
        )

        if not self.quiet:
            status_icon = "✓" if status == "success" else "✗" if status == "error" else "◐"
            print(f"\n--- Stage {stage_log.number}: {stage_name} COMPLETED [{status_icon}] ---")
            print(f"Status: {status}")
            if stage_log.duration_seconds:
                print(f"Duration: {stage_log.duration_seconds:.2f} seconds")
            if item_count is not None:
                print(f"Items: {item_count}")
            if output_files:
                print(f"Outputs: {', '.join(output_files)}")
            if error_message:
                print(f"Error: {error_message}")

        self.current_stage = None

    def log_error(
        self,
        error_type: str,
        message: str,
        recovered: bool = False,
        include_traceback: bool = True,
    ):
        """
        Log an error or exception.

        Args:
            error_type: Type of error (e.g., "APIError", "ValidationError")
            message: Error message
            recovered: Whether the workflow recovered from this error
            include_traceback: Whether to capture full traceback
        """
        tb_str = traceback.format_exc() if include_traceback else None

        error_log = ErrorLog(
            error_type=error_type,
            message=message,
            stage=self.current_stage,
            recovered=recovered,
            traceback_str=tb_str,
        )
        self.errors.append(error_log)

        if not self.quiet:
            recovery_status = "RECOVERED" if recovered else "FATAL"
            print(f"\n!!! ERROR [{recovery_status}] !!!")
            print(f"Type: {error_type}")
            print(f"Message: {message}")
            if self.current_stage:
                print(f"Stage: {self.current_stage}")

    def end_execution(self, success: bool = True):
        """
        Mark workflow execution as completed.

        Args:
            success: Whether the workflow completed successfully
        """
        self.completed_at = datetime.now()
        self.success = success

        if self.started_at and self.completed_at:
            self.total_duration_seconds = (self.completed_at - self.started_at).total_seconds()

        # Determine success based on stage statuses if not explicitly set
        if success:
            for stage in self.stages.values():
                if stage.status == "error":
                    self.success = False
                    break

        if not self.quiet:
            status_icon = "✓" if self.success else "✗"
            print(f"\n{'=' * 60}")
            print(f"WORKFLOW EXECUTION COMPLETED [{status_icon}]")
            print(f"{'=' * 60}")
            print(f"Execution ID: {self.execution_id}")
            print(f"Success: {self.success}")
            print(f"End Time: {self.completed_at.isoformat()}")
            print(f"Total Duration: {self.total_duration_seconds:.2f} seconds")
            print(f"Stages Completed: {sum(1 for s in self.stages.values() if s.status == 'success')}/{len(self.stages)}")
            print(f"Errors: {len(self.errors)}")
            print(f"{'=' * 60}\n")

    def save_execution_log(self, filename: str = "execution_log.json") -> Path:
        """
        Save the execution log to JSON file.

        Args:
            filename: Name of the output file

        Returns:
            Path to the saved log file
        """
        # Ensure execution is ended
        if self.completed_at is None:
            self.end_execution()

        log_data = {
            "execution_id": self.execution_id,
            "team": self.team,
            "mode": self.mode,
            "framework": self.framework,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_duration_seconds": self.total_duration_seconds,
            "stages": [stage.to_dict() for stage in sorted(self.stages.values(), key=lambda s: s.number)],
            "errors": [error.to_dict() for error in self.errors],
            "success": self.success,
            "metadata": self.metadata,
            "summary": {
                "total_stages": len(self.stages),
                "successful_stages": sum(1 for s in self.stages.values() if s.status == "success"),
                "failed_stages": sum(1 for s in self.stages.values() if s.status == "error"),
                "total_errors": len(self.errors),
                "recovered_errors": sum(1 for e in self.errors if e.recovered),
                "total_items": sum(s.item_count or 0 for s in self.stages.values()),
            },
        }

        # Include observability data if collector is attached
        if self.metrics_collector is not None:
            log_data["observability"] = self.metrics_collector.get_summary()

        # Include multi-run metadata if this is part of a batch
        if self.run_number is not None:
            log_data["run_number"] = self.run_number
        if self.total_runs is not None:
            log_data["total_runs"] = self.total_runs

        output_path = self.output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        if not self.quiet:
            print(f"Execution log saved to: {output_path}")
        return output_path

    def get_stage_status(self, stage_name: str) -> Optional[str]:
        """Get the status of a specific stage."""
        if stage_name in self.stages:
            return self.stages[stage_name].status
        return None

    def get_stage_duration(self, stage_name: str) -> Optional[float]:
        """Get the duration of a specific stage in seconds."""
        if stage_name in self.stages:
            stage = self.stages[stage_name]
            if stage.duration_seconds:
                return stage.duration_seconds
            # If stage is still running, calculate current duration
            if stage.started_at and not stage.completed_at:
                return (datetime.now() - stage.started_at).total_seconds()
        return None

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the execution."""
        return {
            "execution_id": self.execution_id,
            "team": self.team,
            "mode": self.mode,
            "success": self.success,
            "duration_seconds": self.total_duration_seconds,
            "stages_completed": sum(1 for s in self.stages.values() if s.status == "success"),
            "total_stages": len(self.stages),
            "error_count": len(self.errors),
        }


# Context manager for stage timing
class StageContext:
    """
    Context manager for automatic stage timing.

    Usage:
        with StageContext(logger, "Sourcing", 1) as ctx:
            # Run stage code
            results = run_sourcing()
            ctx.set_results(item_count=len(results), output_files=["results.csv"])
    """

    def __init__(
        self,
        logger: WorkflowLogger,
        stage_name: str,
        stage_number: int,
    ):
        self.logger = logger
        self.stage_name = stage_name
        self.stage_number = stage_number
        self.item_count: Optional[int] = None
        self.output_files: List[str] = []
        self.error_message: Optional[str] = None
        self.status: str = "success"

    def __enter__(self):
        self.logger.start_stage(self.stage_name, self.stage_number)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.status = "error"
            self.error_message = str(exc_val)
            self.logger.log_error(
                error_type=exc_type.__name__,
                message=str(exc_val),
                recovered=False,
            )

        self.logger.end_stage(
            self.stage_name,
            status=self.status,
            item_count=self.item_count,
            output_files=self.output_files,
            error_message=self.error_message,
        )

        # Don't suppress the exception
        return False

    def set_results(
        self,
        item_count: Optional[int] = None,
        output_files: Optional[List[str]] = None,
    ):
        """Set the results for this stage."""
        if item_count is not None:
            self.item_count = item_count
        if output_files:
            self.output_files = output_files
