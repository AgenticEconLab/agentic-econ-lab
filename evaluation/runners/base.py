# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Base runner interface for workflow execution.

All framework-specific runners inherit from BaseWorkflowRunner.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import uuid


@dataclass
class RunConfig:
    """Configuration for a single workflow run."""

    team: str
    mode: str
    inputs: Dict[str, Any]
    output_dir: Optional[Path] = None
    run_id: Optional[str] = None
    timeout_seconds: int = 3600
    capture_logs: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.run_id is None:
            self.run_id = str(uuid.uuid4())[:8]
        if self.output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(f"outputs/{self.team}/{self.mode}/{timestamp}_{self.run_id}")


@dataclass
class ExecutionResult:
    """Result from a workflow execution."""

    run_id: str
    team: str
    mode: str
    framework: str
    success: bool
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    output_dir: Optional[Path] = None
    log_file: Optional[Path] = None
    error_message: Optional[str] = None
    stages_completed: int = 0
    total_stages: int = 0
    outputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def compute_duration(self):
        """Compute duration from start and end times."""
        if self.end_time and self.start_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()

    @property
    def completion_rate(self) -> float:
        """Get the stage completion rate."""
        if self.total_stages == 0:
            return 0.0
        return self.stages_completed / self.total_stages


class BaseWorkflowRunner(ABC):
    """Abstract base class for workflow runners."""

    # Framework identifier
    FRAMEWORK: str = "base"

    # Supported teams
    SUPPORTED_TEAMS: List[str] = []

    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize runner.

        Args:
            base_path: Base path to workflow codebase
        """
        self.base_path = base_path or Path.cwd()

    @abstractmethod
    def run(
        self,
        config: RunConfig,
    ) -> ExecutionResult:
        """
        Run a workflow with instrumentation.

        Args:
            config: Run configuration

        Returns:
            ExecutionResult with timing and output information
        """
        pass

    @abstractmethod
    def get_supported_modes(self, team: str) -> List[str]:
        """
        Get list of supported modes for a team.

        Args:
            team: Team name

        Returns:
            List of supported mode names
        """
        pass

    def run_batch(
        self,
        configs: List[RunConfig],
    ) -> List[ExecutionResult]:
        """
        Run multiple workflow configurations.

        Args:
            configs: List of run configurations

        Returns:
            List of execution results
        """
        results = []
        for config in configs:
            result = self.run(config)
            results.append(result)
        return results

    def run_repeated(
        self,
        config: RunConfig,
        n_runs: int = 3,
        output_base_dir: Optional[Path] = None,
    ) -> List[ExecutionResult]:
        """
        Run the same configuration multiple times for reliability testing.

        Each run gets an isolated output directory: {base}/run_001/, {base}/run_002/, etc.
        A multirun_manifest.json is written after all runs complete.

        Args:
            config: Run configuration
            n_runs: Number of times to run
            output_base_dir: Base directory for all run outputs. If None, uses
                config.output_dir as base. Each run writes to run_NNN/ under this.

        Returns:
            List of execution results from all runs
        """
        base_dir = output_base_dir or config.output_dir or Path(
            f"outputs/{config.team}/{config.mode}"
        )
        base_dir = Path(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for i in range(n_runs):
            run_number = i + 1
            run_dir = base_dir / f"run_{run_number:03d}"
            run_dir.mkdir(parents=True, exist_ok=True)

            run_config = RunConfig(
                team=config.team,
                mode=config.mode,
                inputs=config.inputs.copy(),
                output_dir=run_dir,
                timeout_seconds=config.timeout_seconds,
                capture_logs=config.capture_logs,
                metadata={
                    **config.metadata,
                    "run_number": run_number,
                    "total_runs": n_runs,
                },
            )
            result = self.run(run_config)
            results.append(result)

        # Write multirun manifest
        manifest = {
            "team": config.team,
            "mode": config.mode,
            "framework": self.FRAMEWORK,
            "n_runs": n_runs,
            "base_dir": str(base_dir),
            "completed_at": datetime.now().isoformat(),
            "runs": [],
        }
        for i, result in enumerate(results):
            manifest["runs"].append({
                "run_number": i + 1,
                "run_id": result.run_id,
                "success": result.success,
                "duration_seconds": result.duration_seconds,
                "stages_completed": result.stages_completed,
                "total_stages": result.total_stages,
                "output_dir": str(result.output_dir) if result.output_dir else None,
                "error_message": result.error_message,
            })
        manifest["summary"] = {
            "successful_runs": sum(1 for r in results if r.success),
            "failed_runs": sum(1 for r in results if not r.success),
            "avg_duration_seconds": (
                sum(r.duration_seconds for r in results if r.duration_seconds) /
                max(1, sum(1 for r in results if r.duration_seconds))
            ),
        }

        manifest_path = base_dir / "multirun_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        return results

    def validate_config(self, config: RunConfig) -> List[str]:
        """
        Validate a run configuration.

        Args:
            config: Run configuration to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if config.team not in self.SUPPORTED_TEAMS:
            errors.append(f"Unsupported team: {config.team}. Supported: {self.SUPPORTED_TEAMS}")

        supported_modes = self.get_supported_modes(config.team)
        if config.mode not in supported_modes:
            errors.append(f"Unsupported mode for {config.team}: {config.mode}. Supported: {supported_modes}")

        if config.timeout_seconds <= 0:
            errors.append(f"Invalid timeout: {config.timeout_seconds}. Must be positive.")

        return errors

    def get_workflow_path(self, team: str, mode: str) -> Path:
        """
        Get the path to a workflow's code directory.

        Args:
            team: Team name
            mode: Mode name

        Returns:
            Path to the workflow code directory
        """
        return self.base_path / team / self.FRAMEWORK / mode

    def ensure_output_dir(self, output_dir: Path) -> Path:
        """
        Ensure output directory exists.

        Args:
            output_dir: Directory path

        Returns:
            The directory path (created if needed)
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
