# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Base parser interface for workflow output parsing.

All framework-specific parsers inherit from BaseWorkflowParser.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..schemas.execution import (
    ExecutionTrace,
    StageExecution,
    ErrorInfo,
    TimingInfo,
    WorkflowOutputs,
)


class BaseWorkflowParser(ABC):
    """Abstract base class for workflow output parsers."""

    # Framework identifier
    FRAMEWORK: str = "base"

    # Supported teams
    SUPPORTED_TEAMS: List[str] = []

    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize parser.

        Args:
            base_path: Base path to workflow codebase
        """
        self.base_path = base_path or Path.cwd()

    @abstractmethod
    def parse_outputs(self, output_dir: Path) -> Dict[str, Any]:
        """
        Parse workflow output files from a directory.

        Args:
            output_dir: Directory containing workflow outputs

        Returns:
            Dictionary with parsed output data
        """
        pass

    @abstractmethod
    def extract_timing(self, output_dir: Path, log_file: Optional[Path] = None) -> TimingInfo:
        """
        Extract execution timing information.

        Args:
            output_dir: Directory containing outputs
            log_file: Optional path to log file

        Returns:
            TimingInfo with start/end times and duration
        """
        pass

    @abstractmethod
    def extract_errors(self, output_dir: Path, log_file: Optional[Path] = None) -> List[ErrorInfo]:
        """
        Extract error information from logs.

        Args:
            output_dir: Directory containing outputs
            log_file: Optional path to log file

        Returns:
            List of ErrorInfo objects
        """
        pass

    @abstractmethod
    def build_execution_trace(
        self,
        team: str,
        mode: str,
        output_dir: Path,
        log_file: Optional[Path] = None,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> ExecutionTrace:
        """
        Build complete execution trace from outputs.

        Args:
            team: Team name (IdeationTeam, LiteratureTeam, etc.)
            mode: Mode name (ModeNoWcNoHITL, etc.)
            output_dir: Directory containing outputs
            log_file: Optional path to log file
            inputs: Input parameters used for the run

        Returns:
            Complete ExecutionTrace object
        """
        pass

    def get_team_config(self, team: str) -> Dict[str, Any]:
        """
        Get team-specific configuration.

        Args:
            team: Team name

        Returns:
            Configuration dictionary for the team
        """
        return {}

    def get_expected_outputs(self, team: str, mode: str) -> List[str]:
        """
        Get list of expected output files for a team/mode.

        Args:
            team: Team name
            mode: Mode name

        Returns:
            List of expected output file names
        """
        return []

    def validate_outputs(self, team: str, mode: str, output_dir: Path) -> Dict[str, bool]:
        """
        Validate that expected outputs exist.

        Args:
            team: Team name
            mode: Mode name
            output_dir: Directory containing outputs

        Returns:
            Dictionary mapping file names to existence status
        """
        expected = self.get_expected_outputs(team, mode)
        results = {}
        for filename in expected:
            filepath = output_dir / filename
            results[filename] = filepath.exists()
        return results

    def collect_output_files(self, output_dir: Path) -> WorkflowOutputs:
        """
        Collect all output files from a directory.

        Args:
            output_dir: Directory to scan

        Returns:
            WorkflowOutputs with categorized files
        """
        outputs = WorkflowOutputs()

        if not output_dir.exists():
            return outputs

        for filepath in output_dir.iterdir():
            if filepath.is_file():
                filename = filepath.name
                str_path = str(filepath)

                if filename.endswith('.json'):
                    outputs.json_files[filename] = str_path
                elif filename.endswith('.csv'):
                    outputs.csv_files[filename] = str_path
                elif filename.endswith(('.txt', '.md')):
                    outputs.text_files[filename] = str_path
                else:
                    outputs.other_files.append(str_path)

        return outputs
