# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
AEL (Agentic Econ Lab) output parser.

Parses outputs from the AEL framework - plain Python implementation
used across IdeationTeam, LiteratureTeam, ModelTeam, and DataTeam.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import csv

from .base import BaseWorkflowParser
from ..schemas.execution import (
    ExecutionTrace,
    StageExecution,
    ErrorInfo,
    TimingInfo,
    WorkflowOutputs,
)


class AELParser(BaseWorkflowParser):
    """Parser for AEL (Agentic Econ Lab) workflow outputs."""

    FRAMEWORK = "ael"

    SUPPORTED_TEAMS = ["IdeationTeam", "LiteratureTeam", "ModelTeam", "DataTeam",
                       "EstimationTeam", "ReportingTeam", "CodeTeam"]

    # Team-specific configurations
    TEAM_CONFIGS = {
        "IdeationTeam": {
            "stages": [
                {"number": 1, "name": "Sourcing", "module": "1-SourcingStage"},
                {"number": 2, "name": "Refinement", "module": "2-RefinementStage"},
                {"number": 3, "name": "Integration", "module": "3-IntegrationStage"},
            ],
            "modes": ["ModeNoWcNoHITL", "ModeNoWcWithHITL", "ModeWithWcNoHITL", "ModeWithWcWithHITL"],
            "outputs": {
                "stage1": ["literature_results_automated.csv", "literature_results.csv"],
                "stage2": ["refinement_results_automated.json", "refinement_results.json"],
                "stage3": [
                    "finalized_research_questions_automated.json",
                    "finalized_research_questions_automated.txt",
                    "finalized_research_questions.json",
                    "finalized_research_questions.txt",
                ],
            },
        },
        "LiteratureTeam": {
            "stages": [
                {"number": 1, "name": "LiteratureGathering", "module": "1-LiteratureGatheringStage"},
                {"number": 2, "name": "GapDetection", "module": "2-GapDetectionStage"},
                {"number": 3, "name": "Synthesis", "module": "3-SynthesisStage"},
            ],
            "modes": ["ModeNoWcNoHITL", "ModeNoWcWithHITL", "ModeWithWcNoHITL", "ModeWithWcWithHITL"],
            "outputs": {
                "stage1": ["literature_batch.json", "literature_items.csv"],
                "stage2": ["gap_analysis_results.json", "research_gaps.csv", "knowledge_graph.json"],
                "stage3": ["synthesis_results.json", "literature_review.txt", "research_plan.txt", "bibliography.txt"],
            },
        },
        "ModelTeam": {
            "stages": [
                {"number": 1, "name": "Theory", "module": "1-TheoryStage"},
                {"number": 2, "name": "ModelDesign", "module": "2-ModelDesignStage"},
                {"number": 3, "name": "Calibration", "module": "3-CalibrationStage"},
            ],
            "modes": ["ModeNoWcNoHITL", "ModeNoWcWithHITL", "ModeWithWcNoHITL", "ModeWithWcWithHITL"],
            "outputs": {
                "stage1": ["theory_output.json"],
                "stage2": ["model_design_output.json"],
                "stage3": ["calibration_output.json", "calibrated_models.txt", "calibrated_parameters.csv"],
            },
        },
        "DataTeam": {
            "stages": [
                {"number": 1, "name": "DataSource", "module": "1-DataSourceStage"},
                {"number": 2, "name": "DataCleaning", "module": "2-DataCleaningStage"},
                {"number": 3, "name": "QualityAssurance", "module": "3-QualityAssuranceStage"},
            ],
            "modes": ["ModeOpenSourceAPI", "ModePremiumSubscribed", "ModeUserUploaded"],
            "outputs": {
                "stage1": ["api_source_output.json", "data_source_output.json"],
                "stage2": ["api_cleaning_output.json", "data_cleaning_output.json"],
                "stage3": ["api_qa_output.json", "quality_assurance_output.json"],
            },
        },
        # No Wc axis: 2 modes only; stage files byte-identical.
        "EstimationTeam": {
            "stages": [
                {"number": 1, "name": "Estimation", "module": "1-EstimationStage"},
                {"number": 2, "name": "ValidationDiagnostics", "module": "2-ValidationDiagnosticsStage"},
                {"number": 3, "name": "InferenceRobustness", "module": "3-InferenceRobustnessStage"},
            ],
            "modes": ["ModeNoWcNoHITL", "ModeNoWcWithHITL"],
            "outputs": {
                "stage1": ["estimation_output.json"],
                "stage2": ["validation_output.json"],
                "stage3": ["inference_output.json"],
            },
        },
        # No Wc axis: 2 modes only; stage files byte-identical.
        "ReportingTeam": {
            "stages": [
                {"number": 1, "name": "Interpretation", "module": "1-InterpretationStage"},
                {"number": 2, "name": "Drafting", "module": "2-DraftingStage"},
                {"number": 3, "name": "Quality", "module": "3-QualityStage"},
            ],
            "modes": ["ModeNoWcNoHITL", "ModeNoWcWithHITL"],
            "outputs": {
                "stage1": ["interpretation_output.json"],
                "stage2": ["drafting_output.json", "research_report.md"],
                "stage3": ["quality_output.json"],
            },
        },
        # No Wc axis: 2 modes only; stage files byte-identical.
        "CodeTeam": {
            "stages": [
                {"number": 1, "name": "CodeGeneration", "module": "1-CodeGenerationStage"},
                {"number": 2, "name": "Validation", "module": "2-ValidationStage"},
                {"number": 3, "name": "Experimentation", "module": "3-ExperimentationStage"},
            ],
            "modes": ["ModeNoWcNoHITL", "ModeNoWcWithHITL"],
            "outputs": {
                "stage1": ["generation_output.json"],
                "stage2": ["validation_output.json"],
                "stage3": ["experimentation_output.json"],
            },
        },
    }

    def __init__(self, base_path: Optional[Path] = None):
        """Initialize AEL parser."""
        super().__init__(base_path)

    def get_team_config(self, team: str) -> Dict[str, Any]:
        """Get configuration for a specific team."""
        return self.TEAM_CONFIGS.get(team, {})

    def get_expected_outputs(self, team: str, mode: str) -> List[str]:
        """Get list of expected output files for a team."""
        config = self.get_team_config(team)
        if not config:
            return []

        outputs = []
        for stage_key in ["stage1", "stage2", "stage3"]:
            outputs.extend(config.get("outputs", {}).get(stage_key, []))
        return outputs

    def parse_outputs(self, output_dir: Path) -> Dict[str, Any]:
        """
        Parse all output files from a workflow run.

        Args:
            output_dir: Directory containing workflow outputs

        Returns:
            Dictionary with parsed data from all output files
        """
        results = {
            "json_outputs": {},
            "csv_outputs": {},
            "text_outputs": {},
            "metadata": {
                "output_dir": str(output_dir),
                "parsed_at": datetime.now().isoformat(),
            }
        }

        if not output_dir.exists():
            return results

        # Parse JSON files
        for json_file in output_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    results["json_outputs"][json_file.name] = json.load(f)
            except (json.JSONDecodeError, Exception) as e:
                results["json_outputs"][json_file.name] = {"error": str(e)}

        # Parse CSV files
        for csv_file in output_dir.glob("*.csv"):
            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    results["csv_outputs"][csv_file.name] = list(reader)
            except Exception as e:
                results["csv_outputs"][csv_file.name] = {"error": str(e)}

        # Read text files
        for text_file in list(output_dir.glob("*.txt")) + list(output_dir.glob("*.md")):
            try:
                with open(text_file, 'r', encoding='utf-8') as f:
                    results["text_outputs"][text_file.name] = f.read()
            except Exception as e:
                results["text_outputs"][text_file.name] = {"error": str(e)}

        return results

    def extract_timing(
        self,
        output_dir: Path,
        log_file: Optional[Path] = None
    ) -> TimingInfo:
        """
        Extract timing information from outputs or logs.

        Looks for timestamps in JSON outputs or parses log file.
        """
        start_time = None
        end_time = None

        # Try to get timing from JSON outputs
        for json_file in output_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        # Look for timestamp fields
                        if 'timestamp' in data:
                            ts = data['timestamp']
                            try:
                                dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                                if start_time is None or dt < start_time:
                                    start_time = dt
                                if end_time is None or dt > end_time:
                                    end_time = dt
                            except (ValueError, TypeError):
                                pass
            except Exception:
                pass

        # Parse log file if available
        if log_file and log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Look for timestamp patterns
                timestamp_pattern = r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}'
                matches = re.findall(timestamp_pattern, content)

                if matches:
                    first_ts = datetime.strptime(matches[0], '%Y-%m-%d %H:%M:%S')
                    last_ts = datetime.strptime(matches[-1], '%Y-%m-%d %H:%M:%S')

                    if start_time is None or first_ts < start_time:
                        start_time = first_ts
                    if end_time is None or last_ts > end_time:
                        end_time = last_ts
            except Exception:
                pass

        # Default to file modification times
        if start_time is None and output_dir.exists():
            file_times = []
            for f in output_dir.iterdir():
                if f.is_file():
                    file_times.append(datetime.fromtimestamp(f.stat().st_mtime))
            if file_times:
                start_time = min(file_times)
                end_time = max(file_times)

        # Create timing info
        timing = TimingInfo(
            start_time=start_time or datetime.now(),
            end_time=end_time,
        )
        timing.compute_duration()
        return timing

    def extract_errors(
        self,
        output_dir: Path,
        log_file: Optional[Path] = None
    ) -> List[ErrorInfo]:
        """
        Extract error information from logs.

        Looks for error patterns in log files or output metadata.
        """
        errors = []

        # Error patterns to look for
        error_patterns = [
            (r'\[ERROR\]\s*(.+)', 'ERROR'),
            (r'Exception:\s*(.+)', 'Exception'),
            (r'Error:\s*(.+)', 'Error'),
            (r'Failed:\s*(.+)', 'Failure'),
            (r'Traceback.+?(?=\n\n|\Z)', 'Traceback'),
        ]

        if log_file and log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                for pattern, error_type in error_patterns:
                    matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
                    for match in matches:
                        errors.append(ErrorInfo(
                            error_type=error_type,
                            error_message=match[:200] if isinstance(match, str) else str(match)[:200],
                            recovered=True  # Assume recovered if execution continued
                        ))
            except Exception:
                pass

        # Check JSON outputs for error fields
        for json_file in output_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and 'error' in data:
                        errors.append(ErrorInfo(
                            error_type="OutputError",
                            error_message=str(data['error'])[:200],
                            recovered=False
                        ))
            except Exception:
                pass

        return errors

    def _parse_stage_outputs(
        self,
        team: str,
        stage_number: int,
        output_dir: Path
    ) -> tuple[List[str], Optional[int]]:
        """
        Parse outputs for a specific stage.

        Returns:
            Tuple of (output_files, item_count)
        """
        config = self.get_team_config(team)
        stage_key = f"stage{stage_number}"
        expected_files = config.get("outputs", {}).get(stage_key, [])

        found_files = []
        item_count = None

        for filename in expected_files:
            filepath = output_dir / filename
            if filepath.exists():
                found_files.append(filename)

                # Try to extract item count
                if filename.endswith('.json'):
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            # Look for list fields
                            if isinstance(data, list):
                                item_count = len(data)
                            elif isinstance(data, dict):
                                for key, value in data.items():
                                    if isinstance(value, list):
                                        item_count = len(value)
                                        break
                    except Exception:
                        pass
                elif filename.endswith('.csv'):
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            item_count = sum(1 for _ in csv.reader(f)) - 1  # Subtract header
                    except Exception:
                        pass

        return found_files, item_count

    def build_execution_trace(
        self,
        team: str,
        mode: str,
        output_dir: Path,
        log_file: Optional[Path] = None,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> ExecutionTrace:
        """
        Build complete execution trace from AEL workflow outputs.

        Args:
            team: Team name
            mode: Mode name
            output_dir: Directory containing outputs
            log_file: Optional path to log file
            inputs: Input parameters

        Returns:
            Complete ExecutionTrace object
        """
        config = self.get_team_config(team)

        # Extract timing
        timing = self.extract_timing(output_dir, log_file)

        # Extract errors
        errors = self.extract_errors(output_dir, log_file)

        # Build stage executions
        stages = []
        for stage_config in config.get("stages", []):
            output_files, item_count = self._parse_stage_outputs(
                team, stage_config["number"], output_dir
            )

            # Determine stage status
            if output_files:
                status = "success"
            elif stage_config["number"] <= len([s for s in stages if s.status == "success"]) + 1:
                status = "pending"
            else:
                status = "error"

            stage = StageExecution(
                stage_name=stage_config["name"],
                stage_number=stage_config["number"],
                timing=TimingInfo(start_time=timing.start_time),
                status=status,
                output_files=output_files,
                item_count=item_count,
                metadata={"module": stage_config["module"]}
            )
            stages.append(stage)

        # Collect all outputs
        outputs = self.collect_output_files(output_dir)

        # Create execution trace
        trace = ExecutionTrace(
            team=team,
            mode=mode,
            framework=self.FRAMEWORK,
            start_time=timing.start_time,
            end_time=timing.end_time,
            total_duration_seconds=timing.duration_seconds,
            stages=stages,
            errors=errors,
            inputs=inputs or {},
            outputs=outputs,
        )

        return trace

    def get_workflow_path(self, team: str, mode: str) -> Path:
        """
        Get the path to a workflow's directory.

        Args:
            team: Team name
            mode: Mode name

        Returns:
            Path to the workflow directory
        """
        return self.base_path / team / "ael" / mode
