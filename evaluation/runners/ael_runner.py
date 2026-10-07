# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
AEL (Agentic Econ Lab) workflow runner.

Runs AEL workflows with instrumentation for evaluation.
AEL is the plain Python implementation used across all teams.
"""

import json
import hashlib
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
import io
import contextlib

from .base import BaseWorkflowRunner, RunConfig, ExecutionResult

import os


class AELRunner(BaseWorkflowRunner):
    """Runner for AEL (Agentic Econ Lab) workflows."""

    FRAMEWORK = "ael"

    SUPPORTED_TEAMS = ["IdeationTeam", "LiteratureTeam", "ModelTeam", "DataTeam"]

    # Team-specific configurations
    TEAM_CONFIGS = {
        "IdeationTeam": {
            "modes": ["ModeNoWcNoHITL", "ModeNoWcWithHITL", "ModeWithWcNoHITL", "ModeWithWcWithHITL"],
            "stages": ["Sourcing", "Refinement", "Integration"],
            "entry_module": "main",
            "default_inputs": {"research_topic": "AI and economics"},
        },
        "LiteratureTeam": {
            "modes": ["ModeNoWcNoHITL", "ModeNoWcWithHITL", "ModeWithWcNoHITL", "ModeWithWcWithHITL"],
            "stages": ["LiteratureGathering", "GapDetection", "Synthesis"],
            "entry_module": "0-MasterOrchestrator",
            "default_inputs": {"research_topic": "AI and economics"},
        },
        "ModelTeam": {
            "modes": ["ModeNoWcNoHITL", "ModeNoWcWithHITL", "ModeWithWcNoHITL", "ModeWithWcWithHITL"],
            "stages": ["Theory", "ModelDesign", "Calibration"],
            "entry_module": "0-MasterOrchestrator",
            "default_inputs": {"research_question": "How does AI affect labor markets?"},
        },
        "DataTeam": {
            "modes": ["ModeOpenSourceAPI", "ModePremiumSubscribed", "ModeUserUploaded"],
            "stages": ["DataSource", "DataCleaning", "QualityAssurance"],
            "entry_module": "main",
            "default_inputs": {"data_source": "FRED", "indicators": ["GDP", "CPI"]},
        },
    }

    # --- Orchestrator stdout markers (ConsoleUI, shared/console_ui.py) ---------
    # workflow_complete() prints "WORKFLOW COMPLETE" / "WORKFLOW FAILED" plus a
    # summary line "Duration: .. | Stages: N/M | Errors: E"; stage_complete()
    # prints "Stage N complete: .." (success) or "Stage N error: .." / ui.error
    # prints "Stage N failed: ..". A caught mid-stage crash prints WORKFLOW FAILED
    # + Stages: N/M (N<M) but the except block returns, so the process can still
    # exit 0 — RC alone masks the failure. We mine these markers to recover truth.
    _RE_STAGES_SUMMARY = re.compile(r"Stages:\s*(\d+)\s*/\s*(\d+)")
    _RE_STAGE_COMPLETE = re.compile(r"Stage\s+(\d+)\s+complete\b", re.IGNORECASE)
    _RE_STAGE_FAILED = re.compile(r"Stage\s+(\d+)\s+(?:failed|error)\b", re.IGNORECASE)

    def __init__(self, base_path: Optional[Path] = None):
        """Initialize AEL runner."""
        super().__init__(base_path)
        self.logger = logging.getLogger(__name__)

    def get_supported_modes(self, team: str) -> List[str]:
        """Get list of supported modes for a team."""
        config = self.TEAM_CONFIGS.get(team, {})
        return config.get("modes", [])

    def get_team_config(self, team: str) -> Dict[str, Any]:
        """Get configuration for a specific team."""
        return self.TEAM_CONFIGS.get(team, {})

    def _find_workflow_path(self, team: str, mode: str) -> Optional[Path]:
        """
        Find the actual path to the workflow code.

        AEL workflows live under ``<team>/ael/<mode>``; a couple of looser
        fallbacks are kept for direct/flat layouts.
        """
        possible_paths = [
            # AEL folder structure (preferred)
            self.base_path / team / "ael" / mode,
            # Direct mode folder
            self.base_path / team / mode,
        ]

        for path in possible_paths:
            if path.exists():
                return path

        return None

    def _find_entry_point(self, workflow_path: Path, team: str) -> Optional[Path]:
        """
        Find the entry point script for a workflow.

        Returns the path to the main Python file to execute.
        """
        config = self.get_team_config(team)
        entry_module = config.get("entry_module", "main")

        # Try different naming patterns
        possible_names = [
            f"{entry_module}.py",
            "0-MasterOrchestrator.py",
            "main.py",
            "__main__.py",
            f"{team}*.py",
        ]

        for name in possible_names:
            if "*" in name:
                # Glob pattern
                matches = list(workflow_path.glob(name))
                if matches:
                    return matches[0]
            else:
                filepath = workflow_path / name
                if filepath.exists():
                    return filepath

        return None

    def run(self, config: RunConfig) -> ExecutionResult:
        """
        Run an AEL workflow with instrumentation.

        Args:
            config: Run configuration

        Returns:
            ExecutionResult with timing and output information
        """
        start_time = datetime.now()

        # Validate configuration
        errors = self.validate_config(config)
        if errors:
            return ExecutionResult(
                run_id=config.run_id,
                team=config.team,
                mode=config.mode,
                framework=self.FRAMEWORK,
                success=False,
                start_time=start_time,
                end_time=datetime.now(),
                error_message="; ".join(errors),
            )

        # Find workflow path
        workflow_path = self._find_workflow_path(config.team, config.mode)
        if not workflow_path:
            return ExecutionResult(
                run_id=config.run_id,
                team=config.team,
                mode=config.mode,
                framework=self.FRAMEWORK,
                success=False,
                start_time=start_time,
                end_time=datetime.now(),
                error_message=f"Workflow path not found for {config.team}/{config.mode}",
            )

        # Find entry point
        entry_point = self._find_entry_point(workflow_path, config.team)
        if not entry_point:
            return ExecutionResult(
                run_id=config.run_id,
                team=config.team,
                mode=config.mode,
                framework=self.FRAMEWORK,
                success=False,
                start_time=start_time,
                end_time=datetime.now(),
                error_message=f"Entry point not found in {workflow_path}",
            )

        # Prepare output directory
        output_dir = self.ensure_output_dir(config.output_dir)

        # Prepare log file
        log_file = output_dir / f"run_{config.run_id}.log" if config.capture_logs else None

        # Run the workflow
        result = self._execute_workflow(
            entry_point=entry_point,
            workflow_path=workflow_path,
            inputs=config.inputs,
            output_dir=output_dir,
            log_file=log_file,
            timeout_seconds=config.timeout_seconds,
            config=config,
            start_time=start_time,
        )

        return result

    def _execute_workflow(
        self,
        entry_point: Path,
        workflow_path: Path,
        inputs: Dict[str, Any],
        output_dir: Path,
        log_file: Optional[Path],
        timeout_seconds: int,
        config: RunConfig,
        start_time: datetime,
    ) -> ExecutionResult:
        """
        Execute the workflow subprocess.

        Args:
            entry_point: Path to the main Python script
            workflow_path: Path to the workflow directory
            inputs: Input parameters
            output_dir: Directory for outputs
            log_file: Path to log file (if capturing logs)
            timeout_seconds: Timeout for execution
            config: Original run configuration
            start_time: Execution start time

        Returns:
            ExecutionResult with execution details
        """
        team_config = self.get_team_config(config.team)
        total_stages = len(team_config.get("stages", []))

        # Build command
        cmd = [sys.executable, str(entry_point)]

        # Add inputs as command line arguments or environment variables
        env = {
            **dict(os.environ),
            "WORKFLOW_INPUTS": json.dumps(inputs),
            "WORKFLOW_OUTPUT_DIR": str(output_dir),
            "WORKFLOW_RUN_ID": config.run_id,
        }

        # Pass multi-run metadata if present
        if "run_number" in config.metadata:
            env["WORKFLOW_RUN_NUMBER"] = str(config.metadata["run_number"])
        if "total_runs" in config.metadata:
            env["WORKFLOW_TOTAL_RUNS"] = str(config.metadata["total_runs"])

        try:
            # Open log file if capturing
            log_handle = None
            if log_file:
                log_handle = open(log_file, "w", encoding="utf-8")
                log_handle.write(f"=== AEL Workflow Execution Log ===\n")
                log_handle.write(f"Run ID: {config.run_id}\n")
                log_handle.write(f"Team: {config.team}\n")
                log_handle.write(f"Mode: {config.mode}\n")
                log_handle.write(f"Start Time: {start_time.isoformat()}\n")
                log_handle.write(f"Inputs: {json.dumps(inputs, indent=2)}\n")
                log_handle.write(f"{'=' * 50}\n\n")

            # Execute subprocess
            process = subprocess.Popen(
                cmd,
                cwd=str(workflow_path),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            # Collect output with timeout
            stdout_lines = []
            stages_completed = 0

            try:
                for line in process.stdout:
                    stdout_lines.append(line)
                    if log_handle:
                        log_handle.write(line)
                        log_handle.flush()

                process.wait(timeout=timeout_seconds)

            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                end_time = datetime.now()

                if log_handle:
                    log_handle.write(f"\n\n=== TIMEOUT after {timeout_seconds}s ===\n")
                    log_handle.close()

                return ExecutionResult(
                    run_id=config.run_id,
                    team=config.team,
                    mode=config.mode,
                    framework=self.FRAMEWORK,
                    success=False,
                    start_time=start_time,
                    end_time=end_time,
                    duration_seconds=(end_time - start_time).total_seconds(),
                    output_dir=output_dir,
                    log_file=log_file,
                    error_message=f"Workflow timed out after {timeout_seconds} seconds",
                    stages_completed=stages_completed,
                    total_stages=total_stages,
                )

            end_time = datetime.now()
            rc_success = process.returncode == 0

            # Copy outputs from workflow source dir to the isolated run dir FIRST.
            # start_time + sibling-rep comparison gate out STALE files (a crashed
            # stage that didn't rewrite its output, or a byte-identical copy of a
            # prior rep's deliverable) — see _copy_outputs_to_run_dir docstring.
            self._copy_outputs_to_run_dir(workflow_path, output_dir, start_time)
            outputs = self._collect_outputs(output_dir)

            # --- Failure detection from BOTH the return code AND stdout markers ---
            stdout_text = "".join(stdout_lines)
            stdout_failure, stdout_stages = self._analyze_stdout(stdout_text, total_stages)

            # Authoritative stage count: orchestrator's own summary > marker count >
            # output-file inference. (The old "STAGE_COMPLETE"/"Stage completed"
            # markers never matched real output, so manifests were wrong for many
            # runs — V0.7 eval finding.)
            if stdout_stages is not None:
                stages_completed = stdout_stages
            if stages_completed == 0:
                stages_completed = self._count_completed_stages(config.team, output_dir)

            # A rep is only valid if its final-stage deliverable was FRESHLY written
            # this run. Stale deliverables are gated out of the copy above, so a
            # missing/old key deliverable means this rep produced no genuine output.
            deliverable_fresh = self._key_deliverable_fresh(
                config.team, output_dir, start_time
            )

            failure_reasons = []
            if not rc_success:
                failure_reasons.append(f"process exited with code {process.returncode}")
            if stdout_failure:
                failure_reasons.append(
                    "orchestrator reported WORKFLOW FAILED / incomplete stages"
                )
            if not deliverable_fresh:
                failure_reasons.append(
                    "key deliverable was not freshly written this run (stale or missing)"
                )

            success = rc_success and not stdout_failure and deliverable_fresh
            error_message = None if success else "; ".join(failure_reasons)

            if log_handle:
                log_handle.write(f"\n\n{'=' * 50}\n")
                log_handle.write(f"End Time: {end_time.isoformat()}\n")
                log_handle.write(f"Duration: {(end_time - start_time).total_seconds():.2f}s\n")
                log_handle.write(f"Return Code: {process.returncode}\n")
                log_handle.write(f"Stdout failure markers: {stdout_failure}\n")
                log_handle.write(f"Stages completed: {stages_completed}/{total_stages}\n")
                log_handle.write(f"Key deliverable fresh: {deliverable_fresh}\n")
                log_handle.write(f"Success: {success}\n")
                if error_message:
                    log_handle.write(f"Failure reason: {error_message}\n")
                log_handle.close()

            return ExecutionResult(
                run_id=config.run_id,
                team=config.team,
                mode=config.mode,
                framework=self.FRAMEWORK,
                success=success,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=(end_time - start_time).total_seconds(),
                output_dir=output_dir,
                log_file=log_file,
                error_message=error_message,
                stages_completed=stages_completed,
                total_stages=total_stages,
                outputs=outputs,
            )

        except Exception as e:
            end_time = datetime.now()
            self.logger.exception(f"Error executing workflow: {e}")

            if log_handle:
                log_handle.write(f"\n\n=== EXCEPTION ===\n{str(e)}\n")
                log_handle.close()

            return ExecutionResult(
                run_id=config.run_id,
                team=config.team,
                mode=config.mode,
                framework=self.FRAMEWORK,
                success=False,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=(end_time - start_time).total_seconds(),
                output_dir=output_dir,
                log_file=log_file,
                error_message=str(e),
                stages_completed=0,
                total_stages=total_stages,
            )

    def _analyze_stdout(
        self, stdout_text: str, total_stages: int
    ) -> Tuple[bool, Optional[int]]:
        """
        Mine orchestrator stdout for authoritative success / stage-completion truth.

        Returns ``(failure_detected, stages_completed)``. ``stages_completed`` is
        ``None`` when stdout carried no usable marker (caller falls back to output
        files). A failure is flagged when the orchestrator printed ``WORKFLOW
        FAILED``, a ``Stages: N/M`` summary with ``N < M``, or any ``Stage N
        failed/error`` marker — covering the masking case where a caught mid-stage
        crash still lets the process exit 0.
        """
        failure = False
        stages_completed: Optional[int] = None

        if "WORKFLOW FAILED" in stdout_text:
            failure = True

        # The orchestrator's own final summary line is authoritative for the count.
        last_summary = None
        for last_summary in self._RE_STAGES_SUMMARY.finditer(stdout_text):
            pass  # keep the LAST match (the closing workflow_complete summary)
        if last_summary is not None:
            done, total = int(last_summary.group(1)), int(last_summary.group(2))
            stages_completed = done
            if done < total:
                failure = True

        # Per-stage failure markers ("Stage 2 failed: .." / "Stage 2 error: ..").
        if self._RE_STAGE_FAILED.search(stdout_text):
            failure = True

        # Fallback count from per-stage success markers when no summary was printed.
        if stages_completed is None:
            done_nums = {int(n) for n in self._RE_STAGE_COMPLETE.findall(stdout_text)}
            if done_nums:
                stages_completed = len(done_nums)

        return failure, stages_completed

    def _key_deliverable_fresh(
        self, team: str, output_dir: Path, start_time=None
    ) -> bool:
        """
        Confirm the team's final-stage deliverable was FRESHLY written this run.

        Stale deliverables are gated out of ``_copy_outputs_to_run_dir``, so a
        deliverable that is absent from ``output_dir`` (or whose mtime predates the
        run start) means this rep produced no genuine final output and must not be
        counted as success. Reuses the parser's stage-3 output list (DRY).
        """
        if start_time is None:
            return True  # cannot verify timing; do not penalize
        start_ts = start_time.timestamp()
        try:
            from ..parsers.ael_parser import AELParser

            cfg = AELParser(self.base_path).get_team_config(team)
            candidates = cfg.get("outputs", {}).get("stage3", [])
        except Exception:
            candidates = []
        if not candidates:
            return True  # unknown layout; do not penalize
        for name in candidates:
            fp = output_dir / name
            try:
                if fp.exists() and fp.stat().st_mtime >= start_ts:
                    return True
            except OSError:
                continue
        return False

    @staticmethod
    def _file_md5(path: Path) -> str:
        """Stream a file's MD5 (used only to detect byte-identical stale copies)."""
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _is_duplicate_of_sibling(
        self, filepath: Path, sibling_dirs: List[Path]
    ) -> bool:
        """
        True if ``filepath`` is byte-identical to a same-named file already present
        in an earlier rep's directory — i.e. a crashed rep re-emitted (or never
        rewrote) a prior rep's deliverable. This catches the case the mtime gate
        cannot (a stale file whose mtime was bumped within this run's window).
        """
        try:
            src_hash = self._file_md5(filepath)
        except OSError:
            return False
        for d in sibling_dirs:
            cand = d / filepath.name
            if not cand.is_file():
                continue
            try:
                if self._file_md5(cand) == src_hash:
                    return True
            except OSError:
                continue
        return False

    def _copy_outputs_to_run_dir(
        self, workflow_path: Path, output_dir: Path, start_time=None
    ) -> None:
        """
        Copy workflow output files from the source directory to the isolated
        run directory.  MasterOrchestrator scripts write to their own
        script_dir (a SHARED, fixed path), so after each subprocess we copy the
        results to the per-run output directory for proper isolation.

        Two stale-output gates keep a crashed rep from inheriting a prior rep's
        deliverable (the shared mode-dir accumulates outputs across runs):
          (1) mtime gate — only files (re)written at/after this run's start_time
              are this run's. (The former ``-2s`` grace let a stale file written
              just before a fast-crashing rep slip through; removed.)
          (2) content gate — a file byte-identical to an earlier rep's same-named
              file is a stale copy even if its mtime falls inside this run's window.
        """
        start_ts = start_time.timestamp() if start_time is not None else None
        if workflow_path.resolve() == output_dir.resolve():
            return  # Same directory, nothing to copy

        # Sibling rep dirs (multirun layout <base>/run_NNN) for the content gate.
        sibling_dirs: List[Path] = []
        if output_dir.name.startswith("run_"):
            try:
                for d in sorted(output_dir.parent.glob("run_*")):
                    if d.is_dir() and d.resolve() != output_dir.resolve():
                        sibling_dirs.append(d)
            except OSError:
                pass

        output_extensions = {".json", ".csv", ".txt", ".md"}
        # Patterns that mark files as workflow outputs (not source code)
        output_patterns = [
            "literature_results",
            "refinement_results",
            "finalized_research_questions",
            "execution_log",
            "synthesis_results",
            "gap_analysis",
            "knowledge_graph",
            "bibliography",
            "literature_review",
            # LiteratureTeam corpus deliverables — were missing, so the gathered
            # corpus / gaps / plan never reached the run dir to be archived/scored.
            "literature_batch",
            "literature_items",
            "research_gaps",
            "research_plan",
            "theory_output",
            "model_design_output",
            "calibration_output",
            "data_output",
            "quality_report",
            "api_data_requirements",
            # DataTeam stage-2/3 outputs (api_/user_/premium_*) — were missing, so the cleaning/QA
            # reasoning the evaluator now loads never reached the run dir.
            "source_output",
            "cleaning_output",
            "qa_output",
            "codebook",
            "methodology",
            # Archivist's replication manifest (env spec, pipeline code fingerprint,
            # audit trail) — also embedded in qa_output, but archive it standalone too.
            "replication_manifest",
        ]

        for filepath in workflow_path.iterdir():
            if not filepath.is_file():
                continue
            if filepath.suffix not in output_extensions:
                continue
            name_lower = filepath.name.lower()
            # Copy if filename matches any known output pattern
            if not any(pat in name_lower for pat in output_patterns):
                continue

            # (1) mtime gate — skip STALE files not (re)written this run.
            try:
                src_mtime = filepath.stat().st_mtime
            except OSError:
                continue
            if start_ts is not None and src_mtime < start_ts:
                self.logger.warning(
                    f"Skipping STALE output (mtime predates run start): {filepath.name}"
                )
                continue

            # (2) content gate — skip a byte-identical copy of an earlier rep's file.
            if sibling_dirs and self._is_duplicate_of_sibling(filepath, sibling_dirs):
                self.logger.warning(
                    f"Skipping STALE output (identical to another rep): {filepath.name}"
                )
                continue

            dest = output_dir / filepath.name
            try:
                shutil.copy2(filepath, dest)
                self.logger.debug(
                    f"Copied output: {filepath.name} -> {output_dir}"
                )
            except Exception as e:
                self.logger.warning(
                    f"Failed to copy {filepath.name}: {e}"
                )

    def _collect_outputs(self, output_dir: Path) -> Dict[str, Any]:
        """
        Collect output files from the output directory.

        Args:
            output_dir: Directory containing outputs

        Returns:
            Dictionary with categorized output files
        """
        outputs = {
            "json_files": [],
            "csv_files": [],
            "text_files": [],
            "other_files": [],
        }

        if not output_dir.exists():
            return outputs

        for filepath in output_dir.iterdir():
            if filepath.is_file():
                filename = filepath.name
                if filename.endswith(".json"):
                    outputs["json_files"].append(filename)
                elif filename.endswith(".csv"):
                    outputs["csv_files"].append(filename)
                elif filename.endswith((".txt", ".md")):
                    outputs["text_files"].append(filename)
                else:
                    outputs["other_files"].append(filename)

        return outputs

    def _count_completed_stages(self, team: str, output_dir: Path) -> int:
        """
        Count completed stages based on output files.

        Args:
            team: Team name
            output_dir: Directory containing outputs

        Returns:
            Number of stages that appear to have completed
        """
        if not output_dir.exists():
            return 0

        # Import parser to check expected outputs
        from ..parsers.ael_parser import AELParser

        parser = AELParser(self.base_path)
        config = parser.get_team_config(team)

        completed = 0
        for stage_key in ["stage1", "stage2", "stage3"]:
            expected_files = config.get("outputs", {}).get(stage_key, [])
            for filename in expected_files:
                if (output_dir / filename).exists():
                    completed += 1
                    break  # Count stage as complete if any expected file exists

        return completed

    def run_with_callback(
        self,
        config: RunConfig,
        on_stage_complete: Optional[callable] = None,
        on_error: Optional[callable] = None,
    ) -> ExecutionResult:
        """
        Run workflow with callbacks for stage completion and errors.

        Args:
            config: Run configuration
            on_stage_complete: Callback for stage completion (stage_name, stage_number)
            on_error: Callback for errors (error_message)

        Returns:
            ExecutionResult
        """
        # For now, just run normally
        # In a full implementation, this would integrate with the workflow
        # to provide real-time callbacks
        return self.run(config)

    def dry_run(self, config: RunConfig) -> Dict[str, Any]:
        """
        Perform a dry run to validate configuration without executing.

        Args:
            config: Run configuration

        Returns:
            Dictionary with validation results and expected behavior
        """
        errors = self.validate_config(config)
        workflow_path = self._find_workflow_path(config.team, config.mode)
        entry_point = None

        if workflow_path:
            entry_point = self._find_entry_point(workflow_path, config.team)

        team_config = self.get_team_config(config.team)

        return {
            "valid": len(errors) == 0 and entry_point is not None,
            "errors": errors,
            "workflow_path": str(workflow_path) if workflow_path else None,
            "entry_point": str(entry_point) if entry_point else None,
            "expected_stages": team_config.get("stages", []),
            "expected_outputs": self._get_expected_outputs(config.team),
            "config": {
                "team": config.team,
                "mode": config.mode,
                "inputs": config.inputs,
                "timeout_seconds": config.timeout_seconds,
            }
        }

    def run_repeated(
        self,
        config: RunConfig,
        n_runs: int = 3,
        output_base_dir: Optional[Path] = None,
    ) -> List[ExecutionResult]:
        """
        Run the same AEL workflow multiple times with proper environment setup.

        Extends the base run_repeated() to set PYTHONPATH (so shared imports work)
        and AUTO_HITL_MODE=true (so HITL workflows don't block on input).
        Passes WORKFLOW_RUN_NUMBER and WORKFLOW_TOTAL_RUNS to the subprocess.

        Args:
            config: Run configuration
            n_runs: Number of times to run
            output_base_dir: Base directory for all run outputs

        Returns:
            List of execution results from all runs
        """
        base_dir = output_base_dir or config.output_dir or Path(
            f"outputs/{config.team}/{config.mode}"
        )
        base_dir = Path(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        # Set up PYTHONPATH so shared imports resolve
        agents_dir = str(self.base_path)
        current_pythonpath = os.environ.get("PYTHONPATH", "")
        if agents_dir not in current_pythonpath:
            os.environ["PYTHONPATH"] = (
                f"{agents_dir};{current_pythonpath}" if current_pythonpath
                else agents_dir
            )

        # Enable auto-HITL so WithHITL modes don't block
        os.environ["AUTO_HITL_MODE"] = "true"

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

            # Pass run_number to subprocess via environment
            os.environ["WORKFLOW_RUN_NUMBER"] = str(run_number)
            os.environ["WORKFLOW_TOTAL_RUNS"] = str(n_runs)

            self.logger.info(
                f"[MultiRun] Starting run {run_number}/{n_runs} "
                f"for {config.team}/{config.mode} -> {run_dir}"
            )
            result = self.run(run_config)
            results.append(result)

            status = "SUCCESS" if result.success else "FAILED"
            duration = f"{result.duration_seconds:.1f}s" if result.duration_seconds else "N/A"
            self.logger.info(
                f"[MultiRun] Run {run_number}/{n_runs} {status} ({duration})"
            )

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

        self.logger.info(
            f"[MultiRun] Complete: {manifest['summary']['successful_runs']}/{n_runs} "
            f"succeeded. Manifest: {manifest_path}"
        )

        return results

    def _get_expected_outputs(self, team: str) -> List[str]:
        """Get list of expected output files for a team."""
        from ..parsers.ael_parser import AELParser

        parser = AELParser(self.base_path)
        return parser.get_expected_outputs(team, "")
