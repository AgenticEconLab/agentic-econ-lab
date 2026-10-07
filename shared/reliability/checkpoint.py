# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Checkpoint Manager — Checkpoint/resume for AEL pipeline execution.

Saves pipeline state after each stage/team completes, enabling
resume from the last successful checkpoint after failure.

Usage:
    from shared.reliability.checkpoint import CheckpointManager

    mgr = CheckpointManager(checkpoint_dir="./checkpoints")

    # Save after each stage
    mgr.save("IdeationTeam:SourcingStage", stage_output, {"duration": 120})

    # Check if resumable
    if mgr.can_resume():
        stage, data, meta = mgr.load_latest()
        print(f"Resuming from {stage}")

    # List all checkpoints
    for cp in mgr.list_checkpoints():
        print(cp)
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class Checkpoint:
    """A single checkpoint entry."""

    __slots__ = ("stage_name", "data", "metadata", "timestamp", "file_path")

    def __init__(
        self,
        stage_name: str,
        data: Dict[str, Any],
        metadata: Dict[str, Any],
        timestamp: float,
        file_path: str = "",
    ):
        self.stage_name = stage_name
        self.data = data
        self.metadata = metadata
        self.timestamp = timestamp
        self.file_path = file_path

    def __repr__(self):
        return f"Checkpoint({self.stage_name}, ts={self.timestamp:.0f})"


class CheckpointManager:
    """
    Manages checkpoint/resume for pipeline execution.

    Saves JSON checkpoint files after each stage or team completes.
    On failure, can resume from the most recent successful checkpoint.
    """

    def __init__(self, checkpoint_dir: str, pipeline_run_id: str = ""):
        """
        Args:
            checkpoint_dir: Directory to store checkpoint files.
            pipeline_run_id: Optional run ID for namespacing.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.pipeline_run_id = pipeline_run_id
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        stage_name: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a checkpoint after a stage completes.

        Args:
            stage_name: Stage identifier (e.g., "IdeationTeam:SourcingStage").
            data: Stage output data to checkpoint.
            metadata: Optional metadata (duration, item count, etc.).

        Returns:
            Path to the saved checkpoint file.
        """
        ts = time.time()
        checkpoint = {
            "stage_name": stage_name,
            "data": data,
            "metadata": metadata or {},
            "timestamp": ts,
            "pipeline_run_id": self.pipeline_run_id,
        }

        # Filename: sequential index + stage name
        existing = list(self.checkpoint_dir.glob("checkpoint_*.json"))
        index = len(existing)
        filename = f"checkpoint_{index:03d}_{stage_name.replace(':', '_')}.json"
        file_path = self.checkpoint_dir / filename

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2, default=str)

        return str(file_path)

    def load_latest(self) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
        """
        Load the most recent checkpoint.

        Returns:
            Tuple of (stage_name, data, metadata), or None if no checkpoints.
        """
        checkpoints = self._load_all_checkpoints()
        if not checkpoints:
            return None
        latest = max(checkpoints, key=lambda c: (c.timestamp, c.file_path))
        return (latest.stage_name, latest.data, latest.metadata)

    def can_resume(self) -> bool:
        """Check if a resumable checkpoint exists."""
        return len(list(self.checkpoint_dir.glob("checkpoint_*.json"))) > 0

    def list_checkpoints(self) -> List[Checkpoint]:
        """List all checkpoints in chronological order."""
        checkpoints = self._load_all_checkpoints()
        return sorted(checkpoints, key=lambda c: c.timestamp)

    def get_completed_stages(self) -> List[str]:
        """Get list of stage names that have checkpoints."""
        return [cp.stage_name for cp in self.list_checkpoints()]

    def clean(self, keep_latest: int = 0):
        """
        Remove checkpoint files.

        Args:
            keep_latest: Number of most recent checkpoints to keep (0 = remove all).
        """
        files = sorted(self.checkpoint_dir.glob("checkpoint_*.json"))
        if keep_latest > 0:
            files_to_remove = files[:-keep_latest]
        else:
            files_to_remove = files
        for f in files_to_remove:
            f.unlink(missing_ok=True)

    def _load_all_checkpoints(self) -> List[Checkpoint]:
        """Load all checkpoint files."""
        checkpoints = []
        for file_path in self.checkpoint_dir.glob("checkpoint_*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                checkpoints.append(Checkpoint(
                    stage_name=data["stage_name"],
                    data=data.get("data", {}),
                    metadata=data.get("metadata", {}),
                    timestamp=data.get("timestamp", 0),
                    file_path=str(file_path),
                ))
            except (json.JSONDecodeError, KeyError):
                continue
        return checkpoints
