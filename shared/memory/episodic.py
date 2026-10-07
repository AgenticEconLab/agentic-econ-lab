# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Episodic Memory — JSON trajectory logs for self-improvement.

Tier 3 of the 3-tier memory system. Records past run trajectories
so future runs can learn from prior execution patterns.

Usage:
    from shared.memory.episodic import EpisodicMemory, RunTrajectory

    em = EpisodicMemory(memory_dir="./memory")
    trajectory = RunTrajectory(
        run_id="pipeline-abc123",
        research_topic="AI in economics",
        teams_completed=["IdeationTeam", "LiteratureTeam"],
        success=True,
        total_duration_sec=120.5,
    )
    em.record_trajectory(trajectory)
    similar = em.recall_similar_runs("AI economics")
"""

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RunTrajectory(BaseModel):
    """Record of a single pipeline run for episodic memory."""

    run_id: str
    research_topic: str
    teams_completed: List[str] = Field(default_factory=list)
    teams_failed: List[str] = Field(default_factory=list)
    success: bool = True
    total_duration_sec: float = 0.0
    artifact_names: List[str] = Field(default_factory=list)
    config_name: str = ""
    mode: str = ""
    key_findings: List[str] = Field(default_factory=list)
    lessons_learned: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class EpisodicMemory:
    """
    JSON file-based trajectory storage for cross-run self-improvement.

    Each trajectory is stored as a separate JSON file in memory_dir/episodic/.
    Recall is based on keyword matching against research topics.
    """

    def __init__(self, memory_dir: str):
        self._dir = os.path.join(memory_dir, "episodic")
        os.makedirs(self._dir, exist_ok=True)

    def record_trajectory(self, trajectory: RunTrajectory) -> str:
        """
        Save a run trajectory to disk.

        Returns:
            Path to the saved trajectory file.
        """
        safe_id = re.sub(r"[^\w\-]", "_", trajectory.run_id)
        filename = f"{safe_id}.json"
        filepath = os.path.join(self._dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(trajectory.model_dump(), f, indent=2, default=str)
        return filepath

    def recall_similar_runs(
        self,
        research_topic: str,
        top_k: int = 3,
    ) -> List[RunTrajectory]:
        """
        Find past runs with similar research topics.

        Uses keyword overlap scoring (words in common / total unique words).
        Returns up to top_k most similar trajectories, most similar first.
        """
        topic_words = set(research_topic.lower().split())
        if not topic_words:
            return []

        scored: List[tuple] = []
        for trajectory in self._load_all():
            traj_words = set(trajectory.research_topic.lower().split())
            if not traj_words:
                continue
            overlap = len(topic_words & traj_words)
            union = len(topic_words | traj_words)
            score = overlap / union if union > 0 else 0.0
            if score > 0:
                scored.append((score, trajectory))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:top_k]]

    def get_trajectory(self, run_id: str) -> Optional[RunTrajectory]:
        """Load a specific trajectory by run ID."""
        safe_id = re.sub(r"[^\w\-]", "_", run_id)
        filepath = os.path.join(self._dir, f"{safe_id}.json")
        if not os.path.exists(filepath):
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return RunTrajectory.model_validate(data)

    def list_runs(self) -> List[str]:
        """List all stored run IDs."""
        runs = []
        for f in os.listdir(self._dir):
            if f.endswith(".json"):
                runs.append(f[:-5])  # strip .json
        return sorted(runs)

    def count(self) -> int:
        """Count stored trajectories."""
        return len([f for f in os.listdir(self._dir) if f.endswith(".json")])

    def _load_all(self) -> List[RunTrajectory]:
        """Load all trajectories from disk."""
        import warnings

        trajectories = []
        for filename in os.listdir(self._dir):
            if not filename.endswith(".json"):
                continue
            filepath = os.path.join(self._dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                trajectories.append(RunTrajectory.model_validate(data))
            except (json.JSONDecodeError, ValueError) as e:
                warnings.warn(f"Skipping corrupted trajectory {filepath}: {e}")
                continue
        return trajectories
