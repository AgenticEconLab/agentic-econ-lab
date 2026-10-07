# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Cross-Team World Model — Structured shared state for multi-team research (V0.6 Phase 3).

Inspired by FutureHouse Kosmos's world-model pattern for long-horizon
autonomous research campaigns. Provides persistent, versioned state
across pipeline stages and runs.

Usage:
    from shared.research.world_model import WorldModel

    wm = WorldModel(research_topic="AI and monetary policy")
    wm.update("IdeationTeam", "RefinementStage", {"questions": [...]})
    view = wm.query("LiteratureTeam")
    wm.persist("./output/world_model.json")

    # Resume from previous run
    wm2 = WorldModel.load("./output/world_model.json")
"""

import json
import os
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Data Models ───────────────────────────────────────────────────────────


class CrossReference(BaseModel):
    """Link between items from different teams."""

    source_team: str
    source_id: str
    target_team: str
    target_id: str
    relationship: str = "related"


class RunSummary(BaseModel):
    """Summary of a single pipeline run."""

    run_id: str = ""
    timestamp: float = 0.0
    teams_run: List[str] = Field(default_factory=list)
    stages_completed: int = 0
    duration_seconds: float = 0.0


class WorldModelState(BaseModel):
    """Structured state shared across all teams."""

    research_topic: str = ""
    research_questions: List[Dict[str, Any]] = Field(default_factory=list)
    literature_findings: List[Dict[str, Any]] = Field(default_factory=list)
    identified_gaps: List[Dict[str, Any]] = Field(default_factory=list)
    data_sources: List[Dict[str, Any]] = Field(default_factory=list)
    data_quality_scores: Dict[str, float] = Field(default_factory=dict)
    model_specifications: List[Dict[str, Any]] = Field(default_factory=list)
    calibration_results: List[Dict[str, Any]] = Field(default_factory=list)
    cross_references: List[CrossReference] = Field(default_factory=list)
    version: int = 0
    last_updated: str = ""
    run_history: List[RunSummary] = Field(default_factory=list)
    custom: Dict[str, Any] = Field(default_factory=dict)


class WorldModelView(BaseModel):
    """Team-specific filtered view of the world model."""

    team: str
    relevant_state: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    last_sync_version: int = 0


# ── Team → State Field Mapping ────────────────────────────────────────────

# Which state fields each team produces and consumes
TEAM_PRODUCES: Dict[str, List[str]] = {
    "IdeationTeam": ["research_questions"],
    "LiteratureTeam": ["literature_findings", "identified_gaps"],
    "DataTeam": ["data_sources", "data_quality_scores"],
    "ModelTeam": ["model_specifications", "calibration_results"],
}

TEAM_CONSUMES: Dict[str, List[str]] = {
    "IdeationTeam": [],  # First in pipeline
    "LiteratureTeam": ["research_questions"],
    "DataTeam": ["research_questions", "model_specifications"],
    "ModelTeam": ["research_questions", "literature_findings", "identified_gaps",
                  "data_sources", "data_quality_scores"],
}


# ── World Model ───────────────────────────────────────────────────────────


class WorldModel:
    """Structured shared state across AEL teams for multi-run research campaigns.

    Args:
        research_topic: The research topic for this campaign.
        persist_dir: Optional directory for auto-persistence.
    """

    def __init__(self, research_topic: str = "", persist_dir: Optional[str] = None):
        self.state = WorldModelState(research_topic=research_topic)
        self.persist_dir = persist_dir

    def update(self, team: str, stage: str, findings: Dict[str, Any]) -> None:
        """Update world model with team/stage output.

        Args:
            team: Team name (e.g., "IdeationTeam").
            stage: Stage name (e.g., "RefinementStage").
            findings: Dict of findings to merge into state.
        """
        # Map team output fields to state fields
        produce_fields = TEAM_PRODUCES.get(team, [])
        for field in produce_fields:
            if field in findings:
                current = getattr(self.state, field, None)
                new_data = findings[field]
                if isinstance(current, list) and isinstance(new_data, list):
                    # Extend (append new items)
                    current.extend(new_data)
                elif isinstance(current, dict) and isinstance(new_data, dict):
                    current.update(new_data)
                else:
                    setattr(self.state, field, new_data)

        # Store any custom/unmapped fields
        for key, value in findings.items():
            if key not in produce_fields:
                self.state.custom[f"{team}:{stage}:{key}"] = value

        self.state.version += 1
        self.state.last_updated = time.strftime("%Y-%m-%dT%H:%M:%S")

    def query(self, team: str, view: str = "full") -> WorldModelView:
        """Get team-specific view of world model state.

        Args:
            team: Team name to get view for.
            view: "full" for all consumed fields, "minimal" for just dependencies.

        Returns:
            WorldModelView with relevant state for this team.
        """
        consume_fields = TEAM_CONSUMES.get(team, [])
        relevant = {}
        for field in consume_fields:
            value = getattr(self.state, field, None)
            if value is not None:
                relevant[field] = deepcopy(value)

        if view == "full":
            relevant["research_topic"] = self.state.research_topic
            relevant["version"] = self.state.version

        dependencies = [
            t for t, fields in TEAM_PRODUCES.items()
            if any(f in consume_fields for f in fields)
        ]

        return WorldModelView(
            team=team,
            relevant_state=relevant,
            dependencies=dependencies,
            last_sync_version=self.state.version,
        )

    def add_cross_reference(
        self,
        source_team: str,
        source_id: str,
        target_team: str,
        target_id: str,
        relationship: str = "related",
    ) -> None:
        """Add a cross-reference between team outputs."""
        self.state.cross_references.append(
            CrossReference(
                source_team=source_team,
                source_id=source_id,
                target_team=target_team,
                target_id=target_id,
                relationship=relationship,
            )
        )

    def add_run_summary(self, run_summary: RunSummary) -> None:
        """Record a completed pipeline run."""
        self.state.run_history.append(run_summary)

    def persist(self, path: Optional[str] = None) -> str:
        """Save world model to disk for cross-run persistence.

        Args:
            path: File path to save to. If None, uses persist_dir.

        Returns:
            Path where the model was saved.
        """
        if path is None and self.persist_dir:
            os.makedirs(self.persist_dir, exist_ok=True)
            path = os.path.join(self.persist_dir, "world_model.json")
        if path is None:
            raise ValueError("No path specified and no persist_dir configured")

        data = self.state.model_dump(mode="json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    @classmethod
    def load(cls, path: str) -> "WorldModel":
        """Load world model from previous run.

        Args:
            path: Path to JSON file.

        Returns:
            Loaded WorldModel instance.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        wm = cls()
        wm.state = WorldModelState(**data)
        return wm

    def merge(self, other: "WorldModel") -> "WorldModel":
        """Merge findings from parallel team runs with conflict resolution.

        Produces a new WorldModel combining both states. Lists are concatenated;
        dicts are merged (self wins on conflict); version is max + 1.

        Args:
            other: Another WorldModel to merge in.

        Returns:
            New merged WorldModel.
        """
        merged = WorldModel(
            research_topic=self.state.research_topic or other.state.research_topic
        )
        # Merge list fields by concatenation
        for field in ["research_questions", "literature_findings", "identified_gaps",
                      "data_sources", "model_specifications", "calibration_results",
                      "cross_references", "run_history"]:
            self_val = getattr(self.state, field, [])
            other_val = getattr(other.state, field, [])
            setattr(merged.state, field, list(self_val) + list(other_val))

        # Merge dict fields (self wins on conflict)
        for field in ["data_quality_scores", "custom"]:
            self_val = getattr(self.state, field, {})
            other_val = getattr(other.state, field, {})
            combined = {**other_val, **self_val}
            setattr(merged.state, field, combined)

        merged.state.version = max(self.state.version, other.state.version) + 1
        merged.state.last_updated = time.strftime("%Y-%m-%dT%H:%M:%S")

        return merged

    def get_state_summary(self) -> Dict[str, Any]:
        """Return a concise summary of current state."""
        return {
            "research_topic": self.state.research_topic,
            "version": self.state.version,
            "research_questions": len(self.state.research_questions),
            "literature_findings": len(self.state.literature_findings),
            "identified_gaps": len(self.state.identified_gaps),
            "data_sources": len(self.state.data_sources),
            "model_specifications": len(self.state.model_specifications),
            "calibration_results": len(self.state.calibration_results),
            "cross_references": len(self.state.cross_references),
            "runs": len(self.state.run_history),
            "last_updated": self.state.last_updated,
        }
