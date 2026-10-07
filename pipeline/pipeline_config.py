# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Pipeline Configuration — Pydantic models + YAML loader for pipeline definitions.

Defines the structure of pipeline YAML configs and provides loading/validation.

Usage:
    from pipeline.pipeline_config import load_pipeline_config

    config = load_pipeline_config("pipeline/configs/full_research.yaml")
    for stage in config.stages:
        print(f"{stage.team} depends on {stage.depends_on}")
"""

import os
import warnings
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Mode-name deprecation aliasing (added 2026-04-20, Firecrawl → WebCrawl refactor).
#
# The old ``Mode*Fc*`` names are still accepted on input (CLI args, YAML config,
# programmatic construction) but normalized to ``Mode*Wc*`` with a one-shot
# DeprecationWarning. Scheduled for removal in a future release.
# ---------------------------------------------------------------------------

_FC_TO_WC = {
    "ModeNoFcNoHITL":     "ModeNoWcNoHITL",
    "ModeNoFcWithHITL":   "ModeNoWcWithHITL",
    "ModeWithFcNoHITL":   "ModeWithWcNoHITL",
    "ModeWithFcWithHITL": "ModeWithWcWithHITL",
}
_DEPRECATION_WARNED: set = set()


def _normalize_mode_name(value: str) -> str:
    """Map legacy ``Mode*Fc*`` names to their ``Mode*Wc*`` replacement.

    Emits a one-shot ``DeprecationWarning`` per unique old name so callers
    know to update their config. Unknown inputs pass through untouched.
    """
    if not isinstance(value, str):
        return value
    if value not in _FC_TO_WC:
        return value
    new_value = _FC_TO_WC[value]
    if value not in _DEPRECATION_WARNED:
        _DEPRECATION_WARNED.add(value)
        warnings.warn(
            f"Mode name {value!r} is deprecated; use {new_value!r} instead. "
            "Legacy names will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
    return new_value


class StageConfig(BaseModel):
    """Configuration for a single team stage in the pipeline."""

    team: str = Field(description="Team name (e.g., 'IdeationTeam')")
    mode: str = Field(description="Execution mode (e.g., 'ModeNoWcNoHITL')")

    @field_validator("mode", mode="before")
    @classmethod
    def _normalize_legacy_mode(cls, v: Any) -> Any:
        return _normalize_mode_name(v) if isinstance(v, str) else v
    inputs: List[str] = Field(
        default_factory=list,
        description="Artifact names this stage consumes",
    )
    outputs: List[str] = Field(
        default_factory=list,
        description="Artifact names this stage produces",
    )
    depends_on: List[str] = Field(
        default_factory=list,
        description="Team names that must complete before this stage",
    )
    enabled: bool = Field(default=True, description="Whether this stage is enabled")


class BudgetConfig(BaseModel):
    """Budget limits for the pipeline."""

    max_total_cost_usd: float = Field(default=5.0)
    max_per_team_cost_usd: float = Field(default=2.0)
    max_total_tokens: int = Field(default=500000)


class PipelineConfig(BaseModel):
    """Full pipeline configuration loaded from YAML."""

    name: str = Field(description="Pipeline name")
    mode: str = Field(
        default="ModeNoWcNoHITL", description="Default mode for all stages"
    )

    @field_validator("mode", mode="before")
    @classmethod
    def _normalize_legacy_mode(cls, v: Any) -> Any:
        return _normalize_mode_name(v) if isinstance(v, str) else v

    stages: List[StageConfig] = Field(description="Ordered list of pipeline stages")
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    output_dir: str = Field(
        default="pipeline_output", description="Base output directory"
    )

    # --- Model<->Data feasibility loop ---
    feasibility_loop: bool = Field(
        default=True,
        description="Run the bounded Model<->Data feasibility loop after the linear pass "
                    "(ModelTeam and DataTeam must both be present). On by default; set false "
                    "to skip. In `auto` HITL mode the loop is cheap (deterministic scout + "
                    "documented-limitation default, no model re-runs).",
    )
    feasibility_max_cycles: int = Field(
        default=3, description="Max Model<->Data cycles before terminating with a documented limitation")
    feasibility_escalation: bool = Field(
        default=False,
        description="Whether the Feasibility Review may escalate to premium/user-uploaded "
                    "tiers. False on HPC (no Refinitiv terminal) — forces the proxy-vs-revise trade-off.")

    @model_validator(mode="after")
    def validate_unique_teams(self) -> "PipelineConfig":
        """Ensure no duplicate team names in stages."""
        seen = set()
        for stage in self.stages:
            if stage.team in seen:
                raise ValueError(f"Duplicate team name in stages: '{stage.team}'")
            seen.add(stage.team)
        return self

    @model_validator(mode="after")
    def validate_dependencies(self) -> "PipelineConfig":
        """Ensure all depends_on references point to existing teams."""
        team_names = {s.team for s in self.stages}
        for stage in self.stages:
            for dep in stage.depends_on:
                if dep not in team_names:
                    raise ValueError(
                        f"Stage '{stage.team}' depends on unknown team '{dep}'. "
                        f"Available: {team_names}"
                    )
        return self

    @model_validator(mode="after")
    def validate_no_circular_deps(self) -> "PipelineConfig":
        """Ensure the dependency graph is a DAG (no cycles).

        Delegates to get_execution_order() which uses Kahn's algorithm.
        """
        try:
            self.get_execution_order()
        except ValueError:
            raise ValueError("Circular dependency detected in pipeline stages")
        return self

    def get_execution_order(self) -> List[str]:
        """Return teams in topological order (respecting depends_on)."""
        graph: Dict[str, List[str]] = {s.team: list(s.depends_on) for s in self.stages}
        order = []
        remaining = set(graph.keys())

        while remaining:
            # Find teams with no unresolved dependencies
            ready = [
                t for t in remaining
                if all(d not in remaining for d in graph[t])
            ]
            if not ready:
                raise ValueError("Circular dependency in pipeline stages")
            # Sort alphabetically for deterministic ordering among peers
            ready.sort()
            order.extend(ready)
            remaining -= set(ready)

        return order

    def get_stage(self, team_name: str) -> Optional[StageConfig]:
        """Get the stage config for a specific team."""
        for stage in self.stages:
            if stage.team == team_name:
                return stage
        return None


def load_pipeline_config(yaml_path: str) -> PipelineConfig:
    """
    Load and validate a pipeline configuration from a YAML file.

    Args:
        yaml_path: Path to the YAML config file.

    Returns:
        Validated PipelineConfig.

    Raises:
        FileNotFoundError: If the YAML file doesn't exist.
        ValidationError: If the config is invalid.
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Pipeline config not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Pipeline config '{yaml_path}' is empty or invalid YAML")

    # Support top-level "pipeline:" wrapper or flat dict
    if "pipeline" in raw:
        raw = raw["pipeline"]

    return PipelineConfig.model_validate(raw)
