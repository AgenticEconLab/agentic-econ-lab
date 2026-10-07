# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Agent Cards — A2A-inspired capability descriptors for each team.

Each team publishes a TeamCard describing its inputs, outputs, supported
modes, estimated cost, and estimated duration. This enables:
- Cross-team discovery: teams know what upstream/downstream teams provide
- Pipeline configuration: orchestrator uses cards to validate connections
- Cost estimation: pre-run cost forecasting

Usage:
    from shared.protocols import get_team_card, list_team_cards

    card = get_team_card("IdeationTeam")
    print(card.description)
    print(card.input_schema)

    # List all teams
    for card in list_team_cards():
        print(f"{card.team_name}: {card.description}")
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TeamCard(BaseModel):
    """
    A2A-inspired capability descriptor for an AEL team.

    Describes what a team does, what it needs as input, what it produces,
    and its operational characteristics.
    """

    team_name: str = Field(description="Team name (e.g., IdeationTeam)")
    description: str = Field(description="What this team does")

    # Schema references (JSON Schema dicts from Pydantic models)
    input_schema: Dict[str, Any] = Field(
        description="JSON schema of expected inputs")
    output_schema: Dict[str, Any] = Field(
        description="JSON schema of produced outputs")

    # Stages
    stages: List[str] = Field(
        description="Ordered list of stage names")

    # Capabilities
    supported_modes: List[str] = Field(
        description="Available workflow modes")
    supported_frameworks: List[str] = Field(
        default_factory=lambda: ["ael"],
        description="Supported frameworks")

    # Operational estimates
    estimated_cost_usd: float = Field(
        description="Estimated cost per run in USD (ModeNoWcNoHITL)")
    estimated_duration_sec: int = Field(
        description="Estimated duration in seconds (NoFcNoHITL)")
    estimated_tokens: int = Field(
        default=0, description="Estimated total tokens per run")

    # Dependencies
    upstream_teams: List[str] = Field(
        default_factory=list,
        description="Teams whose output this team consumes")
    downstream_teams: List[str] = Field(
        default_factory=list,
        description="Teams that consume this team's output")


# ============================================================================
# Team Card Definitions
# ============================================================================

# Lazy-loaded to avoid import-time schema resolution
_TEAM_CARDS: Optional[Dict[str, TeamCard]] = None


def _build_team_cards() -> Dict[str, TeamCard]:
    """Build team card definitions."""

    # IdeationTeam card
    ideation_card = TeamCard(
        team_name="IdeationTeam",
        description="Generate and refine research questions from trending topics or user-provided topics",
        input_schema={
            "type": "object",
            "properties": {
                "research_topic": {"type": "string", "description": "Research topic to explore"},
            },
            "required": ["research_topic"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "prioritized_questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "theoretical_framework": {"type": "string"},
                            "rationale": {"type": "string"},
                            "methodology": {"type": "array", "items": {"type": "string"}},
                            "priority_rank": {"type": "integer"},
                            "priority_score": {"type": "number"},
                        },
                    },
                },
            },
        },
        stages=["SourcingStage", "RefinementStage", "IntegrationStage"],
        supported_modes=[
            "ModeNoWcNoHITL",
            "ModeNoWcWithHITL",
            "ModeWithWcNoHITL",
            "ModeWithWcWithHITL",
        ],
        estimated_cost_usd=0.006,
        estimated_duration_sec=380,
        estimated_tokens=19000,
        upstream_teams=[],
        downstream_teams=["LiteratureTeam", "ModelTeam"],
    )

    # LiteratureTeam card
    literature_card = TeamCard(
        team_name="LiteratureTeam",
        description="Automated literature review with gathering, gap detection, and synthesis",
        input_schema={
            "type": "object",
            "properties": {
                "research_questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "priority_rank": {"type": "integer"},
                        },
                    },
                },
            },
            "required": ["research_questions"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "literature_review": {"type": "object"},
                "research_plan": {"type": "object"},
                "bibliography": {"type": "object"},
                "knowledge_graph": {"type": "object"},
            },
        },
        stages=["LiteratureGatheringStage", "GapDetectionStage", "SynthesisStage"],
        supported_modes=[
            "ModeNoWcNoHITL",
            "ModeNoWcWithHITL",
            "ModeWithWcNoHITL",
            "ModeWithWcWithHITL",
        ],
        estimated_cost_usd=0.015,
        estimated_duration_sec=600,
        estimated_tokens=45000,
        upstream_teams=["IdeationTeam"],
        downstream_teams=["ModelTeam"],
    )

    # ModelTeam card
    model_card = TeamCard(
        team_name="ModelTeam",
        description="Economic model development, calibration, and sensitivity analysis",
        input_schema={
            "type": "object",
            "properties": {
                "research_questions": {
                    "type": "array",
                    "items": {"type": "object"},
                },
                "literature_review": {"type": "object"},
            },
            "required": ["research_questions"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "calibrated_models": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "model_title": {"type": "string"},
                            "calibrated_parameters": {"type": "array"},
                            "fit_metrics": {"type": "object"},
                        },
                    },
                },
            },
        },
        stages=["TheoryStage", "ModelDesignStage", "CalibrationStage"],
        supported_modes=[
            "ModeNoWcNoHITL",
            "ModeNoWcWithHITL",
            "ModeWithWcNoHITL",
            "ModeWithWcWithHITL",
        ],
        estimated_cost_usd=0.012,
        estimated_duration_sec=500,
        estimated_tokens=35000,
        upstream_teams=["IdeationTeam", "LiteratureTeam"],
        downstream_teams=["DataTeam"],
    )

    # DataTeam card
    data_card = TeamCard(
        team_name="DataTeam",
        description="Data collection, cleaning, and processing for economic research",
        input_schema={
            "type": "object",
            "properties": {
                "research_question": {"type": "string"},
                "data_requirements": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["research_question"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "documented_datasets": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "dataset_name": {"type": "string"},
                            "quality_score": {"type": "number"},
                            "codebook": {"type": "object"},
                        },
                    },
                },
            },
        },
        stages=["DataSourceStage", "DataCleaningStage", "QualityAssuranceStage"],
        supported_modes=[
            "open_source_api",
            "premium_subscribed",
            "user_uploaded",
        ],
        estimated_cost_usd=0.008,
        estimated_duration_sec=450,
        estimated_tokens=25000,
        upstream_teams=["ModelTeam"],
        downstream_teams=[],
    )

    return {
        "IdeationTeam": ideation_card,
        "LiteratureTeam": literature_card,
        "ModelTeam": model_card,
        "DataTeam": data_card,
    }


def _get_cards() -> Dict[str, TeamCard]:
    """Get or build the team cards (lazy singleton)."""
    global _TEAM_CARDS
    if _TEAM_CARDS is None:
        _TEAM_CARDS = _build_team_cards()
    return _TEAM_CARDS


def get_team_card(team_name: str) -> Optional[TeamCard]:
    """
    Get the TeamCard for a specific team.

    Args:
        team_name: Team name (e.g., "IdeationTeam").

    Returns:
        TeamCard or None if team not found.
    """
    return _get_cards().get(team_name)


def list_team_cards() -> List[TeamCard]:
    """List all team cards in pipeline order."""
    cards = _get_cards()
    # Return in pipeline order
    order = ["IdeationTeam", "LiteratureTeam", "ModelTeam", "DataTeam"]
    return [cards[name] for name in order if name in cards]


def get_pipeline_order() -> List[str]:
    """Return team names in pipeline execution order."""
    return ["IdeationTeam", "LiteratureTeam", "ModelTeam", "DataTeam"]


def get_a2a_agent_card(team_name: str, host: str = "localhost", port: int = 8000) -> Optional[Dict[str, Any]]:
    """
    Generate an A2A-compliant agent card from a TeamCard.

    Args:
        team_name: Team name.
        host: Server host.
        port: Server port.

    Returns:
        A2A agent card dict, or None if team not found.
    """
    card = get_team_card(team_name)
    if card is None:
        return None

    return {
        "name": f"AEL-{card.team_name}",
        "description": card.description,
        "url": f"http://{host}:{port}/a2a/{card.team_name.lower()}",
        "version": "0.5.0",
        "protocolVersion": "0.3",
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": True,
        },
        "skills": [
            {"id": stage.lower(), "name": stage, "description": f"Execute {stage}"}
            for stage in card.stages
        ],
        "inputModes": ["text/plain", "application/json"],
        "outputModes": ["text/plain", "application/json"],
        "inputSchema": card.input_schema,
        "outputSchema": card.output_schema,
    }


def get_team_dependencies(team_name: str) -> Dict[str, List[str]]:
    """
    Get upstream/downstream dependencies for a team.

    Returns:
        {"upstream": [...], "downstream": [...]}
    """
    card = get_team_card(team_name)
    if card is None:
        return {"upstream": [], "downstream": []}
    return {
        "upstream": card.upstream_teams,
        "downstream": card.downstream_teams,
    }
