# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
A2A Protocol Server — Exposes AEL teams as A2A-compliant agents.

Implements the A2A (Agent-to-Agent) protocol v0.3 for inter-agent
communication. Each team can serve an A2A agent card and handle
task requests via JSON-RPC format.

This module provides the protocol layer — the actual HTTP serving
is the caller's responsibility (e.g., via FastAPI or aiohttp).

Usage:
    from shared.protocols.a2a_server import A2ATeamServer

    server = A2ATeamServer("IdeationTeam")

    # Get A2A-compliant agent card
    card = server.get_agent_card()

    # Handle a task request
    result = server.handle_task({
        "jsonrpc": "2.0",
        "method": "tasks/send",
        "params": {"message": {"role": "user", "parts": [{"text": "AI economics"}]}},
        "id": "req-001",
    })
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# A2A Protocol Data Models (aligned with A2A v0.3 spec)
# ---------------------------------------------------------------------------

class A2ASkill(BaseModel):
    """An agent skill — a discrete capability advertised in the agent card."""
    id: str = Field(description="Unique skill identifier")
    name: str = Field(description="Human-readable skill name")
    description: str = Field(description="What this skill does")
    tags: List[str] = Field(default_factory=list)
    examples: List[str] = Field(default_factory=list)


class A2AAgentCard(BaseModel):
    """A2A-compliant agent card (v0.3 specification)."""
    name: str = Field(description="Agent name")
    description: str = Field(description="Agent description")
    url: str = Field(description="Agent endpoint URL")
    version: str = Field(default="0.5.0")
    protocol_version: str = Field(default="0.3")
    capabilities: Dict[str, Any] = Field(default_factory=dict)
    skills: List[A2ASkill] = Field(default_factory=list)
    input_modes: List[str] = Field(default_factory=lambda: ["text/plain", "application/json"])
    output_modes: List[str] = Field(default_factory=lambda: ["text/plain", "application/json"])
    authentication: Optional[Dict[str, Any]] = None


class TaskState(str, Enum):
    """A2A task lifecycle states."""
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class A2ATask(BaseModel):
    """An A2A task request/response."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    state: TaskState = Field(default=TaskState.SUBMITTED)
    message: Dict[str, Any] = Field(default_factory=dict)
    artifacts: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class A2AJsonRpcRequest(BaseModel):
    """JSON-RPC 2.0 request for A2A protocol."""
    jsonrpc: str = Field(default="2.0")
    method: str = Field(description="A2A method (e.g., tasks/send)")
    params: Dict[str, Any] = Field(default_factory=dict)
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class A2AJsonRpcResponse(BaseModel):
    """JSON-RPC 2.0 response for A2A protocol."""
    jsonrpc: str = Field(default="2.0")
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    id: str = Field(default="")


# ---------------------------------------------------------------------------
# A2A Team Skills (derived from TeamCard stages)
# ---------------------------------------------------------------------------

TEAM_SKILLS: Dict[str, List[A2ASkill]] = {
    "IdeationTeam": [
        A2ASkill(
            id="sourcing", name="Research Sourcing",
            description="Gather sources and identify research trends",
            tags=["search", "trends", "literature"],
            examples=["Find recent papers on AI economics"],
        ),
        A2ASkill(
            id="refinement", name="Question Refinement",
            description="Refine and curate research questions",
            tags=["refinement", "curation"],
        ),
        A2ASkill(
            id="integration", name="Question Integration",
            description="Synthesize and prioritize final research questions",
            tags=["synthesis", "prioritization"],
        ),
    ],
    "LiteratureTeam": [
        A2ASkill(
            id="gathering", name="Literature Gathering",
            description="Collect and organize academic literature",
            tags=["search", "literature", "papers"],
        ),
        A2ASkill(
            id="gap_detection", name="Gap Detection",
            description="Identify research gaps in existing literature",
            tags=["analysis", "gaps"],
        ),
        A2ASkill(
            id="synthesis", name="Literature Synthesis",
            description="Synthesize findings into a literature review",
            tags=["synthesis", "review"],
        ),
    ],
    "ModelTeam": [
        A2ASkill(
            id="theory", name="Theory Development",
            description="Develop theoretical frameworks for economic models",
            tags=["theory", "economics"],
        ),
        A2ASkill(
            id="design", name="Model Design",
            description="Mathematical formulation and specification",
            tags=["modeling", "mathematics"],
        ),
        A2ASkill(
            id="calibration", name="Model Calibration",
            description="Parameter estimation and sensitivity analysis",
            tags=["calibration", "estimation"],
        ),
    ],
    "DataTeam": [
        A2ASkill(
            id="sourcing", name="Data Sourcing",
            description="Connect to and fetch from data sources (FRED, World Bank)",
            tags=["data", "api", "economics"],
        ),
        A2ASkill(
            id="cleaning", name="Data Cleaning",
            description="Clean and transform economic datasets",
            tags=["data", "cleaning", "transformation"],
        ),
        A2ASkill(
            id="quality", name="Quality Assurance",
            description="Validate and ensure data quality",
            tags=["quality", "validation"],
        ),
    ],
}


# ---------------------------------------------------------------------------
# A2A Team Server
# ---------------------------------------------------------------------------

class A2ATeamServer:
    """
    A2A-compliant server for an AEL team.

    Handles agent card discovery and task request processing
    via JSON-RPC format.
    """

    def __init__(
        self,
        team_name: str,
        host: str = "localhost",
        port: int = 8000,
    ):
        self.team_name = team_name
        self.host = host
        self.port = port
        self._tasks: Dict[str, A2ATask] = {}

    def get_agent_card(self) -> A2AAgentCard:
        """Return an A2A-compliant agent card for this team."""
        from shared.protocols.agent_card import get_team_card

        team_card = get_team_card(self.team_name)
        description = team_card.description if team_card else f"AEL {self.team_name}"
        skills = TEAM_SKILLS.get(self.team_name, [])

        return A2AAgentCard(
            name=f"AEL-{self.team_name}",
            description=description,
            url=f"http://{self.host}:{self.port}/a2a/{self.team_name.lower()}",
            version="0.5.0",
            protocol_version="0.3",
            capabilities={
                "streaming": False,
                "pushNotifications": False,
                "stateTransitionHistory": True,
            },
            skills=skills,
        )

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a JSON-RPC request.

        Supported methods:
        - tasks/send: Submit a new task
        - tasks/get: Get task status
        - tasks/cancel: Cancel a task

        Returns:
            JSON-RPC response dict.
        """
        rpc = A2AJsonRpcRequest(**request)

        if rpc.method == "tasks/send":
            return self._handle_task_send(rpc)
        elif rpc.method == "tasks/get":
            return self._handle_task_get(rpc)
        elif rpc.method == "tasks/cancel":
            return self._handle_task_cancel(rpc)
        else:
            return A2AJsonRpcResponse(
                id=rpc.id,
                error={"code": -32601, "message": f"Method not found: {rpc.method}"},
            ).model_dump()

    def _handle_task_send(self, rpc: A2AJsonRpcRequest) -> Dict[str, Any]:
        """Handle tasks/send — create a new task."""
        task = A2ATask(
            message=rpc.params.get("message", {}),
            metadata={
                "team": self.team_name,
                "source": rpc.params.get("source", "external"),
            },
        )
        self._tasks[task.id] = task

        return A2AJsonRpcResponse(
            id=rpc.id,
            result=task.model_dump(),
        ).model_dump()

    def _handle_task_get(self, rpc: A2AJsonRpcRequest) -> Dict[str, Any]:
        """Handle tasks/get — retrieve task status."""
        task_id = rpc.params.get("id", "")
        task = self._tasks.get(task_id)
        if task is None:
            return A2AJsonRpcResponse(
                id=rpc.id,
                error={"code": -32602, "message": f"Task not found: {task_id}"},
            ).model_dump()
        return A2AJsonRpcResponse(
            id=rpc.id,
            result=task.model_dump(),
        ).model_dump()

    def _handle_task_cancel(self, rpc: A2AJsonRpcRequest) -> Dict[str, Any]:
        """Handle tasks/cancel — cancel a task."""
        task_id = rpc.params.get("id", "")
        task = self._tasks.get(task_id)
        if task is None:
            return A2AJsonRpcResponse(
                id=rpc.id,
                error={"code": -32602, "message": f"Task not found: {task_id}"},
            ).model_dump()
        task.state = TaskState.CANCELED
        task.updated_at = datetime.now(timezone.utc).isoformat()
        return A2AJsonRpcResponse(
            id=rpc.id,
            result=task.model_dump(),
        ).model_dump()

    def complete_task(self, task_id: str, artifacts: List[Dict[str, Any]]):
        """Mark a task as completed with artifacts (called by orchestrator)."""
        task = self._tasks.get(task_id)
        if task:
            task.state = TaskState.COMPLETED
            task.artifacts = artifacts
            task.updated_at = datetime.now(timezone.utc).isoformat()

    def fail_task(self, task_id: str, error_message: str):
        """Mark a task as failed (called by orchestrator)."""
        task = self._tasks.get(task_id)
        if task:
            task.state = TaskState.FAILED
            task.metadata["error"] = error_message
            task.updated_at = datetime.now(timezone.utc).isoformat()
