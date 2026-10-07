# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
MCP V0.7 extensions — OAuth 2.1, Tasks (SEP-1686), and .well-known discovery.

Reference: MCP 2026 roadmap
  * OAuth 2.1 resource-server model (SEP-1865)
  * Tasks primitive with retry/expiry (SEP-1686)
  * ``/.well-known/oauth-protected-resource`` discovery
  * ``MCP-Protocol-Version`` header negotiation

These helpers attach to any existing :class:`AELMCPServer` without changing
its public surface. Legacy clients continue to work unchanged.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional


# Current MCP protocol version the AEL server speaks
MCP_PROTOCOL_VERSION = "2026-03-26"


# ---------------------------------------------------------------------------
# OAuth 2.1 resource-server model
# ---------------------------------------------------------------------------

@dataclass
class OAuth21ResourceMetadata:
    """Contents of /.well-known/oauth-protected-resource (SEP-1865)."""
    resource: str                              # canonical resource URI
    authorization_servers: List[str]
    scopes_supported: List[str] = field(default_factory=list)
    bearer_methods_supported: List[str] = field(default_factory=lambda: ["header"])
    resource_documentation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resource": self.resource,
            "authorization_servers": list(self.authorization_servers),
            "scopes_supported": list(self.scopes_supported),
            "bearer_methods_supported": list(self.bearer_methods_supported),
            "resource_documentation": self.resource_documentation,
        }


class OAuth21Authorizer:
    """Minimal OAuth 2.1 bearer-token validator.

    Uses a pluggable ``token_validator`` callable so that production
    deployments can swap in JWT/opaque-token stacks without changing the
    surface. The default validator compares against a static allow-list.
    """

    def __init__(
        self,
        *,
        metadata: OAuth21ResourceMetadata,
        token_validator: Optional[Callable[[str], bool]] = None,
        allowed_tokens: Optional[List[str]] = None,
    ) -> None:
        self.metadata = metadata
        if token_validator is None:
            allowed = set(allowed_tokens or [])
            token_validator = lambda tok: tok in allowed  # noqa: E731
        self._validate = token_validator

    def authorize(self, headers: Dict[str, str]) -> None:
        """Raise PermissionError unless a valid Bearer token is present."""
        auth = headers.get("Authorization") or headers.get("authorization")
        if not auth or not auth.lower().startswith("bearer "):
            raise PermissionError("missing or malformed Bearer token")
        token = auth.split(" ", 1)[1].strip()
        if not self._validate(token):
            raise PermissionError("token rejected")

    def well_known_payload(self) -> Dict[str, Any]:
        return self.metadata.to_dict()


def make_well_known_handler(auth: OAuth21Authorizer) -> Callable[[], Dict[str, Any]]:
    """Return a 0-arg handler for /.well-known/oauth-protected-resource."""
    def handler() -> Dict[str, Any]:
        return auth.well_known_payload()
    return handler


# ---------------------------------------------------------------------------
# Tasks primitive (SEP-1686)
# ---------------------------------------------------------------------------

TaskStatus = Literal["pending", "running", "completed", "failed", "expired"]


@dataclass
class MCPTask:
    id: str
    tool_name: str
    arguments: Dict[str, Any]
    status: TaskStatus = "pending"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    expires_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "tool": self.tool_name,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "expires_at": self.expires_at,
            "retries": self.retries,
            "error": self.error,
        }


class MCPTaskRegistry:
    """In-memory task store for long-running MCP operations.

    Production deployments persist this to disk/SQL; the in-memory impl is
    sufficient for AEL's current use cases (DataTeam long-running fetches
    parked here while the caller polls).
    """

    DEFAULT_TTL_SECONDS = 3_600

    def __init__(self, *, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        self._tasks: Dict[str, MCPTask] = {}
        self._ttl = ttl_seconds

    def create(self, tool_name: str, arguments: Dict[str, Any],
                *, max_retries: int = 2) -> MCPTask:
        task = MCPTask(
            id=str(uuid.uuid4()),
            tool_name=tool_name,
            arguments=arguments,
            max_retries=max_retries,
            expires_at=time.time() + self._ttl,
        )
        self._tasks[task.id] = task
        return task

    def get(self, task_id: str) -> Optional[MCPTask]:
        task = self._tasks.get(task_id)
        if task is None:
            return None
        if task.expires_at and time.time() > task.expires_at and task.status not in {"completed", "failed"}:
            task.status = "expired"
        return task

    def start(self, task_id: str) -> MCPTask:
        task = self._require(task_id)
        task.status = "running"
        task.started_at = time.time()
        return task

    def complete(self, task_id: str, result: Any) -> MCPTask:
        task = self._require(task_id)
        task.status = "completed"
        task.completed_at = time.time()
        task.result = result
        return task

    def fail(self, task_id: str, error: str, *, retryable: bool = True) -> MCPTask:
        task = self._require(task_id)
        if retryable and task.retries < task.max_retries:
            task.retries += 1
            task.status = "pending"
            task.error = error
        else:
            task.status = "failed"
            task.completed_at = time.time()
            task.error = error
        return task

    def list(self, *, status: Optional[TaskStatus] = None) -> List[MCPTask]:
        if status is None:
            return list(self._tasks.values())
        return [t for t in self._tasks.values() if t.status == status]

    def prune_expired(self) -> int:
        """Drop expired tasks; returns the number removed."""
        now = time.time()
        expired = [
            tid for tid, t in self._tasks.items()
            if t.expires_at and now > t.expires_at and t.status in {"pending", "expired"}
        ]
        for tid in expired:
            del self._tasks[tid]
        return len(expired)

    # ------------------------------------------------------------------

    def _require(self, task_id: str) -> MCPTask:
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"unknown task: {task_id}")
        return task


# ---------------------------------------------------------------------------
# Protocol-version negotiation
# ---------------------------------------------------------------------------

def negotiate_protocol_version(
    client_version: Optional[str],
    *,
    supported: List[str] = (MCP_PROTOCOL_VERSION,),
) -> str:
    """Return the highest mutually-supported version.

    ``client_version=None`` falls back to the legacy default (first supported).
    """
    if client_version is None:
        return supported[0]
    if client_version in supported:
        return client_version
    return supported[0]
