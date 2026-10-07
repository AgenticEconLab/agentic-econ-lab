# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
A2A Protocol Client — Consume external A2A-compliant agents.

Enables AEL to discover and communicate with external agents that
implement the A2A protocol. Supports agent card discovery, task
submission, and result retrieval via JSON-RPC.

Usage:
    from shared.protocols.a2a_client import A2AClient

    client = A2AClient("http://external-agent.example.com/a2a")

    # Discover agent capabilities
    card = client.get_agent_card()

    # Submit a task
    task = client.send_task({"role": "user", "parts": [{"text": "Analyze GDP data"}]})

    # Check task status
    status = client.get_task(task["id"])
"""

import uuid
from typing import Any, Dict, Optional

import httpx


class A2AClientError(Exception):
    """Error communicating with an A2A agent."""

    def __init__(self, message: str, code: int = -1):
        super().__init__(message)
        self.code = code


class A2AClient:
    """
    Client for consuming A2A-compliant agents.

    Sends JSON-RPC requests to an A2A agent endpoint.
    """

    def __init__(
        self,
        agent_url: str,
        timeout: float = 60.0,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Args:
            agent_url: Base URL of the A2A agent.
            timeout: HTTP request timeout in seconds.
            headers: Optional extra headers (e.g., auth tokens).
        """
        self.agent_url = agent_url.rstrip("/")
        self.timeout = timeout
        self.headers = headers or {}

    def get_agent_card(self) -> Dict[str, Any]:
        """
        Discover the agent's capabilities via its agent card.

        Returns:
            Agent card as a dict (A2A AgentCard format).
        """
        url = f"{self.agent_url}/.well-known/agent.json"
        try:
            resp = httpx.get(
                url,
                headers=self.headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as e:
            raise A2AClientError(f"Failed to fetch agent card: {e}") from e

    def send_task(
        self,
        message: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Submit a task to the A2A agent.

        Args:
            message: Task message (A2A format: {role, parts}).
            metadata: Optional task metadata.

        Returns:
            Task object with id and state.
        """
        request = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {
                "message": message,
                **({"metadata": metadata} if metadata else {}),
            },
            "id": str(uuid.uuid4()),
        }
        return self._send_rpc(request)

    def get_task(self, task_id: str) -> Dict[str, Any]:
        """
        Get the current status of a task.

        Args:
            task_id: Task identifier.

        Returns:
            Task object with current state and artifacts.
        """
        request = {
            "jsonrpc": "2.0",
            "method": "tasks/get",
            "params": {"id": task_id},
            "id": str(uuid.uuid4()),
        }
        return self._send_rpc(request)

    def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """
        Cancel a running task.

        Args:
            task_id: Task identifier.

        Returns:
            Task object with updated state.
        """
        request = {
            "jsonrpc": "2.0",
            "method": "tasks/cancel",
            "params": {"id": task_id},
            "id": str(uuid.uuid4()),
        }
        return self._send_rpc(request)

    def _send_rpc(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send a JSON-RPC request and return the result."""
        try:
            resp = httpx.post(
                self.agent_url,
                json=request,
                headers={"Content-Type": "application/json", **self.headers},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            if "error" in data and data["error"] is not None:
                error = data["error"]
                raise A2AClientError(
                    error.get("message", "Unknown A2A error"),
                    code=error.get("code", -1),
                )

            return data.get("result", {})

        except httpx.HTTPError as e:
            raise A2AClientError(f"A2A request failed: {e}") from e
