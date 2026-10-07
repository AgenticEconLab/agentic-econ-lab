# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Message Bus — In-process pub/sub for inter-team communication.

Provides structured messaging between teams during pipeline execution.
Teams publish ArtifactMessages when they produce outputs, and downstream
teams subscribe to receive them.

Usage:
    from shared.protocols.message_bus import MessageBus, ArtifactMessage

    bus = MessageBus(pipeline_run_id="run-001")

    # Downstream subscribes
    bus.subscribe("LiteratureTeam", "research_questions", callback)

    # Upstream publishes
    bus.publish(ArtifactMessage(
        source_team="IdeationTeam",
        target_team="LiteratureTeam",
        artifact_name="research_questions",
        artifact_data={...},
        schema_name="IntegrationStageOutput",
    ))
"""

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field


class ArtifactMessage(BaseModel):
    """Message carrying an artifact between teams."""

    source_team: str = Field(description="Team that produced the artifact")
    target_team: str = Field(
        default="", description="Intended recipient team (empty = broadcast)"
    )
    artifact_name: str = Field(description="Artifact identifier")
    artifact_data: Dict[str, Any] = Field(description="Artifact payload")
    schema_name: str = Field(
        default="", description="Pydantic schema name for the data"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp",
    )
    pipeline_run_id: str = Field(default="", description="Pipeline run ID")


class MessageBus:
    """
    Simple in-process message bus for cross-team communication.

    Supports publish/subscribe pattern with optional team/artifact filtering.
    Messages are stored in a log for post-hoc analysis.
    """

    def __init__(self, pipeline_run_id: str = ""):
        self._pipeline_run_id = pipeline_run_id
        # Subscriptions: (team_name, artifact_name) -> list of callbacks
        self._subscriptions: Dict[str, List[Callable[[ArtifactMessage], None]]] = {}
        # Full message log
        self._message_log: List[ArtifactMessage] = []

    @property
    def pipeline_run_id(self) -> str:
        return self._pipeline_run_id

    def subscribe(
        self,
        team_name: str,
        artifact_name: str,
        callback: Callable[[ArtifactMessage], None],
    ):
        """
        Subscribe a team to receive messages for a specific artifact.

        Args:
            team_name: Subscribing team name.
            artifact_name: Artifact to subscribe to.
            callback: Function called when a matching message is published.
        """
        key = self._make_key(team_name, artifact_name)
        if key not in self._subscriptions:
            self._subscriptions[key] = []
        self._subscriptions[key].append(callback)

    def publish(self, message: ArtifactMessage):
        """
        Publish an artifact message to all matching subscribers.

        Args:
            message: The artifact message to publish.
        """
        # Stamp with pipeline run ID if not set (copy to avoid mutating caller's object)
        if not message.pipeline_run_id:
            message = message.model_copy(update={"pipeline_run_id": self._pipeline_run_id})

        self._message_log.append(message)

        if message.target_team:
            # Deliver to specific target team subscribers
            key = self._make_key(message.target_team, message.artifact_name)
            for callback in self._subscriptions.get(key, []):
                callback(message)
        else:
            # True broadcast: deliver to ALL subscribers for this artifact
            suffix = f"::{message.artifact_name}"
            for sub_key, callbacks in self._subscriptions.items():
                if sub_key.endswith(suffix):
                    for callback in callbacks:
                        callback(message)
            return  # Wildcard already covered by broadcast

        # Also deliver to wildcard subscribers (team="*")
        wildcard_key = self._make_key("*", message.artifact_name)
        for callback in self._subscriptions.get(wildcard_key, []):
            callback(message)

    def get_messages(
        self,
        source_team: Optional[str] = None,
        artifact_name: Optional[str] = None,
    ) -> List[ArtifactMessage]:
        """
        Query the message log with optional filters.

        Args:
            source_team: Filter by source team.
            artifact_name: Filter by artifact name.

        Returns:
            List of matching messages.
        """
        results = list(self._message_log)
        if source_team:
            results = [m for m in results if m.source_team == source_team]
        if artifact_name:
            results = [m for m in results if m.artifact_name == artifact_name]
        return results

    def message_count(self) -> int:
        """Total number of messages published."""
        return len(self._message_log)

    def clear(self):
        """Clear all subscriptions and message history."""
        self._subscriptions.clear()
        self._message_log.clear()

    def to_a2a_messages(self) -> list:
        """
        Serialize the message log to A2A-compatible JSON-RPC format.

        Returns:
            List of A2A JSON-RPC message dicts.
        """
        a2a_messages = []
        for msg in self._message_log:
            a2a_messages.append({
                "jsonrpc": "2.0",
                "method": "tasks/send",
                "params": {
                    "message": {
                        "role": "agent",
                        "parts": [{"type": "data", "data": msg.artifact_data}],
                    },
                    "metadata": {
                        "source_team": msg.source_team,
                        "target_team": msg.target_team,
                        "artifact_name": msg.artifact_name,
                        "schema_name": msg.schema_name,
                        "timestamp": msg.timestamp,
                        "pipeline_run_id": msg.pipeline_run_id,
                    },
                },
                "id": f"{msg.source_team}-{msg.artifact_name}-{msg.timestamp}",
            })
        return a2a_messages

    @classmethod
    def from_a2a_message(cls, a2a_msg: dict) -> "ArtifactMessage":
        """
        Deserialize an A2A JSON-RPC message to an ArtifactMessage.

        Args:
            a2a_msg: A2A JSON-RPC dict with params.message and params.metadata.

        Returns:
            ArtifactMessage instance.
        """
        params = a2a_msg.get("params", {})
        metadata = params.get("metadata", {})
        message_data = params.get("message", {})

        # Extract data from A2A message parts
        artifact_data = {}
        parts = message_data.get("parts", [])
        for part in parts:
            if part.get("type") == "data":
                artifact_data = part.get("data", {})
                break

        return ArtifactMessage(
            source_team=metadata.get("source_team", ""),
            target_team=metadata.get("target_team", ""),
            artifact_name=metadata.get("artifact_name", ""),
            artifact_data=artifact_data,
            schema_name=metadata.get("schema_name", ""),
            timestamp=metadata.get("timestamp", ""),
            pipeline_run_id=metadata.get("pipeline_run_id", ""),
        )

    @staticmethod
    def _make_key(team_name: str, artifact_name: str) -> str:
        return f"{team_name}::{artifact_name}"
