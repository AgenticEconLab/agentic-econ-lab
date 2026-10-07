# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Artifact Store — Typed artifact persistence for cross-team data flow.

Each artifact has a name, data payload, optional Pydantic schema for validation,
producer team, and timestamp. Artifacts are stored in memory and optionally
persisted to disk as JSON.

Usage:
    from pipeline.artifact_store import ArtifactStore

    store = ArtifactStore(base_dir="./pipeline_output")
    store.register("research_questions", data, producer="IdeationTeam",
                   schema=IntegrationStageOutput)
    questions = store.get("research_questions")
"""

import copy
import json
import os
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field, ValidationError


class ArtifactIntegrityError(Exception):
    """An artifact's content does not match its HMAC signature."""


class Artifact(BaseModel):
    """A typed artifact produced by a team."""

    name: str = Field(description="Artifact identifier (e.g., 'research_questions')")
    data: Dict[str, Any] = Field(description="Artifact payload")
    schema_name: Optional[str] = Field(
        default=None, description="Name of the Pydantic schema used for validation"
    )
    producer: str = Field(default="", description="Team that produced this artifact")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp of artifact creation",
    )
    pipeline_run_id: str = Field(
        default="", description="Pipeline run ID for traceability"
    )
    # HMAC-SHA256 over the payload, producer
    # and name (shared.security.signed_message), verified on every read.
    signature: str = Field(default="", description="HMAC-SHA256 signature of the artifact")
    signed_at: str = Field(default="", description="Timestamp included in the signature")


class ArtifactStore:
    """
    Persistent store for cross-team artifacts.

    Artifacts are registered with optional Pydantic schema validation,
    stored in memory, and optionally persisted to disk.
    """

    def __init__(self, base_dir: Optional[str] = None, pipeline_run_id: str = "",
                 signing_key: Optional[str] = None):
        self._artifacts: Dict[str, Artifact] = {}
        self._base_dir = base_dir
        self._pipeline_run_id = pipeline_run_id
        # Artifacts handed between teams are HMAC-signed at registration and verified
        # on read. Key: explicit > AEL_PIPELINE_SIGNING_KEY > a fresh per-run random key.
        self._signing_key = (signing_key or os.environ.get("AEL_PIPELINE_SIGNING_KEY")
                             or secrets.token_hex(32))
        self.verified_reads = 0
        self.integrity_failures: List[str] = []

        if base_dir:
            os.makedirs(base_dir, exist_ok=True)

    @property
    def pipeline_run_id(self) -> str:
        return self._pipeline_run_id

    def register(
        self,
        name: str,
        data: Any,
        producer: str = "",
        schema: Optional[Type[BaseModel]] = None,
        persist: bool = True,
    ) -> Artifact:
        """
        Register a typed artifact.

        Args:
            name: Artifact identifier (e.g., 'research_questions').
            data: Raw data dict or Pydantic model instance.
            producer: Team name that produced this artifact.
            schema: Optional Pydantic model class for validation.
            persist: If True and base_dir is set, write to disk.

        Returns:
            The registered Artifact.

        Raises:
            ValidationError: If schema is provided and data fails validation.
        """
        # Convert Pydantic models to dicts
        if isinstance(data, BaseModel):
            payload = data.model_dump()
            schema_name = type(data).__name__
        elif schema is not None:
            # Validate against schema
            validated = schema.model_validate(data)
            payload = validated.model_dump()
            schema_name = schema.__name__
        else:
            payload = data if isinstance(data, dict) else {"value": data}
            schema_name = None

        from shared.security.signed_message import SignedMessage
        signed = SignedMessage.create(payload, f"{producer}:{name}", self._signing_key)
        artifact = Artifact(
            name=name,
            data=copy.deepcopy(payload),
            schema_name=schema_name,
            producer=producer,
            pipeline_run_id=self._pipeline_run_id,
            signature=signed.signature,
            signed_at=signed.timestamp,
        )

        self._artifacts[name] = artifact

        if persist and self._base_dir:
            self._persist(artifact)

        return artifact

    def get(
        self,
        name: str,
        expected_schema: Optional[Type[BaseModel]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve an artifact's data, optionally validating against a schema.

        Args:
            name: Artifact identifier.
            expected_schema: If provided, validate data against this schema.

        Returns:
            Artifact data dict, or None if not found.

        Raises:
            ValidationError: If expected_schema is provided and data fails.
        """
        artifact = self._artifacts.get(name)
        if artifact is None:
            return None
        self._verify(artifact)

        if expected_schema is not None:
            validated = expected_schema.model_validate(copy.deepcopy(artifact.data))
            return validated.model_dump()

        return copy.deepcopy(artifact.data)

    def _verify(self, artifact: "Artifact") -> None:
        """Reject an artifact whose content no longer matches its signature."""
        if not artifact.signature:
            return                      # legacy artifact without a signature
        from shared.security.signed_message import SignedMessage
        msg = SignedMessage(payload=artifact.data, agent_id=f"{artifact.producer}:{artifact.name}",
                            timestamp=artifact.signed_at, signature=artifact.signature)
        if not msg.verify(self._signing_key):
            self.integrity_failures.append(artifact.name)
            raise ArtifactIntegrityError(
                f"artifact '{artifact.name}' (producer {artifact.producer}) failed signature "
                "verification; refusing to hand it to a downstream team")
        self.verified_reads += 1

    def get_artifact(self, name: str) -> Optional[Artifact]:
        """Retrieve the full Artifact object (including metadata)."""
        return self._artifacts.get(name)

    def has(self, name: str) -> bool:
        """Check if an artifact exists."""
        return name in self._artifacts

    def list_artifacts(self) -> List[str]:
        """List all registered artifact names."""
        return list(self._artifacts.keys())

    def all(self) -> Dict[str, Artifact]:
        """Return all artifacts."""
        return dict(self._artifacts)

    def summary(self) -> List[Dict[str, Any]]:
        """Return a summary of all artifacts (name, producer, schema, timestamp)."""
        return [
            {
                "name": a.name,
                "producer": a.producer,
                "schema": a.schema_name,
                "timestamp": a.timestamp,
                "keys": list(a.data.keys()) if isinstance(a.data, dict) else [],
            }
            for a in self._artifacts.values()
        ]

    def clear(self):
        """Remove all artifacts from memory (disk files are not deleted)."""
        self._artifacts.clear()

    def _persist(self, artifact: Artifact):
        """Write artifact to disk as JSON. Logs a warning on I/O failure."""
        filepath = os.path.join(self._base_dir, f"{artifact.name}.json")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(artifact.model_dump(), f, indent=2, default=str)
        except OSError as e:
            import warnings
            warnings.warn(
                f"Failed to persist artifact '{artifact.name}' to disk: {e}"
            )

    def load_from_disk(self, name: str) -> Optional[Artifact]:
        """
        Load a previously persisted artifact from disk.

        Its signature is verified on the next get(): an artifact written by another run
        verifies only when both runs share AEL_PIPELINE_SIGNING_KEY.

        Returns:
            Artifact if file exists, None otherwise.
        """
        if not self._base_dir:
            return None

        filepath = os.path.join(self._base_dir, f"{name}.json")
        if not os.path.exists(filepath):
            return None

        with open(filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)

        artifact = Artifact.model_validate(raw)
        self._artifacts[name] = artifact
        return artifact
