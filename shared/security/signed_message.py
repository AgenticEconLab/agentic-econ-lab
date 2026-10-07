# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Signed Message Bus — HMAC-signed inter-agent communication (V0.6 Phase 4, OWASP ASI07).

Provides message integrity verification for inter-agent communication
via HMAC-SHA256 signatures.

Usage:
    from shared.security.signed_message import SignedMessage, SecureMessageBus

    bus = SecureMessageBus(secret_key="shared-secret")
    bus.send({"action": "analyze"}, sender_id="IdeationTeam")
    msg = bus.receive()
    assert msg.verify("shared-secret")
"""

import hashlib
import hmac
import json
import time
from collections import deque
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SignedMessage(BaseModel):
    """A2A message with integrity signature."""

    payload: Dict[str, Any] = Field(default_factory=dict)
    agent_id: str = ""
    timestamp: str = ""
    signature: str = ""

    def verify(self, secret_key: str) -> bool:
        """Verify message integrity using HMAC-SHA256.

        Args:
            secret_key: Shared secret key for verification.

        Returns:
            True if signature matches, False otherwise.
        """
        expected = self._compute_signature(secret_key)
        return hmac.compare_digest(self.signature, expected)

    @classmethod
    def create(
        cls, payload: Dict[str, Any], agent_id: str, secret_key: str
    ) -> "SignedMessage":
        """Create a signed message.

        Args:
            payload: Message content.
            agent_id: Sender agent ID.
            secret_key: Shared secret for signing.

        Returns:
            Signed message with computed signature.
        """
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        msg = cls(payload=payload, agent_id=agent_id, timestamp=ts)
        msg.signature = msg._compute_signature(secret_key)
        return msg

    def _compute_signature(self, secret_key: str) -> str:
        """Compute HMAC-SHA256 of payload + agent_id + timestamp."""
        data = json.dumps(self.payload, sort_keys=True, default=str)
        message = f"{data}|{self.agent_id}|{self.timestamp}"
        return hmac.new(
            secret_key.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()


class SecurityError(Exception):
    """Raised when message integrity check fails."""

    pass


class SecureMessageBus:
    """MessageBus with signed payloads and validation.

    Args:
        secret_key: Shared secret for HMAC signing.
        max_queue_size: Maximum number of messages to buffer.
    """

    def __init__(self, secret_key: str = "", max_queue_size: int = 1000):
        self.secret_key = secret_key
        self.max_queue_size = max_queue_size
        self._queue: deque = deque(maxlen=max_queue_size)
        self._sent_count: int = 0
        self._received_count: int = 0
        self._verification_failures: int = 0

    def send(self, message: Dict[str, Any], sender_id: str) -> SignedMessage:
        """Sign and enqueue a message.

        Args:
            message: Message payload.
            sender_id: Sender agent/team ID.

        Returns:
            The signed message that was enqueued.
        """
        signed = SignedMessage.create(message, sender_id, self.secret_key)
        self._queue.append(signed)
        self._sent_count += 1
        return signed

    def receive(self, verify: bool = True) -> Optional[SignedMessage]:
        """Dequeue and optionally verify a message.

        Args:
            verify: If True, verify signature before returning.

        Returns:
            Verified signed message, or None if queue is empty.

        Raises:
            SecurityError: If verification fails.
        """
        if not self._queue:
            return None

        signed = self._queue.popleft()
        self._received_count += 1

        if verify and self.secret_key:
            if not signed.verify(self.secret_key):
                self._verification_failures += 1
                raise SecurityError(
                    f"Message integrity check failed: agent={signed.agent_id}"
                )

        return signed

    def peek(self) -> Optional[SignedMessage]:
        """Peek at the next message without removing it."""
        return self._queue[0] if self._queue else None

    def pending_count(self) -> int:
        """Number of messages waiting in queue."""
        return len(self._queue)

    def get_stats(self) -> Dict[str, int]:
        """Return message bus statistics."""
        return {
            "sent": self._sent_count,
            "received": self._received_count,
            "pending": len(self._queue),
            "verification_failures": self._verification_failures,
        }
