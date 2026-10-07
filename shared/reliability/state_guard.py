# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
State Guard — Hash-based state verification between pipeline stages.

Prevents state synchronization failures by signing stage outputs
with content hashes and verifying inputs match expected hashes.

Usage:
    from shared.reliability.state_guard import StateGuard

    guard = StateGuard()

    # After stage completes, sign its output
    output_hash = guard.sign_output(stage_output)

    # Before next stage, verify input matches
    if not guard.verify_input(stage_input, output_hash):
        raise StateGuard.StateMismatchError("Corrupted inter-stage data")
"""

import hashlib
import json
from typing import Any, Dict, Optional


class StateMismatchError(Exception):
    """Raised when state verification fails between stages."""

    def __init__(self, message: str, expected_hash: str = "", actual_hash: str = ""):
        super().__init__(message)
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash


class StateGuard:
    """
    Hash-based state verification between pipeline stages.

    Signs stage outputs with SHA-256 content hashes and verifies
    that downstream inputs match the expected hash from upstream.
    """

    StateMismatchError = StateMismatchError

    def __init__(self):
        self._hashes: Dict[str, str] = {}

    def sign_output(self, output: Any, stage_name: str = "") -> str:
        """
        Generate a content hash for a stage output.

        Args:
            output: Stage output (dict, list, or JSON-serializable).
            stage_name: Optional stage name for tracking.

        Returns:
            SHA-256 hex digest (first 16 chars).
        """
        raw = json.dumps(output, sort_keys=True, default=str, separators=(",", ":"))
        digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
        if stage_name:
            self._hashes[stage_name] = digest
        return digest

    def verify_input(self, input_data: Any, expected_hash: str) -> bool:
        """
        Verify that input data matches the expected hash.

        Args:
            input_data: Data to verify.
            expected_hash: Expected hash from upstream stage.

        Returns:
            True if hashes match, False otherwise.
        """
        actual = self.sign_output(input_data)
        return actual == expected_hash

    def verify_or_raise(self, input_data: Any, expected_hash: str, context: str = ""):
        """
        Verify input and raise StateMismatchError if mismatch.

        Args:
            input_data: Data to verify.
            expected_hash: Expected hash.
            context: Human-readable context for error message.
        """
        actual = self.sign_output(input_data)
        if actual != expected_hash:
            raise StateMismatchError(
                f"State mismatch{f' at {context}' if context else ''}: "
                f"expected {expected_hash}, got {actual}",
                expected_hash=expected_hash,
                actual_hash=actual,
            )

    def get_hash(self, stage_name: str) -> Optional[str]:
        """Get the stored hash for a stage."""
        return self._hashes.get(stage_name)

    def get_all_hashes(self) -> Dict[str, str]:
        """Get all stored stage hashes."""
        return dict(self._hashes)

    def clear(self):
        """Clear all stored hashes."""
        self._hashes.clear()
