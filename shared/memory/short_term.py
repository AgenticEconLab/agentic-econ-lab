# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Short-Term Memory — Session-scoped in-memory context.

Tier 1 of the 3-tier memory system. Holds active context and state
within a single pipeline run. Discarded when the run completes.

Usage:
    from shared.memory.short_term import ShortTermMemory

    stm = ShortTermMemory()
    stm.set("research_topic", "AI in economics")
    topic = stm.get("research_topic")
"""

from typing import Any, Dict, List, Optional


class ShortTermMemory:
    """
    In-memory key-value store scoped to a single pipeline run.

    Thread-safe is not required since each pipeline run is sequential.
    """

    def __init__(self):
        self._store: Dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        """Store a value under a key."""
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value by key, returning default if not found."""
        return self._store.get(key, default)

    def has(self, key: str) -> bool:
        """Check if a key exists."""
        return key in self._store

    def delete(self, key: str) -> bool:
        """Remove a key. Returns True if key existed."""
        if key in self._store:
            del self._store[key]
            return True
        return False

    def keys(self) -> List[str]:
        """List all stored keys."""
        return list(self._store.keys())

    def all(self) -> Dict[str, Any]:
        """Return a copy of all stored data."""
        return dict(self._store)

    def clear(self) -> None:
        """Remove all data."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: str) -> bool:
        return key in self._store
