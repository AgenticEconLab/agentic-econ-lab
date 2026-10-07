# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Tests for shared.memory.short_term.ShortTermMemory.

Validates session-scoped in-memory key-value store operations.
"""

import pytest

from shared.memory.short_term import ShortTermMemory


class TestSetGet:
    """Test basic set/get operations."""

    def test_set_and_get(self):
        stm = ShortTermMemory()
        stm.set("topic", "AI in economics")
        assert stm.get("topic") == "AI in economics"

    def test_get_missing_returns_default(self):
        stm = ShortTermMemory()
        assert stm.get("missing") is None
        assert stm.get("missing", "fallback") == "fallback"

    def test_overwrite(self):
        stm = ShortTermMemory()
        stm.set("key", "v1")
        stm.set("key", "v2")
        assert stm.get("key") == "v2"

    def test_store_complex_value(self):
        stm = ShortTermMemory()
        data = {"questions": [{"q": "How does AI affect GDP?", "rank": 1}]}
        stm.set("research_questions", data)
        assert stm.get("research_questions")["questions"][0]["rank"] == 1


class TestHasDelete:
    """Test has/delete operations."""

    def test_has(self):
        stm = ShortTermMemory()
        stm.set("key", "value")
        assert stm.has("key")
        assert not stm.has("other")

    def test_contains(self):
        stm = ShortTermMemory()
        stm.set("key", "value")
        assert "key" in stm
        assert "other" not in stm

    def test_delete_existing(self):
        stm = ShortTermMemory()
        stm.set("key", "value")
        assert stm.delete("key") is True
        assert not stm.has("key")

    def test_delete_missing(self):
        stm = ShortTermMemory()
        assert stm.delete("missing") is False


class TestListingClear:
    """Test listing and clear operations."""

    def test_keys(self):
        stm = ShortTermMemory()
        stm.set("a", 1)
        stm.set("b", 2)
        assert set(stm.keys()) == {"a", "b"}

    def test_all_returns_copy(self):
        stm = ShortTermMemory()
        stm.set("key", "value")
        data = stm.all()
        data["key"] = "modified"
        assert stm.get("key") == "value"  # Original unchanged

    def test_clear(self):
        stm = ShortTermMemory()
        stm.set("a", 1)
        stm.set("b", 2)
        stm.clear()
        assert len(stm) == 0
        assert stm.keys() == []

    def test_len(self):
        stm = ShortTermMemory()
        assert len(stm) == 0
        stm.set("a", 1)
        assert len(stm) == 1
        stm.set("b", 2)
        assert len(stm) == 2
