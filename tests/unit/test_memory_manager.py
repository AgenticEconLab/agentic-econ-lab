# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Tests for shared.memory.memory_manager.MemoryManager.

Validates the unified 3-tier memory facade.
"""

import pytest

from shared.memory.memory_manager import MemoryManager
from shared.memory.episodic import RunTrajectory


@pytest.fixture
def mm(tmp_path):
    """Create a MemoryManager with temp directory."""
    return MemoryManager(
        run_id="test-run-001",
        memory_dir=str(tmp_path),
    )


class TestShortTermOperations:
    """Test Tier 1: session-scoped context."""

    def test_set_and_get_context(self, mm):
        mm.set_context("research_topic", "AI in economics")
        assert mm.get_context("research_topic") == "AI in economics"

    def test_get_missing_context(self, mm):
        assert mm.get_context("missing") is None
        assert mm.get_context("missing", "default") == "default"

    def test_clear_session(self, mm):
        mm.set_context("key", "value")
        mm.clear_session()
        assert mm.get_context("key") is None


class TestLongTermOperations:
    """Test Tier 2: cross-run persistent facts."""

    def test_remember_and_recall(self, mm):
        mm.remember_fact("FRED provides GDP data", category="data_source")
        facts = mm.recall_facts("GDP data")
        assert len(facts) >= 1
        assert any("GDP" in f.fact for f in facts)

    def test_remember_stamps_run_id(self, mm):
        fact = mm.remember_fact("test fact")
        assert fact.run_id == "test-run-001"

    def test_forget_stale(self, mm):
        mm.remember_fact("recent fact")
        removed = mm.forget_stale(max_age_days=90)
        assert removed == 0  # Just created, not stale


class TestEpisodicOperations:
    """Test Tier 3: trajectory storage."""

    def test_record_and_recall_trajectory(self, mm):
        trajectory = RunTrajectory(
            run_id="run-001",
            research_topic="AI impact on labor markets",
            teams_completed=["IdeationTeam"],
            success=True,
        )
        mm.record_trajectory(trajectory)
        similar = mm.recall_similar_runs("AI labor economics")
        assert len(similar) >= 1

    def test_recall_no_similar(self, mm):
        results = mm.recall_similar_runs("quantum physics")
        assert results == []


class TestSummary:
    """Test memory summary."""

    def test_summary_structure(self, mm):
        mm.set_context("key", "value")
        mm.remember_fact("test fact")
        summary = mm.summary()
        assert "short_term" in summary
        assert "long_term" in summary
        assert "episodic" in summary
        assert summary["short_term"]["count"] == 1
        assert summary["long_term"]["total_facts"] == 1
        assert summary["episodic"]["total_runs"] == 0


class TestLifecycle:
    """Test close and resource management."""

    def test_close(self, mm):
        mm.remember_fact("test")
        mm.close()
        # After close, DB operations should not crash
        # (we just verify close doesn't raise)

    def test_persistence_across_managers(self, tmp_path):
        mm1 = MemoryManager(run_id="run1", memory_dir=str(tmp_path))
        mm1.remember_fact("persistent fact", category="test")
        mm1.record_trajectory(RunTrajectory(
            run_id="run1", research_topic="AI economics"
        ))
        mm1.close()

        mm2 = MemoryManager(run_id="run2", memory_dir=str(tmp_path))
        facts = mm2.recall_facts("persistent")
        assert len(facts) >= 1
        similar = mm2.recall_similar_runs("AI economics")
        assert len(similar) >= 1
        mm2.close()
