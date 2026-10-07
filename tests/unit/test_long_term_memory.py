# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Tests for shared.memory.long_term.LongTermMemory.

Validates SQLite-backed persistent fact storage and recall.
"""

import os
from datetime import datetime, timezone, timedelta

import pytest

from shared.memory.long_term import LongTermMemory, Fact


@pytest.fixture
def ltm(tmp_path):
    """Create a LongTermMemory instance with temp directory."""
    return LongTermMemory(memory_dir=str(tmp_path))


def fake_embed(texts):
    """Deterministic fake embedding function for testing."""
    embeddings = []
    for text in texts:
        # Hash-based pseudo-embedding (deterministic, 4-dimensional)
        words = text.lower().split()
        vec = [0.0] * 4
        for i, w in enumerate(words):
            vec[i % 4] += hash(w) % 100 / 100.0
        # Normalize
        norm = sum(v**2 for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        embeddings.append(vec)
    return embeddings


@pytest.fixture
def ltm_with_embeddings(tmp_path):
    """LongTermMemory with a fake embedding function."""
    return LongTermMemory(memory_dir=str(tmp_path), embed_fn=fake_embed)


class TestRememberFact:
    """Test fact storage."""

    def test_remember_returns_fact(self, ltm):
        fact = ltm.remember_fact("FRED provides GDP data", category="data_source")
        assert isinstance(fact, Fact)
        assert fact.fact == "FRED provides GDP data"
        assert fact.category == "data_source"
        assert fact.fact_id > 0

    def test_remember_multiple(self, ltm):
        ltm.remember_fact("Fact 1")
        ltm.remember_fact("Fact 2")
        assert ltm.count() == 2

    def test_remember_with_source(self, ltm):
        fact = ltm.remember_fact("test", source="DataTeam", run_id="run-123")
        assert fact.source == "DataTeam"
        assert fact.run_id == "run-123"

    def test_remember_with_embeddings(self, ltm_with_embeddings):
        ltm_with_embeddings.remember_fact("GDP data from FRED")
        assert ltm_with_embeddings.count() == 1
        # Embeddings should be saved
        assert os.path.exists(
            os.path.join(ltm_with_embeddings._dir, "fact_embeddings.npy")
        )


class TestRecallFacts:
    """Test fact recall."""

    def test_recall_by_keyword(self, ltm):
        ltm.remember_fact("FRED provides GDP data", category="data_source")
        ltm.remember_fact("ArXiv has AI papers", category="literature")
        ltm.remember_fact("World Bank has development data", category="data_source")

        results = ltm.recall_facts("GDP data sources")
        assert len(results) >= 1
        assert any("GDP" in f.fact for f in results)

    def test_recall_with_category_filter(self, ltm):
        ltm.remember_fact("FRED provides GDP", category="data_source")
        ltm.remember_fact("ArXiv has AI papers", category="literature")

        results = ltm.recall_facts("data", category="data_source")
        assert all(f.category == "data_source" for f in results)

    def test_recall_empty_memory(self, ltm):
        results = ltm.recall_facts("anything")
        assert results == []

    def test_recall_respects_top_k(self, ltm):
        for i in range(10):
            ltm.remember_fact(f"fact about topic {i}")
        results = ltm.recall_facts("topic", top_k=3)
        assert len(results) <= 3

    def test_recall_by_vector(self, ltm_with_embeddings):
        ltm_with_embeddings.remember_fact("FRED provides GDP data")
        ltm_with_embeddings.remember_fact("ArXiv has AI papers")
        ltm_with_embeddings.remember_fact("World Bank development indicators")

        results = ltm_with_embeddings.recall_facts("GDP economic data")
        assert len(results) >= 1


class TestForgetStale:
    """Test stale fact purging."""

    def test_forget_removes_old_facts(self, ltm):
        # Insert a "stale" fact with old timestamp
        old_time = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        ltm._conn.execute(
            "INSERT INTO facts (fact, category, source, run_id, timestamp) VALUES (?, ?, ?, ?, ?)",
            ("old fact", "general", "", "", old_time),
        )
        ltm._conn.commit()

        ltm.remember_fact("recent fact")
        assert ltm.count() == 2

        removed = ltm.forget_stale(max_age_days=90)
        assert removed == 1
        assert ltm.count() == 1

    def test_forget_keeps_recent(self, ltm):
        ltm.remember_fact("recent fact")
        removed = ltm.forget_stale(max_age_days=90)
        assert removed == 0
        assert ltm.count() == 1


class TestGetAllAndCount:
    """Test listing and counting."""

    def test_get_all_facts(self, ltm):
        ltm.remember_fact("Fact A", category="cat1")
        ltm.remember_fact("Fact B", category="cat2")
        all_facts = ltm.get_all_facts()
        assert len(all_facts) == 2

    def test_get_all_filtered(self, ltm):
        ltm.remember_fact("Fact A", category="cat1")
        ltm.remember_fact("Fact B", category="cat2")
        results = ltm.get_all_facts(category="cat1")
        assert len(results) == 1
        assert results[0].category == "cat1"

    def test_count_by_category(self, ltm):
        ltm.remember_fact("A", category="data_source")
        ltm.remember_fact("B", category="data_source")
        ltm.remember_fact("C", category="literature")
        assert ltm.count(category="data_source") == 2
        assert ltm.count(category="literature") == 1
        assert ltm.count() == 3


class TestClearAndClose:
    """Test lifecycle operations."""

    def test_clear(self, ltm):
        ltm.remember_fact("fact")
        ltm.clear()
        assert ltm.count() == 0

    def test_close_and_reopen(self, tmp_path):
        ltm1 = LongTermMemory(memory_dir=str(tmp_path))
        ltm1.remember_fact("persistent fact")
        ltm1.close()

        ltm2 = LongTermMemory(memory_dir=str(tmp_path))
        assert ltm2.count() == 1
        facts = ltm2.get_all_facts()
        assert facts[0].fact == "persistent fact"
        ltm2.close()
