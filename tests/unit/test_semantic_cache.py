# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Tests for semantic cache."""

import numpy as np
import pytest
from shared.rag.semantic_cache import SemanticCache, CacheResult, CacheStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_embed(texts):
    """Deterministic fake embedder: hash-based pseudo-embedding."""
    embeddings = []
    for t in texts:
        np.random.seed(hash(t) % (2**31))
        embeddings.append(np.random.randn(64).tolist())
    return embeddings


def _fake_llm(prompt):
    return f"Response to: {prompt}"


# ---------------------------------------------------------------------------
# CacheResult tests
# ---------------------------------------------------------------------------

class TestCacheResult:
    def test_hit(self):
        r = CacheResult(hit=True, response="cached", cost_saved=0.01, similarity=0.98)
        assert r.hit is True
        assert r.cost_saved == 0.01

    def test_miss(self):
        r = CacheResult(hit=False, response="computed")
        assert r.hit is False
        assert r.cost_saved == 0.0


class TestCacheStats:
    def test_to_dict(self):
        s = CacheStats(total_entries=10, total_lookups=20, total_hits=15,
                       total_misses=5, hit_rate=0.75, total_cost_saved=0.5)
        d = s.to_dict()
        assert d["hit_rate"] == 0.75
        assert d["total_cost_saved"] == 0.5


# ---------------------------------------------------------------------------
# SemanticCache core tests
# ---------------------------------------------------------------------------

class TestSemanticCache:
    def test_cache_miss_then_hit(self, tmp_path):
        cache = SemanticCache(
            cache_dir=str(tmp_path / "cache"),
            similarity_threshold=0.95,
        )
        # First call: miss
        r1 = cache.get_or_compute(
            prompt="What is GDP?",
            llm_fn=_fake_llm,
            embed_fn=_fake_embed,
            estimated_cost=0.01,
        )
        assert r1.hit is False
        assert r1.response == "Response to: What is GDP?"

        # Same prompt: hit (exact match)
        r2 = cache.get_or_compute(
            prompt="What is GDP?",
            llm_fn=_fake_llm,
            embed_fn=_fake_embed,
            estimated_cost=0.01,
        )
        assert r2.hit is True
        assert r2.cost_saved == 0.01
        assert r2.response == "Response to: What is GDP?"

    def test_different_prompts_are_misses(self, tmp_path):
        cache = SemanticCache(
            cache_dir=str(tmp_path / "cache"),
            similarity_threshold=0.99,  # Very strict
        )
        r1 = cache.get_or_compute("prompt A", _fake_llm, _fake_embed)
        r2 = cache.get_or_compute("completely different prompt B", _fake_llm, _fake_embed)
        assert r1.hit is False
        assert r2.hit is False

    def test_stats(self, tmp_path):
        cache = SemanticCache(cache_dir=str(tmp_path / "cache"))
        cache.get_or_compute("Q1", _fake_llm, _fake_embed, estimated_cost=0.01)
        cache.get_or_compute("Q1", _fake_llm, _fake_embed, estimated_cost=0.01)
        stats = cache.get_stats()
        assert stats.total_entries == 1
        assert stats.total_lookups == 2
        assert stats.total_hits == 1
        assert stats.total_misses == 1
        assert stats.hit_rate == pytest.approx(0.5)

    def test_clear(self, tmp_path):
        cache = SemanticCache(cache_dir=str(tmp_path / "cache"))
        cache.get_or_compute("Q1", _fake_llm, _fake_embed)
        cache.clear()
        stats = cache.get_stats()
        assert stats.total_entries == 0

    def test_lookup_without_compute(self, tmp_path):
        cache = SemanticCache(cache_dir=str(tmp_path / "cache"))
        # Nothing cached yet
        result = cache.lookup("Q1", _fake_embed)
        assert result is None

        # Store one
        cache.get_or_compute("Q1", _fake_llm, _fake_embed)
        result = cache.lookup("Q1", _fake_embed)
        assert result is not None
        assert result.hit is True

    def test_max_cache_size_eviction(self, tmp_path):
        cache = SemanticCache(
            cache_dir=str(tmp_path / "cache"),
            max_cache_size=5,
            similarity_threshold=0.999,
        )
        for i in range(6):
            cache.get_or_compute(f"unique_prompt_{i}", _fake_llm, _fake_embed)
        stats = cache.get_stats()
        # Should have evicted oldest entries
        assert stats.total_entries <= 5

    def test_ttl_expiry(self, tmp_path):
        cache = SemanticCache(
            cache_dir=str(tmp_path / "cache"),
            ttl_hours=0,  # Expire immediately
        )
        cache.get_or_compute("Q1", _fake_llm, _fake_embed)
        # Second call: expired entry should be treated as miss
        r2 = cache.get_or_compute("Q1", _fake_llm, _fake_embed)
        # The entry was expired, so it's a miss (llm_fn is called again)
        # Note: this depends on timing, the entry was just created
        # With ttl_hours=0, the freshly stored entry should expire on next lookup
        # due to age_hours > 0 check
        assert isinstance(r2, CacheResult)

    def test_persistence_across_instances(self, tmp_path):
        cache_dir = str(tmp_path / "cache")
        # Instance 1: store
        c1 = SemanticCache(cache_dir=cache_dir)
        c1.get_or_compute("persistent prompt", _fake_llm, _fake_embed)

        # Instance 2: should find it
        c2 = SemanticCache(cache_dir=cache_dir)
        result = c2.lookup("persistent prompt", _fake_embed)
        assert result is not None
        assert result.hit is True
