# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Hardening tests (semantic-cache thread-safety, attack-corpus realism)."""

from __future__ import annotations

import threading

import pytest


class TestSemanticCacheThreadSafety:
    def test_concurrent_puts_do_not_corrupt_stats(self):
        from shared.cache import SemanticCacheGateway

        gateway = SemanticCacheGateway(enabled=True, similarity_threshold=0.5)

        def worker(n: int):
            for i in range(n):
                prompt = f"concurrent prompt {i}"
                gateway.lookup_or_call(
                    prompt, model="m", seed=0, fetch=lambda p: f"ans-{p}",
                )

        threads = [threading.Thread(target=worker, args=(25,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = gateway.stats
        # Each prompt triggers exactly one miss; 25 unique prompts × 4 threads
        # produces up to 100 total lookups (some may hit other threads' puts).
        # Hits + misses must equal total recorded operations.
        assert stats["total"] == stats["hits"] + stats["misses"]
        assert stats["total"] >= 25  # at minimum, 25 unique misses
        # Entries must not exceed configured max (concurrent writes never
        # bypass eviction).
        assert stats["entries"] <= gateway.max_entries

    def test_concurrent_get_does_not_crash(self):
        from shared.cache import SemanticCacheGateway

        gateway = SemanticCacheGateway(enabled=True, similarity_threshold=0.7)
        gateway.put("seed prompt", "seed answer", model="m")

        def reader():
            for _ in range(50):
                gateway.get("seed prompt", model="m")

        threads = [threading.Thread(target=reader) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = gateway.stats
        # At least one thread should observe a hit
        assert stats["hits"] >= 1


class TestMINJAParaphraseCorpus:
    """Test that the detector fires on both attack classes."""

    def test_paraphrased_contradiction_is_detected(self):
        from shared.verification import BeliefDriftDetector, MemoryEntry
        import time as _time
        now = _time.time()
        entries = [
            MemoryEntry(content="Rate hikes increase inflation in the short run",
                        source="ecb", timestamp=now),
            MemoryEntry(content="Rate hikes decrease inflation in the short run",
                        source="attacker", timestamp=now),
        ]
        alerts = BeliefDriftDetector().scan(entries)
        reasons = {a.reason for a in alerts}
        assert "contradicts-prior-entry" in reasons

    def test_bit_identical_attack_still_detected(self):
        from shared.verification import BeliefDriftDetector, MemoryEntry
        import time as _time
        now = _time.time()
        entries = [
            MemoryEntry(content="claim", source="trusted", timestamp=now),
            MemoryEntry(content="claim", source="attacker", timestamp=now),
        ]
        alerts = BeliefDriftDetector().scan(entries)
        reasons = {a.reason for a in alerts}
        assert "fingerprint-collision-different-source" in reasons
