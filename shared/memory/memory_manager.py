# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Memory Manager — Unified facade for the 3-tier memory system.

Coordinates short-term (session), long-term (persistent), and episodic
(trajectory) memory tiers through a single interface.

Usage:
    from shared.memory.memory_manager import MemoryManager

    mm = MemoryManager(run_id="pipeline-abc123", memory_dir="./memory")
    mm.set_context("research_topic", "AI in economics")
    mm.remember_fact("FRED provides GDP data", category="data_source")
    facts = mm.recall_facts("GDP data sources")
"""

from typing import Any, Callable, Dict, List, Optional

from shared.memory.short_term import ShortTermMemory
from shared.memory.long_term import LongTermMemory, Fact
from shared.memory.episodic import EpisodicMemory, RunTrajectory


class MemoryManager:
    """
    Unified facade for the 3-tier memory system.

    Tier 1: Short-term (in-memory, session-scoped)
    Tier 2: Long-term (SQLite + embeddings, cross-run)
    Tier 3: Episodic (JSON trajectories, self-improvement)
    """

    def __init__(
        self,
        run_id: str = "",
        memory_dir: str = "./memory",
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    ):
        """
        Args:
            run_id: Current pipeline run ID for traceability.
            memory_dir: Base directory for persistent storage.
            embed_fn: Optional embedding function for vector similarity.
        """
        self.run_id = run_id
        self.memory_dir = memory_dir

        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory(memory_dir, embed_fn=embed_fn)
        self.episodic = EpisodicMemory(memory_dir)

    # ── Tier 1: Short-Term (session-scoped) ──

    def set_context(self, key: str, value: Any) -> None:
        """Store session-scoped context."""
        self.short_term.set(key, value)

    def get_context(self, key: str, default: Any = None) -> Any:
        """Retrieve session-scoped context."""
        return self.short_term.get(key, default)

    # ── Tier 2: Long-Term (cross-run) ──

    def remember_fact(
        self,
        fact: str,
        category: str = "general",
        source: str = "",
    ) -> Fact:
        """Remember a fact for future runs."""
        return self.long_term.remember_fact(
            fact=fact,
            category=category,
            source=source,
            run_id=self.run_id,
        )

    def recall_facts(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
    ) -> List[Fact]:
        """Recall relevant facts from long-term memory."""
        return self.long_term.recall_facts(query, top_k=top_k, category=category)

    # ── V0.7 MemoryGuard 2.0 — trust-aware retrieval (gated on feature flag) ──

    def recall_facts_trust_aware(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
        *,
        min_trust: float = 0.3,
        temporal_decay_days: float = 90.0,
    ) -> List[Fact]:
        """Trust-aware recall with provenance + decay (MemoryGuard 2.0, V0.7).

        Falls back to the V0.6 ``recall_facts`` when the ``memory_guard_v2``
        feature flag is off or the trust-aware retriever is unavailable.
        Also runs :class:`BeliefDriftDetector` and sets drift alerts on the
        short-term memory under the key ``"memory_drift_alerts"``.
        """
        # Always start from the base candidate set
        candidates = self.long_term.recall_facts(
            query, top_k=max(top_k * 4, 20), category=category,
        )
        try:
            from shared.feature_flags import is_enabled
            if not is_enabled("memory_guard_v2"):
                return candidates[:top_k]
            from shared.verification import (
                BeliefDriftDetector,
                MemoryEntry,
                TrustAwareRetriever,
            )
        except Exception:
            return candidates[:top_k]

        now = 0.0
        entries: List[MemoryEntry] = []
        index: Dict[str, Fact] = {}
        for fact in candidates:
            content = getattr(fact, "fact", None) or getattr(fact, "content", None) or str(fact)
            source = getattr(fact, "source", "") or "unknown"
            ts = getattr(fact, "timestamp", None) or getattr(fact, "created_at", None) or 0.0
            try:
                ts_float = float(ts) if ts is not None else 0.0
            except Exception:
                ts_float = 0.0
            entry = MemoryEntry(
                content=str(content),
                source=str(source),
                timestamp=ts_float,
                trust_score=float(getattr(fact, "trust_score", 1.0)),
            )
            entries.append(entry)
            index[entry.fingerprint()] = fact

        # Belief-drift detection (additive — writes alerts to short-term memory)
        try:
            alerts = BeliefDriftDetector().scan(entries)
            if alerts:
                self.short_term.set("memory_drift_alerts", [
                    {"fingerprint": a.entry_fingerprint, "reason": a.reason,
                     "severity": a.severity}
                    for a in alerts
                ])
        except Exception:
            pass

        retriever = TrustAwareRetriever(
            min_trust=min_trust,
            temporal_decay_days=temporal_decay_days,
        )
        import time as _time
        hits = retriever.retrieve(query, entries, k=top_k, now=_time.time())
        ranked: List[Fact] = []
        for e in hits:
            fp = e.fingerprint()
            if fp in index and index[fp] not in ranked:
                ranked.append(index[fp])
        # Fill in with untrusted-but-relevant if we came up short
        for fact in candidates:
            if fact not in ranked and len(ranked) < top_k:
                ranked.append(fact)
        return ranked[:top_k]

    def forget_stale(self, max_age_days: int = 90) -> int:
        """Purge facts older than max_age_days."""
        return self.long_term.forget_stale(max_age_days)

    # ── Tier 3: Episodic (trajectories) ──

    def record_trajectory(self, trajectory: RunTrajectory) -> str:
        """Save a run trajectory for future learning."""
        return self.episodic.record_trajectory(trajectory)

    def recall_similar_runs(
        self, research_topic: str, top_k: int = 3
    ) -> List[RunTrajectory]:
        """Find past runs with similar research topics."""
        return self.episodic.recall_similar_runs(research_topic, top_k=top_k)

    # ── Lifecycle ──

    def clear_session(self) -> None:
        """Clear short-term memory (typically at end of run)."""
        self.short_term.clear()

    def close(self) -> None:
        """Release resources (close DB connections)."""
        self.long_term.close()

    def summary(self) -> Dict[str, Any]:
        """Return a summary of all memory tiers."""
        return {
            "short_term": {
                "keys": self.short_term.keys(),
                "count": len(self.short_term),
            },
            "long_term": {
                "total_facts": self.long_term.count(),
            },
            "episodic": {
                "total_runs": self.episodic.count(),
            },
        }
