# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
MemoryGuard 2.0 — trust-aware retrieval + provenance + belief-drift detection.

Reference: "Memory Poisoning Attack and Defense on Memory-Based LLM-Agents"
(arXiv 2601.05504, Jan 2026) — MINJA achieves >95% injection success; standard
detectors miss 66% of poisoned entries. arXiv 2603.20357 adds layered defense
with belief-drift detection + context provenance tracking.

This module *extends* the V0.6 MemoryGuard (it does not replace it). Callers
can opt into trust-aware retrieval while keeping V0.6 validation behaviour.

Target: detect ≥ 90% of MINJA-style poisoning on the test corpus.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional


@dataclass
class MemoryEntry:
    content: str
    source: str                              # provenance (agent_id, tool, external_origin)
    timestamp: float                         # unix epoch seconds
    trust_score: float = 1.0                 # 0..1
    signature: Optional[str] = None
    belief_version: int = 1
    tags: Dict[str, str] = field(default_factory=dict)
    contradiction_count: int = 0

    def fingerprint(self) -> str:
        """Content-only hash. Distinct entries with the same content but
        different sources are the signature of MINJA-style poisoning
        (see BeliefDriftDetector collision check)."""
        h = hashlib.sha256()
        h.update(self.content.encode("utf-8", errors="replace"))
        return h.hexdigest()[:16]


@dataclass
class DriftAlert:
    entry_fingerprint: str
    reason: str
    severity: str                            # "low" | "medium" | "high"
    details: Dict[str, Any] = field(default_factory=dict)


class TrustAwareRetriever:
    """Combined relevance + trust + recency scoring for memory retrieval.

    score(entry, query) = relevance(entry, query)
                        * trust_score_with_decay(entry)
                        * freshness(entry)

    Callers supply a ``relevance_fn`` because AEL uses multiple retriever
    stacks (BM25, embeddings, hybrid). When absent, a simple token-overlap
    baseline is used.
    """

    def __init__(
        self,
        *,
        relevance_fn: Optional[Callable[[str, MemoryEntry], float]] = None,
        temporal_decay_days: float = 90.0,
        min_trust: float = 0.3,
    ) -> None:
        self._relevance_fn = relevance_fn or _token_overlap_relevance
        self.temporal_decay_days = temporal_decay_days
        self.min_trust = min_trust

    def retrieve(
        self,
        query: str,
        entries: Iterable[MemoryEntry],
        k: int,
        *,
        now: Optional[float] = None,
    ) -> List[MemoryEntry]:
        now = now if now is not None else time.time()
        scored: List[tuple[float, MemoryEntry]] = []
        for e in entries:
            trust = self._trust_with_decay(e, now=now)
            if trust < self.min_trust:
                continue
            rel = self._relevance_fn(query, e)
            fresh = self._freshness(e, now=now)
            score = rel * trust * fresh
            if score > 0:
                scored.append((score, e))
        scored.sort(key=lambda p: p[0], reverse=True)
        return [e for _, e in scored[:k]]

    # ------------------------------------------------------------------

    def _trust_with_decay(self, e: MemoryEntry, *, now: float) -> float:
        age_days = max(0.0, (now - e.timestamp) / 86_400.0)
        half_life = max(1.0, self.temporal_decay_days)
        decay = math.pow(0.5, age_days / half_life)
        penalty = math.pow(0.7, e.contradiction_count)
        return max(0.0, min(1.0, e.trust_score * decay * penalty))

    def _freshness(self, e: MemoryEntry, *, now: float) -> float:
        age_s = max(0.0, now - e.timestamp)
        # Half-day bonus, decays across the temporal horizon.
        return 1.0 / (1.0 + age_s / (self.temporal_decay_days * 86_400.0))


def _token_overlap_relevance(query: str, entry: MemoryEntry) -> float:
    q = set(query.lower().split())
    e = set(entry.content.lower().split())
    if not q or not e:
        return 0.0
    return len(q & e) / len(q)


# ---------------------------------------------------------------------------
# Belief-drift detector
# ---------------------------------------------------------------------------

class BeliefDriftDetector:
    """Flags entries whose content contradicts a prior entry in the same topic.

    Heuristic: if two entries share >= ``overlap_threshold`` tokens AND
    contain opposite negation markers ("is" vs "is not", "increases" vs
    "decreases"), flag as drift.

    Real deployments can plug in an embedding-based contradiction detector by
    subclassing and overriding ``_contradicts``.
    """

    _NEGATION_PAIRS = [
        ("increase", "decrease"),
        ("increases", "decreases"),
        ("increased", "decreased"),
        ("rise", "fall"),
        ("rises", "falls"),
        ("grow", "shrink"),
        ("grows", "shrinks"),
        ("positive", "negative"),
        ("support", "refute"),
        ("supports", "refutes"),
        ("confirm", "deny"),
        ("confirms", "denies"),
        ("accept", "reject"),
        ("accepts", "rejects"),
    ]

    def __init__(self, *, overlap_threshold: float = 0.4) -> None:
        self.overlap_threshold = overlap_threshold

    def scan(self, entries: List[MemoryEntry]) -> List[DriftAlert]:
        alerts: List[DriftAlert] = []
        for i, a in enumerate(entries):
            for b in entries[i + 1 :]:
                if self._same_topic(a, b) and self._contradicts(a.content, b.content):
                    alerts.append(DriftAlert(
                        entry_fingerprint=b.fingerprint(),
                        reason="contradicts-prior-entry",
                        severity="medium",
                        details={"conflicts_with": a.fingerprint()},
                    ))
        # Explicit MINJA-style high-severity: same fingerprint repeated
        # with diverging trust sources.
        seen: Dict[str, MemoryEntry] = {}
        for e in entries:
            fp = e.fingerprint()
            if fp in seen and seen[fp].source != e.source:
                alerts.append(DriftAlert(
                    entry_fingerprint=fp,
                    reason="fingerprint-collision-different-source",
                    severity="high",
                    details={"first_source": seen[fp].source, "colliding_source": e.source},
                ))
            else:
                seen[fp] = e
        return alerts

    # ------------------------------------------------------------------

    def _same_topic(self, a: MemoryEntry, b: MemoryEntry) -> bool:
        ta = set(a.content.lower().split())
        tb = set(b.content.lower().split())
        if not ta or not tb:
            return False
        overlap = len(ta & tb) / max(len(ta | tb), 1)
        return overlap >= self.overlap_threshold

    def _contradicts(self, text_a: str, text_b: str) -> bool:
        ta = text_a.lower()
        tb = text_b.lower()
        for pos, neg in self._NEGATION_PAIRS:
            if pos in ta and neg in tb:
                return True
            if neg in ta and pos in tb:
                return True
        # "is not" vs "is" for shared subject
        if " is not " in ta and " is " in tb and " not " not in tb:
            return True
        if " is not " in tb and " is " in ta and " not " not in ta:
            return True
        return False


def record_contradiction(entry: MemoryEntry) -> MemoryEntry:
    """Increment contradiction counter and decay trust."""
    entry.contradiction_count += 1
    entry.trust_score = max(0.0, entry.trust_score * 0.7)
    entry.belief_version += 1
    return entry
