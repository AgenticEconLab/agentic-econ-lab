# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Semantic caching gateway (V0.7, Trial).

Lightweight in-memory implementation — a GPTCache / LiteLLM gateway can be
dropped in later by subclassing :class:`SemanticCacheGateway` and overriding
``_embed`` + ``_lookup``. The default deterministic implementation uses
token-overlap similarity so unit tests don't need an embedding model.

Explicit design choices:
  * Disabled by default (``enabled=False``) so K=5 multi-run evaluation keeps
    variance measurements clean.
  * Cache keys include ``(model, temperature, seed)`` so different runs never
    conflate.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CacheEntry:
    prompt: str
    response: str
    model: str
    temperature: float
    seed: Optional[int]
    timestamp: float
    hits: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticCacheGateway:
    """Token-overlap semantic cache for LLM responses.

    The gateway is *additive*: callers supply a ``fetch`` callable (the real
    LLM call) and the gateway either returns a cached response or calls
    through, caching the result.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        similarity_threshold: float = 0.85,
        max_entries: int = 1000,
    ) -> None:
        self.enabled = enabled
        self.threshold = similarity_threshold
        self.max_entries = max_entries
        self._store: Dict[str, List[CacheEntry]] = {}  # key: (model, temp, seed)
        self._hits = 0
        self._misses = 0
        # Thread-safety: guard mutable state so concurrent
        # AEL stages can share a gateway without corrupting stats or entries.
        self._lock = threading.RLock()

    @property
    def stats(self) -> Dict[str, int]:
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total) if total else 0.0
            return {
                "hits": self._hits,
                "misses": self._misses,
                "total": total,
                "hit_rate_pct": int(round(hit_rate * 100)),
                "entries": sum(len(v) for v in self._store.values()),
            }

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    def lookup_or_call(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        fetch: Any,
    ) -> str:
        """Return a cached response if found, else call ``fetch(prompt)`` and cache."""
        if not self.enabled:
            with self._lock:
                self._misses += 1
            return fetch(prompt)

        key = self._key(model, temperature, seed)
        with self._lock:
            hit = self._find_hit(key, prompt)
            if hit is not None:
                hit.hits += 1
                self._hits += 1
                return hit.response
            self._misses += 1

        # Call LLM outside the lock so concurrent callers aren't serialised.
        response = fetch(prompt)

        with self._lock:
            entry = CacheEntry(
                prompt=prompt, response=response, model=model,
                temperature=temperature, seed=seed, timestamp=time.time(),
            )
            self._store.setdefault(key, []).append(entry)
            self._evict_if_needed()
        return response

    def put(
        self,
        prompt: str,
        response: str,
        *,
        model: str,
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ) -> CacheEntry:
        key = self._key(model, temperature, seed)
        entry = CacheEntry(
            prompt=prompt, response=response, model=model,
            temperature=temperature, seed=seed, timestamp=time.time(),
        )
        with self._lock:
            self._store.setdefault(key, []).append(entry)
        return entry

    def get(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ) -> Optional[str]:
        if not self.enabled:
            return None
        with self._lock:
            hit = self._find_hit(self._key(model, temperature, seed), prompt)
            if hit is not None:
                hit.hits += 1
                self._hits += 1
                return hit.response
            self._misses += 1
            return None

    # ------------------------------------------------------------------

    def _key(self, model: str, temperature: float, seed: Optional[int]) -> str:
        return f"{model}|{temperature}|{seed}"

    def _find_hit(self, key: str, prompt: str) -> Optional[CacheEntry]:
        entries = self._store.get(key, [])
        best: Optional[CacheEntry] = None
        best_sim = 0.0
        for e in entries:
            sim = _similarity(prompt, e.prompt)
            if sim > best_sim:
                best_sim = sim
                best = e
        if best_sim >= self.threshold:
            return best
        return None

    def _evict_if_needed(self) -> None:
        total = sum(len(v) for v in self._store.values())
        if total <= self.max_entries:
            return
        # Simple eviction: drop the oldest entry overall
        oldest: Optional[tuple[str, int]] = None
        oldest_ts = float("inf")
        for key, entries in self._store.items():
            for idx, e in enumerate(entries):
                if e.timestamp < oldest_ts:
                    oldest_ts = e.timestamp
                    oldest = (key, idx)
        if oldest:
            k, i = oldest
            del self._store[k][i]


def _similarity(a: str, b: str) -> float:
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union
