# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Semantic Cache — Caches LLM responses for semantically similar queries.

Avoids redundant LLM calls by embedding prompts and matching against
cached responses within a cosine similarity threshold.

Storage: SQLite metadata + numpy embedding vectors on disk.

Usage:
    from shared.rag.semantic_cache import SemanticCache

    cache = SemanticCache(cache_dir="./cache", similarity_threshold=0.95)
    result = cache.get_or_compute(
        prompt="Summarize AI labor economics",
        llm_fn=lambda p: client.invoke([{"role": "user", "content": p}]),
        embed_fn=lambda texts: client.embed(texts),
    )
    if result.hit:
        print(f"Cache hit! Saved ~${result.cost_saved:.4f}")
"""

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass
class CacheResult:
    """Result from a cache lookup."""

    hit: bool
    response: str
    cost_saved: float = 0.0
    similarity: float = 0.0
    cache_key: str = ""


@dataclass
class CacheStats:
    """Cache performance statistics."""

    total_entries: int = 0
    total_lookups: int = 0
    total_hits: int = 0
    total_misses: int = 0
    hit_rate: float = 0.0
    total_cost_saved: float = 0.0
    oldest_entry: str = ""
    newest_entry: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_entries": self.total_entries,
            "total_lookups": self.total_lookups,
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "hit_rate": round(self.hit_rate, 4),
            "total_cost_saved": round(self.total_cost_saved, 6),
        }


class SemanticCache:
    """
    Caches LLM responses keyed by semantic similarity of prompts.

    How it works:
    1. Embed the incoming prompt
    2. Search cache for vectors within cosine similarity threshold
    3. If match: return cached response (no LLM call)
    4. If no match: call LLM, cache prompt embedding + response
    """

    def __init__(
        self,
        cache_dir: str = ".cache/semantic",
        similarity_threshold: float = 0.95,
        max_cache_size: int = 10000,
        ttl_hours: int = 24,
    ):
        """
        Args:
            cache_dir: Directory for SQLite DB and embedding files.
            similarity_threshold: Cosine similarity threshold (0-1) for cache hits.
            max_cache_size: Maximum number of cached entries.
            ttl_hours: Time-to-live in hours for cache entries.
        """
        self.cache_dir = Path(cache_dir)
        self.threshold = similarity_threshold
        self.max_size = max_cache_size
        self.ttl_hours = ttl_hours

        self._total_lookups = 0
        self._total_hits = 0
        self._total_cost_saved = 0.0

        # Ensure directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # SQLite for metadata
        self._db_path = self.cache_dir / "cache.db"
        self._embeddings_path = self.cache_dir / "embeddings.npz"
        self._init_db()

        # In-memory embedding matrix (loaded lazily)
        self._embeddings: Optional[np.ndarray] = None
        self._keys: List[str] = []
        self._loaded = False

    def _init_db(self):
        """Create the cache table if it doesn't exist."""
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    cache_key TEXT PRIMARY KEY,
                    prompt_hash TEXT NOT NULL,
                    prompt_preview TEXT DEFAULT '',
                    response TEXT NOT NULL,
                    model TEXT DEFAULT '',
                    estimated_cost REAL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    access_count INTEGER DEFAULT 1
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def _load_embeddings(self):
        """Load embeddings from disk into memory."""
        if self._loaded:
            return

        if self._embeddings_path.exists():
            data = np.load(str(self._embeddings_path), allow_pickle=True)
            self._embeddings = data.get("embeddings")
            self._keys = list(data.get("keys", []))
        else:
            self._embeddings = None
            self._keys = []

        self._loaded = True

    def _save_embeddings(self):
        """Save embeddings to disk."""
        if self._embeddings is not None and len(self._keys) > 0:
            np.savez(
                str(self._embeddings_path),
                embeddings=self._embeddings,
                keys=np.array(self._keys, dtype=object),
            )

    def get_or_compute(
        self,
        prompt: str,
        llm_fn: Callable[[str], str],
        embed_fn: Callable[[List[str]], List[List[float]]],
        model: str = "",
        estimated_cost: float = 0.0,
    ) -> CacheResult:
        """
        Check cache first; compute and cache on miss.

        Args:
            prompt: The prompt to send to the LLM.
            llm_fn: Function to call on cache miss. Signature: (prompt) -> response.
            embed_fn: Function to embed texts. Signature: (texts) -> embeddings.
            model: Model name for metadata.
            estimated_cost: Estimated cost of the LLM call (for cost tracking).

        Returns:
            CacheResult with hit status, response, and cost saved.
        """
        self._total_lookups += 1
        self._load_embeddings()

        # Embed the prompt
        prompt_embedding = np.array(embed_fn([prompt])[0], dtype=np.float32)

        # Search cache
        cache_key, similarity = self._find_similar(prompt_embedding)

        if cache_key is not None:
            # Cache hit
            response = self._get_response(cache_key)
            if response is not None:
                self._total_hits += 1
                self._total_cost_saved += estimated_cost
                self._update_access(cache_key)
                return CacheResult(
                    hit=True,
                    response=response,
                    cost_saved=estimated_cost,
                    similarity=similarity,
                    cache_key=cache_key,
                )

        # Cache miss — call LLM
        response = llm_fn(prompt)

        # Store in cache
        new_key = self._store(prompt, prompt_embedding, response, model, estimated_cost)

        return CacheResult(
            hit=False,
            response=response,
            cost_saved=0.0,
            similarity=0.0,
            cache_key=new_key,
        )

    def lookup(
        self,
        prompt: str,
        embed_fn: Callable[[List[str]], List[List[float]]],
    ) -> Optional[CacheResult]:
        """
        Look up a prompt in the cache without computing on miss.

        Returns CacheResult if found, None if not.
        """
        self._load_embeddings()
        prompt_embedding = np.array(embed_fn([prompt])[0], dtype=np.float32)
        cache_key, similarity = self._find_similar(prompt_embedding)

        if cache_key is not None:
            response = self._get_response(cache_key)
            if response is not None:
                return CacheResult(
                    hit=True,
                    response=response,
                    similarity=similarity,
                    cache_key=cache_key,
                )
        return None

    def _find_similar(
        self, embedding: np.ndarray
    ) -> tuple:
        """Find the most similar cached embedding above threshold."""
        if self._embeddings is None or len(self._keys) == 0:
            return None, 0.0

        # Cosine similarity
        norm_q = np.linalg.norm(embedding)
        if norm_q == 0:
            return None, 0.0

        norms = np.linalg.norm(self._embeddings, axis=1)
        # Avoid division by zero
        valid = norms > 0
        if not np.any(valid):
            return None, 0.0

        sims = np.zeros(len(self._keys))
        sims[valid] = (
            self._embeddings[valid] @ embedding
        ) / (norms[valid] * norm_q)

        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= self.threshold:
            return self._keys[best_idx], best_sim

        return None, best_sim

    def _get_response(self, cache_key: str) -> Optional[str]:
        """Get cached response by key, respecting TTL."""
        conn = sqlite3.connect(str(self._db_path))
        try:
            row = conn.execute(
                "SELECT response, created_at FROM cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if row is None:
                return None
            response, created_at = row
            # Check TTL
            created = datetime.fromisoformat(created_at)
            now = datetime.now(timezone.utc)
            age_hours = (now - created).total_seconds() / 3600
            if age_hours > self.ttl_hours:
                # Expired — remove
                self._evict(cache_key, conn)
                return None
            return response
        finally:
            conn.close()

    def _update_access(self, cache_key: str):
        """Update last_accessed and access_count."""
        conn = sqlite3.connect(str(self._db_path))
        try:
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "UPDATE cache SET last_accessed = ?, access_count = access_count + 1 "
                "WHERE cache_key = ?",
                (now, cache_key),
            )
            conn.commit()
        finally:
            conn.close()

    def _store(
        self,
        prompt: str,
        embedding: np.ndarray,
        response: str,
        model: str,
        estimated_cost: float,
    ) -> str:
        """Store a new cache entry."""
        # Enforce max size
        self._enforce_max_size()

        cache_key = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        now = datetime.now(timezone.utc).isoformat()

        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute(
                "INSERT OR REPLACE INTO cache "
                "(cache_key, prompt_hash, prompt_preview, response, model, "
                "estimated_cost, created_at, last_accessed, access_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)",
                (
                    cache_key,
                    hashlib.sha256(prompt.encode()).hexdigest(),
                    prompt[:200],
                    response,
                    model,
                    estimated_cost,
                    now,
                    now,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        # Update in-memory embeddings
        if self._embeddings is not None:
            self._embeddings = np.vstack([self._embeddings, embedding.reshape(1, -1)])
        else:
            self._embeddings = embedding.reshape(1, -1)
        self._keys.append(cache_key)

        self._save_embeddings()
        return cache_key

    def _enforce_max_size(self):
        """Evict oldest entries if cache exceeds max size."""
        conn = sqlite3.connect(str(self._db_path))
        try:
            count = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            if count >= self.max_size:
                # Remove oldest 10%
                to_remove = max(1, count // 10)
                rows = conn.execute(
                    "SELECT cache_key FROM cache ORDER BY last_accessed ASC LIMIT ?",
                    (to_remove,),
                ).fetchall()
                for (key,) in rows:
                    self._evict(key, conn)
        finally:
            conn.close()

    def _evict(self, cache_key: str, conn: sqlite3.Connection):
        """Remove a single entry from DB and in-memory embeddings."""
        conn.execute("DELETE FROM cache WHERE cache_key = ?", (cache_key,))
        conn.commit()
        if cache_key in self._keys:
            idx = self._keys.index(cache_key)
            self._keys.pop(idx)
            if self._embeddings is not None and len(self._embeddings) > idx:
                self._embeddings = np.delete(self._embeddings, idx, axis=0)

    def clear(self):
        """Clear all cache entries."""
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute("DELETE FROM cache")
            conn.commit()
        finally:
            conn.close()
        self._embeddings = None
        self._keys = []
        self._save_embeddings()

    def get_stats(self) -> CacheStats:
        """Get cache performance statistics."""
        conn = sqlite3.connect(str(self._db_path))
        try:
            count = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            oldest = conn.execute(
                "SELECT MIN(created_at) FROM cache"
            ).fetchone()[0] or ""
            newest = conn.execute(
                "SELECT MAX(created_at) FROM cache"
            ).fetchone()[0] or ""
        finally:
            conn.close()

        hit_rate = (
            self._total_hits / self._total_lookups
            if self._total_lookups > 0
            else 0.0
        )

        return CacheStats(
            total_entries=count,
            total_lookups=self._total_lookups,
            total_hits=self._total_hits,
            total_misses=self._total_lookups - self._total_hits,
            hit_rate=hit_rate,
            total_cost_saved=self._total_cost_saved,
            oldest_entry=oldest,
            newest_entry=newest,
        )
