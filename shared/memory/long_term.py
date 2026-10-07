# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Long-Term Memory — SQLite + embeddings for cross-run persistent facts.

Tier 2 of the 3-tier memory system. Stores discovered facts, data sources,
and literature findings across pipeline runs with vector similarity recall.

Usage:
    from shared.memory.long_term import LongTermMemory

    ltm = LongTermMemory(memory_dir="./memory")
    ltm.remember_fact("FRED provides GDP data", category="data_source", source="DataTeam")
    facts = ltm.recall_facts("GDP data sources", top_k=5)
    ltm.forget_stale(max_age_days=90)
"""

import json
import os
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field


class Fact(BaseModel):
    """A persistent fact stored in long-term memory."""

    fact_id: int = Field(default=0, description="Auto-assigned row ID")
    fact: str = Field(description="The factual statement")
    category: str = Field(default="general", description="e.g. data_source, literature, method")
    source: str = Field(default="", description="Team or agent that produced this fact")
    run_id: str = Field(default="", description="Pipeline run that produced this fact")
    relevance_score: float = Field(default=0.0, description="Similarity score from recall")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class LongTermMemory:
    """
    SQLite-backed persistent memory with optional vector similarity recall.

    Facts are stored in SQLite with text and metadata. When an embed_fn is
    provided, facts also get vector embeddings for semantic recall.
    """

    def __init__(
        self,
        memory_dir: str,
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    ):
        """
        Args:
            memory_dir: Directory for the SQLite database and embeddings.
            embed_fn: Optional embedding function (texts -> embeddings).
                      If None, recall falls back to keyword matching.
        """
        self._dir = os.path.join(memory_dir, "long_term")
        os.makedirs(self._dir, exist_ok=True)

        self._embed_fn = embed_fn
        self._db_path = os.path.join(self._dir, "facts.db")
        self._embeddings_path = os.path.join(self._dir, "fact_embeddings.npy")
        self._ids_path = os.path.join(self._dir, "fact_ids.json")

        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

        # Load embeddings
        self._embeddings: Optional[np.ndarray] = None
        self._embedding_ids: List[int] = []
        self._load_embeddings()

    def _init_db(self):
        """Create the facts table if it doesn't exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                source TEXT DEFAULT '',
                run_id TEXT DEFAULT '',
                timestamp TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def _load_embeddings(self):
        """Load fact embeddings from disk."""
        if os.path.exists(self._embeddings_path) and os.path.exists(self._ids_path):
            self._embeddings = np.load(self._embeddings_path)
            with open(self._ids_path, "r", encoding="utf-8") as f:
                self._embedding_ids = json.load(f)

    def _save_embeddings(self):
        """Save fact embeddings to disk, or remove stale files if empty."""
        if self._embeddings is not None and len(self._embedding_ids) > 0:
            np.save(self._embeddings_path, self._embeddings)
            with open(self._ids_path, "w", encoding="utf-8") as f:
                json.dump(self._embedding_ids, f)
        else:
            for path in [self._embeddings_path, self._ids_path]:
                if os.path.exists(path):
                    os.remove(path)

    def remember_fact(
        self,
        fact: str,
        category: str = "general",
        source: str = "",
        run_id: str = "",
    ) -> Fact:
        """
        Store a new fact in long-term memory.

        Args:
            fact: The factual statement to remember.
            category: Category of the fact (data_source, literature, method, etc.).
            source: Team or agent that produced this fact.
            run_id: Pipeline run ID for traceability.

        Returns:
            The stored Fact with assigned ID.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        cursor = self._conn.execute(
            """INSERT INTO facts (fact, category, source, run_id, timestamp)
               VALUES (?, ?, ?, ?, ?)""",
            (fact, category, source, run_id, timestamp),
        )
        self._conn.commit()
        fact_id = cursor.lastrowid

        # Compute and store embedding
        if self._embed_fn is not None:
            embeddings = self._embed_fn([fact])
            new_array = np.array(embeddings, dtype=np.float32)

            if self._embeddings is not None and self._embeddings.size > 0:
                self._embeddings = np.vstack([self._embeddings, new_array])
            else:
                self._embeddings = new_array

            self._embedding_ids.append(fact_id)
            self._save_embeddings()

        return Fact(
            fact_id=fact_id,
            fact=fact,
            category=category,
            source=source,
            run_id=run_id,
            timestamp=timestamp,
        )

    def recall_facts(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
    ) -> List[Fact]:
        """
        Recall facts relevant to a query.

        Uses vector similarity if embeddings are available, otherwise
        falls back to keyword matching.

        Args:
            query: The query string.
            top_k: Maximum number of facts to return.
            category: Optional filter by category.

        Returns:
            List of Fact objects, most relevant first.
        """
        # Try vector recall first
        if self._embed_fn is not None and self._embeddings is not None and len(self._embedding_ids) > 0:
            return self._recall_by_vector(query, top_k, category)

        # Fallback to keyword matching
        return self._recall_by_keyword(query, top_k, category)

    def forget_stale(self, max_age_days: int = 90) -> int:
        """
        Remove facts older than max_age_days.

        Returns:
            Number of facts removed.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()

        # Get IDs to remove
        cursor = self._conn.execute(
            "SELECT fact_id FROM facts WHERE timestamp < ?", (cutoff,)
        )
        stale_ids = {row["fact_id"] for row in cursor.fetchall()}

        if not stale_ids:
            return 0

        # Remove from DB
        self._conn.execute(
            f"DELETE FROM facts WHERE timestamp < ?", (cutoff,)
        )
        self._conn.commit()

        # Remove from embeddings
        if self._embedding_ids:
            keep_mask = [
                i for i, fid in enumerate(self._embedding_ids)
                if fid not in stale_ids
            ]
            self._embedding_ids = [self._embedding_ids[i] for i in keep_mask]
            if self._embeddings is not None and keep_mask:
                self._embeddings = self._embeddings[keep_mask]
            elif not keep_mask:
                self._embeddings = None
                self._embedding_ids = []
            self._save_embeddings()

        return len(stale_ids)

    def get_all_facts(self, category: Optional[str] = None) -> List[Fact]:
        """Retrieve all facts, optionally filtered by category."""
        if category:
            cursor = self._conn.execute(
                "SELECT * FROM facts WHERE category = ? ORDER BY timestamp DESC",
                (category,),
            )
        else:
            cursor = self._conn.execute("SELECT * FROM facts ORDER BY timestamp DESC")
        return [self._row_to_fact(row) for row in cursor.fetchall()]

    def count(self, category: Optional[str] = None) -> int:
        """Count facts, optionally by category."""
        if category:
            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM facts WHERE category = ?", (category,)
            )
        else:
            cursor = self._conn.execute("SELECT COUNT(*) FROM facts")
        return cursor.fetchone()[0]

    def clear(self):
        """Remove all facts and embeddings."""
        self._conn.execute("DELETE FROM facts")
        self._conn.commit()
        self._embeddings = None
        self._embedding_ids = []
        for path in [self._embeddings_path, self._ids_path]:
            if os.path.exists(path):
                os.remove(path)

    def close(self):
        """Close the SQLite connection."""
        self._conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit — ensures connection is closed."""
        self.close()
        return False

    def __del__(self):
        """Safety net: close connection if not already closed."""
        try:
            self._conn.close()
        except Exception:
            pass

    def _recall_by_vector(
        self, query: str, top_k: int, category: Optional[str]
    ) -> List[Fact]:
        """Recall facts by cosine similarity to query embedding."""
        query_embedding = np.array(
            self._embed_fn([query])[0], dtype=np.float32
        )

        # Compute cosine similarities
        norms = np.linalg.norm(self._embeddings, axis=1)
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []

        similarities = self._embeddings @ query_embedding / (norms * query_norm + 1e-10)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in top_indices:
            if len(results) >= top_k:
                break
            fact_id = self._embedding_ids[idx]
            fact = self._get_fact_by_id(fact_id)
            if fact is None:
                continue
            if category and fact.category != category:
                continue
            fact.relevance_score = float(similarities[idx])
            results.append(fact)

        return results

    def _recall_by_keyword(
        self, query: str, top_k: int, category: Optional[str]
    ) -> List[Fact]:
        """Recall facts by keyword overlap with query."""
        query_words = set(query.lower().split())
        if not query_words:
            return []

        all_facts = self.get_all_facts(category=category)
        scored = []
        for fact in all_facts:
            fact_words = set(fact.fact.lower().split())
            overlap = len(query_words & fact_words)
            union = len(query_words | fact_words)
            score = overlap / union if union > 0 else 0.0
            if score > 0:
                fact.relevance_score = score
                scored.append(fact)

        scored.sort(key=lambda f: f.relevance_score, reverse=True)
        return scored[:top_k]

    def _get_fact_by_id(self, fact_id: int) -> Optional[Fact]:
        """Retrieve a single fact by ID."""
        cursor = self._conn.execute(
            "SELECT * FROM facts WHERE fact_id = ?", (fact_id,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_fact(row)

    @staticmethod
    def _row_to_fact(row) -> Fact:
        """Convert a SQLite row to a Fact."""
        return Fact(
            fact_id=row["fact_id"],
            fact=row["fact"],
            category=row["category"] or "general",
            source=row["source"] or "",
            run_id=row["run_id"] or "",
            timestamp=row["timestamp"] or "",
        )
