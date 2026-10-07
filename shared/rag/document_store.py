# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Document Store — SQLite metadata + numpy embeddings for literature.

Provides persistent storage for literature documents with both structured
metadata (SQLite) and vector embeddings (numpy arrays on disk).

Usage:
    from shared.rag.document_store import DocumentStore, Document

    store = DocumentStore(store_dir="./memory/documents")
    store.add_documents([
        Document(doc_id="arxiv:2301.01234", title="AI Economics",
                 abstract="...", authors=["Smith"], year=2023)
    ])
    doc = store.get_by_id("arxiv:2301.01234")
    stats = store.get_stats()
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field


class Document(BaseModel):
    """A document in the store (e.g., a literature item)."""

    doc_id: str = Field(description="Unique document identifier")
    title: str = Field(default="")
    abstract: str = Field(default="")
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    source: str = Field(default="", description="e.g. arXiv, Scholar, OpenAlex")
    url: str = Field(default="")
    venue: Optional[str] = None
    citation_count: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class StoreStats(BaseModel):
    """Statistics about the document store."""

    total_documents: int = 0
    documents_with_embeddings: int = 0
    embedding_dimension: int = 0
    store_dir: str = ""


class DocumentStore:
    """
    Local document store with SQLite metadata and numpy embeddings.

    Documents are stored in SQLite for structured queries. Embeddings
    are stored as a numpy .npy file for efficient vector operations.
    """

    def __init__(
        self,
        store_dir: str,
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    ):
        """
        Args:
            store_dir: Directory for SQLite DB and embedding files.
            embed_fn: Optional embedding function. Signature: (texts) -> embeddings.
                      If None, documents are stored without embeddings.
        """
        self._store_dir = store_dir
        self._embed_fn = embed_fn
        os.makedirs(store_dir, exist_ok=True)

        self._db_path = os.path.join(store_dir, "documents.db")
        self._embeddings_path = os.path.join(store_dir, "embeddings.npy")
        self._ids_path = os.path.join(store_dir, "embedding_ids.json")

        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

        # Load embeddings index
        self._embeddings: Optional[np.ndarray] = None
        self._embedding_ids: List[str] = []
        self._load_embeddings()

    def _init_db(self):
        """Create the documents table if it doesn't exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                title TEXT,
                abstract TEXT,
                authors TEXT,
                year INTEGER,
                source TEXT,
                url TEXT,
                venue TEXT,
                citation_count INTEGER,
                metadata TEXT,
                timestamp TEXT
            )
        """)
        self._conn.commit()

    def _load_embeddings(self):
        """Load embeddings and ID mapping from disk."""
        if os.path.exists(self._embeddings_path) and os.path.exists(self._ids_path):
            self._embeddings = np.load(self._embeddings_path)
            with open(self._ids_path, "r", encoding="utf-8") as f:
                self._embedding_ids = json.load(f)

    def _save_embeddings(self):
        """Save embeddings and ID mapping to disk, or remove stale files if empty."""
        if self._embeddings is not None and len(self._embedding_ids) > 0:
            np.save(self._embeddings_path, self._embeddings)
            with open(self._ids_path, "w", encoding="utf-8") as f:
                json.dump(self._embedding_ids, f)
        else:
            for path in [self._embeddings_path, self._ids_path]:
                if os.path.exists(path):
                    os.remove(path)

    def add_documents(
        self,
        documents: List[Document],
        compute_embeddings: bool = True,
    ) -> int:
        """
        Add documents to the store.

        Args:
            documents: List of Document objects to add.
            compute_embeddings: If True and embed_fn is set, compute embeddings.

        Returns:
            Number of documents added (excludes duplicates).
        """
        added = 0
        new_docs = []

        for doc in documents:
            try:
                self._conn.execute(
                    """INSERT OR REPLACE INTO documents
                       (doc_id, title, abstract, authors, year, source, url,
                        venue, citation_count, metadata, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        doc.doc_id,
                        doc.title,
                        doc.abstract,
                        json.dumps(doc.authors),
                        doc.year,
                        doc.source,
                        doc.url,
                        doc.venue,
                        doc.citation_count,
                        json.dumps(doc.metadata),
                        doc.timestamp,
                    ),
                )
                added += 1
                new_docs.append(doc)
            except sqlite3.Error:
                continue

        self._conn.commit()

        # Compute embeddings for new documents
        if compute_embeddings and self._embed_fn and new_docs:
            texts = [self._make_embed_text(d) for d in new_docs]
            new_embeddings = self._embed_fn(texts)
            new_ids = [d.doc_id for d in new_docs]
            self._merge_embeddings(new_ids, new_embeddings)

        return added

    def get_by_id(self, doc_id: str) -> Optional[Document]:
        """Retrieve a document by its ID."""
        cursor = self._conn.execute(
            "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_document(row)

    def search_by_text(self, query: str, limit: int = 20) -> List[Document]:
        """
        Simple SQL LIKE search on title and abstract.

        For proper retrieval, use HybridRetriever instead.
        """
        pattern = f"%{query}%"
        cursor = self._conn.execute(
            """SELECT * FROM documents
               WHERE title LIKE ? OR abstract LIKE ?
               LIMIT ?""",
            (pattern, pattern, limit),
        )
        return [self._row_to_document(row) for row in cursor.fetchall()]

    def get_all_documents(self) -> List[Document]:
        """Retrieve all documents from the store."""
        cursor = self._conn.execute("SELECT * FROM documents")
        return [self._row_to_document(row) for row in cursor.fetchall()]

    def get_embeddings(self) -> tuple:
        """
        Return the embedding matrix and corresponding doc IDs.

        Returns:
            (embeddings_array, doc_ids_list) or (None, []) if no embeddings.
        """
        return self._embeddings, list(self._embedding_ids)

    def count(self) -> int:
        """Count total documents."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM documents")
        return cursor.fetchone()[0]

    def get_stats(self) -> StoreStats:
        """Return store statistics."""
        dim = 0
        if self._embeddings is not None and self._embeddings.ndim == 2:
            dim = self._embeddings.shape[1]
        return StoreStats(
            total_documents=self.count(),
            documents_with_embeddings=len(self._embedding_ids),
            embedding_dimension=dim,
            store_dir=self._store_dir,
        )

    def delete(self, doc_id: str) -> bool:
        """Remove a document by ID. Returns True if document existed."""
        cursor = self._conn.execute(
            "DELETE FROM documents WHERE doc_id = ?", (doc_id,)
        )
        self._conn.commit()
        deleted = cursor.rowcount > 0

        # Remove from embeddings
        if deleted and doc_id in self._embedding_ids:
            idx = self._embedding_ids.index(doc_id)
            self._embedding_ids.pop(idx)
            if self._embeddings is not None and self._embeddings.shape[0] > idx:
                self._embeddings = np.delete(self._embeddings, idx, axis=0)
            if len(self._embedding_ids) == 0:
                self._embeddings = None
            self._save_embeddings()

        return deleted

    def clear(self):
        """Remove all documents and embeddings."""
        self._conn.execute("DELETE FROM documents")
        self._conn.commit()
        self._embeddings = None
        self._embedding_ids = []
        # Clean up files
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

    def _make_embed_text(self, doc: Document) -> str:
        """Create the text to embed for a document."""
        parts = []
        if doc.title:
            parts.append(doc.title)
        if doc.abstract:
            parts.append(doc.abstract)
        if doc.authors:
            parts.append(", ".join(doc.authors))
        return " ".join(parts) if parts else doc.doc_id

    def _merge_embeddings(
        self, new_ids: List[str], new_embeddings: List[List[float]]
    ):
        """Merge new embeddings into the existing embedding matrix."""
        new_array = np.array(new_embeddings, dtype=np.float32)

        # Remove any existing entries for these IDs
        for doc_id in new_ids:
            if doc_id in self._embedding_ids:
                idx = self._embedding_ids.index(doc_id)
                self._embedding_ids.pop(idx)
                if self._embeddings is not None:
                    self._embeddings = np.delete(self._embeddings, idx, axis=0)

        # Append new embeddings
        if self._embeddings is not None and self._embeddings.size > 0:
            self._embeddings = np.vstack([self._embeddings, new_array])
        else:
            self._embeddings = new_array

        self._embedding_ids.extend(new_ids)
        self._save_embeddings()

    @staticmethod
    def _row_to_document(row) -> Document:
        """Convert a SQLite row to a Document."""
        authors = json.loads(row["authors"]) if row["authors"] else []
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        return Document(
            doc_id=row["doc_id"],
            title=row["title"] or "",
            abstract=row["abstract"] or "",
            authors=authors,
            year=row["year"],
            source=row["source"] or "",
            url=row["url"] or "",
            venue=row["venue"],
            citation_count=row["citation_count"],
            metadata=metadata,
            timestamp=row["timestamp"] or "",
        )
