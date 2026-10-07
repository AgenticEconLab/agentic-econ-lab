# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Tests for shared.rag.document_store.DocumentStore.

Validates SQLite metadata storage and embedding operations.
"""

import os
import json

import numpy as np
import pytest

from shared.rag.document_store import DocumentStore, Document, StoreStats


def fake_embed(texts):
    """Deterministic fake embedding function (8-dimensional)."""
    embeddings = []
    for text in texts:
        words = text.lower().split()
        vec = [0.0] * 8
        for i, w in enumerate(words):
            vec[i % 8] += hash(w) % 100 / 100.0
        norm = sum(v**2 for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        embeddings.append(vec)
    return embeddings


@pytest.fixture
def store(tmp_path):
    """DocumentStore without embeddings."""
    s = DocumentStore(store_dir=str(tmp_path))
    yield s
    s.close()


@pytest.fixture
def store_with_embed(tmp_path):
    """DocumentStore with fake embedding function."""
    s = DocumentStore(store_dir=str(tmp_path), embed_fn=fake_embed)
    yield s
    s.close()


@pytest.fixture
def sample_docs():
    return [
        Document(
            doc_id="arxiv:2301.01234",
            title="AI and Labor Markets",
            abstract="This paper studies the impact of AI on employment.",
            authors=["Smith, John", "Doe, Jane"],
            year=2023,
            source="arXiv",
        ),
        Document(
            doc_id="arxiv:2302.05678",
            title="Fiscal Policy and GDP Growth",
            abstract="We analyze the effects of fiscal stimulus on GDP.",
            authors=["Johnson, Bob"],
            year=2024,
            source="arXiv",
        ),
        Document(
            doc_id="scholar:9999",
            title="Machine Learning in Economics",
            abstract="A survey of ML applications in economic research.",
            authors=["Williams, Alice"],
            year=2023,
            source="Scholar",
        ),
    ]


class TestAddDocuments:
    """Test document insertion."""

    def test_add_documents(self, store, sample_docs):
        added = store.add_documents(sample_docs)
        assert added == 3
        assert store.count() == 3

    def test_add_with_embeddings(self, store_with_embed, sample_docs):
        store_with_embed.add_documents(sample_docs)
        embeddings, ids = store_with_embed.get_embeddings()
        assert embeddings is not None
        assert embeddings.shape[0] == 3
        assert len(ids) == 3

    def test_add_duplicate_replaces(self, store, sample_docs):
        store.add_documents(sample_docs[:1])
        assert store.count() == 1
        # Re-add with updated data
        updated = Document(
            doc_id="arxiv:2301.01234",
            title="Updated Title",
            abstract="Updated abstract",
        )
        store.add_documents([updated])
        assert store.count() == 1
        doc = store.get_by_id("arxiv:2301.01234")
        assert doc.title == "Updated Title"

    def test_add_empty_list(self, store):
        added = store.add_documents([])
        assert added == 0


class TestRetrieve:
    """Test document retrieval."""

    def test_get_by_id(self, store, sample_docs):
        store.add_documents(sample_docs)
        doc = store.get_by_id("arxiv:2301.01234")
        assert doc is not None
        assert doc.title == "AI and Labor Markets"
        assert doc.authors == ["Smith, John", "Doe, Jane"]
        assert doc.year == 2023

    def test_get_missing(self, store):
        assert store.get_by_id("nonexistent") is None

    def test_search_by_text(self, store, sample_docs):
        store.add_documents(sample_docs)
        results = store.search_by_text("GDP")
        assert len(results) >= 1
        assert any("GDP" in d.title for d in results)

    def test_get_all_documents(self, store, sample_docs):
        store.add_documents(sample_docs)
        all_docs = store.get_all_documents()
        assert len(all_docs) == 3


class TestEmbeddings:
    """Test embedding storage and retrieval."""

    def test_embeddings_saved_to_disk(self, store_with_embed, sample_docs, tmp_path):
        store_with_embed.add_documents(sample_docs)
        assert os.path.exists(os.path.join(str(tmp_path), "embeddings.npy"))
        assert os.path.exists(os.path.join(str(tmp_path), "embedding_ids.json"))

    def test_embeddings_persist_across_instances(self, tmp_path, sample_docs):
        store1 = DocumentStore(store_dir=str(tmp_path), embed_fn=fake_embed)
        store1.add_documents(sample_docs)
        store1.close()

        store2 = DocumentStore(store_dir=str(tmp_path))
        embeddings, ids = store2.get_embeddings()
        assert embeddings is not None
        assert len(ids) == 3
        store2.close()

    def test_get_embeddings_empty(self, store):
        embeddings, ids = store.get_embeddings()
        assert embeddings is None
        assert ids == []


class TestDeleteClear:
    """Test deletion and clearing."""

    def test_delete_existing(self, store, sample_docs):
        store.add_documents(sample_docs)
        assert store.delete("arxiv:2301.01234") is True
        assert store.count() == 2
        assert store.get_by_id("arxiv:2301.01234") is None

    def test_delete_missing(self, store):
        assert store.delete("nonexistent") is False

    def test_delete_removes_embedding(self, store_with_embed, sample_docs):
        store_with_embed.add_documents(sample_docs)
        embeddings_before, ids_before = store_with_embed.get_embeddings()
        assert len(ids_before) == 3

        store_with_embed.delete("arxiv:2301.01234")
        embeddings_after, ids_after = store_with_embed.get_embeddings()
        assert len(ids_after) == 2
        assert "arxiv:2301.01234" not in ids_after

    def test_clear(self, store, sample_docs):
        store.add_documents(sample_docs)
        store.clear()
        assert store.count() == 0


class TestStats:
    """Test store statistics."""

    def test_stats_empty(self, store):
        stats = store.get_stats()
        assert stats.total_documents == 0
        assert stats.documents_with_embeddings == 0

    def test_stats_with_documents(self, store_with_embed, sample_docs):
        store_with_embed.add_documents(sample_docs)
        stats = store_with_embed.get_stats()
        assert stats.total_documents == 3
        assert stats.documents_with_embeddings == 3
        assert stats.embedding_dimension == 8  # fake_embed produces 8-dim
