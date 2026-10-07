# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Tests for shared.rag.hybrid_retriever.

Validates BM25 indexing, vector search, RRF fusion, and hybrid retrieval.
"""

import numpy as np
import pytest

from shared.rag.document_store import DocumentStore, Document
from shared.rag.hybrid_retriever import (
    BM25Index,
    VectorIndex,
    HybridRetriever,
    reciprocal_rank_fusion,
)


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
def sample_docs():
    return [
        Document(
            doc_id="doc1",
            title="Artificial Intelligence and Labor Markets",
            abstract="This paper studies the impact of AI on employment and wages.",
        ),
        Document(
            doc_id="doc2",
            title="Fiscal Policy and GDP Growth",
            abstract="We analyze the effects of fiscal stimulus on GDP growth.",
        ),
        Document(
            doc_id="doc3",
            title="Machine Learning for Economic Forecasting",
            abstract="A survey of ML applications in macroeconomic predictions.",
        ),
        Document(
            doc_id="doc4",
            title="Climate Change and Agriculture",
            abstract="Impact of climate change on crop yields in developing nations.",
        ),
    ]


@pytest.fixture
def store_with_docs(tmp_path, sample_docs):
    store = DocumentStore(store_dir=str(tmp_path), embed_fn=fake_embed)
    store.add_documents(sample_docs)
    yield store
    store.close()


class TestBM25Index:
    """Test BM25 keyword search."""

    def test_build_and_search(self, sample_docs):
        bm25 = BM25Index()
        bm25.build(sample_docs)
        results = bm25.search("AI labor employment")
        assert len(results) >= 1
        assert results[0][0] == "doc1"  # Best match

    def test_no_results_for_unrelated_query(self, sample_docs):
        bm25 = BM25Index()
        bm25.build(sample_docs)
        results = bm25.search("quantum physics")
        assert len(results) == 0

    def test_empty_index(self):
        bm25 = BM25Index()
        bm25.build([])
        results = bm25.search("anything")
        assert results == []

    def test_respects_top_k(self, sample_docs):
        bm25 = BM25Index()
        bm25.build(sample_docs)
        results = bm25.search("economic", top_k=2)
        assert len(results) <= 2


class TestVectorIndex:
    """Test vector similarity search."""

    def test_search_returns_results(self, store_with_docs):
        vi = VectorIndex(store_with_docs)
        query_emb = np.array(fake_embed(["AI labor markets"])[0], dtype=np.float32)
        results = vi.search(query_emb, top_k=3)
        assert len(results) >= 1

    def test_search_empty_store(self, tmp_path):
        store = DocumentStore(store_dir=str(tmp_path))
        vi = VectorIndex(store)
        query_emb = np.array([0.1] * 8, dtype=np.float32)
        results = vi.search(query_emb)
        assert results == []
        store.close()


class TestReciprocalRankFusion:
    """Test RRF merging algorithm."""

    def test_single_list(self):
        results = reciprocal_rank_fusion(
            [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        )
        assert results[0][0] == "doc1"
        assert results[1][0] == "doc2"

    def test_two_lists_merge(self):
        list1 = [("doc1", 0.9), ("doc2", 0.5)]
        list2 = [("doc2", 0.9), ("doc3", 0.5)]
        results = reciprocal_rank_fusion(list1, list2)
        # doc2 appears in both lists, should rank high
        doc_ids = [r[0] for r in results]
        assert "doc2" in doc_ids

    def test_empty_lists(self):
        results = reciprocal_rank_fusion([], [])
        assert results == []

    def test_rrf_scores_are_positive(self):
        results = reciprocal_rank_fusion(
            [("doc1", 1.0)], [("doc2", 1.0)]
        )
        assert all(score > 0 for _, score in results)


class TestHybridRetriever:
    """Test full hybrid retrieval pipeline."""

    def test_bm25_only_search(self, store_with_docs):
        retriever = HybridRetriever(document_store=store_with_docs)
        # Search without query_embedding → BM25 only
        results = retriever.search("AI labor markets")
        assert len(results) >= 1
        assert isinstance(results[0], Document)

    def test_hybrid_search_with_embedding(self, store_with_docs):
        retriever = HybridRetriever(document_store=store_with_docs)
        query_emb = np.array(fake_embed(["AI labor employment"])[0], dtype=np.float32)
        results = retriever.search("AI labor employment", query_embedding=query_emb)
        assert len(results) >= 1

    def test_respects_top_k(self, store_with_docs):
        retriever = HybridRetriever(document_store=store_with_docs)
        results = retriever.search("economic", top_k=2)
        assert len(results) <= 2

    def test_auto_builds_index(self, store_with_docs):
        retriever = HybridRetriever(document_store=store_with_docs)
        assert not retriever._index_built
        results = retriever.search("test")
        assert retriever._index_built

    def test_empty_store(self, tmp_path):
        store = DocumentStore(store_dir=str(tmp_path))
        retriever = HybridRetriever(document_store=store)
        results = retriever.search("anything")
        assert results == []
        store.close()
