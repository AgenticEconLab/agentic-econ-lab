# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Hybrid Retriever — BM25 + vector search with Reciprocal Rank Fusion.

Combines keyword-based (BM25) and semantic (vector cosine) retrieval
for improved recall and precision over either method alone.

Target: ~0.91 recall vs ~0.72 BM25-only (+27%)

Usage:
    from shared.rag.hybrid_retriever import HybridRetriever
    from shared.rag.document_store import DocumentStore

    store = DocumentStore(store_dir="./memory/documents", embed_fn=llm.embed)
    retriever = HybridRetriever(document_store=store)
    results = retriever.search("fiscal policy GDP", top_k=20)
"""

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from shared.rag.document_store import Document, DocumentStore


class BM25Index:
    """
    BM25 (Okapi BM25) index for keyword-based retrieval.

    Implements the BM25 scoring formula without external dependencies.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self._k1 = k1
        self._b = b
        self._doc_ids: List[str] = []
        self._doc_lengths: List[int] = []
        self._avg_dl: float = 0.0
        self._term_freqs: List[Dict[str, int]] = []  # Per-document term frequencies
        self._doc_freq: Dict[str, int] = defaultdict(int)  # Document frequency per term
        self._n_docs: int = 0

    def build(self, documents: List[Document]) -> None:
        """Build the BM25 index from a list of documents."""
        self._doc_ids = []
        self._doc_lengths = []
        self._term_freqs = []
        self._doc_freq = defaultdict(int)

        for doc in documents:
            text = self._make_text(doc)
            tokens = self._tokenize(text)
            tf = Counter(tokens)

            self._doc_ids.append(doc.doc_id)
            self._doc_lengths.append(len(tokens))
            self._term_freqs.append(dict(tf))

            for term in set(tokens):
                self._doc_freq[term] += 1

        self._n_docs = len(documents)
        self._avg_dl = (
            sum(self._doc_lengths) / self._n_docs if self._n_docs > 0 else 0.0
        )

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Search the index and return ranked results.

        Returns:
            List of (doc_id, score) tuples, highest score first.
        """
        if self._n_docs == 0:
            return []

        query_tokens = self._tokenize(query)
        scores = [0.0] * self._n_docs

        for term in query_tokens:
            if term not in self._doc_freq:
                continue
            df = self._doc_freq[term]
            idf = math.log((self._n_docs - df + 0.5) / (df + 0.5) + 1.0)

            for i in range(self._n_docs):
                tf = self._term_freqs[i].get(term, 0)
                if tf == 0:
                    continue
                dl = self._doc_lengths[i]
                numerator = tf * (self._k1 + 1)
                denominator = tf + self._k1 * (1 - self._b + self._b * dl / self._avg_dl)
                scores[i] += idf * (numerator / denominator)

        # Rank by score
        ranked = sorted(
            [(self._doc_ids[i], scores[i]) for i in range(self._n_docs) if scores[i] > 0],
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer with lowercasing."""
        text = text.lower()
        tokens = re.findall(r"\b[a-z0-9]+\b", text)
        return tokens

    @staticmethod
    def _make_text(doc: Document) -> str:
        """Create searchable text from a document."""
        parts = []
        if doc.title:
            parts.append(doc.title)
        if doc.abstract:
            parts.append(doc.abstract)
        if doc.authors:
            parts.append(" ".join(doc.authors))
        return " ".join(parts)


class VectorIndex:
    """
    Vector similarity index using cosine similarity on embeddings.

    Operates on the embeddings stored in the DocumentStore.
    """

    def __init__(self, document_store: DocumentStore):
        self._store = document_store

    def search(self, query_embedding: np.ndarray, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Find documents most similar to the query embedding.

        Args:
            query_embedding: Query vector (1D numpy array).
            top_k: Number of results to return.

        Returns:
            List of (doc_id, similarity_score) tuples, highest first.
        """
        embeddings, doc_ids = self._store.get_embeddings()
        if embeddings is None or len(doc_ids) == 0:
            return []

        # Cosine similarity
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []

        norms = np.linalg.norm(embeddings, axis=1)
        similarities = embeddings @ query_embedding / (norms * query_norm + 1e-10)

        # Top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [
            (doc_ids[i], float(similarities[i]))
            for i in top_indices
            if similarities[i] > 0
        ]
        return results


def reciprocal_rank_fusion(
    *result_lists: List[Tuple[str, float]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion (RRF) to merge multiple ranked result lists.

    RRF score = sum over lists of 1 / (k + rank_i)

    Args:
        *result_lists: Variable number of ranked result lists,
                       each containing (doc_id, score) tuples.
        k: RRF constant (default 60, standard in literature).

    Returns:
        Merged (doc_id, rrf_score) list, highest score first.
    """
    fused_scores: Dict[str, float] = defaultdict(float)

    for results in result_lists:
        for rank, (doc_id, _score) in enumerate(results):
            fused_scores[doc_id] += 1.0 / (k + rank + 1)  # 1-indexed rank

    ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


class HybridRetriever:
    """
    Hybrid retrieval combining BM25 keyword search and vector similarity.

    Architecture:
    1. BM25 retrieval (keyword matching)
    2. Vector retrieval (semantic similarity)
    3. Reciprocal Rank Fusion (merge results)

    Both indexes are built lazily from the DocumentStore.
    """

    def __init__(
        self,
        document_store: DocumentStore,
        rrf_k: int = 60,
    ):
        """
        Args:
            document_store: The document store with documents and embeddings.
            rrf_k: RRF constant for rank fusion (default 60).
        """
        self._store = document_store
        self._rrf_k = rrf_k
        self._bm25 = BM25Index()
        self._vector = VectorIndex(document_store)
        self._index_built = False

    def build_index(self) -> None:
        """Build/rebuild the BM25 index from the document store."""
        documents = self._store.get_all_documents()
        self._bm25.build(documents)
        self._index_built = True

    def search(
        self,
        query: str,
        top_k: int = 20,
        query_embedding: Optional[np.ndarray] = None,
    ) -> List[Document]:
        """
        Search using hybrid BM25 + vector retrieval with RRF fusion.

        Args:
            query: Text query for BM25 and embedding.
            top_k: Number of results to return.
            query_embedding: Pre-computed query embedding. If None, only
                             BM25 results are used (no vector component).

        Returns:
            List of Document objects, most relevant first.
        """
        if not self._index_built:
            self.build_index()

        # BM25 retrieval
        bm25_results = self._bm25.search(query, top_k=top_k * 2)

        # Vector retrieval (if embedding provided)
        vector_results: List[Tuple[str, float]] = []
        if query_embedding is not None:
            vector_results = self._vector.search(query_embedding, top_k=top_k * 2)

        # Fuse results
        if vector_results:
            fused = reciprocal_rank_fusion(
                bm25_results, vector_results, k=self._rrf_k
            )
        else:
            fused = bm25_results

        # Resolve doc IDs to Document objects
        documents = []
        for doc_id, _score in fused[:top_k]:
            doc = self._store.get_by_id(doc_id)
            if doc is not None:
                documents.append(doc)

        return documents
