# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
RAG (Retrieval-Augmented Generation) system for AEL workflows.

- DocumentStore: SQLite metadata + numpy embeddings for literature
- HybridRetriever: BM25 + vector + Reciprocal Rank Fusion
- SemanticCache: Cache LLM responses by semantic prompt similarity
"""

from shared.rag.document_store import DocumentStore, Document
from shared.rag.hybrid_retriever import HybridRetriever
from shared.rag.semantic_cache import SemanticCache, CacheResult, CacheStats

__all__ = [
    "DocumentStore",
    "Document",
    "HybridRetriever",
    "SemanticCache",
    "CacheResult",
    "CacheStats",
]
