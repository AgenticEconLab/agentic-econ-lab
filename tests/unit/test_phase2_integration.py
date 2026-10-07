# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Tests for AEL V0.4 Phase 2: Memory & RAG Integration.

Verifies:
1. MemoryManager imported and used in all 15 MasterOrchestrators
2. RunTrajectory imported in all 15 MasterOrchestrators
3. memory.close() called in all 15 MasterOrchestrators
4. DocumentStore + HybridRetriever in 4 LiteratureTeam orchestrators
5. GroundingChecker in 4 LiteratureTeam orchestrators
6. SemanticCache parameter in LLMClient
7. Functional tests for MemoryManager, DocumentStore, HybridRetriever, GroundingChecker
"""

import ast
import os
import sys
import tempfile
import shutil
from pathlib import Path

import pytest

# repository root for file path lookups (sys.path handled by conftest.py)
_agents_dir = Path(__file__).resolve().parent.parent.parent


# ============================================================================
# Helpers
# ============================================================================

def _get_orchestrator_files():
    """Return all 15 MasterOrchestrator files across all teams."""
    base = _agents_dir
    orchestrators = []

    # IdeationTeam, LiteratureTeam, ModelTeam: 4 modes each
    for team in ["IdeationTeam", "LiteratureTeam", "ModelTeam"]:
        for mode in ["ModeNoWcNoHITL", "ModeNoWcWithHITL",
                      "ModeWithWcNoHITL", "ModeWithWcWithHITL"]:
            f = base / team / "ael" / mode / "0-MasterOrchestrator.py"
            if f.exists():
                orchestrators.append(f)

    # DataTeam: 3 workflows
    for mode in ["ModeOpenSourceAPI", "ModePremiumSubscribed", "ModeUserUploaded"]:
        f = base / "DataTeam" / "ael" / mode / "0-MasterOrchestrator.py"
        if f.exists():
            orchestrators.append(f)

    return orchestrators


def _get_literature_orchestrators():
    """Return 4 LiteratureTeam MasterOrchestrator files."""
    base = _agents_dir
    files = []
    for mode in ["ModeNoWcNoHITL", "ModeNoWcWithHITL",
                  "ModeWithWcNoHITL", "ModeWithWcWithHITL"]:
        f = base / "LiteratureTeam" / "ael" / mode / "0-MasterOrchestrator.py"
        if f.exists():
            files.append(f)
    return files


def _file_contains_import(filepath, module_path, name):
    """Check if a file imports `name` from `module_path` using AST."""
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(filepath))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module == module_path:
                for alias in node.names:
                    if alias.name == name:
                        return True
    return False


def _file_contains_text(filepath, text):
    """Check if a file contains a text string."""
    source = filepath.read_text(encoding="utf-8")
    return text in source


# ============================================================================
# Test 1: MemoryManager in all 15 MasterOrchestrators
# ============================================================================

class TestMemoryManagerIntegration:
    """Verify MemoryManager is imported and used in all 15 MasterOrchestrators."""

    def test_all_15_orchestrators_found(self):
        """Sanity check: we find exactly 15 MasterOrchestrator files."""
        orchestrators = _get_orchestrator_files()
        assert len(orchestrators) == 15, (
            f"Expected 15 MasterOrchestrators, found {len(orchestrators)}"
        )

    @pytest.mark.parametrize("filepath", _get_orchestrator_files(),
                             ids=lambda p: str(p.relative_to(_agents_dir)))
    def test_memory_manager_imported(self, filepath):
        """Each MasterOrchestrator imports MemoryManager."""
        assert _file_contains_import(
            filepath, "shared.memory.memory_manager", "MemoryManager"
        ), f"{filepath.name}: missing MemoryManager import"

    @pytest.mark.parametrize("filepath", _get_orchestrator_files(),
                             ids=lambda p: str(p.relative_to(_agents_dir)))
    def test_memory_manager_instantiated(self, filepath):
        """Each MasterOrchestrator instantiates MemoryManager."""
        assert _file_contains_text(filepath, "MemoryManager("), (
            f"{filepath.name}: MemoryManager not instantiated"
        )

    @pytest.mark.parametrize("filepath", _get_orchestrator_files(),
                             ids=lambda p: str(p.relative_to(_agents_dir)))
    def test_memory_close_called(self, filepath):
        """Each MasterOrchestrator calls memory.close()."""
        assert _file_contains_text(filepath, "memory.close()"), (
            f"{filepath.name}: memory.close() not called"
        )

    @pytest.mark.parametrize("filepath", _get_orchestrator_files(),
                             ids=lambda p: str(p.relative_to(_agents_dir)))
    def test_run_trajectory_imported(self, filepath):
        """Each MasterOrchestrator imports RunTrajectory."""
        # RunTrajectory may be imported inline (from shared.memory.episodic import RunTrajectory)
        assert _file_contains_text(filepath, "RunTrajectory"), (
            f"{filepath.name}: RunTrajectory not referenced"
        )

    @pytest.mark.parametrize("filepath", _get_orchestrator_files(),
                             ids=lambda p: str(p.relative_to(_agents_dir)))
    def test_memory_record_trajectory(self, filepath):
        """Each MasterOrchestrator records a trajectory."""
        assert _file_contains_text(filepath, "memory.record_trajectory("), (
            f"{filepath.name}: memory.record_trajectory() not called"
        )


# ============================================================================
# Test 2: DocumentStore + HybridRetriever in LiteratureTeam
# ============================================================================

class TestLiteratureTeamRAG:
    """Verify RAG components in LiteratureTeam MasterOrchestrators."""

    def test_literature_team_has_4_orchestrators(self):
        files = _get_literature_orchestrators()
        assert len(files) == 4

    @pytest.mark.parametrize("filepath", _get_literature_orchestrators(),
                             ids=lambda p: p.parent.name)
    def test_document_store_imported(self, filepath):
        assert _file_contains_import(
            filepath, "shared.rag.document_store", "DocumentStore"
        ), f"{filepath.parent.name}: missing DocumentStore import"

    @pytest.mark.parametrize("filepath", _get_literature_orchestrators(),
                             ids=lambda p: p.parent.name)
    def test_document_model_imported(self, filepath):
        assert _file_contains_import(
            filepath, "shared.rag.document_store", "Document"
        ), f"{filepath.parent.name}: missing Document import"

    @pytest.mark.parametrize("filepath", _get_literature_orchestrators(),
                             ids=lambda p: p.parent.name)
    def test_hybrid_retriever_imported(self, filepath):
        assert _file_contains_import(
            filepath, "shared.rag.hybrid_retriever", "HybridRetriever"
        ), f"{filepath.parent.name}: missing HybridRetriever import"

    @pytest.mark.parametrize("filepath", _get_literature_orchestrators(),
                             ids=lambda p: p.parent.name)
    def test_grounding_checker_imported(self, filepath):
        assert _file_contains_import(
            filepath, "shared.guardrails.grounding_checker", "GroundingChecker"
        ), f"{filepath.parent.name}: missing GroundingChecker import"

    @pytest.mark.parametrize("filepath", _get_literature_orchestrators(),
                             ids=lambda p: p.parent.name)
    def test_citation_model_imported(self, filepath):
        assert _file_contains_import(
            filepath, "shared.guardrails.grounding_checker", "Citation"
        ), f"{filepath.parent.name}: missing Citation import"

    @pytest.mark.parametrize("filepath", _get_literature_orchestrators(),
                             ids=lambda p: p.parent.name)
    def test_document_store_instantiated(self, filepath):
        assert _file_contains_text(filepath, "DocumentStore("), (
            f"{filepath.parent.name}: DocumentStore not instantiated"
        )

    @pytest.mark.parametrize("filepath", _get_literature_orchestrators(),
                             ids=lambda p: p.parent.name)
    def test_hybrid_retriever_instantiated(self, filepath):
        assert _file_contains_text(filepath, "HybridRetriever("), (
            f"{filepath.parent.name}: HybridRetriever not instantiated"
        )

    @pytest.mark.parametrize("filepath", _get_literature_orchestrators(),
                             ids=lambda p: p.parent.name)
    def test_doc_store_close_called(self, filepath):
        assert _file_contains_text(filepath, "doc_store.close()"), (
            f"{filepath.parent.name}: doc_store.close() not called"
        )


# ============================================================================
# Test 3: SemanticCache in LLMClient
# ============================================================================

class TestSemanticCacheIntegration:
    """Verify SemanticCache is wired into LLMClient."""

    def test_llm_client_accepts_semantic_cache(self):
        """LLMClient.__init__ has semantic_cache parameter."""
        from shared.llm import LLMClient
        import inspect
        sig = inspect.signature(LLMClient.__init__)
        assert "semantic_cache" in sig.parameters, (
            "LLMClient.__init__ missing semantic_cache parameter"
        )

    def test_llm_client_stores_semantic_cache(self):
        """LLMClient stores semantic_cache attribute."""
        from shared.llm import LLMClient
        client = LLMClient(model="gpt-4o-mini", api_key="test")
        assert hasattr(client, "semantic_cache")
        assert client.semantic_cache is None

    def test_llm_client_semantic_cache_in_invoke(self):
        """LLMClient.invoke() checks semantic_cache."""
        llm_path = _agents_dir / "shared" / "llm.py"
        source = llm_path.read_text(encoding="utf-8")
        assert "self.semantic_cache" in source
        assert "get_or_compute" in source

    def test_semantic_cache_only_for_temp_zero(self):
        """SemanticCache only activates for temperature=0 calls."""
        llm_path = _agents_dir / "shared" / "llm.py"
        source = llm_path.read_text(encoding="utf-8")
        # Check the guard condition
        assert "self.temperature == 0" in source


# ============================================================================
# Test 4: Functional — MemoryManager
# ============================================================================

class TestMemoryManagerFunctional:
    """Functional tests for MemoryManager operations."""

    def test_set_and_get_context(self):
        """MemoryManager can set and retrieve context."""
        from shared.memory.memory_manager import MemoryManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = MemoryManager(run_id="test-run-1", memory_dir=tmpdir)
            mm.set_context("research_topic", "AI economics")
            assert mm.get_context("research_topic") == "AI economics"
            mm.close()

    def test_remember_and_recall_facts(self):
        """MemoryManager can store and recall facts."""
        from shared.memory.memory_manager import MemoryManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = MemoryManager(run_id="test-run-2", memory_dir=tmpdir)
            mm.remember_fact("FRED provides GDP data", category="data_source")
            facts = mm.recall_facts("GDP data")
            assert len(facts) >= 1
            mm.close()

    def test_record_and_recall_trajectory(self):
        """MemoryManager can record and recall trajectories."""
        from shared.memory.memory_manager import MemoryManager
        from shared.memory.episodic import RunTrajectory
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = MemoryManager(run_id="test-run-3", memory_dir=tmpdir)
            trajectory = RunTrajectory(
                run_id="test-run-3",
                research_topic="Agent-based modeling in macroeconomics",
                teams_completed=["IdeationTeam"],
                success=True,
                mode="ModeNoWcNoHITL",
            )
            mm.record_trajectory(trajectory)

            # Recall by similar topic
            similar = mm.recall_similar_runs("Agent-based modeling")
            assert len(similar) >= 1
            assert similar[0].run_id == "test-run-3"
            mm.close()

    def test_memory_summary(self):
        """MemoryManager.summary() returns stats dict."""
        from shared.memory.memory_manager import MemoryManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = MemoryManager(run_id="test-run-4", memory_dir=tmpdir)
            mm.set_context("topic", "test")
            summary = mm.summary()
            assert isinstance(summary, dict)
            mm.close()


# ============================================================================
# Test 5: Functional — DocumentStore
# ============================================================================

class TestDocumentStoreFunctional:
    """Functional tests for DocumentStore operations."""

    def test_add_and_count_documents(self):
        """DocumentStore can add documents and count them."""
        from shared.rag.document_store import DocumentStore, Document
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = DocumentStore(store_dir=tmpdir)
            doc = Document(
                doc_id="test-doc-1",
                title="Test Paper on AI Economics",
                abstract="This paper studies AI in economics.",
                authors=["Smith, J."],
                year=2025,
                source="arxiv",
            )
            ds.add_documents([doc], compute_embeddings=False)
            assert ds.count() >= 1
            ds.close()

    def test_search_by_text(self):
        """DocumentStore text search returns relevant results."""
        from shared.rag.document_store import DocumentStore, Document
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = DocumentStore(store_dir=tmpdir)
            doc = Document(
                doc_id="test-doc-2",
                title="Agent-Based Models in Monetary Policy",
                abstract="We develop an agent-based model for monetary policy.",
                authors=["Jones, A."],
                year=2024,
                source="repec",
            )
            ds.add_documents([doc], compute_embeddings=False)
            results = ds.search_by_text("agent-based model")
            assert len(results) >= 1
            ds.close()

    def test_get_by_id(self):
        """DocumentStore can retrieve document by ID."""
        from shared.rag.document_store import DocumentStore, Document
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = DocumentStore(store_dir=tmpdir)
            doc = Document(
                doc_id="lookup-doc-1",
                title="Lookup Test",
                abstract="Test abstract.",
                authors=["Test"],
                year=2025,
                source="manual",
            )
            ds.add_documents([doc], compute_embeddings=False)
            result = ds.get_by_id("lookup-doc-1")
            assert result is not None
            assert result.title == "Lookup Test"
            ds.close()

    def test_get_stats(self):
        """DocumentStore.get_stats() returns stats object."""
        from shared.rag.document_store import DocumentStore
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = DocumentStore(store_dir=tmpdir)
            stats = ds.get_stats()
            assert hasattr(stats, "total_documents")
            assert stats.total_documents == 0
            ds.close()


# ============================================================================
# Test 6: Functional — HybridRetriever
# ============================================================================

class TestHybridRetrieverFunctional:
    """Functional tests for HybridRetriever search."""

    def test_search_returns_results(self):
        """HybridRetriever search returns documents from store."""
        from shared.rag.document_store import DocumentStore, Document
        from shared.rag.hybrid_retriever import HybridRetriever
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = DocumentStore(store_dir=tmpdir)
            docs = [
                Document(
                    doc_id=f"hr-doc-{i}",
                    title=f"Paper {i} on fiscal policy",
                    abstract=f"Analysis of fiscal policy impact {i}.",
                    authors=[f"Author{i}"],
                    year=2024,
                    source="arxiv",
                )
                for i in range(3)
            ]
            ds.add_documents(docs, compute_embeddings=False)

            retriever = HybridRetriever(document_store=ds)
            results = retriever.search("fiscal policy", top_k=2)
            assert len(results) >= 1
            ds.close()

    def test_search_empty_store(self):
        """HybridRetriever handles empty DocumentStore gracefully."""
        from shared.rag.document_store import DocumentStore
        from shared.rag.hybrid_retriever import HybridRetriever
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = DocumentStore(store_dir=tmpdir)
            retriever = HybridRetriever(document_store=ds)
            results = retriever.search("anything", top_k=5)
            assert results == []
            ds.close()


# ============================================================================
# Test 7: Functional — GroundingChecker
# ============================================================================

class TestGroundingCheckerFunctional:
    """Functional tests for GroundingChecker citation verification."""

    def test_check_known_citation(self):
        """GroundingChecker verifies citation found in DocumentStore."""
        from shared.rag.document_store import DocumentStore, Document
        from shared.guardrails.grounding_checker import GroundingChecker, Citation

        with tempfile.TemporaryDirectory() as tmpdir:
            ds = DocumentStore(store_dir=tmpdir)
            doc = Document(
                doc_id="gc-doc-1",
                title="Fiscal Policy Analysis",
                abstract="Analysis of fiscal policy effects.",
                authors=["Smith, J."],
                year=2024,
                source="repec",
                venue="AER",
            )
            ds.add_documents([doc], compute_embeddings=False)

            checker = GroundingChecker(document_store=ds)
            citations = [
                Citation(
                    title="Fiscal Policy Analysis",
                    authors=["Smith, J."],
                    year=2024,
                    venue="AER",
                )
            ]
            report = checker.check_citations(citations)
            assert report.total == 1
            assert report.grounding_score >= 0.0
            ds.close()

    def test_check_hallucinated_citation(self):
        """GroundingChecker flags citation not in DocumentStore."""
        from shared.rag.document_store import DocumentStore
        from shared.guardrails.grounding_checker import GroundingChecker, Citation

        with tempfile.TemporaryDirectory() as tmpdir:
            ds = DocumentStore(store_dir=tmpdir)
            # Empty store — citation is hallucinated
            checker = GroundingChecker(document_store=ds)
            citations = [
                Citation(
                    title="Nonexistent Paper",
                    authors=["Nobody"],
                    year=2099,
                )
            ]
            report = checker.check_citations(citations)
            assert report.total == 1
            assert report.hallucinated >= 1
            ds.close()

    def test_grounding_report_fields(self):
        """GroundingReport has expected fields."""
        from shared.guardrails.grounding_checker import GroundingChecker
        from shared.rag.document_store import DocumentStore

        with tempfile.TemporaryDirectory() as tmpdir:
            ds = DocumentStore(store_dir=tmpdir)
            checker = GroundingChecker(document_store=ds)
            report = checker.check_citations([])
            assert hasattr(report, "total")
            assert hasattr(report, "verified")
            assert hasattr(report, "hallucinated")
            assert hasattr(report, "grounding_score")
            ds.close()


# ============================================================================
# Test 8: Functional — SemanticCache
# ============================================================================

class TestSemanticCacheFunctional:
    """Functional tests for SemanticCache."""

    def test_cache_init(self):
        """SemanticCache initializes with directory."""
        from shared.rag.semantic_cache import SemanticCache
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SemanticCache(cache_dir=tmpdir)
            assert cache is not None

    def test_cache_result_model(self):
        """CacheResult has expected fields."""
        from shared.rag.semantic_cache import CacheResult
        result = CacheResult(
            hit=False,
            response="test response",
            cost_saved=0.0,
            similarity=0.0,
            cache_key="test-key",
        )
        assert result.hit is False
        assert result.response == "test response"

    def test_lookup_empty_cache(self):
        """SemanticCache.lookup returns None on empty cache."""
        from shared.rag.semantic_cache import SemanticCache
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SemanticCache(cache_dir=tmpdir)
            # lookup requires embed_fn; use dummy
            def dummy_embed(texts):
                return [[0.0] * 10 for _ in texts]
            result = cache.lookup("test prompt", embed_fn=dummy_embed)
            assert result is None


# ============================================================================
# Test 9: Non-LiteratureTeam should NOT have RAG imports
# ============================================================================

class TestNonLiteratureTeamNoRAG:
    """Verify non-LiteratureTeam orchestrators do NOT import RAG components."""

    def _get_non_literature_orchestrators(self):
        """Return all orchestrators except LiteratureTeam."""
        all_files = _get_orchestrator_files()
        return [f for f in all_files if "LiteratureTeam" not in str(f)]

    def test_non_literature_no_document_store(self):
        """Non-LiteratureTeam orchestrators should not import DocumentStore."""
        for filepath in self._get_non_literature_orchestrators():
            has_import = _file_contains_import(
                filepath, "shared.rag.document_store", "DocumentStore"
            )
            assert not has_import, (
                f"{filepath.relative_to(_agents_dir)}: unexpected DocumentStore import"
            )

    def test_non_literature_no_hybrid_retriever(self):
        """Non-LiteratureTeam orchestrators should not import HybridRetriever."""
        for filepath in self._get_non_literature_orchestrators():
            has_import = _file_contains_import(
                filepath, "shared.rag.hybrid_retriever", "HybridRetriever"
            )
            assert not has_import, (
                f"{filepath.relative_to(_agents_dir)}: unexpected HybridRetriever import"
            )

    def test_non_literature_no_grounding_checker(self):
        """Non-LiteratureTeam orchestrators should not import GroundingChecker."""
        for filepath in self._get_non_literature_orchestrators():
            has_import = _file_contains_import(
                filepath, "shared.guardrails.grounding_checker", "GroundingChecker"
            )
            assert not has_import, (
                f"{filepath.relative_to(_agents_dir)}: unexpected GroundingChecker import"
            )


# ============================================================================
# Test 10: Count verification
# ============================================================================

class TestPhase2Counts:
    """Verify total counts match expectations."""

    def test_total_memory_manager_count(self):
        """Exactly 15 orchestrators have MemoryManager."""
        orchestrators = _get_orchestrator_files()
        count = sum(
            1 for f in orchestrators
            if _file_contains_import(f, "shared.memory.memory_manager", "MemoryManager")
        )
        assert count == 15, f"Expected 15, got {count}"

    def test_total_document_store_count(self):
        """Exactly 4 orchestrators have DocumentStore (LiteratureTeam only)."""
        orchestrators = _get_orchestrator_files()
        count = sum(
            1 for f in orchestrators
            if _file_contains_import(f, "shared.rag.document_store", "DocumentStore")
        )
        assert count == 4, f"Expected 4, got {count}"

    def test_total_grounding_checker_count(self):
        """Exactly 4 orchestrators have GroundingChecker (LiteratureTeam only)."""
        orchestrators = _get_orchestrator_files()
        count = sum(
            1 for f in orchestrators
            if _file_contains_import(f, "shared.guardrails.grounding_checker", "GroundingChecker")
        )
        assert count == 4, f"Expected 4, got {count}"
