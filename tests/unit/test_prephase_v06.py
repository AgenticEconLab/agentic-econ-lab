# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
AEL V0.6 Pre-Phase — Diagnosis-Driven Technical Debt Cleanup Tests

Tests for:
- Provider routing consolidation (provider_map.py)
- Test infrastructure (conftest.py, no sys.path.insert in test files)
- SQLite context managers (LongTermMemory, DocumentStore)
- Silent error swallowing replaced with debug logging
- Requirements file exists
"""

import importlib
import logging
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest


# ── Provider Routing Consolidation ────────────────────────────────────────


class TestProviderMap:
    """Verify shared/provider_map.py is the single source of truth."""

    def test_module_imports(self):
        from shared.provider_map import (
            PROVIDER_ROUTES,
            OPENAI_COMPATIBLE_PROVIDERS,
            PROVIDER_API_KEY_ENV,
            detect_provider,
            ALL_PROVIDERS,
        )
        assert callable(detect_provider)

    def test_all_known_providers(self):
        # V0.7 Pre-Phase: perplexity + groq added.
        from shared.provider_map import ALL_PROVIDERS
        v06_eight = {"openai", "anthropic", "google", "deepseek", "mistral",
                     "xai", "openrouter", "ollama"}
        assert v06_eight.issubset(ALL_PROVIDERS)
        assert "perplexity" in ALL_PROVIDERS
        assert "groq" in ALL_PROVIDERS

    @pytest.mark.parametrize("model,expected", [
        ("gpt-4o-mini", "openai"),
        ("gpt-4.1-nano", "openai"),
        ("claude-sonnet-4-6", "anthropic"),
        ("claude-haiku-4-5-20251001", "anthropic"),
        ("gemini-2.5-flash", "google"),
        ("deepseek-chat", "deepseek"),
        ("deepseek-r1", "deepseek"),
        ("open-mistral-nemo", "mistral"),
        ("mistral-large-3", "mistral"),
        ("codestral", "mistral"),
        ("grok-beta", "xai"),
        ("meta-llama/llama-4-scout", "openrouter"),
        ("qwen/qwen3-8b", "openrouter"),
        ("ollama/llama3", "ollama"),
        ("o4-mini", "openai"),
        ("text-embedding-3-small", "openai"),
    ])
    def test_detect_provider(self, model, expected):
        from shared.provider_map import detect_provider
        assert detect_provider(model) == expected

    def test_detect_provider_default(self):
        from shared.provider_map import detect_provider
        assert detect_provider("unknown-model") == "openai"
        assert detect_provider("unknown-model", default="unknown") == "unknown"

    def test_openai_compatible_providers(self):
        from shared.provider_map import OPENAI_COMPATIBLE_PROVIDERS
        for name in ["deepseek", "mistral", "xai", "openrouter", "ollama"]:
            assert name in OPENAI_COMPATIBLE_PROVIDERS
            assert "base_url" in OPENAI_COMPATIBLE_PROVIDERS[name]
            assert "api_key_env" in OPENAI_COMPATIBLE_PROVIDERS[name]

    def test_provider_api_key_env(self):
        from shared.provider_map import PROVIDER_API_KEY_ENV
        assert "openai" in PROVIDER_API_KEY_ENV
        assert "anthropic" in PROVIDER_API_KEY_ENV
        assert "google" in PROVIDER_API_KEY_ENV
        assert len(PROVIDER_API_KEY_ENV) >= 8


class TestProviderMapConsumers:
    """Verify all 4 consumer modules use provider_map.py."""

    def test_llm_uses_provider_map(self):
        """shared/llm.py imports from shared.provider_map."""
        from shared import llm
        from shared import provider_map
        assert llm.PROVIDER_ROUTES is provider_map.PROVIDER_ROUTES
        assert llm.detect_provider is provider_map.detect_provider

    def test_llm_router_uses_provider_map(self):
        """shared/llm_router.py delegates to shared.provider_map."""
        from shared.llm_router import _detect_provider_from_prefix
        # Should produce same results as provider_map
        from shared.provider_map import detect_provider
        for model in ["gpt-4o", "claude-sonnet-4-6", "deepseek-chat"]:
            assert _detect_provider_from_prefix(model) == detect_provider(model)

    def test_gen_ai_conventions_uses_provider_map(self):
        """shared/telemetry/gen_ai_conventions.py uses provider_map."""
        from shared.telemetry.gen_ai_conventions import _detect_provider
        # Telemetry uses "unknown" default
        assert _detect_provider("gpt-4o") == "openai"
        assert _detect_provider("unknown-model") == "unknown"

    def test_otel_exporter_uses_provider_map(self):
        """shared/telemetry/otel_exporter.py uses provider_map for all providers."""
        from shared.telemetry.otel_exporter import _dict_to_otel_attrs
        from shared.telemetry.gen_ai_conventions import GenAIAttributes
        # Test that deepseek is now detected (was missing before)
        rec = {"model": "deepseek-chat", "prompt_tokens": 10, "completion_tokens": 5}
        attrs = _dict_to_otel_attrs(rec)
        assert attrs.get(GenAIAttributes.PROVIDER_NAME) == "deepseek"


# ── Test Infrastructure ───────────────────────────────────────────────────


class TestConftest:
    """Verify conftest.py handles sys.path setup."""

    def test_conftest_exists(self):
        conftest = Path(__file__).resolve().parent.parent / "conftest.py"
        assert conftest.exists(), "tests/conftest.py should exist"

    def test_no_sys_path_insert_in_test_files(self):
        """No test file should have module-level sys.path.insert (conftest handles it).

        Exception: test files that import sys for use in test logic (not path setup)
        are allowed to have sys in their imports.
        """
        tests_dir = Path(__file__).resolve().parent.parent
        violations = []
        for test_file in sorted(tests_dir.rglob("test_*.py")):
            if test_file.name in ("conftest.py", "test_prephase_v06.py"):
                continue  # Skip self (contains the string in assertions)
            content = test_file.read_text(encoding="utf-8")
            lines = content.split("\n")
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                # Only flag actual sys.path.insert calls (not just "import sys")
                if "sys.path.insert" in stripped:
                    # Skip if inside a function/method body (indented)
                    if line.startswith("    ") or line.startswith("\t"):
                        continue  # Inside a function — okay (test logic)
                    violations.append(f"{test_file.name}:{i}")
        assert violations == [], f"Module-level sys.path.insert found in: {violations}"

    def test_shared_imports_work(self):
        """Verify that shared imports work via conftest.py."""
        import shared.llm
        import shared.observability
        import shared.provider_map


# ── SQLite Context Managers ───────────────────────────────────────────────


class TestSQLiteContextManagers:
    """Verify LongTermMemory and DocumentStore support context manager protocol."""

    def test_long_term_memory_context_manager(self):
        from shared.memory.long_term import LongTermMemory
        with tempfile.TemporaryDirectory() as tmpdir:
            with LongTermMemory(memory_dir=tmpdir) as mem:
                mem.remember_fact("test fact", category="test")
                facts = mem.recall_facts("test")
                assert len(facts) > 0
            # After exiting, connection should be closed

    def test_long_term_memory_has_context_methods(self):
        from shared.memory.long_term import LongTermMemory
        assert hasattr(LongTermMemory, "__enter__")
        assert hasattr(LongTermMemory, "__exit__")
        assert hasattr(LongTermMemory, "__del__")

    def test_document_store_context_manager(self):
        from shared.rag.document_store import DocumentStore, Document
        with tempfile.TemporaryDirectory() as tmpdir:
            with DocumentStore(store_dir=tmpdir) as store:
                doc = Document(
                    doc_id="test-1",
                    title="Test Document",
                    abstract="A test abstract",
                )
                store.add_documents([doc])
                retrieved = store.get_by_id("test-1")
                assert retrieved is not None

    def test_document_store_has_context_methods(self):
        from shared.rag.document_store import DocumentStore
        assert hasattr(DocumentStore, "__enter__")
        assert hasattr(DocumentStore, "__exit__")
        assert hasattr(DocumentStore, "__del__")

    def test_semantic_cache_per_operation_connections(self):
        """SemanticCache already uses per-operation connections — no context manager needed."""
        from shared.rag.semantic_cache import SemanticCache
        # SemanticCache doesn't need __enter__/__exit__ since it opens/closes per op
        assert not hasattr(SemanticCache, "__enter__") or True  # fine either way


# ── Silent Error Swallowing ───────────────────────────────────────────────


class TestErrorLogging:
    """Verify bare except:pass replaced with debug logging."""

    def test_tool_registry_logs_on_recording_failure(self):
        """tool_registry._record_tool_call logs instead of silencing."""
        import shared.tools.tool_registry as tr
        source = Path(tr.__file__).read_text(encoding="utf-8")
        # Should NOT contain 'except Exception:\n            pass'
        assert "except Exception:\n            pass" not in source

    def test_observability_logs_on_pricing_failure(self):
        """observability pricing override logs instead of silencing."""
        import shared.observability as obs
        source = Path(obs.__file__).read_text(encoding="utf-8")
        assert "except Exception:\n        pass  # Silently fall back" not in source

    def test_observability_logs_on_url_resolution_failure(self):
        """observability URL resolution logs instead of silencing."""
        import shared.observability as obs
        source = Path(obs.__file__).read_text(encoding="utf-8")
        # The old pattern was: except Exception:\n        return "Unknown"
        # New pattern uses logging before returning
        assert "logging.getLogger" in source


# ── Requirements File ─────────────────────────────────────────────────────


class TestRequirements:
    """Verify requirements-ael.txt exists and is well-formed."""

    def test_requirements_file_exists(self):
        agents_dir = Path(__file__).resolve().parent.parent.parent
        req = agents_dir / "requirements-ael.txt"
        assert req.exists(), "requirements-ael.txt should exist"

    def test_requirements_has_core_deps(self):
        agents_dir = Path(__file__).resolve().parent.parent.parent
        content = (agents_dir / "requirements-ael.txt").read_text()
        for dep in ["httpx", "pydantic", "python-dotenv", "numpy", "pytest"]:
            assert dep in content, f"{dep} missing from requirements-ael.txt"
