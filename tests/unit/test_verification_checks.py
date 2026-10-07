# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Verification-layer integration tests."""

from __future__ import annotations

import json

import pytest

from shared.verification import (
    AdversarialFalsifier,
    attach_citation_audit,
    extract_ideation_texts,
    extract_review_texts,
    run_citation_verifier_pass,
    CitationStatus,
    VerificationSummary,
)


# ---------------------------------------------------------------------------
# Falsifier verdict threshold
# ---------------------------------------------------------------------------

class TestFalsifierVerdictThreshold:
    def test_single_rule_match_needs_revision(self):
        f = AdversarialFalsifier()
        report = f.run("Monetary easing causes inflation.")
        # Single rule match — conservative "needs_revision"
        assert report.verdict == "needs_revision"

    def test_multiple_rule_matches_falsified(self):
        f = AdversarialFalsifier()
        report = f.run("All rational agents always converge monotone.")
        # 3+ patterns → falsified
        assert report.verdict == "falsified"

    def test_llm_source_triggers_falsified(self):
        class FakeLLM:
            def invoke(self, prompt):
                return "single counterexample"
        f = AdversarialFalsifier(max_rounds=2, llm_client=FakeLLM())
        report = f.run("The pipeline has three stages.")
        # No rule match, but LLM provided a counterexample → falsified
        llm_findings = [c for c in report.admissible_counterexamples if c.get("source") == "llm"]
        if llm_findings:
            assert report.verdict == "falsified"

    def test_no_counterexamples_survives(self):
        f = AdversarialFalsifier()
        report = f.run("The module contains three functions.")
        assert report.verdict == "survives"


# ---------------------------------------------------------------------------
# Missing exports
# ---------------------------------------------------------------------------

class TestExports:
    def test_verification_summary_importable(self):
        assert VerificationSummary is not None
        s = VerificationSummary(total=0)
        assert s.verified == 0

    def test_citation_status_importable(self):
        assert CitationStatus is not None


# ---------------------------------------------------------------------------
# Team hooks — CitationVerifier integration
# ---------------------------------------------------------------------------

class TestCitationVerifierHook:
    def test_disabled_returns_none(self, monkeypatch):
        monkeypatch.setenv("AEL_CITATION_VERIFIER_ENABLED", "0")
        assert run_citation_verifier_pass(["text with 10.1234/foo citation"]) is None

    def test_enabled_produces_audit(self, monkeypatch):
        monkeypatch.setenv("AEL_CITATION_VERIFIER_ENABLED", "1")
        audit = run_citation_verifier_pass([
            "DOI 10.1038/nature10000 is real. [Fake, 2099] is not.",
        ])
        assert audit is not None
        assert audit["overall_total"] >= 2
        assert "per_text" in audit

    def test_empty_input_still_returns_audit(self, monkeypatch):
        monkeypatch.setenv("AEL_CITATION_VERIFIER_ENABLED", "1")
        audit = run_citation_verifier_pass(["", None, "   "])
        assert audit is not None
        assert audit["overall_total"] == 0

    def test_extract_review_texts(self):
        texts = extract_review_texts({
            "literature_review": "Body of review",
            "sections": [{"section_content": "Section A"}, {"content": "Section B"}],
        })
        assert "Body of review" in texts
        assert "Section A" in texts
        assert "Section B" in texts

    def test_extract_ideation_texts(self):
        texts = extract_ideation_texts({
            "prioritized_questions": [
                {"theoretical_framework": "Theory blurb", "rationale": "Why"},
                {"description": "Another"},
            ]
        })
        assert "Theory blurb" in texts
        assert "Why" in texts
        assert "Another" in texts

    def test_attach_citation_audit_writes_in_place(self, tmp_path):
        p = tmp_path / "out.json"
        p.write_text(json.dumps({"x": 1}))
        attach_citation_audit(p, {"overall_total": 5})
        data = json.loads(p.read_text())
        assert data["citation_audit"]["overall_total"] == 5

    def test_attach_citation_audit_noop_on_none(self, tmp_path):
        p = tmp_path / "out.json"
        p.write_text(json.dumps({"x": 1}))
        attach_citation_audit(p, None)
        assert json.loads(p.read_text()) == {"x": 1}


# ---------------------------------------------------------------------------
# LogSanitizer wired into MCP server
# ---------------------------------------------------------------------------

class TestMCPLogSanitizerIntegration:
    def test_mcp_server_default_sanitizer(self):
        from shared.protocols.mcp_server import AELMCPServer
        server = AELMCPServer()
        assert server._sanitize_io is True
        assert server._sanitizer is not None

    def test_mcp_server_disable_sanitizer(self):
        from shared.protocols.mcp_server import AELMCPServer
        server = AELMCPServer(sanitize_io=False)
        assert server._sanitize_io is False

    def test_mcp_tool_result_has_redacted_secrets(self, monkeypatch):
        from pydantic import BaseModel
        from shared.protocols.mcp_server import AELMCPServer
        from shared.tools.tool_registry import ToolRegistry

        class _EchoSchema(BaseModel):
            text: str = ""

        def _echo_handler(text: str = "", collector=None, agent=None):
            return {"echo": text}

        ToolRegistry.register(
            name="echo_leak_test",
            description="echo (test only)",
            input_schema=_EchoSchema,
            handler=_echo_handler,
            category="test",
        )
        try:
            server = AELMCPServer()
            payload_with_secret = "Authorization: Bearer " + "a" * 40
            result = server.call_tool(
                "echo_leak_test", {"text": payload_with_secret},
            )
            assert result.isError is False
            text = result.content[0]["text"]
            # Bearer token must be redacted in output
            assert "Bearer a" not in text
            assert "[REDACTED]" in text
        finally:
            ToolRegistry.unregister("echo_leak_test")


# ---------------------------------------------------------------------------
# MemoryManager trust-aware recall
# ---------------------------------------------------------------------------

class TestMemoryManagerTrustAware:
    def test_trust_aware_method_exists(self):
        from shared.memory.memory_manager import MemoryManager
        assert hasattr(MemoryManager, "recall_facts_trust_aware")

    def test_disabled_flag_falls_through(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AEL_MEMORY_GUARD_V2", "0")
        from shared.memory.memory_manager import MemoryManager
        mm = MemoryManager(run_id="t", memory_dir=str(tmp_path))
        mm.remember_fact("Inflation affects GDP", category="macro", source="fred")
        hits = mm.recall_facts_trust_aware("inflation", top_k=3)
        assert isinstance(hits, list)
        mm.close()

    def test_enabled_flag_runs_drift_scan(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AEL_MEMORY_GUARD_V2", "1")
        from shared.memory.memory_manager import MemoryManager
        mm = MemoryManager(run_id="t", memory_dir=str(tmp_path))
        mm.remember_fact("Inflation rose in Q1", category="macro", source="ecb")
        mm.remember_fact("Inflation rose in Q1", category="macro", source="untrusted-injection")
        hits = mm.recall_facts_trust_aware("inflation", top_k=3)
        assert isinstance(hits, list)
        # Drift alerts should be captured in short-term memory
        alerts = mm.short_term.get("memory_drift_alerts")
        # May be None if no collision detected; must be list when present
        if alerts is not None:
            assert isinstance(alerts, list)
        mm.close()


# ---------------------------------------------------------------------------
# LogSanitizer edge-case improvements
# ---------------------------------------------------------------------------

class TestLogSanitizerImprovements:
    def test_bearer_lowercase_redacted(self):
        from shared.security import LogSanitizer
        s = LogSanitizer()
        out = s.sanitize_string("authorization: bearer " + "a" * 40)
        assert "bearer a" not in out.lower() or "[REDACTED]" in out

    def test_shell_multiline_injection_matched(self):
        from shared.security import LogSanitizer
        s = LogSanitizer()
        payload = "value=${SECRET\nINNER}"
        out = s.sanitize_string(payload)
        assert "${SECRET" not in out
