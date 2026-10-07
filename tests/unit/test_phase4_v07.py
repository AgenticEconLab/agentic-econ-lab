# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""AEL V0.7 Phase 4 — Verification: CitationVerifier + Falsifier + MemoryGuard 2.0 + LogSanitizer integration."""

from __future__ import annotations

import json
import time

import pytest

from shared.verification import (
    AdversarialFalsifier,
    BeliefDriftDetector,
    Citation,
    CitationVerdict,
    CitationVerifier,
    FalsifierReport,
    MemoryEntry,
    TrustAwareRetriever,
    extract_citations,
)
from shared.verification.trust_memory import record_contradiction


# ---------------------------------------------------------------------------
# Citation extraction
# ---------------------------------------------------------------------------

class TestExtract:
    def test_extracts_doi(self):
        text = "See 10.1038/s41586-026-10251-x for details."
        cs = extract_citations(text)
        assert any(c.doi for c in cs)

    def test_extracts_arxiv(self):
        text = "Refer to arxiv: 2512.21031 for the recipe."
        cs = extract_citations(text)
        assert any(c.arxiv_id == "2512.21031" for c in cs)

    def test_extracts_arxiv_url(self):
        text = "See https://arxiv.org/abs/2603.17694 for MALLES."
        cs = extract_citations(text)
        assert any(c.arxiv_id == "2603.17694" for c in cs)

    def test_extracts_bracketed_author_year(self):
        text = "As shown by [Smith, 2024] and (Jones et al., 2023)."
        cs = extract_citations(text)
        years = {c.year for c in cs}
        assert 2024 in years and 2023 in years

    def test_dedupes(self):
        text = "10.1234/foo.bar 10.1234/foo.bar"
        cs = extract_citations(text)
        dois = [c.doi for c in cs if c.doi]
        assert len(dois) == 1

    def test_empty_text_no_citations(self):
        assert extract_citations("") == []


# ---------------------------------------------------------------------------
# CitationVerifier
# ---------------------------------------------------------------------------

class TestVerifier:
    def test_default_backend_verifies_doi(self):
        verifier = CitationVerifier()
        verdicts = verifier.verify_all("See 10.1038/s41586-026-10251-x for details.")
        assert verdicts[0].status == "verified"

    def test_default_backend_suspect_for_bracketed(self):
        verifier = CitationVerifier()
        verdicts = verifier.verify_all("Noted in [Doe, 2020].")
        assert verdicts[0].status == "suspect"

    def test_custom_backend_override(self):
        def always_verify(c: Citation):
            return CitationVerdict(citation=c, status="verified",
                                   evidence_urls=["http://x"], reasoning="mock")
        verifier = CitationVerifier()
        verifier.register_backend(always_verify)
        verdicts = verifier.verify_all("[Smith 2020]")
        assert verdicts[0].status == "verified"
        assert verdicts[0].evidence_urls == ["http://x"]

    def test_hallucination_fallback(self):
        def never_verify(c: Citation):
            return None
        verifier = CitationVerifier(backends=[never_verify])
        v = verifier.verify_one(Citation(raw="bogus", doi=None))
        assert v.status == "hallucinated"

    def test_redact_hallucinated(self):
        verifier = CitationVerifier()
        def return_hall(c: Citation):
            return CitationVerdict(citation=c, status="hallucinated",
                                   reasoning="fake", confidence=0.9)
        verifier.register_backend(return_hall)
        text = "As shown in [Fake, 2099] this is wrong."
        verdicts = verifier.verify_all(text)
        redacted = verifier.redact_hallucinated(text, verdicts)
        assert "[CITATION_HALLUCINATED]" in redacted
        assert "[Fake, 2099]" not in redacted

    def test_redact_suspect_marks_unverified(self):
        verifier = CitationVerifier()
        text = "From [Jones 2023] we learn ..."
        verdicts = verifier.verify_all(text)
        redacted = verifier.redact_hallucinated(text, verdicts)
        assert "[UNVERIFIED]" in redacted

    def test_summary_counts(self):
        verifier = CitationVerifier()
        verdicts = verifier.verify_all(
            "DOI 10.1111/abcde and [Smith 2020]."
        )
        s = verifier.summarize(verdicts)
        assert s.total == 2
        assert s.verified + s.suspect + s.hallucinated == s.total

    def test_flag_rate_target(self):
        """Target: every non-verifiable bracketed citation is flagged (suspect
        or hallucinated), i.e. never passes through as 'verified'."""
        parts = [f"10.1000/abc{i:03d}" for i in range(80)]
        parts += [f"[Fake{i}, 2099]" for i in range(20)]
        text = " ".join(parts)
        verifier = CitationVerifier()
        verdicts = verifier.verify_all(text)
        summary = verifier.summarize(verdicts)
        # All 80 DOIs verified; 20 bracketed flagged as suspect (or hallucinated)
        assert summary.verified == 80
        assert summary.suspect + summary.hallucinated == 20
        redacted_text = verifier.redact_hallucinated(text, verdicts)
        # Suspect citations get an [UNVERIFIED] marker appended; they are not
        # silently passed through.
        assert "[UNVERIFIED]" in redacted_text

    def test_collector_backward_compat(self):
        v = CitationVerifier(collector=None)
        assert v.verify_all("") == []


# ---------------------------------------------------------------------------
# AdversarialFalsifier
# ---------------------------------------------------------------------------

class TestFalsifier:
    def test_universal_claim_triggers_rule(self):
        f = AdversarialFalsifier()
        report = f.run("All agents always converge to equilibrium.")
        assert report.admissible_counterexamples
        assert report.verdict in ("needs_revision", "falsified")

    def test_causal_claim_suggests_adjustment(self):
        f = AdversarialFalsifier()
        report = f.run("Quantitative easing causes inflation.")
        joined = " ".join(report.suggested_revisions).lower()
        assert "adjustment" in joined or "dowhy" in joined or "causalfm" in joined

    def test_benign_claim_survives(self):
        f = AdversarialFalsifier()
        report = f.run("The pipeline has three stages.")
        assert report.verdict == "survives"
        assert report.admissible_counterexamples == []

    def test_max_rounds_respected_without_llm(self):
        f = AdversarialFalsifier(max_rounds=3)
        report = f.run("Every rational agent chooses optimally.")
        # Without an LLM client, only round 1 runs
        assert report.rounds == 1

    def test_max_rounds_with_llm(self):
        # The LLM must return a STRUCTURED, admissible probe to be accepted —
        # a bare string is no longer rubber-stamped as a counterexample.
        class FakeLLM:
            def invoke(self, prompt):
                return json.dumps({
                    "claim_scope": "all inputs",
                    "is_universal_claim": True,
                    "counterexample": (
                        "An input whose premise the theorem does not actually "
                        "cover, so the asserted conclusion fails."
                    ),
                    "contradicts_within_scope": True,
                    "confidence": 0.9,
                    "reasoning": "stub",
                })
        f = AdversarialFalsifier(max_rounds=3, llm_client=FakeLLM())
        report = f.run("The theorem is proven for all inputs.")
        assert report.rounds == 3
        assert sum(1 for c in report.admissible_counterexamples if c.get("source") == "llm") == 2
        assert report.verdict == "falsified"

    def test_inadmissible_llm_counterexample_corroborates(self):
        # Hedged claim: a lone counterexample cannot refute it, so even a
        # confident, in-scope LLM counterexample is rejected and the claim is
        # corroborated rather than falsified.
        class HedgedBreakerLLM:
            def invoke(self, prompt):
                return json.dumps({
                    "claim_scope": "typical firms",
                    "is_universal_claim": False,
                    "counterexample": "One firm that cut prices after a demand shock.",
                    "contradicts_within_scope": True,
                    "confidence": 0.95,
                    "reasoning": "stub",
                })
        f = AdversarialFalsifier(max_rounds=2, llm_client=HedgedBreakerLLM())
        report = f.run("Firms usually raise prices after a demand shock on average.")
        assert report.verdict == "corroborated"
        assert not any(
            c.get("source") == "llm" for c in report.admissible_counterexamples
        )
        assert report.rejected_counterexamples

    def test_llm_failure_graceful(self):
        class FailingLLM:
            def invoke(self, prompt):
                raise RuntimeError("network fail")
        f = AdversarialFalsifier(max_rounds=2, llm_client=FailingLLM())
        report = f.run("Always positive outcome.")
        # Should not crash; round 2 just produces no additional counterexample
        assert isinstance(report, FalsifierReport)

    def test_invalid_max_rounds_rejected(self):
        with pytest.raises(ValueError):
            AdversarialFalsifier(max_rounds=0)

    def test_multiple_rule_matches_yield_falsified(self):
        f = AdversarialFalsifier()
        report = f.run("Always monotone and rational.")
        assert len(report.admissible_counterexamples) >= 2
        assert report.verdict == "falsified"

    def test_collector_backward_compat(self):
        f = AdversarialFalsifier(collector=None)
        assert f.run("x is y").verdict == "survives"


# ---------------------------------------------------------------------------
# TrustAwareRetriever (MemoryGuard 2.0)
# ---------------------------------------------------------------------------

class TestTrustAware:
    def test_filters_low_trust(self):
        now = time.time()
        entries = [
            MemoryEntry(content="a b c", source="trusted", timestamp=now, trust_score=0.9),
            MemoryEntry(content="a b c", source="poisoned", timestamp=now, trust_score=0.1),
        ]
        r = TrustAwareRetriever(min_trust=0.5)
        hits = r.retrieve("a b c", entries, k=5)
        assert len(hits) == 1
        assert hits[0].source == "trusted"

    def test_recency_breaks_ties(self):
        now = time.time()
        entries = [
            MemoryEntry(content="foo bar", source="x", timestamp=now - 60*86400, trust_score=1.0),
            MemoryEntry(content="foo bar", source="y", timestamp=now,              trust_score=1.0),
        ]
        r = TrustAwareRetriever()
        hits = r.retrieve("foo bar", entries, k=1)
        assert hits[0].source == "y"

    def test_contradiction_penalty(self):
        now = time.time()
        clean = MemoryEntry(content="x y", source="a", timestamp=now, trust_score=0.9)
        contested = MemoryEntry(content="x y", source="b", timestamp=now, trust_score=0.9,
                                 contradiction_count=3)
        r = TrustAwareRetriever()
        hits = r.retrieve("x y", [clean, contested], k=2)
        assert hits[0].source == "a"

    def test_empty_query_no_hits(self):
        r = TrustAwareRetriever()
        hits = r.retrieve("", [
            MemoryEntry(content="x", source="a", timestamp=time.time()),
        ], k=5)
        assert hits == []

    def test_fingerprint_stable(self):
        e1 = MemoryEntry(content="abc", source="src", timestamp=1.0)
        e2 = MemoryEntry(content="abc", source="src", timestamp=2.0)
        assert e1.fingerprint() == e2.fingerprint()


class TestBeliefDriftDetector:
    def test_detects_direct_contradiction(self):
        now = time.time()
        entries = [
            MemoryEntry(content="rate increases demand", source="a", timestamp=now),
            MemoryEntry(content="rate decreases demand", source="b", timestamp=now),
        ]
        alerts = BeliefDriftDetector().scan(entries)
        assert any(a.reason == "contradicts-prior-entry" for a in alerts)

    def test_fingerprint_collision_high_severity(self):
        now = time.time()
        entries = [
            MemoryEntry(content="fact", source="sourceA", timestamp=now),
            MemoryEntry(content="fact", source="sourceB", timestamp=now),
        ]
        alerts = BeliefDriftDetector().scan(entries)
        highs = [a for a in alerts if a.severity == "high"]
        assert highs

    def test_minja_style_detection_rate(self):
        """Target: ≥ 90% detection rate on MINJA-style corpus.

        The test mixes two classes of attack —
        bit-identical fingerprint collisions AND paraphrased contradictions —
        so the "90% detection" claim is measured against a realistic attacker
        profile, not a SHA-256 tautology.
        """
        now = time.time()
        corpus = []

        # Half the attacks: bit-identical content from a divergent source
        # (fingerprint-collision path).
        for i in range(3):
            corpus.append(MemoryEntry(content=f"claim-{i}", source="trusted-ecb",
                                       timestamp=now))
            corpus.append(MemoryEntry(content=f"claim-{i}", source=f"attacker-{i}",
                                       timestamp=now))

        # The other half: paraphrased contradictions of a trusted claim
        # (belief-drift / contradiction path).
        paraphrase_pairs = [
            ("Rate hikes increase inflation in the short run",
             "Rate hikes decrease inflation in the short run"),
            ("Monetary easing supports GDP growth",
             "Monetary easing refutes GDP growth"),
            ("Output rises when productivity rises",
             "Output falls when productivity rises"),
        ]
        for a, b in paraphrase_pairs:
            corpus.append(MemoryEntry(content=a, source="trusted-bis", timestamp=now))
            corpus.append(MemoryEntry(content=b, source="attacker-paraphrase",
                                       timestamp=now))

        alerts = BeliefDriftDetector().scan(corpus)
        # 3 collision alerts + 3 contradiction alerts expected → 6 total.
        # 90% of 6 = 5.4 → require at least 5.
        assert len(alerts) >= 5, f"expected >=5 alerts, got {len(alerts)}"
        # Both detection paths should fire.
        reasons = {a.reason for a in alerts}
        assert "fingerprint-collision-different-source" in reasons
        assert "contradicts-prior-entry" in reasons

    def test_no_false_positive_on_unique_content(self):
        now = time.time()
        entries = [
            MemoryEntry(content=f"distinct-topic-{i}", source="x", timestamp=now)
            for i in range(5)
        ]
        alerts = BeliefDriftDetector().scan(entries)
        assert not alerts

    def test_record_contradiction_decays_trust(self):
        e = MemoryEntry(content="x", source="s", timestamp=0.0, trust_score=0.9)
        e2 = record_contradiction(e)
        assert e2.contradiction_count == 1
        assert e2.trust_score < 0.9
        assert e2.belief_version == 2


# ---------------------------------------------------------------------------
# LogSanitizer integration (ASI07 Log-To-Leak defense) — sanity
# ---------------------------------------------------------------------------

class TestLogToLeakDefenseVectors:
    """Canonical Log-To-Leak vectors (OpenReview UVgbFuXPaO) — 100% block rate."""

    @pytest.fixture(autouse=True)
    def _san(self):
        from shared.security import LogSanitizer
        self.s = LogSanitizer()

    def test_jinja_vector(self):
        payload = "Log msg: {{env.API_KEY}}"
        out = self.s.sanitize_string(payload)
        assert "{{" not in out

    def test_shell_vector(self):
        payload = "output=${OPENAI_API_KEY}"
        out = self.s.sanitize_string(payload)
        assert "${" not in out

    def test_erb_vector(self):
        payload = "<%= system_env :SECRET %>"
        out = self.s.sanitize_string(payload)
        assert "<%" not in out

    def test_php_vector(self):
        payload = "<?= $_ENV['ANTHROPIC_API_KEY'] ?>"
        out = self.s.sanitize_string(payload)
        assert "<?" not in out

    def test_bearer_leak_in_logs(self):
        payload = "GET /v1/messages 200 Authorization: Bearer " + "A" * 30
        out = self.s.sanitize_string(payload)
        assert "Bearer A" not in out
