# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Unit tests for shared/verification/backends.py — the real Crossref/arXiv/
OpenAlex/Semantic Scholar CitationVerifier backends. Mocked network
throughout (fast, deterministic); the one-off live-network check that these
same backends resolve a real DOI, a real arXiv id, and correctly fall a
fabricated citation through to `hallucinated` is recorded in the commit that
wired them into team_hooks.py, not re-run here on every CI pass.
"""
from types import SimpleNamespace

import shared.observability as obs
from shared.verification.backends import (
    ArxivIdBackend,
    CrossrefBackend,
    OpenAlexBackend,
    SemanticScholarBackend,
    default_production_backends,
)
from shared.verification.citation_verifier import Citation, CitationVerifier


def _resp(status_code=200, json_data=None, text=""):
    return SimpleNamespace(
        status_code=status_code,
        json=lambda: json_data if json_data is not None else {},
        text=text,
    )


class TestCrossrefBackend:
    def test_not_applicable_without_doi(self, monkeypatch):
        monkeypatch.setattr(obs, "tracked_get", lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("should not call network without a DOI")))
        out = CrossrefBackend()(Citation(raw="(Smith, 2024)"))
        assert out is None

    def test_verified_on_real_doi(self, monkeypatch):
        monkeypatch.setattr(obs, "tracked_get", lambda *a, **k: _resp(
            200, {"message": {"title": ["Deep learning"]}}))
        out = CrossrefBackend()(Citation(raw="10.1038/nature14539", doi="10.1038/nature14539"))
        assert out is not None and out.status == "verified"
        assert "doi.org/10.1038/nature14539" in out.evidence_urls[0]

    def test_defers_on_404(self, monkeypatch):
        """A miss must return None (defer), never a negative verdict of its own."""
        monkeypatch.setattr(obs, "tracked_get", lambda *a, **k: _resp(404))
        out = CrossrefBackend()(Citation(raw="10.9999/fake", doi="10.9999/fake"))
        assert out is None

    def test_defers_on_network_error(self, monkeypatch):
        def boom(*a, **k):
            raise ConnectionError("no route")
        monkeypatch.setattr(obs, "tracked_get", boom)
        out = CrossrefBackend()(Citation(raw="10.1/x", doi="10.1/x"))
        assert out is None


class TestArxivIdBackend:
    def test_not_applicable_without_arxiv_id(self):
        out = ArxivIdBackend()(Citation(raw="(Smith, 2024)"))
        assert out is None

    def test_verified_on_real_id(self, monkeypatch):
        atom = ('<feed xmlns="http://www.w3.org/2005/Atom">'
                '<entry><title>Welfare Analysis in Dynamic Models</title></entry></feed>')
        monkeypatch.setattr(obs, "tracked_get", lambda *a, **k: _resp(200, text=atom))
        out = ArxivIdBackend()(Citation(raw="arxiv:1908.09173", arxiv_id="1908.09173"))
        assert out is not None and out.status == "verified"
        assert "1908.09173" in out.evidence_urls[0]

    def test_defers_on_arxiv_error_entry(self, monkeypatch):
        """arXiv's own not-found shape: a single entry titled 'Error'."""
        atom = ('<feed xmlns="http://www.w3.org/2005/Atom">'
                '<entry><title>Error</title></entry></feed>')
        monkeypatch.setattr(obs, "tracked_get", lambda *a, **k: _resp(200, text=atom))
        out = ArxivIdBackend()(Citation(raw="arxiv:9999.99999", arxiv_id="9999.99999"))
        assert out is None

    def test_defers_on_no_entry(self, monkeypatch):
        atom = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
        monkeypatch.setattr(obs, "tracked_get", lambda *a, **k: _resp(200, text=atom))
        out = ArxivIdBackend()(Citation(raw="arxiv:0000.00000", arxiv_id="0000.00000"))
        assert out is None


class TestOpenAlexBackend:
    def test_doi_hit_short_circuits_search(self, monkeypatch):
        monkeypatch.setattr(obs, "tracked_get", lambda *a, **k: _resp(
            200, {"id": "https://openalex.org/W123", "title": "Deep learning"}))
        import shared.tools.openalex_tool as oat
        monkeypatch.setattr(oat, "openalex_search_handler", lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("should not fall through to search after a DOI hit")))
        out = OpenAlexBackend()(Citation(raw="10.1038/nature14539", doi="10.1038/nature14539"))
        assert out is not None and out.status == "verified"

    def test_falls_through_to_search_on_doi_miss(self, monkeypatch):
        monkeypatch.setattr(obs, "tracked_get", lambda *a, **k: _resp(404))
        import shared.tools.openalex_tool as oat
        monkeypatch.setattr(oat, "openalex_search_handler", lambda **k: [
            {"title": "Some Paper", "year": 2024, "authors": ["Jane Smith"], "url": "https://x"}])
        out = OpenAlexBackend()(Citation(raw="(Smith, 2024)", doi="10.9999/miss", year=2024))
        assert out is not None and out.status == "verified"
        assert out.confidence < 0.97  # heuristic match, weaker than an exact id hit

    def test_search_requires_year_and_author_match(self, monkeypatch):
        import shared.tools.openalex_tool as oat
        monkeypatch.setattr(oat, "openalex_search_handler", lambda **k: [
            {"title": "Unrelated Paper", "year": 2010, "authors": ["Someone Else"], "url": "https://x"}])
        out = OpenAlexBackend()(Citation(raw="(Smith, 2024)", year=2024))
        assert out is None  # year mismatch on the only hit -> defer, not hallucinated

    def test_defers_when_search_empty(self, monkeypatch):
        import shared.tools.openalex_tool as oat
        monkeypatch.setattr(oat, "openalex_search_handler", lambda **k: [])
        out = OpenAlexBackend()(Citation(raw="(Fakeauthor, 2099)", year=2099))
        assert out is None


class TestSemanticScholarBackend:
    def test_verified_on_match(self, monkeypatch):
        monkeypatch.setattr(obs, "tracked_get", lambda *a, **k: _resp(200, {
            "data": [{"title": "Welfare Analysis", "year": 2019,
                      "authors": [{"name": "Victor Chernozhukov"}],
                      "externalIds": {"DOI": "10.48550/arXiv.1908.09173"}}]}))
        out = SemanticScholarBackend()(Citation(raw="(Chernozhukov et al., 2019)", year=2019))
        assert out is not None and out.status == "verified"

    def test_defers_on_429(self, monkeypatch):
        """A rate limit must defer, not fail the whole citation as hallucinated."""
        monkeypatch.setattr(obs, "tracked_get", lambda *a, **k: _resp(429))
        out = SemanticScholarBackend()(Citation(raw="(Someone, 2024)", year=2024))
        assert out is None

    def test_defers_on_empty_query(self):
        out = SemanticScholarBackend()(Citation(raw=""))
        assert out is None


class TestChainIntegration:
    """The properties team_hooks.py actually relies on: order, deferral, and
    that 'hallucinated' is only ever the chain's own exhaustion verdict."""

    def test_default_order_is_crossref_arxiv_openalex_semanticscholar(self):
        chain = default_production_backends()
        assert [b.name for b in chain] == ["crossref", "arxiv", "openalex", "semantic_scholar"]

    def test_earlier_backend_miss_does_not_block_later_backends(self, monkeypatch):
        """Regression guard for the exact bug this module fixes: one backend's
        coverage gap must not brand a real citation as hallucinated."""
        monkeypatch.setattr(obs, "tracked_get", lambda *a, **k: _resp(404))  # Crossref/arXiv miss
        import shared.tools.openalex_tool as oat
        monkeypatch.setattr(oat, "openalex_search_handler", lambda **k: [
            {"title": "Real Paper", "year": 2024, "authors": ["Jane Smith"], "url": "https://x"}])
        verifier = CitationVerifier(backends=default_production_backends())
        v = verifier.verify_one(Citation(raw="(Smith, 2024)", doi="10.1/not-in-crossref", year=2024))
        assert v.status == "verified"  # OpenAlex still got a chance despite Crossref's miss

    def test_full_exhaustion_yields_hallucinated_not_a_backend_verdict(self, monkeypatch):
        monkeypatch.setattr(obs, "tracked_get", lambda *a, **k: _resp(404))
        import shared.tools.openalex_tool as oat
        monkeypatch.setattr(oat, "openalex_search_handler", lambda **k: [])
        verifier = CitationVerifier(backends=default_production_backends())
        v = verifier.verify_one(Citation(raw="(Fakeauthor et al., 2099)", year=2099))
        assert v.status == "hallucinated"
        assert v.reasoning == "no backend could verify"  # the chain's own fallback, not a backend's
