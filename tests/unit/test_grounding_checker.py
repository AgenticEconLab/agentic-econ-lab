# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Tests for shared.guardrails.grounding_checker.GroundingChecker.

Validates citation verification against a document store.
"""

import pytest

from shared.rag.document_store import DocumentStore, Document
from shared.guardrails.grounding_checker import (
    GroundingChecker,
    Citation,
    CitationCheck,
    GroundingReport,
)


@pytest.fixture
def store(tmp_path):
    """DocumentStore pre-loaded with sample papers."""
    s = DocumentStore(store_dir=str(tmp_path))
    s.add_documents([
        Document(
            doc_id="arxiv:2301.01234",
            title="AI and Labor Markets",
            abstract="This paper studies AI impact on employment.",
            authors=["Smith, John", "Doe, Jane"],
            year=2023,
            venue="American Economic Review",
        ),
        Document(
            doc_id="arxiv:2302.05678",
            title="Fiscal Policy and GDP Growth",
            abstract="Analysis of fiscal stimulus effects.",
            authors=["Johnson, Bob"],
            year=2024,
            venue="Quarterly Journal of Economics",
        ),
    ], compute_embeddings=False)
    yield s
    s.close()


@pytest.fixture
def checker(store):
    return GroundingChecker(document_store=store)


class TestCitationVerification:
    """Test citation existence and accuracy checks."""

    def test_verified_citation(self, checker):
        citation = Citation(
            title="AI and Labor Markets",
            authors=["Smith, John"],
            year=2023,
            venue="American Economic Review",
        )
        report = checker.check_citations([citation])
        assert report.total == 1
        assert report.verified == 1
        assert report.hallucinated == 0
        assert report.grounding_score == 1.0

    def test_hallucinated_citation(self, checker):
        citation = Citation(
            title="Completely Made Up Paper That Does Not Exist",
            authors=["Nobody, No One"],
            year=2099,
        )
        report = checker.check_citations([citation])
        assert report.total == 1
        assert report.hallucinated == 1
        assert report.verified == 0
        assert report.grounding_score == 0.0

    def test_inaccurate_citation_wrong_year(self, checker):
        citation = Citation(
            title="AI and Labor Markets",
            authors=["Smith, John"],
            year=2020,  # Wrong year (actual: 2023)
        )
        report = checker.check_citations([citation])
        assert report.total == 1
        assert report.inaccurate == 1
        assert report.verified == 0
        assert len(report.checks[0].issues) > 0

    def test_inaccurate_citation_wrong_authors(self, checker):
        citation = Citation(
            title="AI and Labor Markets",
            authors=["Wrong, Author"],
            year=2023,
        )
        report = checker.check_citations([citation])
        assert report.total == 1
        assert report.inaccurate == 1

    def test_mixed_citations(self, checker):
        citations = [
            Citation(title="AI and Labor Markets", authors=["Smith, John"], year=2023),
            Citation(title="Nonexistent Paper", authors=["Nobody"]),
            Citation(title="Fiscal Policy and GDP Growth", authors=["Johnson, Bob"], year=2024),
        ]
        report = checker.check_citations(citations)
        assert report.total == 3
        assert report.verified == 2
        assert report.hallucinated == 1

    def test_empty_citations(self, checker):
        report = checker.check_citations([])
        assert report.total == 0
        assert report.grounding_score == 0.0


class TestTitleMatching:
    """Test fuzzy title matching."""

    def test_exact_match(self, checker):
        citation = Citation(title="AI and Labor Markets")
        report = checker.check_citations([citation])
        assert report.checks[0].exists

    def test_case_insensitive(self, checker):
        citation = Citation(title="ai and labor markets")
        report = checker.check_citations([citation])
        assert report.checks[0].exists

    def test_partial_title_match(self, checker):
        citation = Citation(title="AI and Labor Markets: A Comprehensive Study")
        report = checker.check_citations([citation])
        # Should find via containment matching
        assert report.checks[0].exists


class TestAuthorMatching:
    """Test author verification logic."""

    def test_last_name_matching(self, checker):
        """Author 'John Smith' should match 'Smith, John'."""
        citation = Citation(
            title="AI and Labor Markets",
            authors=["John Smith"],  # Different format
            year=2023,
        )
        report = checker.check_citations([citation])
        # Should find the paper and match via last-name comparison
        assert report.checks[0].exists

    def test_empty_authors_passes(self, checker):
        """Empty author list should not trigger mismatch."""
        citation = Citation(
            title="AI and Labor Markets",
            authors=[],
            year=2023,
        )
        report = checker.check_citations([citation])
        assert report.checks[0].metadata_accurate


class TestExternalLookup:
    """Test external lookup function integration."""

    def test_external_lookup_finds_paper(self, tmp_path):
        store = DocumentStore(store_dir=str(tmp_path))

        def mock_lookup(citation):
            if "Quantum" in citation.title:
                return {"title": "Quantum Economics", "authors": ["Feynman"], "year": 2025}
            return None

        checker = GroundingChecker(document_store=store, external_lookup_fn=mock_lookup)
        citation = Citation(title="Quantum Economics", authors=["Feynman"], year=2025)
        report = checker.check_citations([citation])
        assert report.checks[0].exists
        assert report.verified == 1
        store.close()

    def test_no_store_no_lookup(self):
        checker = GroundingChecker()
        citation = Citation(title="Any Paper")
        report = checker.check_citations([citation])
        assert report.hallucinated == 1


class TestGroundingReport:
    """Test report structure."""

    def test_report_has_checks(self, checker):
        citations = [Citation(title="AI and Labor Markets")]
        report = checker.check_citations(citations)
        assert len(report.checks) == 1
        assert isinstance(report.checks[0], CitationCheck)

    def test_report_grounding_score(self, checker):
        citations = [
            Citation(title="AI and Labor Markets", year=2023),
            Citation(title="Nonexistent Paper"),
        ]
        report = checker.check_citations(citations)
        assert 0.0 <= report.grounding_score <= 1.0
