# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Grounding Checker — Verifies factual claims in agent outputs.

Detects hallucinated citations by checking whether cited papers exist
in the document store or via external lookup, and whether metadata
(authors, year, venue) matches.

Usage:
    from shared.guardrails.grounding_checker import GroundingChecker

    checker = GroundingChecker(document_store=store)
    report = checker.check_citations(citations)
    print(f"Verified: {report.verified}, Hallucinated: {report.hallucinated}")
"""

import re
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from shared.rag.document_store import DocumentStore


class Citation(BaseModel):
    """A citation to verify."""

    title: str = Field(default="", description="Paper title")
    authors: List[str] = Field(default_factory=list, description="Author names")
    year: Optional[int] = Field(default=None, description="Publication year")
    venue: Optional[str] = Field(default=None, description="Journal/conference")
    doi: Optional[str] = Field(default=None, description="DOI if available")
    url: Optional[str] = Field(default=None, description="URL if available")
    claim: str = Field(default="", description="The claim attributed to this citation")


class CitationCheck(BaseModel):
    """Result of checking a single citation."""

    citation: Citation
    exists: bool = Field(description="Whether the paper was found in store or lookup")
    metadata_accurate: bool = Field(
        default=False, description="Whether author/year/venue match"
    )
    matched_doc_id: Optional[str] = Field(
        default=None, description="Doc ID of matched document"
    )
    issues: List[str] = Field(
        default_factory=list, description="Specific issues found"
    )


class GroundingReport(BaseModel):
    """Aggregated result of grounding verification."""

    total: int = Field(description="Total citations checked")
    verified: int = Field(description="Citations both existing and accurate")
    hallucinated: int = Field(description="Citations with no matching paper")
    inaccurate: int = Field(description="Citations found but with wrong metadata")
    checks: List[CitationCheck] = Field(default_factory=list)
    grounding_score: float = Field(
        default=0.0,
        description="Fraction of citations that are verified (0.0-1.0)",
    )


class GroundingChecker:
    """
    Verifies factual claims in agent outputs against the document store.

    Checks:
    1. Citation existence: Does the cited paper exist in the store?
    2. Author verification: Do author names match?
    3. Date accuracy: Does the publication year match?
    4. Venue accuracy: Does the venue/journal match?
    """

    def __init__(
        self,
        document_store: Optional[DocumentStore] = None,
        external_lookup_fn: Optional[Callable[[Citation], Optional[Dict]]] = None,
    ):
        """
        Args:
            document_store: DocumentStore for local lookup.
            external_lookup_fn: Optional function for external verification
                (e.g., ArXiv API lookup). Signature: Citation -> dict or None.
        """
        self._store = document_store
        self._external_lookup = external_lookup_fn

    def check_citations(self, citations: List[Citation]) -> GroundingReport:
        """
        Verify a list of citations.

        Args:
            citations: List of Citation objects to verify.

        Returns:
            GroundingReport with per-citation results and aggregate scores.
        """
        checks = []
        for citation in citations:
            check = self._check_single(citation)
            checks.append(check)

        verified = sum(1 for c in checks if c.exists and c.metadata_accurate)
        hallucinated = sum(1 for c in checks if not c.exists)
        inaccurate = sum(1 for c in checks if c.exists and not c.metadata_accurate)
        total = len(citations)

        return GroundingReport(
            total=total,
            verified=verified,
            hallucinated=hallucinated,
            inaccurate=inaccurate,
            checks=checks,
            grounding_score=verified / total if total > 0 else 0.0,
        )

    def _check_single(self, citation: Citation) -> CitationCheck:
        """Check a single citation against the store and external sources."""
        # Try document store first
        matched_doc = None
        matched_doc_id = None

        if self._store is not None and citation.title:
            matched_doc = self._find_in_store(citation)
            if matched_doc is not None:
                matched_doc_id = matched_doc.get("doc_id")

        # Try external lookup if not found in store
        if matched_doc is None and self._external_lookup is not None:
            matched_doc = self._external_lookup(citation)

        if matched_doc is None:
            return CitationCheck(
                citation=citation,
                exists=False,
                metadata_accurate=False,
                issues=["Paper not found in document store or external sources"],
            )

        # Verify metadata accuracy
        issues = []
        metadata_accurate = True

        # Author check
        if citation.authors and matched_doc.get("authors"):
            if not self._authors_match(citation.authors, matched_doc["authors"]):
                issues.append(
                    f"Author mismatch: cited {citation.authors}, "
                    f"found {matched_doc['authors']}"
                )
                metadata_accurate = False

        # Year check
        if citation.year is not None and matched_doc.get("year") is not None:
            if citation.year != matched_doc["year"]:
                issues.append(
                    f"Year mismatch: cited {citation.year}, "
                    f"found {matched_doc['year']}"
                )
                metadata_accurate = False

        # Venue check
        if citation.venue and matched_doc.get("venue"):
            if not self._venue_matches(citation.venue, matched_doc["venue"]):
                issues.append(
                    f"Venue mismatch: cited '{citation.venue}', "
                    f"found '{matched_doc['venue']}'"
                )
                metadata_accurate = False

        return CitationCheck(
            citation=citation,
            exists=True,
            metadata_accurate=metadata_accurate,
            matched_doc_id=matched_doc_id,
            issues=issues,
        )

    def _find_in_store(self, citation: Citation) -> Optional[Dict]:
        """Search the document store for a matching paper."""
        # Try full title search
        results = self._store.search_by_text(citation.title, limit=5)

        # Also try with first few significant words (handles cases where
        # the cited title is longer than the stored title)
        clean_title = re.sub(r"[^\w\s]", "", citation.title)
        title_words = clean_title.split()
        if len(title_words) > 3:
            short_query = " ".join(title_words[:4])
            results.extend(self._store.search_by_text(short_query, limit=5))

        # Deduplicate
        seen = set()
        unique_results = []
        for doc in results:
            if doc.doc_id not in seen:
                seen.add(doc.doc_id)
                unique_results.append(doc)

        for doc in unique_results:
            if self._title_matches(citation.title, doc.title):
                return {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "authors": doc.authors,
                    "year": doc.year,
                    "venue": doc.venue,
                }

        return None

    @staticmethod
    def _title_matches(cited_title: str, found_title: str) -> bool:
        """Check if two titles are substantially similar."""
        ct = cited_title.lower().strip()
        ft = found_title.lower().strip()
        if ct == ft:
            return True
        # Fuzzy: check if one contains the other (handles truncation)
        if len(ct) > 10 and len(ft) > 10:
            if ct in ft or ft in ct:
                return True
        # Word overlap > 80%
        ct_words = set(ct.split())
        ft_words = set(ft.split())
        if ct_words and ft_words:
            overlap = len(ct_words & ft_words)
            max_len = max(len(ct_words), len(ft_words))
            if overlap / max_len > 0.8:
                return True
        return False

    @staticmethod
    def _authors_match(cited: List[str], found: List[str]) -> bool:
        """Check if author lists overlap substantially."""
        cited_lower = {a.lower().strip() for a in cited}
        found_lower = {a.lower().strip() for a in found}
        if not cited_lower or not found_lower:
            return True  # Can't verify, assume OK
        # At least one author in common (handles partial author lists)
        overlap = cited_lower & found_lower
        if overlap:
            return True
        # Check last-name matching (handles "John Smith" vs "Smith, J.")
        def _get_last_name(name: str) -> str:
            name = name.strip()
            if "," in name:
                return name.split(",")[0].strip()  # "smith, j." -> "smith"
            parts = name.split()
            return parts[-1] if parts else name

        cited_last = {_get_last_name(a) for a in cited_lower}
        found_last = {_get_last_name(a) for a in found_lower}
        return bool(cited_last & found_last)

    @staticmethod
    def _venue_matches(cited: str, found: str) -> bool:
        """Check if venues match (case-insensitive substring match)."""
        ct = cited.lower().strip()
        ft = found.lower().strip()
        return ct in ft or ft in ct
