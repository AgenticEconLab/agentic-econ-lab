# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
CitationVerifier — post-processor that flags hallucinated citations.

Reference pattern: OpenScholar / CiteLLM / SemanticCite / Feynman AI citation
verifiers (2026). ICLR 2026 review analysis found 50+ hallucinated citations
per paper missed by 3–5 reviewers; NeurIPS 2025 found 100+ across 53 of 4,000
accepted papers. V0.7 requires this verifier to be a mandatory
post-processor for LiteratureTeam and IdeationTeam outputs.

Backends are pluggable; the default ``NullBackend`` treats any citation as
suspect unless it already carries a DOI or arXiv ID, which is sufficient for
CI tests. Production callers register Semantic Scholar / Crossref / arXiv /
OpenAlex backends via :meth:`CitationVerifier.register_backend`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional


CitationStatus = Literal["verified", "suspect", "hallucinated"]


@dataclass
class Citation:
    raw: str
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    source_url: Optional[str] = None


@dataclass
class CitationVerdict:
    citation: Citation
    status: CitationStatus
    evidence_urls: List[str] = field(default_factory=list)
    reasoning: str = ""
    confidence: float = 0.5

    def is_ok(self) -> bool:
        return self.status == "verified"


Backend = Callable[[Citation], Optional[CitationVerdict]]


# Regex patterns for citation extraction
DOI_RE      = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
ARXIV_RE    = re.compile(r"\barxiv:\s*(\d{4}\.\d{4,5})\b", re.IGNORECASE)
ARXIV_ABS_RE = re.compile(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", re.IGNORECASE)
YEAR_RE     = re.compile(r"\b(19|20)\d{2}\b")
# Bibliographic brackets: "(Smith et al., 2024)", "[Doe 2023]", "[Fake0, 2099]".
BRACKETED_CITATION_RE = re.compile(
    r"[\[(]"
    r"([A-Z][A-Za-z0-9\-]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][A-Za-z0-9\-]+))?"
    r"(?:[,]?\s+(?:19|20)\d{2})?)"
    r"[\])]"
)


def extract_citations(text: str) -> List[Citation]:
    """Heuristic citation extractor — DOIs, arXiv IDs, bracketed author-year."""
    seen_raw: set[str] = set()
    out: List[Citation] = []

    for m in DOI_RE.finditer(text):
        raw = m.group(0)
        if raw in seen_raw:
            continue
        seen_raw.add(raw)
        out.append(Citation(raw=raw, doi=raw))

    for m in ARXIV_RE.finditer(text):
        raw = m.group(0)
        if raw in seen_raw:
            continue
        seen_raw.add(raw)
        out.append(Citation(raw=raw, arxiv_id=m.group(1)))

    for m in ARXIV_ABS_RE.finditer(text):
        raw = m.group(0)
        if raw in seen_raw:
            continue
        seen_raw.add(raw)
        out.append(Citation(raw=raw, arxiv_id=m.group(1),
                             source_url=f"https://{raw}"))

    for m in BRACKETED_CITATION_RE.finditer(text):
        raw = m.group(0)
        if raw in seen_raw:
            continue
        seen_raw.add(raw)
        yr = YEAR_RE.search(m.group(1))
        out.append(Citation(
            raw=raw,
            title=None,
            year=int(yr.group(0)) if yr else None,
        ))

    return out


class _NullBackend:
    """Default backend: accept citations with DOI or arxiv ID; mark rest suspect."""

    name = "null"

    def __call__(self, c: Citation) -> Optional[CitationVerdict]:
        if c.doi or c.arxiv_id:
            return CitationVerdict(
                citation=c,
                status="verified",
                evidence_urls=[
                    f"https://doi.org/{c.doi}" if c.doi else f"https://arxiv.org/abs/{c.arxiv_id}"
                ],
                reasoning="identifier present",
                confidence=0.95,
            )
        return CitationVerdict(
            citation=c,
            status="suspect",
            reasoning="no DOI or arXiv identifier",
            confidence=0.4,
        )


@dataclass
class VerificationSummary:
    total: int = 0
    verified: int = 0
    suspect: int = 0
    hallucinated: int = 0


class CitationVerifier:
    """Verify each citation via registered backends in sequence.

    First backend to return a non-None verdict wins. If none return, the
    citation is marked ``hallucinated``.
    """

    def __init__(
        self,
        backends: Optional[List[Backend]] = None,
        *,
        collector: Any = None,
    ) -> None:
        self._backends: List[Backend] = list(backends) if backends else [_NullBackend()]
        self.collector = collector

    def register_backend(self, backend: Backend) -> None:
        self._backends.insert(0, backend)

    def verify_all(self, text: str) -> List[CitationVerdict]:
        citations = extract_citations(text)
        return [self.verify_one(c) for c in citations]

    def verify_one(self, citation: Citation) -> CitationVerdict:
        for backend in self._backends:
            verdict = backend(citation)
            if verdict is not None:
                return verdict
        return CitationVerdict(
            citation=citation,
            status="hallucinated",
            reasoning="no backend could verify",
            confidence=0.9,
        )

    def redact_hallucinated(self, text: str, verdicts: List[CitationVerdict]) -> str:
        """Replace hallucinated citations with an inline marker."""
        for v in verdicts:
            if v.status == "hallucinated":
                text = text.replace(v.citation.raw, "[CITATION_HALLUCINATED]")
            elif v.status == "suspect":
                text = text.replace(v.citation.raw, f"{v.citation.raw} [UNVERIFIED]")
        return text

    def summarize(self, verdicts: List[CitationVerdict]) -> VerificationSummary:
        s = VerificationSummary(total=len(verdicts))
        for v in verdicts:
            if v.status == "verified":
                s.verified += 1
            elif v.status == "suspect":
                s.suspect += 1
            else:
                s.hallucinated += 1
        return s

    def hallucination_rate(self, verdicts: List[CitationVerdict]) -> float:
        if not verdicts:
            return 0.0
        return sum(1 for v in verdicts if v.status == "hallucinated") / len(verdicts)
