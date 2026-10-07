# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Real bibliographic backends for :class:`CitationVerifier`.

Design: chained, not ensemble. Each backend answers only what it is authoritative for
and returns ``None`` (not a negative verdict) the moment it cannot confirm a citation —
whether because the citation doesn't carry the identifier this backend needs, the lookup
missed, or the request failed. ``None`` means "defer to the next backend," never "this
citation is fake." Only ``CitationVerifier.verify_one`` is allowed to conclude
``hallucinated``, and only after every registered backend has deferred — i.e. the
citation drew a miss from every source checked, not just the first one. A single
backend's coverage gap (real, even for OpenAlex — see the abstract-coverage
notes in README.md) must never by itself brand a real citation as fabricated.

Order registered in ``team_hooks.py`` (cheapest/most authoritative for the citation's
own shape, first): Crossref (DOI exact) -> arXiv (ID exact) -> OpenAlex (DOI exact,
then broad search; already used elsewhere in this project, keyed via
OPENALEX_API_KEY) -> Semantic Scholar (broad search, last resort — this project's own
LiteratureTeam logs have hit real 429s from this endpoint under load, so it only ever
runs on citations nothing earlier could confirm).

``Backend = Callable[[Citation], Optional[CitationVerdict]]`` (citation_verifier.py) —
``verify_one`` calls a backend with only the citation, so collector/agent (for
observability) are constructor state on each backend instance, not call-time arguments.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from typing import Any, Optional

from .citation_verifier import Citation, CitationVerdict


def _year_matches(candidate_year: Optional[int], citation_year: Optional[int]) -> bool:
    """Conservative year check for the heuristic (no-DOI/no-arXiv-ID) match path."""
    if citation_year is None or candidate_year is None:
        return True  # nothing to contradict with; don't penalize a citation with no year
    return abs(int(candidate_year) - int(citation_year)) <= 1


def _author_surname_hint(raw: str) -> Optional[str]:
    """Pull a plausible author surname out of a bracketed citation's raw text,
    e.g. '(Smith et al., 2024)' -> 'Smith'. Best-effort only."""
    m = re.search(r"[\[(]\s*([A-Z][A-Za-z\-]+)", raw)
    return m.group(1) if m else None


class CrossrefBackend:
    """DOI-exact lookup against Crossref, the DOI registration agency. Keyless."""

    name = "crossref"

    def __init__(self, collector: Any = None, agent: str = "CitationVerifier"):
        self.collector = collector
        self.agent = agent

    def __call__(self, c: Citation) -> Optional[CitationVerdict]:
        if not c.doi:
            return None
        from shared.observability import tracked_get

        try:
            resp = tracked_get(
                f"https://api.crossref.org/works/{c.doi}",
                collector=self.collector, agent=self.agent, retries=2,
            )
        except Exception:
            return None
        if getattr(resp, "status_code", None) != 200:
            return None  # 404 or transient failure: defer, don't declare hallucinated
        try:
            msg = resp.json().get("message") or {}
        except Exception:
            return None
        if not msg.get("title"):
            return None
        return CitationVerdict(
            citation=c, status="verified",
            evidence_urls=[f"https://doi.org/{c.doi}"],
            reasoning="DOI resolves in Crossref", confidence=0.97,
        )


class ArxivIdBackend:
    """arXiv-ID-exact lookup against the real arXiv API (stdlib XML, no extra
    dependency) — confirms the id resolves to a real paper rather than only
    matching the id's regex shape, which is all the previous default did."""

    name = "arxiv"

    def __init__(self, collector: Any = None, agent: str = "CitationVerifier"):
        self.collector = collector
        self.agent = agent

    def __call__(self, c: Citation) -> Optional[CitationVerdict]:
        if not c.arxiv_id:
            return None
        from shared.observability import tracked_get

        try:
            resp = tracked_get(
                "http://export.arxiv.org/api/query",
                params={"id_list": c.arxiv_id}, collector=self.collector,
                agent=self.agent, retries=2,
            )
        except Exception:
            return None
        if getattr(resp, "status_code", None) != 200:
            return None
        try:
            root = ET.fromstring(resp.text)
        except Exception:
            return None
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entry = root.find("atom:entry", ns)
        if entry is None:
            return None
        title_el = entry.find("atom:title", ns)
        title = (title_el.text or "").strip() if title_el is not None else ""
        if not title or title.lower().startswith("error"):
            return None  # arXiv's own not-found shape: an entry titled "Error"
        return CitationVerdict(
            citation=c, status="verified",
            evidence_urls=[f"https://arxiv.org/abs/{c.arxiv_id}"],
            reasoning="arXiv id resolves", confidence=0.97,
        )


class OpenAlexBackend:
    """DOI-exact lookup by id, else a conservative broad search. Reuses this
    project's existing, keyed OpenAlex integration (OPENALEX_API_KEY)."""

    name = "openalex"

    def __init__(self, collector: Any = None, agent: str = "CitationVerifier"):
        self.collector = collector
        self.agent = agent

    def __call__(self, c: Citation) -> Optional[CitationVerdict]:
        if c.doi:
            v = self._by_doi(c)
            if v is not None:
                return v
        return self._by_search(c)

    def _by_doi(self, c: Citation) -> Optional[CitationVerdict]:
        from shared.observability import tracked_get
        from shared.tools.openalex_tool import _auth_params

        try:
            resp = tracked_get(
                f"https://api.openalex.org/works/doi:{c.doi}",
                params=_auth_params(), collector=self.collector,
                agent=self.agent, retries=2,
            )
        except Exception:
            return None
        if getattr(resp, "status_code", None) != 200:
            return None
        try:
            w = resp.json()
        except Exception:
            return None
        if not w.get("title"):
            return None
        return CitationVerdict(
            citation=c, status="verified",
            evidence_urls=[w.get("id", f"https://doi.org/{c.doi}")],
            reasoning="DOI resolves in OpenAlex", confidence=0.95,
        )

    def _by_search(self, c: Citation) -> Optional[CitationVerdict]:
        query = c.title or c.raw
        if not query or not query.strip():
            return None
        try:
            from shared.tools.openalex_tool import openalex_search_handler
            hits = openalex_search_handler(
                query=query, max_results=5, econ_only=False,
                collector=self.collector, agent=self.agent,
            )
        except Exception:
            return None
        surname = _author_surname_hint(c.raw)
        for h in hits or []:
            if not _year_matches(h.get("year"), c.year):
                continue
            if surname and not any(surname.lower() in (a or "").lower() for a in (h.get("authors") or [])):
                continue
            return CitationVerdict(
                citation=c, status="verified",
                evidence_urls=[h.get("url")] if h.get("url") else [],
                reasoning=f"heuristic match in OpenAlex search ('{h.get('title', '')[:80]}')",
                confidence=0.6,  # weaker: no exact identifier, year/author heuristic only
            )
        return None


class SemanticScholarBackend:
    """Broad search, last resort. This project's own runs have hit real 429s from
    this endpoint under load (see LiteratureTeam TopicCrawler logs) — ordered last
    and called only on citations every earlier backend already deferred on, to
    keep call volume low."""

    name = "semantic_scholar"
    _URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    def __init__(self, collector: Any = None, agent: str = "CitationVerifier"):
        self.collector = collector
        self.agent = agent

    def __call__(self, c: Citation) -> Optional[CitationVerdict]:
        query = c.title or c.raw
        if not query or not query.strip():
            return None
        from shared.observability import tracked_get

        params = {"query": query, "limit": 5, "fields": "title,year,authors,externalIds"}
        key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        headers = {"x-api-key": key} if key else None
        try:
            resp = tracked_get(
                self._URL, params=params, headers=headers,
                collector=self.collector, agent=self.agent, retries=1,
            )
        except Exception:
            return None
        if getattr(resp, "status_code", None) != 200:
            return None  # incl. 429 — defer rather than block on retry storms
        try:
            hits = (resp.json() or {}).get("data") or []
        except Exception:
            return None
        surname = _author_surname_hint(c.raw)
        for h in hits:
            if not _year_matches(h.get("year"), c.year):
                continue
            authors = [(a or {}).get("name", "") for a in (h.get("authors") or [])]
            if surname and not any(surname.lower() in a.lower() for a in authors):
                continue
            ext = h.get("externalIds") or {}
            url = (f"https://doi.org/{ext['DOI']}" if ext.get("DOI")
                   else f"https://www.semanticscholar.org/paper/{h.get('paperId', '')}")
            return CitationVerdict(
                citation=c, status="verified",
                evidence_urls=[url],
                reasoning=f"heuristic match in Semantic Scholar search ('{h.get('title', '')[:80]}')",
                confidence=0.55,
            )
        return None


def default_production_backends(collector: Any = None, agent: str = "CitationVerifier") -> list:
    """The real chain, in priority order (first = checked first)."""
    return [
        CrossrefBackend(collector=collector, agent=agent),
        ArxivIdBackend(collector=collector, agent=agent),
        OpenAlexBackend(collector=collector, agent=agent),
        SemanticScholarBackend(collector=collector, agent=agent),
    ]
