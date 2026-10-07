# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Shared helpers for the literature searches of IdeationTeam (SourcingStage) and
LiteratureTeam (LiteratureGatheringStage).

Problems addressed here:

* Crossref results were requested with ``select=...published-print...`` only, so any work
  without a print date (working papers, online-first articles, SSRN postings) came back
  without a year and the report printed "(n.d.)"; the venue was never requested, so the
  report printed the provider name "Crossref" as the venue. :func:`crossref_select`,
  :func:`crossref_year` and :func:`crossref_venue` request and read ``issued``,
  ``published-print``, ``published-online``, ``posted``, ``container-title`` and ``type``.
* The Crossref User-Agent carried a placeholder address; :func:`crossref_headers` uses
  ``CROSSREF_MAILTO`` / ``OPENALEX_MAILTO`` when set and omits the mailto otherwise.
* Search failures (HTTP 429/5xx, empty results) were silent. :func:`fetch_json` checks the
  status, retries 429/5xx a bounded number of times with backoff (``tracked_get`` handles
  the wait), logs failures per provider, and :class:`ProviderStats` keeps per-provider
  counts that the stages record in their metadata.
* :func:`dedupe_key` normalizes title + first-author surname, so the same paper listed
  under two DOIs (e.g. two SSRN versions) is kept once.
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional

_log = logging.getLogger("ael.scholarly_search")

# Providers' own names — never a publication venue.
PROVIDER_NAMES = {
    "crossref", "crossref api", "openalex", "openalex api", "arxiv", "arxiv api",
    "semantic scholar", "semantic scholar api", "semanticscholar", "google scholar",
    "ssrn search", "web", "unknown", "n/a", "none", "",
}

CROSSREF_FIELDS = (
    "DOI", "title", "author", "abstract", "issued", "published-print", "published-online",
    "posted", "container-title", "type", "is-referenced-by-count", "URL",
)

DEFAULT_RETRIES = 3

# arXiv categories for economics work — the filter LiteratureTeam's _search_arxiv uses.
ARXIV_ECON_CATEGORIES = "cat:econ.* OR cat:q-fin.* OR cat:cs.MA"


def arxiv_econ_query(query: str) -> str:
    """Restrict an arXiv query to the economics / quantitative-finance categories."""
    q = (query or "").strip()
    if "cat:" in q:
        return q
    return f"({q}) AND ({ARXIV_ECON_CATEGORIES})"


def crossref_items(
    query: str,
    max_results: int,
    *,
    stats: Optional[ProviderStats] = None,
    collector: Any = None,
    agent: str = "",
    sort_by_citations: bool = False,
) -> List[Dict[str, Any]]:
    """Crossref /works search returning normalized dicts with title, authors, abstract,
    url, year, venue, citation_count, doi. Failures are logged and counted."""
    params: Dict[str, Any] = {"query": query, "rows": max_results, "select": crossref_select()}
    if sort_by_citations:
        params.update({"sort": "is-referenced-by-count", "order": "desc"})
    data = fetch_json("https://api.crossref.org/works", provider="Crossref", params=params,
                      headers=crossref_headers(), stats=stats, collector=collector, agent=agent)
    if not isinstance(data, dict):
        return []
    out: List[Dict[str, Any]] = []
    for it in (data.get("message") or {}).get("items", []) or []:
        try:
            title = (it.get("title") or ["Untitled"])[0]
            authors = [f"{a.get('given', '')} {a.get('family', '')}".strip()
                       for a in it.get("author", []) or []]
            out.append({
                "title": title,
                "authors": [a for a in authors if a],
                "abstract": re.sub(r"<[^>]+>", "", it.get("abstract") or "No abstract available"),
                "url": it.get("URL") or f"https://doi.org/{it.get('DOI', '')}",
                "doi": it.get("DOI"),
                "year": crossref_year(it),
                "venue": crossref_venue(it),
                "type": it.get("type"),
                "citation_count": it.get("is-referenced-by-count"),
            })
        except Exception as e:  # one malformed record never drops the rest
            _log.debug("Crossref: skipped a malformed record: %s", e)
    if stats is not None:
        stats.record("Crossref", len(out))
    return out


def crossref_select() -> str:
    return ",".join(CROSSREF_FIELDS)


def contact_email() -> Optional[str]:
    """The operator's contact address for polite API pools, if configured."""
    for var in ("CROSSREF_MAILTO", "OPENALEX_MAILTO"):
        v = (os.environ.get(var) or "").strip()
        if v and "@" in v and not v.endswith("example.com"):
            return v
    return None


def crossref_headers() -> Dict[str, str]:
    mail = contact_email()
    ua = f"agentic-econ-lab/0.7.3 (mailto:{mail})" if mail else "agentic-econ-lab/0.7.3"
    return {"User-Agent": ua}


def _year_of(part: Any) -> Optional[int]:
    try:
        dp = (part or {}).get("date-parts") or []
        y = dp[0][0] if dp and dp[0] else None
        return int(y) if y else None
    except (TypeError, ValueError, IndexError, AttributeError):
        return None


def crossref_year(item: Dict[str, Any]) -> Optional[int]:
    """First present of issued / published-print / published-online / posted."""
    for key in ("issued", "published-print", "published-online", "posted", "created"):
        y = _year_of(item.get(key))
        if y:
            return y
    return None


def crossref_venue(item: Dict[str, Any]) -> Optional[str]:
    """Journal / book / repository title from container-title (None if absent)."""
    ct = item.get("container-title")
    if isinstance(ct, list):
        ct = next((c for c in ct if isinstance(c, str) and c.strip()), None)
    if isinstance(ct, str) and ct.strip():
        return ct.strip()
    inst = item.get("institution")
    if isinstance(inst, list) and inst and isinstance(inst[0], dict) and inst[0].get("name"):
        return str(inst[0]["name"]).strip()
    return None


def is_provider_name(venue: Optional[str]) -> bool:
    return (venue or "").strip().lower() in PROVIDER_NAMES


def clean_venue(venue: Optional[str]) -> Optional[str]:
    """The venue, or None when it is empty or only a search provider's name."""
    if venue is None or is_provider_name(venue):
        return None
    return str(venue).strip()


class ProviderStats:
    """Per-provider request/result/failure counts for a search stage's metadata."""

    def __init__(self) -> None:
        self.counts: Dict[str, Dict[str, Any]] = {}

    def _row(self, provider: str) -> Dict[str, Any]:
        return self.counts.setdefault(
            provider, {"requests": 0, "results": 0, "failures": 0, "last_error": None})

    def record(self, provider: str, n_results: int = 0, error: Optional[str] = None) -> None:
        row = self._row(provider)
        row["requests"] += 1
        row["results"] += int(n_results or 0)
        if error:
            row["failures"] += 1
            row["last_error"] = str(error)[:300]

    def as_dict(self) -> Dict[str, Dict[str, Any]]:
        return {k: dict(v) for k, v in self.counts.items()}


def fetch_json(
    url: str,
    *,
    provider: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    stats: Optional[ProviderStats] = None,
    collector: Any = None,
    agent: str = "",
    retries: int = DEFAULT_RETRIES,
) -> Optional[Any]:
    """GET ``url`` through the ``web_fetch`` tool and return the parsed JSON body.

    Retries 429/5xx up to ``retries`` times with backoff (``tracked_get``), checks the
    final status, and logs + records a failure for ``provider``. Returns None on failure.
    Callers record the result count with ``stats.record(provider, n)`` on success."""
    from shared.tools.tool_registry import ToolRegistry

    try:
        res = ToolRegistry.invoke("web_fetch", {
            "url": url, "params": params or {}, "headers": headers or {}, "retries": retries,
        }, collector=collector, agent=agent)
    except Exception as e:  # registry/tool failure
        res = None
        err = f"{type(e).__name__}: {e}"
    else:
        err = None if res.success else (res.error or "request failed")
    status = None
    data = None
    if res is not None and res.success:
        status = (res.data or {}).get("status_code")
        data = (res.data or {}).get("data")
        if status is not None and not (200 <= int(status) < 300):
            err = f"HTTP {status}"
        elif not isinstance(data, (dict, list)):
            err = "non-JSON response"
    if err:
        _log.warning("%s search failed: %s", provider, err)
        print(f"    [{provider}] search failed: {err}")
        if stats is not None:
            stats.record(provider, 0, error=err)
        return None
    return data


# --------------------------------------------------------------------------- #
# Deduplication
# --------------------------------------------------------------------------- #

def _ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode("ascii")


def normalize_title(title: str) -> str:
    t = _ascii(title).lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def first_author_surname(authors: Optional[Iterable[Any]]) -> str:
    for a in authors or []:
        name = a.get("name", "") if isinstance(a, dict) else str(a or "")
        name = _ascii(name).strip()
        if not name:
            continue
        if "," in name:                      # "Surname, Given"
            sur = name.split(",", 1)[0]
        else:                                # "Given Surname"
            sur = name.split()[-1]
        return re.sub(r"[^a-z]", "", sur.lower())
    return ""


def dedupe_key(title: str, authors: Optional[Iterable[Any]] = None) -> str:
    """Normalized title + first-author surname (the DOI is deliberately not part of it:
    the same paper can carry two DOIs, e.g. two SSRN postings)."""
    return f"{normalize_title(title)}|{first_author_surname(authors)}"


def dedupe_papers(papers: List[Any]) -> List[Any]:
    """Keep the first paper per normalized title + first-author surname (a record without
    authors matches any author with the same title); a later duplicate only fills fields
    (year, venue, abstract, authors) the kept one is missing. Works for objects with
    attributes or dicts."""
    def get(p, k):
        return p.get(k) if isinstance(p, dict) else getattr(p, k, None)

    def put(p, k, v):
        if isinstance(p, dict):
            p[k] = v
        else:
            try:
                setattr(p, k, v)
            except Exception:
                pass

    kept: List[Any] = []
    by_title: Dict[str, List[int]] = {}
    for p in papers:
        nt = normalize_title(get(p, "title") or "")
        sur = first_author_surname(get(p, "authors"))
        match = None
        if nt:
            for idx in by_title.get(nt, []):
                other = first_author_surname(get(kept[idx], "authors"))
                # same title and same first author, or one side has no author list
                if not sur or not other or sur == other:
                    match = idx
                    break
        if match is None:
            if nt:
                by_title.setdefault(nt, []).append(len(kept))
            kept.append(p)
            continue
        first = kept[match]
        for field in ("year", "venue", "authors"):
            if not get(first, field) and get(p, field):
                put(first, field, get(p, field))
        ab = get(first, "abstract")
        if (not ab or ab == "No abstract available") and get(p, "abstract") not in (None, "", "No abstract available"):
            put(first, "abstract", get(p, "abstract"))
    return kept
