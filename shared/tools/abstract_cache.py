# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Generic local abstract cache — OPTIONAL, user-supplied DOI->abstract store.

A user who has an institutional abstracts database (e.g. Elsevier abstracts they are
licensed to use) can point ``ABSTRACT_CACHE`` at a CSV to fill abstracts that the open
sources (OpenAlex/S2/Crossref) lack — on THEIR runs only. The AEL repository ships NO
data and references no specific dataset: if ``ABSTRACT_CACHE`` is unset or the file is
missing, every lookup returns "" (a no-op), so the framework stays fully self-sufficient
on the keyless layers for everyone else.

CSV format: any delimiter (auto-sniffed) with a DOI column and an abstract column. Column
names are auto-detected (doi / DOI / DI / doi_url; abstract / description / dc:description)
and can be overridden via ``ABSTRACT_CACHE_DOI_COL`` / ``ABSTRACT_CACHE_ABS_COL``.
"""

from __future__ import annotations

import csv
import os
from functools import lru_cache
from typing import Dict

_DOI_CANDIDATES = ("doi", "di", "doi_url", "doiurl")
_ABS_CANDIDATES = ("abstract", "description", "dc:description", "ab")


def normalize_doi(doi: str) -> str:
    """Canonical DOI key for matching: lowercase, no resolver prefix."""
    d = (doi or "").strip().lower()
    for pfx in ("https://doi.org/", "http://dx.doi.org/", "http://doi.org/", "doi:"):
        if d.startswith(pfx):
            d = d[len(pfx):]
    return d.strip()


@lru_cache(maxsize=1)
def _load_cache() -> Dict[str, str]:
    path = os.getenv("ABSTRACT_CACHE")
    if not path or not os.path.exists(path):
        return {}
    out: Dict[str, str] = {}
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            sample = f.read(8192)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            except Exception:
                dialect = csv.excel
            reader = csv.DictReader(f, dialect=dialect)
            if not reader.fieldnames:
                return {}
            lower = {c.lower().strip(): c for c in reader.fieldnames}
            doi_col = os.getenv("ABSTRACT_CACHE_DOI_COL") or next(
                (lower[c] for c in _DOI_CANDIDATES if c in lower), None)
            abs_col = os.getenv("ABSTRACT_CACHE_ABS_COL") or next(
                (lower[c] for c in _ABS_CANDIDATES if c in lower), None)
            if not doi_col or not abs_col:
                print(f"[abstract_cache] could not find DOI/abstract columns in {path} "
                      f"(headers: {reader.fieldnames}); set ABSTRACT_CACHE_DOI_COL / _ABS_COL")
                return {}
            for row in reader:
                d = normalize_doi(row.get(doi_col, ""))
                a = (row.get(abs_col) or "").strip()
                if d and a:
                    out[d] = a
        print(f"[abstract_cache] loaded {len(out)} abstracts from {path}")
    except Exception as e:
        print(f"[abstract_cache] failed to load {path}: {type(e).__name__}: {e}")
        return {}
    return out


def abstract_from_cache(doi: str) -> str:
    """Cached abstract for a DOI, or '' if none / cache unconfigured."""
    if not doi:
        return ""
    return _load_cache().get(normalize_doi(doi), "")


def cache_size() -> int:
    """Number of cached abstracts (0 if unconfigured)."""
    return len(_load_cache())


# Reminder text reused across console / output-file / docs so the message is identical.
ABSTRACT_GAP_REMINDER = (
    "Some abstracts are publisher-restricted and NOT available from open sources — "
    "notably Elsevier economics journals (J. Econometrics, J. Monetary Economics, "
    "European Economic Review, J. Public Economics, Games and Economic Behavior, …). "
    "Papers without an abstract are dropped (an abstract-less record is near-useless), "
    "which can thin out the review. To recover them, configure EITHER a local cache "
    "(ABSTRACT_CACHE = a DOI->abstract CSV) OR institutional access "
    "(ELSEVIER_API_KEY + ELSEVIER_INSTTOKEN). See README.md, 'Literature abstracts'."
)


def extract_doi(url: str) -> str:
    """Pull a DOI out of a URL/identifier (for the abstract cascade)."""
    if not url:
        return ""
    u = url.strip()
    low = u.lower()
    if "doi.org/" in low:
        return u[low.index("doi.org/") + len("doi.org/"):].strip()
    if low.startswith("10."):
        return u
    return ""


def has_usable_abstract(item) -> bool:
    """True if an item (anything with a .abstract attr) has a meaningful abstract."""
    ab = (getattr(item, "abstract", "") or "").strip()
    return bool(ab) and ab != "No abstract available" and len(ab) > 50


def fill_missing_abstracts(items, agent_name: str = "", collector=None) -> int:
    """Fill .abstract on items lacking one via the OPTIONAL cascade: local cache
    (ABSTRACT_CACHE, user CSV) -> Elsevier API (key+insttoken). No-op without those.

    Works on any objects exposing ``.abstract`` and ``.url``. Returns the count filled.
    Shared by LiteratureTeam and IdeationTeam so the cascade behaves identically.
    """
    from shared.tools.tool_registry import ToolRegistry
    filled = 0
    for p in items:
        if has_usable_abstract(p):
            continue
        doi = extract_doi(getattr(p, "url", "") or "")
        if not doi:
            continue
        new_ab = abstract_from_cache(doi)
        if not new_ab:
            try:
                r = ToolRegistry.invoke("elsevier_abstract", {"doi": doi},
                                        collector=collector, agent=agent_name)
                new_ab = (r.data or {}).get("abstract", "") if r.success else ""
            except Exception:
                new_ab = ""
        if new_ab:
            p.abstract = new_ab
            filled += 1
    if filled and agent_name:
        print(f"[{agent_name}] Abstract cascade filled {filled} missing abstracts (cache/Elsevier)")
    return filled


def report_and_remind(items, agent_name: str = "") -> dict:
    """Compute abstract coverage over ``items``, print the summary + reminder, return the dict.
    One call for any team's gather step to surface the gap on the console."""
    total = len(items)
    kept = sum(1 for p in items if has_usable_abstract(p))
    cov = abstract_coverage(total, kept)
    tag = f"[{agent_name}] " if agent_name else ""
    print(f"{tag}Abstract coverage: {cov['summary']}")
    if cov["reminder"]:
        print(f"{tag}{cov['reminder']}")
    return cov


def abstract_coverage(total_found: int, kept: int) -> dict:
    """Abstract-coverage stats + a user-facing reminder about the (Elsevier) abstract gap.

    Returned dict is meant to be both printed to the console AND written into the output
    files so the limitation is visible wherever the results are read.
    """
    dropped = max(0, total_found - kept)
    cache_on = bool(os.getenv("ABSTRACT_CACHE"))
    els_on = bool(os.getenv("ELSEVIER_API_KEY"))
    summary = f"{kept}/{total_found} papers have a usable abstract ({dropped} lack one)."
    reminder = ""
    if dropped:
        if cache_on or els_on:
            reminder = ("Note: " + ("ABSTRACT_CACHE" if cache_on else "") +
                        ("+Elsevier" if (cache_on and els_on) else ("Elsevier" if els_on else "")) +
                        " is active; remaining gaps are papers not covered by your cache/subscription.")
        else:
            reminder = "⚠ " + ABSTRACT_GAP_REMINDER
    return {
        "total_found": total_found,
        "with_abstract": kept,
        "dropped_no_abstract": dropped,
        "abstract_cache_active": cache_on,
        "elsevier_active": els_on,
        "summary": summary,
        "reminder": reminder,
    }
