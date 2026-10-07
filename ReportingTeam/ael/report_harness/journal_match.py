# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Deterministic journal shortlisting (§4.7 JournalAdvisor's numeric core).

Keyword overlap between the report's own content (research question + estimated dependent
variable) and a small, STATIC, illustrative reference table of journal scopes -- not live
journal-database intelligence (no acceptance rates, impact factors, or "recent publication
patterns" are claimed or available). The LLM narrates adaptation strategy for the deterministic
shortlist; it never re-ranks or invents a journal not in this table.
"""

from __future__ import annotations

import re
from typing import Dict, List

from .types import JournalMatch

# Static, illustrative only. Field tags are broad keyword buckets a report's own text/topic
# can plausibly match against -- not a claim of current editorial scope or fit.
JOURNAL_PROFILES: List[Dict] = [
    {"name": "American Economic Review", "tags": [
        "growth", "welfare", "policy", "general", "macroeconomic", "microeconomic",
        "inequality", "labor", "trade"],
     "scope": "General-interest flagship; broad theoretical and empirical contributions."},
    {"name": "Journal of Monetary Economics", "tags": [
        "monetary", "inflation", "interest", "business cycle", "central bank", "macroeconomic",
        "growth", "output"],
     "scope": "Monetary theory and policy, business cycles, macroeconomic dynamics."},
    {"name": "Journal of Applied Econometrics", "tags": [
        "econometric", "estimation", "regression", "panel", "time series", "identification",
        "instrument", "robustness"],
     "scope": "Applied econometric methodology and empirical technique."},
    {"name": "Journal of Development Economics", "tags": [
        "development", "poverty", "inequality", "growth", "informal", "emerging",
        "developing", "welfare"],
     "scope": "Economic development, poverty, and growth in developing economies."},
    {"name": "Journal of International Economics", "tags": [
        "trade", "exchange rate", "international", "capital flow", "tariff", "globalization",
        "current account"],
     "scope": "International trade, finance, and open-economy macroeconomics."},
    {"name": "Review of Economics and Statistics", "tags": [
        "empirical", "applied", "labor", "public", "microeconomic", "policy evaluation",
        "causal"],
     "scope": "Broad applied empirical economics across fields."},
    {"name": "Journal of Public Economics", "tags": [
        "public", "tax", "taxation", "welfare", "inequality", "fiscal", "government",
        "redistribution"],
     "scope": "Public finance, taxation, government policy, and welfare."},
    {"name": "Journal of Financial Economics", "tags": [
        "finance", "asset", "pricing", "corporate", "investment", "market", "risk", "return"],
     "scope": "Corporate finance and asset pricing."},
]

_WORD_RE = re.compile(r"[a-z]+")


def _tokenize(text: str) -> set:
    return set(_WORD_RE.findall((text or "").lower()))


def shortlist_journals(research_question: str, dependent_name: str,
                       top_n: int = 3) -> List[JournalMatch]:
    """Deterministic keyword-overlap ranking. Ties broken alphabetically by journal name
    for reproducibility."""
    tokens = _tokenize(research_question) | _tokenize(dependent_name)
    matches: List[JournalMatch] = []
    for profile in JOURNAL_PROFILES:
        matched = sorted(t for t in profile["tags"]
                         if any(word in tokens for word in t.split()))
        if matched:
            matches.append(JournalMatch(
                name=profile["name"], score=len(matched),
                matched_tags=matched, scope=profile["scope"]))
    matches.sort(key=lambda m: (-m.score, m.name))
    return matches[:top_n]
