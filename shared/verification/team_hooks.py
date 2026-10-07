# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
LiteratureTeam / IdeationTeam post-stage hooks (V0.7).

Wire the V0.7 CitationVerifier into team pipelines with 2 lines of MO code.
Gated on ``citation_verifier_enabled``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from shared.feature_flags import is_enabled


def run_citation_verifier_pass(
    texts: Iterable[str],
    *,
    collector: Any = None,
) -> Optional[Dict[str, Any]]:
    """Aggregate CitationVerifier results across a list of text blobs.

    Returns a summary dict with per-text verdicts and an overall hallucination
    rate, or None when the feature flag is off.
    """
    if not is_enabled("citation_verifier_enabled"):
        return None
    try:
        from shared.verification import CitationVerifier
        from shared.verification.backends import default_production_backends
    except Exception:
        return None

    # Real bibliographic backends (Crossref -> arXiv -> OpenAlex -> Semantic
    # Scholar), not the NullBackend default: see backends.py for why the order
    # is a chain (first confirmed match wins) rather than a parallel ensemble,
    # and why an individual backend defers via None instead of ever declaring
    # a citation hallucinated on its own.
    verifier = CitationVerifier(
        backends=default_production_backends(collector=collector, agent="CitationVerifier"),
        collector=collector,
    )
    combined: List[Dict[str, Any]] = []
    per_text: List[Dict[str, Any]] = []
    for idx, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            continue
        verdicts = verifier.verify_all(text)
        summary = verifier.summarize(verdicts)
        hallucination_rate = verifier.hallucination_rate(verdicts)
        per_text.append({
            "index": idx,
            "total": summary.total,
            "verified": summary.verified,
            "suspect": summary.suspect,
            "hallucinated": summary.hallucinated,
            "hallucination_rate": hallucination_rate,
        })
        for v in verdicts:
            combined.append({
                "citation": v.citation.raw,
                "status": v.status,
                "reasoning": v.reasoning,
                "confidence": v.confidence,
                "evidence_urls": v.evidence_urls,
            })

    total = sum(p["total"] for p in per_text)
    hallucinated = sum(p["hallucinated"] for p in per_text)
    return {
        "overall_total": total,
        "overall_hallucinated": hallucinated,
        "overall_hallucination_rate": (hallucinated / total) if total else 0.0,
        "per_text": per_text,
        "verdicts": combined,
    }


def attach_citation_audit(
    output_file: str | Path,
    audit: Optional[Dict[str, Any]],
    *,
    key: str = "citation_audit",
) -> None:
    """Merge a citation-audit summary into an existing JSON output file."""
    if not audit:
        return
    path = Path(output_file)
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return
    if not isinstance(data, dict):
        return
    data[key] = audit
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def extract_review_texts(literature_output: Dict[str, Any]) -> List[str]:
    """Extract review/synthesis text blobs from a LiteratureTeam synthesis output."""
    texts: List[str] = []
    for key in ("literature_review", "synthesis", "summary", "review_text"):
        value = literature_output.get(key)
        if isinstance(value, str) and value.strip():
            texts.append(value)
    sections = literature_output.get("sections") or []
    if isinstance(sections, list):
        for s in sections:
            if isinstance(s, dict):
                for k in ("section_content", "content", "body"):
                    v = s.get(k)
                    if isinstance(v, str) and v.strip():
                        texts.append(v)
    return texts


def extract_ideation_texts(ideation_output: Dict[str, Any]) -> List[str]:
    """Extract theoretical_framework / rationale from IdeationTeam prioritized questions."""
    texts: List[str] = []
    questions = ideation_output.get("prioritized_questions") or ideation_output.get("questions") or []
    if isinstance(questions, list):
        for q in questions:
            if isinstance(q, dict):
                for k in ("theoretical_framework", "rationale", "description"):
                    v = q.get(k)
                    if isinstance(v, str) and v.strip():
                        texts.append(v)
    return texts
