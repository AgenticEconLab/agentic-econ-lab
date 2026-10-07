# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Verification-grade output helpers (V0.7).

Modules:
  - citation_verifier : post-processor catching hallucinated citations
    (OpenScholar / CiteLLM / SemanticCite pattern)
  - falsifier : DASES-style adversarial counterexample generator
  - trust_memory : trust-aware MemoryGuard 2.0 extension
"""

from shared.verification.citation_verifier import (
    Citation,
    CitationStatus,
    CitationVerdict,
    CitationVerifier,
    VerificationSummary,
    extract_citations,
)
from shared.verification.backends import (
    ArxivIdBackend,
    CrossrefBackend,
    OpenAlexBackend,
    SemanticScholarBackend,
    default_production_backends,
)
from shared.verification.falsifier import (
    AdversarialFalsifier,
    FalsifierReport,
)
from shared.verification.trust_memory import (
    BeliefDriftDetector,
    DriftAlert,
    MemoryEntry,
    TrustAwareRetriever,
)
from shared.verification.team_hooks import (
    attach_citation_audit,
    extract_ideation_texts,
    extract_review_texts,
    run_citation_verifier_pass,
)

__all__ = [
    "Citation",
    "CitationStatus",
    "CitationVerdict",
    "CitationVerifier",
    "VerificationSummary",
    "extract_citations",
    "CrossrefBackend",
    "ArxivIdBackend",
    "OpenAlexBackend",
    "SemanticScholarBackend",
    "default_production_backends",
    "AdversarialFalsifier",
    "FalsifierReport",
    "MemoryEntry",
    "TrustAwareRetriever",
    "BeliefDriftDetector",
    "DriftAlert",
    "run_citation_verifier_pass",
    "extract_review_texts",
    "extract_ideation_texts",
    "attach_citation_audit",
]
