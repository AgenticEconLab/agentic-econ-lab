# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
AEL Guardrails — Safety and validation layer.

Provides schema validation, budget control, grounding verification,
and data quality checks for the AEL pipeline.
"""

from shared.guardrails.schema_validator import SchemaValidator, ValidationResult
from shared.guardrails.budget_controller import BudgetController, BudgetStatus, BudgetExceededError
from shared.guardrails.grounding_checker import GroundingChecker, GroundingReport, Citation, CitationCheck
from shared.guardrails.pii_detector import PIIDetector, PIIFinding, PIIScanReport

__all__ = [
    "SchemaValidator",
    "ValidationResult",
    "BudgetController",
    "BudgetStatus",
    "BudgetExceededError",
    "GroundingChecker",
    "GroundingReport",
    "Citation",
    "CitationCheck",
    "PIIDetector",
    "PIIFinding",
    "PIIScanReport",
]
