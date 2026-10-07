# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Deterministic code harness (§4.5): modules DERIVED from the parsed sympy system —
no LLM-written code — validated by an executed check battery, swept by comparative statics."""

from .experiment import comparative_statics
from .generate import generate_module
from .types import (
    ComparativeStatic,
    ExperimentResult,
    GenerationResult,
    RefusedEquation,
    ValidationCheck,
    ValidationResult,
)
from .validate import cleanup_workdir, validate_module

__all__ = [
    "cleanup_workdir", "comparative_statics", "generate_module", "validate_module",
    "ComparativeStatic", "ExperimentResult", "GenerationResult",
    "RefusedEquation", "ValidationCheck", "ValidationResult",
]
