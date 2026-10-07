# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Deterministic estimation harness (§4.6): the LLM proposes an EstimationSpec, this package
validates, estimates (statsmodels), diagnoses, and issues the honest verdict."""

from .data_loader import find_retrieved_data, load_panel
from .diagnostics import apply_diagnostics_verdict, run_diagnostics
from .harness import (build_design, dependent_in_regressors, missing_main_effects,
                      refit_from_outcome, repair_departures, resolve_ref, run_estimation)
from .inference import run_hypotheses, run_inference, run_robustness
from .types import (
    VERDICT_ESTIMATED,
    VERDICT_FRAGILE,
    VERDICT_INESTIMABLE,
    CoefficientResult,
    DiagnosticResult,
    DiagnosticsReport,
    EstimationOutcome,
    EstimationSpec,
    HypothesisResult,
    HypothesisSpec,
    InferenceReport,
    PerformanceProfile,
    RobustnessCheck,
    VariableSpec,
)

__all__ = [
    "find_retrieved_data", "load_panel",
    "apply_diagnostics_verdict", "run_diagnostics",
    "build_design", "dependent_in_regressors", "missing_main_effects", "refit_from_outcome", "repair_departures", "resolve_ref", "run_estimation",
    "run_hypotheses", "run_inference", "run_robustness",
    "VERDICT_ESTIMATED", "VERDICT_FRAGILE", "VERDICT_INESTIMABLE",
    "CoefficientResult", "DiagnosticResult", "DiagnosticsReport",
    "EstimationOutcome", "EstimationSpec", "HypothesisResult", "HypothesisSpec",
    "InferenceReport", "PerformanceProfile",
    "RobustnessCheck", "VariableSpec",
]
