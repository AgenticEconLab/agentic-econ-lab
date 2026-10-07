# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Multi-evaluator evaluation module.

Provides consensus engine for aggregating scores from multiple LLM evaluators
with score calibration and inter-evaluator agreement measurement.
"""

from evaluation.consensus.consensus_engine import (
    ConsensusEngine,
    CalibrationMethod,
    ConsensusResult,
)
from evaluation.consensus.adversarial_reviewer import (
    AdversarialReviewer,
    AdversarialReport,
)

__all__ = [
    "ConsensusEngine",
    "CalibrationMethod",
    "ConsensusResult",
    "AdversarialReviewer",
    "AdversarialReport",
]
