# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""External benchmark hooks (V0.7)."""

from evaluation.external.hypobench_runner import HypoBenchRunner, HypoBenchReport
from evaluation.external.hle_runner import HLERunner, HLEReport

__all__ = [
    "HypoBenchRunner",
    "HypoBenchReport",
    "HLERunner",
    "HLEReport",
]
