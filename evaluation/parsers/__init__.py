# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Output parsers for AEL (Agentic Econ Lab) workflow logs.

`AELParser` turns a stage-instrumented execution log into the evaluation
schema. The `BaseWorkflowParser` interface is kept generic so additional
parsers could be added, but AEL is the only implementation the project ships.
"""

from .base import BaseWorkflowParser
from .ael_parser import AELParser

__all__ = [
    "BaseWorkflowParser",
    "AELParser",
]
