# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Deterministic reporting harness (§4.7): magnitudes, figures, assembly, and the
number-consistency check — the LLM writes narrative, the harness writes numbers."""

from .assemble import assemble_report, collect_limitations, derived_numbers, render_spec_terms
from .consistency import artifact_number_pool, check_consistency, extract_numbers
from .figures import make_figures
from .formatting import build_availability_statement, format_for_journal
from .interpret import interpret_estimation
from .journal_match import shortlist_journals
from .types import (
    ConsistencyReport,
    DraftingResult,
    EffectSize,
    InterpretationResult,
    JournalMatch,
    QualityResult,
    UnverifiedNumber,
)

__all__ = [
    "assemble_report", "collect_limitations", "derived_numbers", "render_spec_terms",
    "artifact_number_pool", "check_consistency", "extract_numbers",
    "make_figures", "interpret_estimation",
    "build_availability_statement", "format_for_journal", "shortlist_journals",
    "ConsistencyReport", "DraftingResult", "EffectSize",
    "InterpretationResult", "JournalMatch", "QualityResult", "UnverifiedNumber",
]
