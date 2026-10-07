# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Data contracts for the deterministic reporting harness (§4.7).

Same architecture as calib_harness/estim_harness: every number in the final report is either
computed here from upstream artifacts or cross-checked against them; the LLM writes clearly
delimited NARRATIVE blocks whose numbers the consistency checker verifies after the fact.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class EffectSize(BaseModel):
    """Economic magnitude of one regressor, computed from the estimation artifact."""
    param: str
    estimate: float
    p_value: float
    sd_x: Optional[float] = None
    one_sd_effect: Optional[float] = None       # b * sd_x, in units of the dependent
    standardized_beta: Optional[float] = None   # b * sd_x / sd_y
    elasticity_at_means: Optional[float] = None # b * mean_x / mean_y (level-level only)
    interpretation_kind: str = "marginal_effect"  # by transforms: elasticity | semi_elasticity | ...
    # Interaction rows — b_base + b_interaction * mean(moderator), when computable
    conditional_marginal_effect: Optional[float] = None
    note: str = ""                               # why a magnitude is n/a, or how it was derived


class InterpretationResult(BaseModel):
    """Stage-1 output: deterministic magnitudes + descriptives + figure files."""
    verdict: str = ""                            # carried from estimation (estimated/fragile/inestimable)
    dependent: str = ""
    effects: List[EffectSize] = Field(default_factory=list)
    descriptives: Dict[str, Dict[str, float]] = Field(default_factory=dict)  # var -> {mean, sd, min, max}
    n_obs: int = 0
    r_squared: Optional[float] = None
    figure_files: List[str] = Field(default_factory=list)
    narrative: str = ""                          # LLM narration of the COMPUTED numbers only
    notes: List[str] = Field(default_factory=list)


class UnverifiedNumber(BaseModel):
    value: float
    context: str                                 # +/- chars around the number in the draft


class VerifiedNumber(BaseModel):
    """A draft number that matched a value in the source artifacts. The check is value
    occurrence, not claim-level support: ``sources`` are the artifacts in which the matched
    value appears (possibly several), ``match`` how it matched."""
    value: float
    context: str
    match: Literal["exact", "rounding", "significant_figures", "percent_scaling"]
    matched_value: float
    sources: List[str] = Field(default_factory=list)


class ConsistencyReport(BaseModel):
    """The deterministic number cross-check of the final draft vs source artifacts."""
    total_numbers: int = 0
    verified: int = 0
    skipped_small_ints: int = 0                  # |n| <= small_int_floor: counts/enumerations
    unverified: List[UnverifiedNumber] = Field(default_factory=list)
    # Per-number provenance and match type, so a
    # reader can see WHICH artifact a value occurs in and whether it matched only by rounding.
    verified_numbers: List[VerifiedNumber] = Field(default_factory=list)
    match_kinds: Dict[str, int] = Field(default_factory=dict)
    verified_by_source: Dict[str, int] = Field(default_factory=dict)
    verdict: Literal["consistent", "has_unverified", "empty"] = "empty"


class DraftingResult(BaseModel):
    """Stage-2 output: the assembled report + the narrative blocks that went into it."""
    report_file: str = ""
    narratives: Dict[str, str] = Field(default_factory=dict)   # slot -> LLM text
    section_count: int = 0
    limitation_count: int = 0
    metadata: Dict = Field(default_factory=dict)


class JournalMatch(BaseModel):
    """One candidate journal, scored deterministically against the report's own content
    (§4.7 JournalAdvisor's numeric core) — keyword overlap against a static, illustrative
    reference table, NOT live journal-database intelligence."""
    name: str
    score: int
    matched_tags: List[str] = Field(default_factory=list)
    scope: str = ""


class QualityResult(BaseModel):
    """Stage-3 output: consistency verdict + reviewer narration."""
    consistency: ConsistencyReport
    report_file: str = ""
    proofreader_notes: str = ""
    # §4.7 JournalAdvisor: deterministic shortlist + LLM adaptation-strategy narration.
    journal_shortlist: List[JournalMatch] = Field(default_factory=list)
    journal_narrative: str = ""
    # §4.7 Formatter: deterministic re-render of the verified report into a specific
    # journal-style variant; writes a SEPARATE file, never overwrites the verified original.
    formatted_report_file: str = ""
    availability_statement: str = ""
    metadata: Dict = Field(default_factory=dict)
