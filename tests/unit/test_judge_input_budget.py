# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Tier-2 judge input budget (fix from the v0.7.1 run audit).

EstimationTeam: analysis_data rows (stored in all three stage files) filled the 24k budget, so
inference_output.json (hypotheses, robustness) never reached the judge in 17 runs.
ReportingTeam: the ~36k report alone filled the budget; quality/interpretation/drafting
outputs and the consistency result were unseen. Now bulky arrays are stripped for every
team, each output file gets a share of the budget, and the consistency result leads.
"""

from evaluation.scoring.llm_evaluator import LLMEvaluator, _allocate


def _rows(n):
    return [{"date": f"{1900 + i}-01-01", "y": 1.234567 * i, "x": 7.654321 * i} for i in range(n)]


def _evaluator():
    return LLMEvaluator(model="vllm/judge")


def test_estimation_inference_output_reaches_the_judge():
    outcome = {"verdict": "estimated", "analysis_data": _rows(900), "coefficients": []}
    outputs = {
        "estimation_output.json": {"outcome": outcome},
        "validation_output.json": {"outcome": outcome, "diagnostics": {"overall": "clean"}},
        "inference_output.json": {"outcome": outcome, "inference": {
            "hypotheses": [{"name": "MARKER_HYPOTHESIS", "outcome": "inconclusive"}],
            "robustness": [{"name": "MARKER_ROBUSTNESS"}]}},
    }
    text = _evaluator()._format_outputs(outputs, "EstimationTeam")
    assert "MARKER_HYPOTHESIS" in text and "MARKER_ROBUSTNESS" in text
    assert "sample rows omitted" in text
    assert len(text) < 30000


def test_reporting_every_file_and_consistency_result_represented():
    report = "# Research Report\n" + ("Narrative sentence about results. " * 2000)  # ~68k
    outputs = {
        "research_report.md": report,
        "quality_output.json": {"quality": {"consistency": {
            "verdict": "has_unverified", "verified": 187, "total_numbers": 250,
            "skipped_small_ints": 40, "match_kinds": {"exact": 150},
            "unverified": [{"value": 7.77, "context": "x"}],
            "verified_numbers": [{"value": float(i)} for i in range(3000)]},
            "availability_statement": "MARKER_AVAILABILITY"}},
        "interpretation_output.json": {"interpretation": {"effects": [], "narrative": "MARKER_INTERP"}},
        "drafting_output.json": {"drafting": {"narratives": {"intro": "MARKER_DRAFT"}}},
    }
    text = _evaluator()._format_outputs(outputs, "ReportingTeam")
    for marker in ("MARKER_AVAILABILITY", "MARKER_INTERP", "MARKER_DRAFT", "# Research Report"):
        assert marker in text, marker
    head = text.split("### File:")[0]
    assert "Number-consistency check result" in head and '"has_unverified"' in head
    assert "truncated" in text and len(text) < 40000


def test_allocate_water_fills():
    assert _allocate([10, 20], 100) == [10, 20]
    caps = _allocate([100, 5000, 50000], 6000)
    assert caps[0] == 100 and sum(caps) <= 6000 and caps[1] == caps[2]
