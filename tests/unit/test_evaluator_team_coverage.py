# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Tier-2 once silently skipped the execution teams (Code/Estimation/Reporting) — coverage
reached the parser and workflows.yaml but not the LLM evaluator's deliverable map. This
invariant keeps the three layers in lockstep."""

from evaluation.parsers.ael_parser import AELParser
from evaluation.scoring.llm_evaluator import TEAM_OUTPUT_FILES


def test_every_supported_team_has_judge_deliverables():
    supported = getattr(AELParser, "SUPPORTED_TEAMS", None) or [
        "IdeationTeam", "LiteratureTeam", "ModelTeam", "DataTeam",
        "EstimationTeam", "ReportingTeam", "CodeTeam"]
    for team in supported:
        assert team in TEAM_OUTPUT_FILES, (
            f"{team} is parser-supported but has no Tier-2 deliverable map — "
            f"the judge would silently score nothing")
        assert TEAM_OUTPUT_FILES[team], f"{team} deliverable list is empty"


def test_markdown_deliverables_load_as_text(tmp_path):
    from evaluation.scoring.llm_evaluator import LLMEvaluator
    ev = object.__new__(LLMEvaluator)
    rep = tmp_path
    (rep / "research_report.md").write_text("# Report\n\nBody text.")
    (rep / "quality_output.json").write_text('{"ok": true}')
    out = ev._load_outputs("ReportingTeam", "ModeNoWcWithHITL", tmp_path, output_dir=rep)
    assert out["research_report.md"].startswith("# Report")
    assert out["quality_output.json"] == {"ok": True}


def test_new_teams_have_team_specific_rubrics():
    """Code/Estimation/Reporting were formerly scored against IdeationTeam rubrics (the
    get_rubric fallback) — generated code judged as research questions."""
    from evaluation.scoring.rubrics import get_rubric
    for team in ("CodeTeam", "EstimationTeam", "ReportingTeam"):
        for dim in ("correctness", "soundness", "economic_rigor"):
            assert get_rubric(dim, team).team == team, f"{team}/{dim} falls back to Ideation"
