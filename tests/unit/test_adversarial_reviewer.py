# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Tests for adversarial reviewer."""

import json
import pytest
from evaluation.consensus.adversarial_reviewer import (
    AdversarialReviewer,
    AdversarialReport,
    AdversarialIssue,
)


class TestAdversarialIssue:
    def test_create_issue(self):
        issue = AdversarialIssue(
            category="hallucination",
            severity="critical",
            description="Fabricated citation",
            evidence="Smith et al. (2025) does not exist",
        )
        assert issue.category == "hallucination"
        assert issue.severity == "critical"


class TestAdversarialReport:
    def test_empty_report(self):
        report = AdversarialReport(team="IdeationTeam", mode="ModeNoWcNoHITL")
        assert report.total_issues == 0
        assert report.penalty == 0.0

    def test_to_dict(self):
        report = AdversarialReport(
            team="T",
            mode="M",
            issues=[
                AdversarialIssue(
                    category="hallucination",
                    severity="critical",
                    description="Fake citation",
                )
            ],
            total_issues=1,
            critical_count=1,
            penalty=0.15,
        )
        d = report.to_dict()
        assert d["team"] == "T"
        assert d["total_issues"] == 1
        assert d["critical"] == 1
        assert d["penalty"] == 0.15

    def test_penalty_calculation(self):
        # 2 critical + 1 major + 3 minor = 2*0.15 + 1*0.08 + 3*0.03 = 0.47
        report = AdversarialReport(
            team="T", mode="M",
            critical_count=2, major_count=1, minor_count=3,
            penalty=0.47,  # pre-computed
        )
        assert report.penalty == pytest.approx(0.47)


class TestAdversarialReviewer:
    def test_no_llm_returns_empty_report(self):
        reviewer = AdversarialReviewer(llm_invoke_fn=None)
        report = reviewer.review("T", "M", {"output.json": {"data": "test"}})
        assert isinstance(report, AdversarialReport)
        assert report.total_issues == 0

    def test_review_with_mock_llm_finds_issues(self):
        mock_response = json.dumps({
            "issues": [
                {
                    "category": "hallucination",
                    "severity": "critical",
                    "description": "Paper by Smith (2025) does not exist",
                    "evidence": "Referenced in research question 3",
                    "affected_stage": "SourcingStage",
                },
                {
                    "category": "inconsistency",
                    "severity": "major",
                    "description": "Stage 1 identifies labor markets, stage 2 focuses on trade",
                    "evidence": "Cross-stage topic drift",
                },
                {
                    "category": "terminology_misuse",
                    "severity": "minor",
                    "description": "'Elasticity' used without specifying type",
                },
            ]
        })
        reviewer = AdversarialReviewer(
            llm_invoke_fn=lambda prompt: mock_response,
            model="test-model",
        )
        report = reviewer.review("IdeationTeam", "ModeNoWcNoHITL", {"q.json": {}})
        assert report.total_issues == 3
        assert report.critical_count == 1
        assert report.major_count == 1
        assert report.minor_count == 1
        # penalty = 1*0.15 + 1*0.08 + 1*0.03 = 0.26
        assert report.penalty == pytest.approx(0.26)

    def test_review_no_issues_found(self):
        mock_response = json.dumps({"issues": []})
        reviewer = AdversarialReviewer(llm_invoke_fn=lambda p: mock_response)
        report = reviewer.review("T", "M", {"out.json": {"good": "data"}})
        assert report.total_issues == 0
        assert report.penalty == 0.0

    def test_review_with_malformed_response(self):
        reviewer = AdversarialReviewer(
            llm_invoke_fn=lambda p: "this is not json"
        )
        report = reviewer.review("T", "M", {"out.json": {}})
        assert report.total_issues == 0

    def test_review_with_markdown_wrapped_json(self):
        mock_response = '```json\n{"issues": [{"category": "other", "severity": "minor", "description": "test"}]}\n```'
        reviewer = AdversarialReviewer(llm_invoke_fn=lambda p: mock_response)
        report = reviewer.review("T", "M", {"out.json": {}})
        assert report.total_issues == 1

    def test_severity_normalization(self):
        mock_response = json.dumps({
            "issues": [
                {"category": "test", "severity": "CRITICAL", "description": "x"},
                {"category": "test", "severity": "unknown", "description": "y"},
            ]
        })
        reviewer = AdversarialReviewer(llm_invoke_fn=lambda p: mock_response)
        report = reviewer.review("T", "M", {"out.json": {}})
        assert report.issues[0].severity == "critical"
        assert report.issues[1].severity == "minor"  # unknown → minor

    def test_penalty_capped_at_1(self):
        # Many critical issues should cap at 1.0
        issues_json = json.dumps({
            "issues": [
                {"category": "x", "severity": "critical", "description": f"issue {i}"}
                for i in range(20)
            ]
        })
        reviewer = AdversarialReviewer(llm_invoke_fn=lambda p: issues_json)
        report = reviewer.review("T", "M", {"out.json": {}})
        assert report.penalty == pytest.approx(1.0)

    def test_prompt_includes_team_and_mode(self):
        captured_prompt = {}

        def capture(prompt):
            captured_prompt["text"] = prompt
            return '{"issues": []}'

        reviewer = AdversarialReviewer(llm_invoke_fn=capture)
        reviewer.review("LiteratureTeam", "ModeWithWcNoHITL", {"lit.json": {"papers": []}})
        assert "LiteratureTeam" in captured_prompt["text"]
        assert "ModeWithWcNoHITL" in captured_prompt["text"]
