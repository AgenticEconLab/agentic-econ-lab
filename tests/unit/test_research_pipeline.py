# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Tests for pipeline.research_pipeline.ResearchPipelineOrchestrator.

Validates that:
- Pipeline runs teams in dependency order
- Artifacts flow between teams
- Budget checks are enforced
- Team failures stop the pipeline
- Unregistered teams are handled gracefully
- Pipeline manifest is saved
- Instance-level runner registration works
"""

import json
import os

import pytest

from pipeline.research_pipeline import (
    ResearchPipelineOrchestrator,
    PipelineResult,
    TeamResult,
    register_team_runner,
    _TEAM_RUNNERS,
)
from pipeline.pipeline_config import PipelineConfig, StageConfig, BudgetConfig
from shared.observability import MetricsCollector


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def clean_runners():
    """Clear module-level team runners before and after each test."""
    _TEAM_RUNNERS.clear()
    yield
    _TEAM_RUNNERS.clear()


def make_config(stages=None, budget=None):
    """Helper to build a PipelineConfig."""
    if stages is None:
        stages = [
            StageConfig(
                team="IdeationTeam",
                mode="ModeNoWcNoHITL",
                inputs=["research_topic"],
                outputs=["research_questions"],
            ),
            StageConfig(
                team="LiteratureTeam",
                mode="ModeNoWcNoHITL",
                inputs=["research_questions"],
                outputs=["literature_review"],
                depends_on=["IdeationTeam"],
            ),
        ]
    return PipelineConfig(
        name="test_pipeline",
        stages=stages,
        budget=budget or BudgetConfig(),
        output_dir="test_output",
    )


def mock_ideation_runner(upstream_artifacts, mode, output_dir, collector=None):
    """Mock ideation runner that returns sample questions."""
    return {
        "research_questions": {
            "questions": [
                {"question": "How does AI affect GDP?", "priority_rank": 1},
                {"question": "What are LLM labor impacts?", "priority_rank": 2},
            ]
        }
    }


def mock_literature_runner(upstream_artifacts, mode, output_dir, collector=None):
    """Mock literature runner that returns a review."""
    questions = upstream_artifacts.get("research_questions", {})
    return {
        "literature_review": {
            "sections": ["Introduction", "Methods"],
            "source_questions": questions,
        }
    }


def mock_failing_runner(upstream_artifacts, mode, output_dir, collector=None):
    """Mock runner that always fails."""
    raise RuntimeError("Team execution failed")


# ============================================================================
# Tests
# ============================================================================

class TestPipelineExecution:
    """Test end-to-end pipeline execution with mocked teams."""

    def test_single_team_pipeline(self, tmp_path):
        config = make_config(stages=[
            StageConfig(
                team="IdeationTeam",
                mode="ModeNoWcNoHITL",
                inputs=["research_topic"],
                outputs=["research_questions"],
            ),
        ])
        config.output_dir = str(tmp_path)

        register_team_runner("IdeationTeam", mock_ideation_runner)

        pipeline = ResearchPipelineOrchestrator(config)
        result = pipeline.run("AI in economics")

        assert result.success
        assert "IdeationTeam" in result.teams_completed
        assert len(result.teams_failed) == 0

    def test_two_team_pipeline_with_artifact_flow(self, tmp_path):
        config = make_config()
        config.output_dir = str(tmp_path)

        register_team_runner("IdeationTeam", mock_ideation_runner)
        register_team_runner("LiteratureTeam", mock_literature_runner)

        pipeline = ResearchPipelineOrchestrator(config)
        result = pipeline.run("AI in economics")

        assert result.success
        assert result.teams_completed == ["IdeationTeam", "LiteratureTeam"]

        # Verify artifacts were stored
        assert pipeline.artifact_store.has("research_questions")
        assert pipeline.artifact_store.has("literature_review")

        # Verify literature_review received the questions
        lit_review = pipeline.artifact_store.get("literature_review")
        assert "source_questions" in lit_review

    def test_pipeline_stops_on_failure(self, tmp_path):
        config = make_config()
        config.output_dir = str(tmp_path)

        register_team_runner("IdeationTeam", mock_failing_runner)
        register_team_runner("LiteratureTeam", mock_literature_runner)

        pipeline = ResearchPipelineOrchestrator(config)
        result = pipeline.run("AI in economics")

        assert not result.success
        assert "IdeationTeam" in result.teams_failed
        assert "LiteratureTeam" not in result.teams_completed

    def test_unregistered_team_fails_gracefully(self, tmp_path):
        config = make_config(stages=[
            StageConfig(team="UnknownTeam", mode="M", inputs=[], outputs=[]),
        ])
        config.output_dir = str(tmp_path)

        pipeline = ResearchPipelineOrchestrator(config)
        result = pipeline.run("test")

        assert not result.success
        assert "UnknownTeam" in result.teams_failed
        assert "No runner registered" in result.team_results["UnknownTeam"].error


class TestInstanceRunnerRegistration:
    """Test instance-level runner registration for isolation."""

    def test_instance_runners_via_constructor(self, tmp_path):
        config = make_config(stages=[
            StageConfig(team="IdeationTeam", mode="M", inputs=[], outputs=["research_questions"]),
        ])
        config.output_dir = str(tmp_path)

        pipeline = ResearchPipelineOrchestrator(
            config, team_runners={"IdeationTeam": mock_ideation_runner}
        )
        result = pipeline.run("test")
        assert result.success

    def test_instance_register_method(self, tmp_path):
        config = make_config(stages=[
            StageConfig(team="IdeationTeam", mode="M", inputs=[], outputs=["research_questions"]),
        ])
        config.output_dir = str(tmp_path)

        pipeline = ResearchPipelineOrchestrator(config)
        pipeline.register_team_runner("IdeationTeam", mock_ideation_runner)
        result = pipeline.run("test")
        assert result.success

    def test_instance_runners_isolated(self, tmp_path):
        """Instance runners don't leak between orchestrator instances."""
        config = make_config(stages=[
            StageConfig(team="IdeationTeam", mode="M", inputs=[], outputs=[]),
        ])
        config.output_dir = str(tmp_path)

        pipeline1 = ResearchPipelineOrchestrator(config)
        pipeline1.register_team_runner("IdeationTeam", mock_ideation_runner)

        pipeline2 = ResearchPipelineOrchestrator(config)
        # pipeline2 should NOT have the runner from pipeline1
        result2 = pipeline2.run("test")
        assert not result2.success  # No runner registered


class TestArtifactFlow:
    """Test artifact propagation between teams."""

    def test_initial_artifacts_loaded(self, tmp_path):
        config = make_config(stages=[
            StageConfig(
                team="LiteratureTeam",
                mode="ModeNoWcNoHITL",
                inputs=["research_questions"],
                outputs=["literature_review"],
            ),
        ])
        config.output_dir = str(tmp_path)

        register_team_runner("LiteratureTeam", mock_literature_runner)

        pipeline = ResearchPipelineOrchestrator(config)
        result = pipeline.run(
            "AI in economics",
            initial_artifacts={
                "research_questions": {
                    "questions": [{"question": "pre-loaded question"}]
                }
            },
        )

        assert result.success
        lit_review = pipeline.artifact_store.get("literature_review")
        assert lit_review is not None

    def test_research_topic_always_registered(self, tmp_path):
        config = make_config(stages=[])
        config.output_dir = str(tmp_path)

        pipeline = ResearchPipelineOrchestrator(config)
        result = pipeline.run("AI in economics")

        assert pipeline.artifact_store.has("research_topic")
        topic_data = pipeline.artifact_store.get("research_topic")
        assert topic_data["research_topic"] == "AI in economics"

    def test_missing_upstream_artifact_is_empty_dict(self, tmp_path):
        """A team with a missing input artifact receives an empty upstream dict."""
        received_upstream = {}

        def capture_runner(upstream_artifacts, mode, output_dir, collector=None):
            received_upstream.update(upstream_artifacts)
            return {"output": {"data": "test"}}

        config = make_config(stages=[
            StageConfig(
                team="LiteratureTeam",
                mode="M",
                inputs=["nonexistent_artifact"],
                outputs=["output"],
            ),
        ])
        config.output_dir = str(tmp_path)

        pipeline = ResearchPipelineOrchestrator(
            config, team_runners={"LiteratureTeam": capture_runner}
        )
        result = pipeline.run("test")
        assert result.success
        assert "nonexistent_artifact" not in received_upstream


class TestMessageBus:
    """Test message bus integration."""

    def test_messages_published_on_success(self, tmp_path):
        config = make_config(stages=[
            StageConfig(
                team="IdeationTeam",
                mode="ModeNoWcNoHITL",
                inputs=["research_topic"],
                outputs=["research_questions"],
            ),
        ])
        config.output_dir = str(tmp_path)

        register_team_runner("IdeationTeam", mock_ideation_runner)

        pipeline = ResearchPipelineOrchestrator(config)
        pipeline.run("AI in economics")

        messages = pipeline.message_bus.get_messages(source_team="IdeationTeam")
        assert len(messages) >= 1
        assert messages[0].artifact_name == "research_questions"


class TestManifest:
    """Test pipeline manifest output."""

    def test_manifest_saved(self, tmp_path):
        config = make_config(stages=[
            StageConfig(
                team="IdeationTeam",
                mode="ModeNoWcNoHITL",
                inputs=[],
                outputs=["research_questions"],
            ),
        ])
        config.output_dir = str(tmp_path)

        register_team_runner("IdeationTeam", mock_ideation_runner)

        pipeline = ResearchPipelineOrchestrator(config)
        pipeline.run("test topic")

        manifest_path = os.path.join(str(tmp_path), "pipeline_manifest.json")
        assert os.path.exists(manifest_path)

        with open(manifest_path) as f:
            manifest = json.load(f)
        assert manifest["success"] is True
        assert manifest["research_topic"] == "test topic"
        assert "IdeationTeam" in manifest["teams_completed"]


class TestDuration:
    """Test timing tracking."""

    def test_duration_recorded(self, tmp_path):
        config = make_config(stages=[
            StageConfig(
                team="IdeationTeam",
                mode="ModeNoWcNoHITL",
                inputs=[],
                outputs=[],
            ),
        ])
        config.output_dir = str(tmp_path)

        register_team_runner("IdeationTeam", mock_ideation_runner)

        pipeline = ResearchPipelineOrchestrator(config)
        result = pipeline.run("test")

        assert result.total_duration_sec >= 0
        assert result.team_results["IdeationTeam"].duration_sec >= 0


def test_team_output_files_recorded_in_execution_log(tmp_path):
    """The chained orchestrator formerly did not pass the
    files a team wrote to the execution log, so Tier-1 output coverage, reliability and
    transparency were always 0 for chained runs."""
    def writing_runner(upstream_artifacts, mode, output_dir, collector=None):
        with open(os.path.join(output_dir, "research_questions.json"), "w") as f:
            f.write("{}")
        return {"research_questions": {"questions": ["q"]}}

    config = make_config(stages=[StageConfig(team="IdeationTeam", mode="ModeNoWcNoHITL",
                                             inputs=["research_topic"],
                                             outputs=["research_questions"])])
    config.output_dir = str(tmp_path)
    register_team_runner("IdeationTeam", writing_runner)
    pipeline = ResearchPipelineOrchestrator(config)
    assert pipeline.run("AI in economics").success
    stage = pipeline.logger.stages["IdeationTeam"]
    assert os.path.join("IdeationTeam", "research_questions.json") in stage.output_files
