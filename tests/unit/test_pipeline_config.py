# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Tests for pipeline.pipeline_config.

Validates that:
- YAML configs load correctly
- PipelineConfig validation works
- Dependency ordering is computed correctly
- Circular dependencies are rejected
- Stage retrieval works
"""

import os
import tempfile

import pytest

from pipeline.pipeline_config import (
    PipelineConfig,
    StageConfig,
    BudgetConfig,
    load_pipeline_config,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def full_config():
    """A valid full research pipeline config."""
    return PipelineConfig(
        name="test_pipeline",
        mode="ModeNoWcNoHITL",
        stages=[
            StageConfig(
                team="IdeationTeam",
                mode="ModeNoWcNoHITL",
                inputs=["research_topic"],
                outputs=["research_questions"],
                depends_on=[],
            ),
            StageConfig(
                team="LiteratureTeam",
                mode="ModeNoWcNoHITL",
                inputs=["research_questions"],
                outputs=["literature_review"],
                depends_on=["IdeationTeam"],
            ),
            StageConfig(
                team="ModelTeam",
                mode="ModeNoWcNoHITL",
                inputs=["research_questions", "literature_review"],
                outputs=["model_specification"],
                depends_on=["IdeationTeam", "LiteratureTeam"],
            ),
            StageConfig(
                team="DataTeam",
                mode="open_source_api",
                inputs=["model_specification"],
                outputs=["validated_dataset"],
                depends_on=["ModelTeam"],
            ),
        ],
        budget=BudgetConfig(
            max_total_cost_usd=5.0,
            max_per_team_cost_usd=2.0,
            max_total_tokens=500000,
        ),
    )


@pytest.fixture
def yaml_file(tmp_path):
    """Write a valid YAML config to a temp file."""
    yaml_content = """
pipeline:
  name: "test_pipeline"
  mode: "ModeNoWcNoHITL"
  output_dir: "test_output"
  stages:
    - team: IdeationTeam
      mode: ModeNoWcNoHITL
      inputs: [research_topic]
      outputs: [research_questions]
      depends_on: []
    - team: LiteratureTeam
      mode: ModeNoWcNoHITL
      inputs: [research_questions]
      outputs: [literature_review]
      depends_on: [IdeationTeam]
  budget:
    max_total_cost_usd: 3.0
    max_per_team_cost_usd: 1.5
    max_total_tokens: 200000
"""
    filepath = os.path.join(str(tmp_path), "test.yaml")
    with open(filepath, "w") as f:
        f.write(yaml_content)
    return filepath


# ============================================================================
# Tests
# ============================================================================

class TestPipelineConfigCreation:
    """Test PipelineConfig creation and validation."""

    def test_valid_config(self, full_config):
        assert full_config.name == "test_pipeline"
        assert len(full_config.stages) == 4

    def test_budget_defaults(self):
        config = PipelineConfig(
            name="minimal",
            stages=[
                StageConfig(team="IdeationTeam", mode="ModeNoWcNoHITL"),
            ],
        )
        assert config.budget.max_total_cost_usd == 5.0

    def test_invalid_dependency_raises(self):
        with pytest.raises(ValueError, match="unknown team"):
            PipelineConfig(
                name="bad",
                stages=[
                    StageConfig(
                        team="LiteratureTeam",
                        mode="M",
                        depends_on=["NonexistentTeam"],
                    ),
                ],
            )

    def test_circular_dependency_raises(self):
        with pytest.raises(ValueError, match="[Cc]ircular"):
            PipelineConfig(
                name="bad",
                stages=[
                    StageConfig(team="A", mode="M", depends_on=["B"]),
                    StageConfig(team="B", mode="M", depends_on=["A"]),
                ],
            )

    def test_duplicate_team_names_raises(self):
        with pytest.raises(ValueError, match="Duplicate team name"):
            PipelineConfig(
                name="bad",
                stages=[
                    StageConfig(team="IdeationTeam", mode="M"),
                    StageConfig(team="IdeationTeam", mode="M"),
                ],
            )


class TestExecutionOrder:
    """Test topological sort of pipeline stages."""

    def test_linear_order(self, full_config):
        order = full_config.get_execution_order()
        assert order == ["IdeationTeam", "LiteratureTeam", "ModelTeam", "DataTeam"]

    def test_single_stage(self):
        config = PipelineConfig(
            name="single",
            stages=[StageConfig(team="IdeationTeam", mode="M")],
        )
        assert config.get_execution_order() == ["IdeationTeam"]

    def test_parallel_stages_sorted_alphabetically(self):
        """Stages with no dependencies among them are sorted alphabetically."""
        config = PipelineConfig(
            name="parallel",
            stages=[
                StageConfig(team="Zebra", mode="M"),
                StageConfig(team="Alpha", mode="M"),
            ],
        )
        order = config.get_execution_order()
        assert order == ["Alpha", "Zebra"]

    def test_diamond_dependency(self):
        """A depends on nothing, B and C depend on A, D depends on B and C."""
        config = PipelineConfig(
            name="diamond",
            stages=[
                StageConfig(team="A", mode="M"),
                StageConfig(team="B", mode="M", depends_on=["A"]),
                StageConfig(team="C", mode="M", depends_on=["A"]),
                StageConfig(team="D", mode="M", depends_on=["B", "C"]),
            ],
        )
        order = config.get_execution_order()
        assert order[0] == "A"
        assert order[-1] == "D"
        assert set(order[1:3]) == {"B", "C"}


class TestStageRetrieval:
    """Test stage lookup."""

    def test_get_stage(self, full_config):
        stage = full_config.get_stage("LiteratureTeam")
        assert stage is not None
        assert stage.mode == "ModeNoWcNoHITL"

    def test_get_stage_missing(self, full_config):
        assert full_config.get_stage("FakeTeam") is None


class TestYAMLLoading:
    """Test loading from YAML files."""

    def test_load_valid_yaml(self, yaml_file):
        config = load_pipeline_config(yaml_file)
        assert config.name == "test_pipeline"
        assert len(config.stages) == 2
        assert config.budget.max_total_cost_usd == 3.0

    def test_load_full_research_yaml(self):
        """Load the actual full_research.yaml config."""
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "pipeline", "configs", "full_research.yaml"
        )
        if os.path.exists(config_path):
            config = load_pipeline_config(config_path)
            assert config.name == "full_research_pipeline"
            assert len(config.stages) == 7

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_pipeline_config("/nonexistent/path.yaml")

    def test_load_empty_yaml_raises(self, tmp_path):
        empty_file = os.path.join(str(tmp_path), "empty.yaml")
        with open(empty_file, "w") as f:
            f.write("")
        with pytest.raises(ValueError, match="empty or invalid"):
            load_pipeline_config(empty_file)

    def test_load_ideation_only_yaml(self):
        """Load the ideation_only.yaml config."""
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "pipeline", "configs", "ideation_only.yaml"
        )
        if os.path.exists(config_path):
            config = load_pipeline_config(config_path)
            assert config.name == "ideation_only"
            assert len(config.stages) == 1
