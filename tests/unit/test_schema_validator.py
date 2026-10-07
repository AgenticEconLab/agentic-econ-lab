# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Tests for shared.guardrails.schema_validator.SchemaValidator.

Validates that:
- Known team/stage pairs resolve to correct schemas
- Valid data passes validation
- Invalid data fails with descriptive errors
- Unknown team/stage pairs return schema_not_found
- Custom schemas can be registered at runtime
"""

import pytest
from pydantic import BaseModel, Field

from shared.guardrails.schema_validator import SchemaValidator, ValidationResult


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_sourcing_output():
    """Minimal valid SourcingStageOutput data."""
    return {
        "literature_items": [
            {
                "title": "AI in Economics: A Survey",
                "authors": ["Smith, J.", "Doe, A."],
                "url": "https://arxiv.org/abs/2301.00001",
                "source": "arXiv",
                "agent": "CorpusScout",
            }
        ],
        "metadata": {"topic": "AI economics"},
    }


@pytest.fixture
def sample_refinement_output():
    """Minimal valid RefinementStageOutput data."""
    return {
        "concepts": [
            {
                "concept_title": "LLM Impact on Labor Markets",
                "description": "How large language models affect employment",
                "key_themes": ["labor", "AI", "automation"],
                "literature_support": ["Smith2023"],
            }
        ],
        "questions": [
            {
                "question": "How do LLMs affect wage inequality?",
                "rationale": "Growing AI adoption requires understanding labor impacts",
                "methodology_hints": ["panel data", "diff-in-diff"],
                "related_concepts": ["LLM Impact on Labor Markets"],
            }
        ],
        "metadata": {},
    }


@pytest.fixture
def sample_integration_output():
    """Minimal valid IntegrationStageOutput data."""
    return {
        "prioritized_questions": [
            {
                "question": "How do LLMs affect wage inequality?",
                "theoretical_framework": "Skill-biased technological change",
                "rationale": "Important for policy design",
                "methodology": ["panel data"],
                "expected_impact": "Inform labor policy",
                "feasibility": "High — data available from BLS",
                "priority_rank": 1,
                "priority_score": 0.92,
            }
        ],
        "metadata": {},
    }


# ============================================================================
# Tests
# ============================================================================

class TestSchemaDiscovery:
    """Test schema lookup and listing."""

    def test_list_schemas_returns_all_teams(self):
        schemas = SchemaValidator.list_schemas()
        assert "IdeationTeam" in schemas
        assert "LiteratureTeam" in schemas
        assert "ModelTeam" in schemas
        assert "DataTeam" in schemas

    def test_each_team_has_three_stages(self):
        schemas = SchemaValidator.list_schemas()
        for team, stages in schemas.items():
            assert len(stages) == 3, f"{team} should have 3 stages, got {len(stages)}"

    def test_get_schema_returns_model(self):
        schema = SchemaValidator.get_schema("IdeationTeam", "SourcingStage")
        assert schema is not None
        assert issubclass(schema, BaseModel)

    def test_get_schema_unknown_team_returns_none(self):
        assert SchemaValidator.get_schema("UnknownTeam", "SourcingStage") is None

    def test_get_schema_unknown_stage_returns_none(self):
        assert SchemaValidator.get_schema("IdeationTeam", "UnknownStage") is None


class TestIdeationTeamValidation:
    """Test validation for IdeationTeam stages."""

    def test_sourcing_valid(self, sample_sourcing_output):
        result = SchemaValidator.validate("IdeationTeam", "SourcingStage", sample_sourcing_output)
        assert result.valid
        assert result.data is not None
        assert len(result.data["literature_items"]) == 1

    def test_sourcing_empty_items(self):
        result = SchemaValidator.validate("IdeationTeam", "SourcingStage", {
            "literature_items": [],
            "metadata": {},
        })
        assert result.valid

    def test_refinement_valid(self, sample_refinement_output):
        result = SchemaValidator.validate("IdeationTeam", "RefinementStage", sample_refinement_output)
        assert result.valid
        assert len(result.data["questions"]) == 1

    def test_integration_valid(self, sample_integration_output):
        result = SchemaValidator.validate("IdeationTeam", "IntegrationStage", sample_integration_output)
        assert result.valid
        assert result.data["prioritized_questions"][0]["priority_rank"] == 1

    def test_sourcing_missing_required_field(self):
        """literature_items is required."""
        result = SchemaValidator.validate("IdeationTeam", "SourcingStage", {
            "metadata": {},
        })
        assert not result.valid
        assert len(result.errors) > 0

    def test_integration_missing_required_fields(self):
        """A PrioritizedQuestion missing a genuinely-required field (question) is invalid.

        NOTE: priority_rank/priority_score are now Optional — the LLM-robustness mixin maps
        dict/list/non-numeric values to None (Optional absorbs them), so their absence is no
        longer an error. We instead assert a truly-required field (question) is enforced.
        """
        result = SchemaValidator.validate("IdeationTeam", "IntegrationStage", {
            "prioritized_questions": [
                {
                    # Missing: question (required)
                    "theoretical_framework": "TF",
                    "rationale": "R",
                    "methodology": "M",
                    "expected_impact": "EI",
                    "feasibility": "F",
                }
            ],
        })
        assert not result.valid


class TestUnknownSchemas:
    """Test error handling for unknown team/stage pairs."""

    def test_unknown_team(self):
        result = SchemaValidator.validate("FakeTeam", "SourcingStage", {})
        assert not result.valid
        assert result.errors[0]["type"] == "schema_not_found"

    def test_unknown_stage(self):
        result = SchemaValidator.validate("IdeationTeam", "FakeStage", {})
        assert not result.valid
        assert result.errors[0]["type"] == "schema_not_found"


class TestValidateOrRaise:
    """Test the validate_or_raise convenience method."""

    def test_valid_data_returns_dict(self, sample_sourcing_output):
        data = SchemaValidator.validate_or_raise(
            "IdeationTeam", "SourcingStage", sample_sourcing_output
        )
        assert isinstance(data, dict)
        assert "literature_items" in data

    def test_invalid_data_raises(self):
        with pytest.raises(ValueError, match="Schema validation failed"):
            SchemaValidator.validate_or_raise("IdeationTeam", "SourcingStage", {})


class TestOtherTeamsBasicValidation:
    """Smoke tests: each non-Ideation team accepts minimal valid data."""

    def test_gathering_stage_empty_batch(self):
        result = SchemaValidator.validate("LiteratureTeam", "LiteratureGatheringStage", {
            "literature_batch": {
                "research_questions": [],
                "literature_items": [],
                "trend_analyses": [],
                "citations": [],
                "insights": [],
            },
        })
        assert result.valid

    def test_gap_detection_stage_minimal(self):
        result = SchemaValidator.validate("LiteratureTeam", "GapDetectionStage", {
            "gap_analysis": {
                "paper_structures": [],
                "research_gaps": [],
                "knowledge_graph": {"nodes": [], "edges": []},
            },
        })
        assert result.valid

    def test_theory_stage_minimal(self):
        result = SchemaValidator.validate("ModelTeam", "TheoryStage", {
            "research_questions": [],
            "theoretical_frameworks": [],
        })
        assert result.valid

    def test_calibration_stage_minimal(self):
        result = SchemaValidator.validate("ModelTeam", "CalibrationStage", {
            "formal_models": [],
            "calibrated_models": [],
        })
        assert result.valid

    def test_data_source_stage_minimal(self):
        result = SchemaValidator.validate("DataTeam", "DataSourceStage", {
            "research_question": "How does GDP affect employment?",
            "data_requirements": [],
            "discovered_sources": [],
            "selected_series": [],
            "retrieved_data": [],
        })
        assert result.valid

    def test_quality_assurance_stage_minimal(self):
        result = SchemaValidator.validate("DataTeam", "QualityAssuranceStage", {
            "integrated_datasets": [],
            "quality_assessments": [],
            "documented_datasets": [],
        })
        assert result.valid


class TestCustomSchemaRegistration:
    """Test runtime schema registration."""

    def test_register_custom_schema(self):
        class CustomOutput(BaseModel):
            result: str

        SchemaValidator.register_schema("CustomTeam", "CustomStage", CustomOutput)
        result = SchemaValidator.validate("CustomTeam", "CustomStage", {"result": "ok"})
        assert result.valid
        assert result.data["result"] == "ok"

        # Cleanup
        SchemaValidator.reset()
