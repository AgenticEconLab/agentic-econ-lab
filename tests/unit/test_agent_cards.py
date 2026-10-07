# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Tests for shared.protocols.agent_card.

Validates that:
- All 4 teams have cards defined
- Cards have required fields populated
- Pipeline order is correct
- Dependencies form a valid DAG
"""

import pytest

from shared.protocols.agent_card import (
    TeamCard,
    get_team_card,
    list_team_cards,
    get_pipeline_order,
    get_team_dependencies,
)


# ============================================================================
# Tests
# ============================================================================

class TestTeamCardRetrieval:
    """Test team card lookup."""

    def test_get_ideation_card(self):
        card = get_team_card("IdeationTeam")
        assert card is not None
        assert card.team_name == "IdeationTeam"

    def test_get_literature_card(self):
        card = get_team_card("LiteratureTeam")
        assert card is not None
        assert card.team_name == "LiteratureTeam"

    def test_get_model_card(self):
        card = get_team_card("ModelTeam")
        assert card is not None
        assert card.team_name == "ModelTeam"

    def test_get_data_card(self):
        card = get_team_card("DataTeam")
        assert card is not None
        assert card.team_name == "DataTeam"

    def test_get_unknown_card(self):
        assert get_team_card("FakeTeam") is None


class TestTeamCardFields:
    """Test that all required fields are populated."""

    @pytest.fixture(params=["IdeationTeam", "LiteratureTeam", "ModelTeam", "DataTeam"])
    def card(self, request):
        return get_team_card(request.param)

    def test_has_description(self, card):
        assert len(card.description) > 10

    def test_has_input_schema(self, card):
        assert isinstance(card.input_schema, dict)
        assert "type" in card.input_schema

    def test_has_output_schema(self, card):
        assert isinstance(card.output_schema, dict)
        assert "type" in card.output_schema

    def test_has_three_stages(self, card):
        assert len(card.stages) == 3

    def test_has_supported_modes(self, card):
        assert len(card.supported_modes) >= 3

    def test_has_cost_estimate(self, card):
        assert card.estimated_cost_usd > 0

    def test_has_duration_estimate(self, card):
        assert card.estimated_duration_sec > 0


class TestPipelineOrder:
    """Test pipeline ordering and dependencies."""

    def test_pipeline_order(self):
        order = get_pipeline_order()
        assert order == ["IdeationTeam", "LiteratureTeam", "ModelTeam", "DataTeam"]

    def test_list_team_cards_count(self):
        cards = list_team_cards()
        assert len(cards) == 4

    def test_list_team_cards_order(self):
        cards = list_team_cards()
        names = [c.team_name for c in cards]
        assert names == ["IdeationTeam", "LiteratureTeam", "ModelTeam", "DataTeam"]


class TestDependencies:
    """Test dependency graph."""

    def test_ideation_has_no_upstream(self):
        deps = get_team_dependencies("IdeationTeam")
        assert deps["upstream"] == []

    def test_ideation_has_downstream(self):
        deps = get_team_dependencies("IdeationTeam")
        assert "LiteratureTeam" in deps["downstream"]
        assert "ModelTeam" in deps["downstream"]

    def test_literature_upstream_is_ideation(self):
        deps = get_team_dependencies("LiteratureTeam")
        assert "IdeationTeam" in deps["upstream"]

    def test_model_upstream_includes_ideation_and_literature(self):
        deps = get_team_dependencies("ModelTeam")
        assert "IdeationTeam" in deps["upstream"]
        assert "LiteratureTeam" in deps["upstream"]

    def test_data_upstream_is_model(self):
        deps = get_team_dependencies("DataTeam")
        assert "ModelTeam" in deps["upstream"]

    def test_data_has_no_downstream(self):
        deps = get_team_dependencies("DataTeam")
        assert deps["downstream"] == []

    def test_unknown_team_returns_empty(self):
        deps = get_team_dependencies("FakeTeam")
        assert deps["upstream"] == []
        assert deps["downstream"] == []

    def test_no_circular_dependencies(self):
        """Verify the dependency graph is a DAG (no cycles)."""
        visited = set()
        order = get_pipeline_order()
        for team in order:
            deps = get_team_dependencies(team)
            for upstream in deps["upstream"]:
                assert upstream in visited, \
                    f"{team} depends on {upstream} which hasn't been processed yet"
            visited.add(team)
