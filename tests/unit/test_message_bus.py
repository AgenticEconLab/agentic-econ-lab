# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Tests for shared.protocols.message_bus.MessageBus.

Validates that:
- Messages can be published and received by subscribers
- Wildcard subscriptions work
- Message log queries work
- Pipeline run ID is propagated
"""

import pytest

from shared.protocols.message_bus import MessageBus, ArtifactMessage


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def bus():
    return MessageBus(pipeline_run_id="test-run-001")


# ============================================================================
# Tests
# ============================================================================

class TestPublishSubscribe:
    """Test the pub/sub pattern."""

    def test_subscribe_and_receive(self, bus):
        received = []
        bus.subscribe("LiteratureTeam", "research_questions", lambda m: received.append(m))

        bus.publish(ArtifactMessage(
            source_team="IdeationTeam",
            target_team="LiteratureTeam",
            artifact_name="research_questions",
            artifact_data={"questions": ["q1"]},
        ))

        assert len(received) == 1
        assert received[0].source_team == "IdeationTeam"
        assert received[0].artifact_data["questions"] == ["q1"]

    def test_no_delivery_to_wrong_team(self, bus):
        received = []
        bus.subscribe("ModelTeam", "research_questions", lambda m: received.append(m))

        bus.publish(ArtifactMessage(
            source_team="IdeationTeam",
            target_team="LiteratureTeam",
            artifact_name="research_questions",
            artifact_data={"q": 1},
        ))

        assert len(received) == 0

    def test_no_delivery_for_wrong_artifact(self, bus):
        received = []
        bus.subscribe("LiteratureTeam", "literature_review", lambda m: received.append(m))

        bus.publish(ArtifactMessage(
            source_team="IdeationTeam",
            target_team="LiteratureTeam",
            artifact_name="research_questions",
            artifact_data={"q": 1},
        ))

        assert len(received) == 0

    def test_wildcard_subscription(self, bus):
        received = []
        bus.subscribe("*", "research_questions", lambda m: received.append(m))

        bus.publish(ArtifactMessage(
            source_team="IdeationTeam",
            target_team="LiteratureTeam",
            artifact_name="research_questions",
            artifact_data={"q": 1},
        ))

        assert len(received) == 1

    def test_multiple_subscribers(self, bus):
        received_a = []
        received_b = []
        bus.subscribe("LiteratureTeam", "research_questions", lambda m: received_a.append(m))
        bus.subscribe("LiteratureTeam", "research_questions", lambda m: received_b.append(m))

        bus.publish(ArtifactMessage(
            source_team="IdeationTeam",
            target_team="LiteratureTeam",
            artifact_name="research_questions",
            artifact_data={"q": 1},
        ))

        assert len(received_a) == 1
        assert len(received_b) == 1

    def test_broadcast_without_target(self, bus):
        """Messages without target_team are broadcast to ALL matching subscribers."""
        received_specific = []
        received_wildcard = []
        bus.subscribe("LiteratureTeam", "data", lambda m: received_specific.append(m))
        bus.subscribe("*", "data", lambda m: received_wildcard.append(m))

        bus.publish(ArtifactMessage(
            source_team="DataTeam",
            artifact_name="data",
            artifact_data={"v": 1},
        ))

        assert len(received_specific) == 1  # True broadcast delivers to all
        assert len(received_wildcard) == 1


class TestMessageLog:
    """Test message history and queries."""

    def test_message_count(self, bus):
        assert bus.message_count() == 0
        bus.publish(ArtifactMessage(
            source_team="A", artifact_name="x", artifact_data={}))
        assert bus.message_count() == 1

    def test_get_messages_all(self, bus):
        bus.publish(ArtifactMessage(source_team="A", artifact_name="x", artifact_data={}))
        bus.publish(ArtifactMessage(source_team="B", artifact_name="y", artifact_data={}))
        assert len(bus.get_messages()) == 2

    def test_get_messages_filter_source(self, bus):
        bus.publish(ArtifactMessage(source_team="A", artifact_name="x", artifact_data={}))
        bus.publish(ArtifactMessage(source_team="B", artifact_name="y", artifact_data={}))
        assert len(bus.get_messages(source_team="A")) == 1

    def test_get_messages_filter_artifact(self, bus):
        bus.publish(ArtifactMessage(source_team="A", artifact_name="x", artifact_data={}))
        bus.publish(ArtifactMessage(source_team="A", artifact_name="y", artifact_data={}))
        assert len(bus.get_messages(artifact_name="x")) == 1


class TestPipelineRunId:
    """Test pipeline run ID propagation."""

    def test_auto_stamp_pipeline_run_id(self, bus):
        bus.publish(ArtifactMessage(
            source_team="A", artifact_name="x", artifact_data={}))
        messages = bus.get_messages()
        assert messages[0].pipeline_run_id == "test-run-001"

    def test_preserve_existing_pipeline_run_id(self, bus):
        bus.publish(ArtifactMessage(
            source_team="A", artifact_name="x", artifact_data={},
            pipeline_run_id="custom-id"))
        messages = bus.get_messages()
        assert messages[0].pipeline_run_id == "custom-id"

    def test_clear(self, bus):
        bus.subscribe("A", "x", lambda m: None)
        bus.publish(ArtifactMessage(source_team="A", artifact_name="x", artifact_data={}))
        bus.clear()
        assert bus.message_count() == 0
