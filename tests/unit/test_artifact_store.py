# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Tests for pipeline.artifact_store.ArtifactStore.

Validates that:
- Artifacts can be registered and retrieved
- Pydantic validation works on registration
- Disk persistence works
- Disk loading works
- Metadata tracking (producer, schema, timestamp) works
"""

import json
import os
import tempfile

import pytest
from pydantic import BaseModel, Field, ValidationError

from pipeline.artifact_store import ArtifactStore, Artifact


# ============================================================================
# Test schemas
# ============================================================================

class QuestionsOutput(BaseModel):
    questions: list = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class InvalidSchema(BaseModel):
    required_field: str


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def store():
    """In-memory artifact store."""
    return ArtifactStore()


@pytest.fixture
def disk_store(tmp_path):
    """Artifact store with disk persistence."""
    return ArtifactStore(base_dir=str(tmp_path), pipeline_run_id="test-run-001")


# ============================================================================
# Tests
# ============================================================================

class TestRegistration:
    """Test artifact registration."""

    def test_register_dict(self, store):
        artifact = store.register("questions", {"questions": ["q1", "q2"]})
        assert artifact.name == "questions"
        assert artifact.data["questions"] == ["q1", "q2"]

    def test_register_with_producer(self, store):
        artifact = store.register("questions", {"q": 1}, producer="IdeationTeam")
        assert artifact.producer == "IdeationTeam"

    def test_register_with_schema_validation(self, store):
        artifact = store.register(
            "questions",
            {"questions": ["q1"], "metadata": {"topic": "AI"}},
            schema=QuestionsOutput,
        )
        assert artifact.schema_name == "QuestionsOutput"
        assert artifact.data["questions"] == ["q1"]

    def test_register_pydantic_model_instance(self, store):
        model = QuestionsOutput(questions=["q1"], metadata={"k": "v"})
        artifact = store.register("questions", model)
        assert artifact.schema_name == "QuestionsOutput"
        assert artifact.data["questions"] == ["q1"]

    def test_register_invalid_data_raises(self, store):
        with pytest.raises(ValidationError):
            store.register("bad", {}, schema=InvalidSchema)

    def test_register_non_dict(self, store):
        artifact = store.register("count", 42)
        assert artifact.data == {"value": 42}

    def test_register_overwrites(self, store):
        store.register("x", {"v": 1})
        store.register("x", {"v": 2})
        assert store.get("x")["v"] == 2

    def test_pipeline_run_id_propagated(self, disk_store):
        artifact = disk_store.register("q", {"data": 1})
        assert artifact.pipeline_run_id == "test-run-001"


class TestRetrieval:
    """Test artifact retrieval."""

    def test_get_existing(self, store):
        store.register("q", {"val": 42})
        data = store.get("q")
        assert data["val"] == 42

    def test_get_missing_returns_none(self, store):
        assert store.get("nonexistent") is None

    def test_get_with_schema_validation(self, store):
        store.register("q", {"questions": ["q1"], "metadata": {}})
        data = store.get("q", expected_schema=QuestionsOutput)
        assert data["questions"] == ["q1"]

    def test_get_with_invalid_schema_raises(self, store):
        store.register("bad", {"wrong": "data"})
        with pytest.raises(ValidationError):
            store.get("bad", expected_schema=InvalidSchema)

    def test_get_artifact_returns_full_object(self, store):
        store.register("q", {"v": 1}, producer="TestTeam")
        artifact = store.get_artifact("q")
        assert isinstance(artifact, Artifact)
        assert artifact.producer == "TestTeam"

    def test_has(self, store):
        store.register("q", {"v": 1})
        assert store.has("q")
        assert not store.has("missing")


class TestListing:
    """Test artifact listing and summary."""

    def test_list_artifacts_empty(self, store):
        assert store.list_artifacts() == []

    def test_list_artifacts(self, store):
        store.register("a", {"v": 1})
        store.register("b", {"v": 2})
        names = store.list_artifacts()
        assert "a" in names
        assert "b" in names

    def test_all_returns_dict(self, store):
        store.register("a", {"v": 1})
        all_artifacts = store.all()
        assert "a" in all_artifacts
        assert isinstance(all_artifacts["a"], Artifact)

    def test_summary(self, store):
        store.register("q", {"questions": []}, producer="IdeationTeam", schema=QuestionsOutput)
        summary = store.summary()
        assert len(summary) == 1
        assert summary[0]["name"] == "q"
        assert summary[0]["producer"] == "IdeationTeam"
        assert summary[0]["schema"] == "QuestionsOutput"

    def test_clear(self, store):
        store.register("a", {"v": 1})
        store.clear()
        assert store.list_artifacts() == []


class TestPersistence:
    """Test disk persistence and loading."""

    def test_persist_to_disk(self, disk_store, tmp_path):
        disk_store.register("questions", {"q": ["q1"]})
        filepath = os.path.join(str(tmp_path), "questions.json")
        assert os.path.exists(filepath)

        with open(filepath, "r") as f:
            data = json.load(f)
        assert data["name"] == "questions"
        assert data["data"]["q"] == ["q1"]

    def test_load_from_disk(self, disk_store, tmp_path):
        disk_store.register("questions", {"q": ["q1"]}, producer="Test")
        disk_store.clear()
        assert not disk_store.has("questions")

        artifact = disk_store.load_from_disk("questions")
        assert artifact is not None
        assert artifact.data["q"] == ["q1"]
        assert disk_store.has("questions")

    def test_load_from_disk_missing(self, disk_store):
        assert disk_store.load_from_disk("nonexistent") is None

    def test_no_persist_without_base_dir(self, store):
        # In-memory store shouldn't try to persist
        store.register("q", {"v": 1}, persist=True)
        assert store.has("q")

    def test_persist_false_skips_disk(self, disk_store, tmp_path):
        disk_store.register("q", {"v": 1}, persist=False)
        filepath = os.path.join(str(tmp_path), "q.json")
        assert not os.path.exists(filepath)
