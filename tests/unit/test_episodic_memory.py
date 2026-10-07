# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Tests for shared.memory.episodic.EpisodicMemory.

Validates JSON trajectory storage and recall operations.
"""

import os
import json

import pytest

from shared.memory.episodic import EpisodicMemory, RunTrajectory


@pytest.fixture
def em(tmp_path):
    """Create an EpisodicMemory instance with a temp directory."""
    return EpisodicMemory(memory_dir=str(tmp_path))


@pytest.fixture
def sample_trajectory():
    return RunTrajectory(
        run_id="pipeline-test1",
        research_topic="AI impact on labor markets",
        teams_completed=["IdeationTeam", "LiteratureTeam"],
        success=True,
        total_duration_sec=120.5,
        artifact_names=["research_questions", "literature_review"],
        config_name="full_research",
        mode="ModeNoWcNoHITL",
        key_findings=["AI displaces routine tasks"],
        lessons_learned=["ArXiv has more recent papers than Scholar"],
    )


class TestRecordTrajectory:
    """Test recording run trajectories."""

    def test_record_saves_file(self, em, sample_trajectory):
        filepath = em.record_trajectory(sample_trajectory)
        assert os.path.exists(filepath)
        assert filepath.endswith(".json")

    def test_recorded_data_is_valid(self, em, sample_trajectory):
        filepath = em.record_trajectory(sample_trajectory)
        with open(filepath) as f:
            data = json.load(f)
        assert data["run_id"] == "pipeline-test1"
        assert data["research_topic"] == "AI impact on labor markets"
        assert data["success"] is True

    def test_count(self, em, sample_trajectory):
        assert em.count() == 0
        em.record_trajectory(sample_trajectory)
        assert em.count() == 1

    def test_list_runs(self, em, sample_trajectory):
        em.record_trajectory(sample_trajectory)
        runs = em.list_runs()
        assert "pipeline-test1" in runs


class TestGetTrajectory:
    """Test retrieving specific trajectories."""

    def test_get_existing(self, em, sample_trajectory):
        em.record_trajectory(sample_trajectory)
        traj = em.get_trajectory("pipeline-test1")
        assert traj is not None
        assert traj.research_topic == "AI impact on labor markets"

    def test_get_missing(self, em):
        assert em.get_trajectory("nonexistent") is None


class TestRecallSimilarRuns:
    """Test similarity-based trajectory recall."""

    def test_recall_by_topic_overlap(self, em):
        em.record_trajectory(RunTrajectory(
            run_id="run1", research_topic="AI impact on labor markets"
        ))
        em.record_trajectory(RunTrajectory(
            run_id="run2", research_topic="Climate change economics"
        ))
        em.record_trajectory(RunTrajectory(
            run_id="run3", research_topic="AI automation and labor"
        ))

        results = em.recall_similar_runs("AI labor economics")
        assert len(results) >= 1
        # Both AI-related runs should rank higher than climate
        topic_ids = [r.run_id for r in results]
        assert "run2" not in topic_ids or topic_ids.index("run2") > 0

    def test_recall_empty_memory(self, em):
        results = em.recall_similar_runs("any topic")
        assert results == []

    def test_recall_respects_top_k(self, em):
        for i in range(5):
            em.record_trajectory(RunTrajectory(
                run_id=f"run{i}", research_topic=f"topic keyword{i}"
            ))
        results = em.recall_similar_runs("keyword0 keyword1", top_k=2)
        assert len(results) <= 2

    def test_recall_empty_query(self, em, sample_trajectory):
        em.record_trajectory(sample_trajectory)
        results = em.recall_similar_runs("")
        assert results == []
