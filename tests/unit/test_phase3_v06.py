# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
AEL V0.6 Phase 3 — Cross-Team World Model Tests

Tests for WorldModel, WorldModelState, team-specific views, persistence,
merge with conflict resolution, cross-references, and backward compatibility.
"""

import json
import os
import tempfile

import pytest


# ── World Model: Core ─────────────────────────────────────────────────────


class TestWorldModelImports:
    def test_module_imports(self):
        from shared.research.world_model import (
            WorldModel,
            WorldModelState,
            WorldModelView,
            CrossReference,
            RunSummary,
            TEAM_PRODUCES,
            TEAM_CONSUMES,
        )

    def test_state_model_defaults(self):
        from shared.research.world_model import WorldModelState
        state = WorldModelState()
        assert state.version == 0
        assert state.research_questions == []
        assert state.data_quality_scores == {}

    def test_cross_reference_model(self):
        from shared.research.world_model import CrossReference
        cr = CrossReference(
            source_team="IdeationTeam",
            source_id="q1",
            target_team="LiteratureTeam",
            target_id="paper-42",
            relationship="supports",
        )
        assert cr.source_team == "IdeationTeam"
        assert cr.relationship == "supports"

    def test_run_summary_model(self):
        from shared.research.world_model import RunSummary
        rs = RunSummary(
            run_id="run-001",
            teams_run=["IdeationTeam", "LiteratureTeam"],
            stages_completed=6,
        )
        assert rs.run_id == "run-001"
        assert len(rs.teams_run) == 2


class TestWorldModelInit:
    def test_create_empty(self):
        from shared.research.world_model import WorldModel
        wm = WorldModel()
        assert wm.state.research_topic == ""
        assert wm.state.version == 0

    def test_create_with_topic(self):
        from shared.research.world_model import WorldModel
        wm = WorldModel(research_topic="AI and monetary policy")
        assert wm.state.research_topic == "AI and monetary policy"


class TestWorldModelUpdate:
    def test_update_ideation(self):
        from shared.research.world_model import WorldModel
        wm = WorldModel(research_topic="Test")
        wm.update("IdeationTeam", "RefinementStage", {
            "research_questions": [{"q": "How does AI affect GDP?"}],
        })
        assert len(wm.state.research_questions) == 1
        assert wm.state.version == 1

    def test_update_appends_lists(self):
        from shared.research.world_model import WorldModel
        wm = WorldModel()
        wm.update("IdeationTeam", "RefinementStage", {
            "research_questions": [{"q": "Q1"}],
        })
        wm.update("IdeationTeam", "RefinementStage", {
            "research_questions": [{"q": "Q2"}],
        })
        assert len(wm.state.research_questions) == 2

    def test_update_merges_dicts(self):
        from shared.research.world_model import WorldModel
        wm = WorldModel()
        wm.update("DataTeam", "QualityAssuranceStage", {
            "data_quality_scores": {"GDP": 0.95},
        })
        wm.update("DataTeam", "QualityAssuranceStage", {
            "data_quality_scores": {"CPI": 0.88},
        })
        assert wm.state.data_quality_scores == {"GDP": 0.95, "CPI": 0.88}

    def test_update_stores_custom_fields(self):
        from shared.research.world_model import WorldModel
        wm = WorldModel()
        wm.update("IdeationTeam", "SourcingStage", {
            "trend_count": 42,
        })
        assert "IdeationTeam:SourcingStage:trend_count" in wm.state.custom

    def test_update_increments_version(self):
        from shared.research.world_model import WorldModel
        wm = WorldModel()
        wm.update("IdeationTeam", "RefinementStage", {"research_questions": []})
        wm.update("LiteratureTeam", "SynthesisStage", {"literature_findings": []})
        assert wm.state.version == 2


class TestWorldModelQuery:
    def test_query_literature_team(self):
        from shared.research.world_model import WorldModel
        wm = WorldModel(research_topic="Test")
        wm.update("IdeationTeam", "RefinementStage", {
            "research_questions": [{"q": "How does AI affect GDP?"}],
        })
        view = wm.query("LiteratureTeam")
        assert view.team == "LiteratureTeam"
        assert "research_questions" in view.relevant_state
        assert len(view.relevant_state["research_questions"]) == 1
        assert "IdeationTeam" in view.dependencies

    def test_query_ideation_team_empty_deps(self):
        from shared.research.world_model import WorldModel
        wm = WorldModel()
        view = wm.query("IdeationTeam")
        assert view.dependencies == []

    def test_query_model_team_full_deps(self):
        from shared.research.world_model import WorldModel
        wm = WorldModel()
        view = wm.query("ModelTeam")
        assert "IdeationTeam" in view.dependencies
        assert "LiteratureTeam" in view.dependencies
        assert "DataTeam" in view.dependencies

    def test_query_includes_topic_in_full_view(self):
        from shared.research.world_model import WorldModel
        wm = WorldModel(research_topic="Test topic")
        view = wm.query("LiteratureTeam", view="full")
        assert view.relevant_state.get("research_topic") == "Test topic"

    def test_query_tracks_sync_version(self):
        from shared.research.world_model import WorldModel
        wm = WorldModel()
        wm.update("IdeationTeam", "RefinementStage", {"research_questions": []})
        view = wm.query("LiteratureTeam")
        assert view.last_sync_version == 1


class TestWorldModelPersistence:
    def test_persist_and_load(self):
        from shared.research.world_model import WorldModel
        wm = WorldModel(research_topic="Persistence test")
        wm.update("IdeationTeam", "RefinementStage", {
            "research_questions": [{"q": "Q1"}, {"q": "Q2"}],
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            path = wm.persist(os.path.join(tmpdir, "wm.json"))
            assert os.path.exists(path)

            wm2 = WorldModel.load(path)
            assert wm2.state.research_topic == "Persistence test"
            assert len(wm2.state.research_questions) == 2
            assert wm2.state.version == 1

    def test_persist_with_persist_dir(self):
        from shared.research.world_model import WorldModel
        with tempfile.TemporaryDirectory() as tmpdir:
            wm = WorldModel(research_topic="Test", persist_dir=tmpdir)
            path = wm.persist()
            assert path.endswith("world_model.json")
            assert os.path.exists(path)

    def test_persist_no_path_raises(self):
        from shared.research.world_model import WorldModel
        wm = WorldModel()
        with pytest.raises(ValueError):
            wm.persist()


class TestWorldModelMerge:
    def test_merge_combines_lists(self):
        from shared.research.world_model import WorldModel
        wm1 = WorldModel(research_topic="Topic A")
        wm1.update("IdeationTeam", "RefinementStage", {
            "research_questions": [{"q": "Q1"}],
        })
        wm2 = WorldModel(research_topic="Topic A")
        wm2.update("LiteratureTeam", "SynthesisStage", {
            "literature_findings": [{"p": "Paper1"}],
        })
        merged = wm1.merge(wm2)
        assert len(merged.state.research_questions) == 1
        assert len(merged.state.literature_findings) == 1
        assert merged.state.version > max(wm1.state.version, wm2.state.version)

    def test_merge_dict_self_wins(self):
        from shared.research.world_model import WorldModel
        wm1 = WorldModel()
        wm1.update("DataTeam", "QualityAssuranceStage", {
            "data_quality_scores": {"GDP": 0.95},
        })
        wm2 = WorldModel()
        wm2.update("DataTeam", "QualityAssuranceStage", {
            "data_quality_scores": {"GDP": 0.80, "CPI": 0.90},
        })
        merged = wm1.merge(wm2)
        assert merged.state.data_quality_scores["GDP"] == 0.95  # self wins
        assert merged.state.data_quality_scores["CPI"] == 0.90

    def test_merge_preserves_topic(self):
        from shared.research.world_model import WorldModel
        wm1 = WorldModel(research_topic="AI economics")
        wm2 = WorldModel()
        merged = wm1.merge(wm2)
        assert merged.state.research_topic == "AI economics"


class TestWorldModelCrossReferences:
    def test_add_cross_reference(self):
        from shared.research.world_model import WorldModel
        wm = WorldModel()
        wm.add_cross_reference(
            "IdeationTeam", "q1", "LiteratureTeam", "paper-42", "supports"
        )
        assert len(wm.state.cross_references) == 1
        assert wm.state.cross_references[0].relationship == "supports"

    def test_state_summary(self):
        from shared.research.world_model import WorldModel
        wm = WorldModel(research_topic="Test")
        wm.update("IdeationTeam", "RefinementStage", {
            "research_questions": [{"q": "Q1"}],
        })
        summary = wm.get_state_summary()
        assert summary["research_topic"] == "Test"
        assert summary["research_questions"] == 1
        assert summary["version"] == 1


class TestWorldModelRunHistory:
    def test_add_run_summary(self):
        from shared.research.world_model import WorldModel, RunSummary
        wm = WorldModel()
        wm.add_run_summary(RunSummary(
            run_id="run-001",
            teams_run=["IdeationTeam"],
            stages_completed=3,
        ))
        assert len(wm.state.run_history) == 1
        assert wm.state.run_history[0].run_id == "run-001"


class TestWorldModelBackwardCompat:
    def test_world_model_optional(self):
        """World model should be optional — existing workflows work without it."""
        from shared.research.world_model import WorldModel
        # Can create without any arguments
        wm = WorldModel()
        assert wm.state.version == 0

    def test_team_mapping_covers_all_teams(self):
        from shared.research.world_model import TEAM_PRODUCES, TEAM_CONSUMES
        for team in ["IdeationTeam", "LiteratureTeam", "DataTeam", "ModelTeam"]:
            assert team in TEAM_PRODUCES
            assert team in TEAM_CONSUMES
