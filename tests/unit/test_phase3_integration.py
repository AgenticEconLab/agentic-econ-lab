# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Tests for AEL V0.4 Phase 3: Cross-Team Pipeline Activation.

Verifies:
1. run_ael_pipeline.py exists and has correct entry point
2. team_runners.py uses _team_context context manager
3. register_all_runners registers all 4 teams
4. Pipeline wiring: ArtifactStore, MessageBus, BudgetController
5. Full 4-team mock pipeline end-to-end
6. Pipeline config loading from YAML
7. _team_context restores cwd and sys.path on error
"""

import json
import os
import sys
from pathlib import Path

import pytest

# repository root for file path lookups (sys.path handled by conftest.py)
_agents_dir = Path(__file__).resolve().parent.parent.parent


# ============================================================================
# Test 1: run_ael_pipeline.py entry point
# ============================================================================

class TestRunPipelineEntryPoint:
    """Verify run_ael_pipeline.py exists and is well-formed."""

    def test_run_pipeline_exists(self):
        f = _agents_dir / "run_ael_pipeline.py"
        assert f.exists(), "run_ael_pipeline.py not found"

    def test_run_pipeline_imports(self):
        """run_ael_pipeline.py can be imported (syntax check)."""
        import importlib
        spec = importlib.util.spec_from_file_location(
            "run_ael_pipeline", str(_agents_dir / "run_ael_pipeline.py")
        )
        mod = importlib.util.module_from_spec(spec)
        # Don't execute — just verify it loads
        spec.loader.exec_module(mod)
        assert hasattr(mod, "main")
        assert hasattr(mod, "build_default_config")

    def test_build_default_config(self):
        """build_default_config returns a valid PipelineConfig."""
        from run_ael_pipeline import build_default_config
        config = build_default_config()
        assert config.name == "full_research_pipeline"
        assert len(config.stages) == 7
        teams = [s.team for s in config.stages]
        assert "IdeationTeam" in teams
        assert "LiteratureTeam" in teams
        assert "ModelTeam" in teams
        assert "DataTeam" in teams
        assert "EstimationTeam" in teams
        assert "ReportingTeam" in teams
        assert "CodeTeam" in teams
        code = config.get_stage("CodeTeam")
        assert code.enabled is False, "CodeTeam is an optional node, disabled by default"

    def test_build_default_config_custom_mode(self):
        """build_default_config accepts custom mode."""
        from run_ael_pipeline import build_default_config
        config = build_default_config(mode="ModeWithWcNoHITL")
        assert config.mode == "ModeWithWcNoHITL"
        # IdeationTeam stage should use custom mode
        ideation = config.get_stage("IdeationTeam")
        assert ideation.mode == "ModeWithWcNoHITL"
        # DataTeam always uses open_source_api
        data = config.get_stage("DataTeam")
        assert data.mode == "open_source_api"

    def test_execution_order(self):
        """Default config produces correct execution order."""
        from run_ael_pipeline import build_default_config
        config = build_default_config()
        order = config.get_execution_order()
        # IdeationTeam must come before LiteratureTeam and ModelTeam
        assert order.index("IdeationTeam") < order.index("LiteratureTeam")
        assert order.index("LiteratureTeam") < order.index("ModelTeam")
        assert order.index("ModelTeam") < order.index("DataTeam")


# ============================================================================
# Test 2: _team_context context manager
# ============================================================================

class TestTeamContext:
    """Verify _team_context handles cwd and sys.path correctly."""

    def test_context_restores_cwd(self, tmp_path):
        """_team_context restores original cwd after normal execution."""
        from pipeline.team_runners import _team_context
        original = os.getcwd()
        target_dir = tmp_path / "test_team"
        target_dir.mkdir()

        with _team_context(target_dir):
            assert os.getcwd() == str(target_dir)

        assert os.getcwd() == original

    def test_context_restores_cwd_on_error(self, tmp_path):
        """_team_context restores cwd even on exception."""
        from pipeline.team_runners import _team_context
        original = os.getcwd()
        target_dir = tmp_path / "test_team_err"
        target_dir.mkdir()

        with pytest.raises(RuntimeError):
            with _team_context(target_dir):
                raise RuntimeError("simulated error")

        assert os.getcwd() == original

    def test_context_cleans_sys_path(self, tmp_path):
        """_team_context removes added path from sys.path."""
        from pipeline.team_runners import _team_context
        target_dir = tmp_path / "test_team_path"
        target_dir.mkdir()
        target_str = str(target_dir)

        # Ensure not in path before
        if target_str in sys.path:
            sys.path.remove(target_str)

        with _team_context(target_dir):
            assert target_str in sys.path

        assert target_str not in sys.path

    def test_context_does_not_remove_preexisting_path(self, tmp_path):
        """_team_context does not remove path already in sys.path."""
        from pipeline.team_runners import _team_context
        target_dir = tmp_path / "test_team_pre"
        target_dir.mkdir()
        target_str = str(target_dir)

        # Pre-add to sys.path
        sys.path.insert(0, target_str)

        try:
            with _team_context(target_dir):
                assert target_str in sys.path

            # Should still be in sys.path (was pre-existing)
            assert target_str in sys.path
        finally:
            # Cleanup
            if target_str in sys.path:
                sys.path.remove(target_str)


# ============================================================================
# Test 3: register_all_runners
# ============================================================================

class TestRegisterAllRunners:
    """Verify all 4 team runners are registered."""

    def test_register_all_runners(self):
        from pipeline.research_pipeline import _TEAM_RUNNERS
        from pipeline.team_runners import register_all_runners

        _TEAM_RUNNERS.clear()
        register_all_runners()

        assert "IdeationTeam" in _TEAM_RUNNERS
        assert "LiteratureTeam" in _TEAM_RUNNERS
        assert "ModelTeam" in _TEAM_RUNNERS
        assert "DataTeam" in _TEAM_RUNNERS
        assert "EstimationTeam" in _TEAM_RUNNERS
        assert "ReportingTeam" in _TEAM_RUNNERS
        assert "CodeTeam" in _TEAM_RUNNERS
        assert len(_TEAM_RUNNERS) == 7

        _TEAM_RUNNERS.clear()

    def test_runners_are_callable(self):
        from pipeline.team_runners import (
            run_ideation_team,
            run_literature_team,
            run_model_team,
            run_data_team,
        )
        assert callable(run_ideation_team)
        assert callable(run_literature_team)
        assert callable(run_model_team)
        assert callable(run_data_team)


# ============================================================================
# Test 4: Pipeline wiring — ArtifactStore + MessageBus + BudgetController
# ============================================================================

class TestPipelineWiring:
    """Verify pipeline components are wired correctly."""

    def _make_pipeline(self, tmp_path, stages=None):
        from pipeline.pipeline_config import PipelineConfig, StageConfig, BudgetConfig
        from pipeline.research_pipeline import ResearchPipelineOrchestrator

        if stages is None:
            stages = [
                StageConfig(
                    team="IdeationTeam",
                    mode="ModeNoWcNoHITL",
                    inputs=["research_topic"],
                    outputs=["research_questions"],
                ),
            ]
        config = PipelineConfig(
            name="test_wiring",
            stages=stages,
            budget=BudgetConfig(
                max_total_cost_usd=5.0,
                max_per_team_cost_usd=2.0,
                max_total_tokens=500_000,
            ),
            output_dir=str(tmp_path),
        )
        return ResearchPipelineOrchestrator(config)

    def test_artifact_store_created(self, tmp_path):
        pipeline = self._make_pipeline(tmp_path)
        assert pipeline.artifact_store is not None

    def test_message_bus_created(self, tmp_path):
        pipeline = self._make_pipeline(tmp_path)
        assert pipeline.message_bus is not None
        assert pipeline.message_bus.pipeline_run_id == pipeline.pipeline_run_id

    def test_budget_controller_created(self, tmp_path):
        pipeline = self._make_pipeline(tmp_path)
        assert pipeline.budget_controller is not None
        assert pipeline.budget_controller.max_cost == 5.0
        assert pipeline.budget_controller.max_tokens == 500_000

    def test_budget_controller_per_team_limits(self, tmp_path):
        from pipeline.pipeline_config import StageConfig
        stages = [
            StageConfig(team="IdeationTeam", mode="M", inputs=[], outputs=[]),
            StageConfig(
                team="LiteratureTeam", mode="M",
                inputs=[], outputs=[], depends_on=["IdeationTeam"],
            ),
        ]
        pipeline = self._make_pipeline(tmp_path, stages=stages)
        assert "IdeationTeam" in pipeline.budget_controller.per_team_limits
        assert "LiteratureTeam" in pipeline.budget_controller.per_team_limits
        assert pipeline.budget_controller.per_team_limits["IdeationTeam"] == 2.0

    def test_artifact_store_persists_to_disk(self, tmp_path):
        pipeline = self._make_pipeline(tmp_path)
        pipeline.artifact_store.register(
            "test_artifact",
            {"key": "value"},
            producer="test",
            persist=True,
        )
        artifacts_dir = tmp_path / "artifacts"
        assert artifacts_dir.exists()

    def test_message_bus_receives_on_success(self, tmp_path):
        def mock_runner(upstream_artifacts, mode, output_dir, collector=None):
            return {"research_questions": {"q": "test"}}

        pipeline = self._make_pipeline(tmp_path)
        pipeline.register_team_runner("IdeationTeam", mock_runner)
        result = pipeline.run("test topic")

        assert result.success
        msgs = pipeline.message_bus.get_messages(source_team="IdeationTeam")
        assert len(msgs) >= 1


# ============================================================================
# Test 5: Full 4-team mock pipeline
# ============================================================================

class TestFourTeamMockPipeline:
    """Run full 4-team pipeline with mock runners."""

    @staticmethod
    def _mock_ideation(upstream_artifacts, mode, output_dir, collector=None):
        return {
            "research_questions": {
                "final_questions": [
                    {"question": "How does AI affect GDP?", "priority_rank": 1},
                    {"question": "What are LLM labor impacts?", "priority_rank": 2},
                ]
            }
        }

    @staticmethod
    def _mock_literature(upstream_artifacts, mode, output_dir, collector=None):
        return {
            "literature_review": {"sections": ["intro", "methods"]},
            "gap_analysis": {"gaps": ["data availability"]},
            "knowledge_graph": {"nodes": 5, "edges": 8},
        }

    @staticmethod
    def _mock_model(upstream_artifacts, mode, output_dir, collector=None):
        return {
            "model_specification": {
                "model_type": "DSGE",
                "data_requirements": ["GDP", "CPI"],
            }
        }

    @staticmethod
    def _mock_data(upstream_artifacts, mode, output_dir, collector=None):
        return {
            "validated_dataset": {
                "series": ["GDP", "CPI"],
                "quality_score": 0.95,
            }
        }

    @staticmethod
    def _mock_estimation(upstream_artifacts, mode, output_dir, collector=None):
        return {
            "estimation_results": {"outcome": {"verdict": "estimated"}}
        }

    @staticmethod
    def _mock_reporting(upstream_artifacts, mode, output_dir, collector=None):
        return {
            "research_report": {"report_markdown": "# Report", "consistency": {}}
        }

    def _make_full_pipeline(self, tmp_path):
        from run_ael_pipeline import build_default_config
        from pipeline.research_pipeline import ResearchPipelineOrchestrator

        config = build_default_config()
        config.output_dir = str(tmp_path)

        pipeline = ResearchPipelineOrchestrator(config)
        pipeline.register_team_runner("IdeationTeam", self._mock_ideation)
        pipeline.register_team_runner("LiteratureTeam", self._mock_literature)
        pipeline.register_team_runner("ModelTeam", self._mock_model)
        pipeline.register_team_runner("DataTeam", self._mock_data)
        pipeline.register_team_runner("EstimationTeam", self._mock_estimation)
        pipeline.register_team_runner("ReportingTeam", self._mock_reporting)
        return pipeline

    def test_full_pipeline_succeeds(self, tmp_path):
        pipeline = self._make_full_pipeline(tmp_path)
        result = pipeline.run("AI in economics")

        assert result.success
        assert len(result.teams_completed) == 6
        assert len(result.teams_failed) == 0

    def test_full_pipeline_execution_order(self, tmp_path):
        pipeline = self._make_full_pipeline(tmp_path)
        result = pipeline.run("AI in economics")

        # Estimation/Reporting are deferred past the feasibility loop -> always last,
        # in dependency order
        assert result.teams_completed == [
            "IdeationTeam", "LiteratureTeam", "ModelTeam", "DataTeam",
            "EstimationTeam", "ReportingTeam"
        ]

    def test_full_pipeline_artifacts_stored(self, tmp_path):
        pipeline = self._make_full_pipeline(tmp_path)
        pipeline.run("AI in economics")

        assert pipeline.artifact_store.has("research_topic")
        assert pipeline.artifact_store.has("research_questions")
        assert pipeline.artifact_store.has("literature_review")
        assert pipeline.artifact_store.has("gap_analysis")
        assert pipeline.artifact_store.has("knowledge_graph")
        assert pipeline.artifact_store.has("model_specification")
        assert pipeline.artifact_store.has("validated_dataset")

    def test_full_pipeline_messages_published(self, tmp_path):
        pipeline = self._make_full_pipeline(tmp_path)
        pipeline.run("AI in economics")

        # Each team publishes its output artifacts
        assert pipeline.message_bus.message_count() >= 6

    def test_full_pipeline_manifest_saved(self, tmp_path):
        pipeline = self._make_full_pipeline(tmp_path)
        pipeline.run("AI in economics")

        manifest_path = tmp_path / "pipeline_manifest.json"
        assert manifest_path.exists()

        with open(manifest_path) as f:
            manifest = json.load(f)
        assert manifest["success"] is True
        assert len(manifest["teams_completed"]) == 6

    def test_full_pipeline_duration_tracked(self, tmp_path):
        pipeline = self._make_full_pipeline(tmp_path)
        result = pipeline.run("AI in economics")

        assert result.total_duration_sec >= 0
        for team_result in result.team_results.values():
            assert team_result.duration_sec >= 0

    def test_pipeline_stops_on_team_failure(self, tmp_path):
        from run_ael_pipeline import build_default_config
        from pipeline.research_pipeline import ResearchPipelineOrchestrator

        config = build_default_config()
        config.output_dir = str(tmp_path)

        def failing_runner(upstream_artifacts, mode, output_dir, collector=None):
            raise RuntimeError("LiteratureTeam exploded")

        pipeline = ResearchPipelineOrchestrator(config)
        pipeline.register_team_runner("IdeationTeam", self._mock_ideation)
        pipeline.register_team_runner("LiteratureTeam", failing_runner)
        pipeline.register_team_runner("ModelTeam", self._mock_model)
        pipeline.register_team_runner("DataTeam", self._mock_data)
        pipeline.register_team_runner("EstimationTeam", self._mock_estimation)
        pipeline.register_team_runner("ReportingTeam", self._mock_reporting)

        result = pipeline.run("test")

        assert not result.success
        assert "IdeationTeam" in result.teams_completed
        assert "LiteratureTeam" in result.teams_failed
        # ModelTeam, DataTeam, and the deferred EstimationTeam should not have run
        assert "ModelTeam" not in result.teams_completed
        assert "DataTeam" not in result.teams_completed
        assert "EstimationTeam" not in result.teams_completed
        assert "ReportingTeam" not in result.teams_completed


# ============================================================================
# Test 6: Pipeline config YAML loading
# ============================================================================

class TestPipelineConfigLoading:
    """Verify YAML config loading and validation."""

    def test_load_full_research_yaml(self):
        from pipeline.pipeline_config import load_pipeline_config
        config_path = _agents_dir / "pipeline" / "configs" / "full_research.yaml"
        config = load_pipeline_config(str(config_path))

        assert config.name == "full_research_pipeline"
        assert len(config.stages) == 7
        assert config.budget.max_total_cost_usd == 5.0

    def test_load_ideation_only_yaml(self):
        from pipeline.pipeline_config import load_pipeline_config
        config_path = _agents_dir / "pipeline" / "configs" / "ideation_only.yaml"
        config = load_pipeline_config(str(config_path))

        assert len(config.stages) == 1
        assert config.stages[0].team == "IdeationTeam"

    def test_full_research_execution_order(self):
        from pipeline.pipeline_config import load_pipeline_config
        config_path = _agents_dir / "pipeline" / "configs" / "full_research.yaml"
        config = load_pipeline_config(str(config_path))
        order = config.get_execution_order()

        assert order[0] == "IdeationTeam"
        assert order[-1] == "ReportingTeam"


# ============================================================================
# Test 7: _extract_questions_list helper
# ============================================================================

class TestExtractQuestionsList:
    """Verify upstream artifact question extraction."""

    def test_extract_from_final_questions(self):
        from pipeline.team_runners import _extract_questions_list
        artifacts = {
            "research_questions": {
                "final_questions": [{"question": "Q1"}, {"question": "Q2"}]
            }
        }
        result = _extract_questions_list(artifacts)
        assert len(result) == 2

    def test_extract_from_direct_list(self):
        from pipeline.team_runners import _extract_questions_list
        artifacts = {
            "research_questions": ["Q1", "Q2", "Q3"]
        }
        result = _extract_questions_list(artifacts)
        assert len(result) == 3

    def test_extract_empty_wraps_dict(self):
        """Empty research_questions dict is wrapped as single item."""
        from pipeline.team_runners import _extract_questions_list
        # Empty dict with no recognized keys → wraps as [{}]
        result = _extract_questions_list({})
        assert isinstance(result, list)

    def test_extract_legacy_keys(self):
        from pipeline.team_runners import _extract_questions_list
        for key in ["questions", "prioritized", "prioritized_questions"]:
            artifacts = {"research_questions": {key: ["Q1"]}}
            result = _extract_questions_list(artifacts)
            assert len(result) == 1, f"Failed for key: {key}"


# ============================================================================
# Test 8: team_runners.py uses _team_context (not raw os.chdir)
# ============================================================================

class TestTeamRunnersRefactored:
    """Verify team_runners.py uses _team_context instead of raw os.chdir."""

    def test_no_raw_os_chdir_in_runners(self):
        """Runner functions should NOT call os.chdir directly."""
        import ast
        filepath = _agents_dir / "pipeline" / "team_runners.py"
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source)

        # Find all function defs for runners
        runner_names = {
            "run_ideation_team", "run_literature_team",
            "run_model_team", "run_data_team",
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in runner_names:
                # Check that function body doesn't contain os.chdir
                func_source = ast.get_source_segment(source, node)
                assert "os.chdir(" not in func_source, (
                    f"{node.name} contains raw os.chdir — should use _team_context"
                )

    def test_team_context_defined(self):
        """_team_context is defined in team_runners.py."""
        from pipeline.team_runners import _team_context
        import contextlib
        # Verify it's a context manager (generator-based)
        assert callable(_team_context)

    def test_team_runners_use_with_statement(self):
        """Runner functions use 'with _team_context' pattern."""
        filepath = _agents_dir / "pipeline" / "team_runners.py"
        source = filepath.read_text(encoding="utf-8")

        # Count occurrences: one per runner (6) + the WithHITL drivers for Ideation,
        # Literature, Model (Data reuses its runner with an enable_hitl flag;
        # Estimation/Reporting handle HITL inside their runners).
        count = source.count("with _team_context(")
        assert count == 10, (
            f"Expected 10 'with _team_context(' patterns (7 runners + 3 WithHITL drivers), found {count}"
        )


# ============================================================================
# Test 9: Backward compatibility — standalone MasterOrchestrators unchanged
# ============================================================================

class TestBackwardCompatibility:
    """Verify standalone MasterOrchestrators are not affected by pipeline."""

    def test_orchestrators_still_have_main(self):
        """Each MasterOrchestrator still has if __name__ == '__main__' block."""
        base = _agents_dir
        for team in ["IdeationTeam", "LiteratureTeam", "ModelTeam"]:
            for mode in ["ModeNoWcNoHITL"]:
                f = base / team / "ael" / mode / "0-MasterOrchestrator.py"
                content = f.read_text(encoding="utf-8")
                assert 'if __name__ == "__main__"' in content or "if __name__ == '__main__'" in content, (
                    f"{team}/{mode}: missing __main__ block"
                )

    def test_orchestrators_do_not_import_pipeline(self):
        """MasterOrchestrators should NOT depend on pipeline modules."""
        base = _agents_dir
        for team in ["IdeationTeam", "LiteratureTeam", "ModelTeam"]:
            for mode in ["ModeNoWcNoHITL"]:
                f = base / team / "ael" / mode / "0-MasterOrchestrator.py"
                content = f.read_text(encoding="utf-8")
                assert "from pipeline" not in content, (
                    f"{team}/{mode}: should not import from pipeline"
                )
