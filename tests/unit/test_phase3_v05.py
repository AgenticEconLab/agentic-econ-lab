# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
AEL V0.5 Phase 3 — Reliability Hardening Tests

Tests for StateGuard, RoleEnforcer, and CheckpointManager.
"""

import json
import os
import sys
import tempfile
import time

import pytest


# ── StateGuard ────────────────────────────────────────────────────────

class TestStateGuardCore:
    """Core hash signing and verification."""

    def test_sign_output_returns_hex_string(self):
        from shared.reliability.state_guard import StateGuard
        guard = StateGuard()
        h = guard.sign_output({"key": "value"})
        assert isinstance(h, str)
        assert len(h) == 16
        # hex characters only
        assert all(c in "0123456789abcdef" for c in h)

    def test_sign_output_deterministic(self):
        from shared.reliability.state_guard import StateGuard
        guard = StateGuard()
        data = {"items": [1, 2, 3], "name": "test"}
        h1 = guard.sign_output(data)
        h2 = guard.sign_output(data)
        assert h1 == h2

    def test_different_data_different_hash(self):
        from shared.reliability.state_guard import StateGuard
        guard = StateGuard()
        h1 = guard.sign_output({"a": 1})
        h2 = guard.sign_output({"a": 2})
        assert h1 != h2

    def test_verify_input_matching(self):
        from shared.reliability.state_guard import StateGuard
        guard = StateGuard()
        data = {"output": "some result"}
        h = guard.sign_output(data)
        assert guard.verify_input(data, h) is True

    def test_verify_input_mismatch(self):
        from shared.reliability.state_guard import StateGuard
        guard = StateGuard()
        data = {"output": "some result"}
        assert guard.verify_input(data, "0000000000000000") is False

    def test_verify_or_raise_success(self):
        from shared.reliability.state_guard import StateGuard
        guard = StateGuard()
        data = {"x": 42}
        h = guard.sign_output(data)
        guard.verify_or_raise(data, h, context="test")  # Should not raise

    def test_verify_or_raise_failure(self):
        from shared.reliability.state_guard import StateGuard, StateMismatchError
        guard = StateGuard()
        with pytest.raises(StateMismatchError) as exc_info:
            guard.verify_or_raise({"x": 1}, "badhash_________", context="Stage2→Stage3")
        assert "State mismatch" in str(exc_info.value)
        assert "Stage2→Stage3" in str(exc_info.value)

    def test_state_mismatch_error_attributes(self):
        from shared.reliability.state_guard import StateMismatchError
        err = StateMismatchError("test", expected_hash="aaa", actual_hash="bbb")
        assert err.expected_hash == "aaa"
        assert err.actual_hash == "bbb"


class TestStateGuardStageTracking:
    """Stage name tracking with stored hashes."""

    def test_sign_with_stage_name(self):
        from shared.reliability.state_guard import StateGuard
        guard = StateGuard()
        h = guard.sign_output({"data": 1}, stage_name="Stage1")
        assert guard.get_hash("Stage1") == h

    def test_get_hash_unknown_stage(self):
        from shared.reliability.state_guard import StateGuard
        guard = StateGuard()
        assert guard.get_hash("NonexistentStage") is None

    def test_get_all_hashes(self):
        from shared.reliability.state_guard import StateGuard
        guard = StateGuard()
        guard.sign_output({"a": 1}, stage_name="S1")
        guard.sign_output({"b": 2}, stage_name="S2")
        all_h = guard.get_all_hashes()
        assert "S1" in all_h
        assert "S2" in all_h
        assert len(all_h) == 2

    def test_clear(self):
        from shared.reliability.state_guard import StateGuard
        guard = StateGuard()
        guard.sign_output({"a": 1}, stage_name="S1")
        guard.clear()
        assert guard.get_all_hashes() == {}

    def test_sign_output_sort_keys(self):
        """Key order shouldn't matter."""
        from shared.reliability.state_guard import StateGuard
        guard = StateGuard()
        h1 = guard.sign_output({"b": 2, "a": 1})
        h2 = guard.sign_output({"a": 1, "b": 2})
        assert h1 == h2


class TestStateGuardEdgeCases:
    """Edge cases for state guard."""

    def test_empty_dict(self):
        from shared.reliability.state_guard import StateGuard
        guard = StateGuard()
        h = guard.sign_output({})
        assert isinstance(h, str) and len(h) == 16

    def test_nested_data(self):
        from shared.reliability.state_guard import StateGuard
        guard = StateGuard()
        data = {"nested": {"deep": {"value": [1, 2, 3]}}}
        h = guard.sign_output(data)
        assert guard.verify_input(data, h)

    def test_list_data(self):
        from shared.reliability.state_guard import StateGuard
        guard = StateGuard()
        h = guard.sign_output([1, 2, 3])
        assert guard.verify_input([1, 2, 3], h)

    def test_string_data(self):
        from shared.reliability.state_guard import StateGuard
        guard = StateGuard()
        h = guard.sign_output("hello world")
        assert guard.verify_input("hello world", h)

    def test_class_attribute_access(self):
        """StateMismatchError accessible via StateGuard.StateMismatchError."""
        from shared.reliability.state_guard import StateGuard
        assert StateGuard.StateMismatchError is not None


# ── RoleEnforcer ──────────────────────────────────────────────────────

class TestRoleEnforcerCore:
    """Core role enforcement checks."""

    def test_no_violations_clean_output(self):
        from shared.reliability.role_enforcer import RoleEnforcer
        enforcer = RoleEnforcer()
        violations = enforcer.check_output("CorpusScout", "Found 10 relevant papers.")
        assert len(violations) == 0

    def test_forbidden_pattern_detected(self):
        from shared.reliability.role_enforcer import RoleEnforcer
        enforcer = RoleEnforcer()
        violations = enforcer.check_output("CorpusScout", "I believe this is the best approach.")
        assert len(violations) == 1
        assert violations[0].violation_type == "forbidden_pattern"

    def test_multiple_forbidden_patterns(self):
        from shared.reliability.role_enforcer import RoleEnforcer
        enforcer = RoleEnforcer()
        violations = enforcer.check_output(
            "DataCollector",
            "The data suggests we can conclude that X is true."
        )
        assert len(violations) == 2  # both patterns match

    def test_output_too_long(self):
        from shared.reliability.role_enforcer import RoleEnforcer
        enforcer = RoleEnforcer()
        long_output = "x" * 60000
        violations = enforcer.check_output("TrendSurfer", long_output)
        assert any(v.violation_type == "output_too_long" for v in violations)

    def test_output_within_limit(self):
        from shared.reliability.role_enforcer import RoleEnforcer
        enforcer = RoleEnforcer()
        short_output = "x" * 100
        violations = enforcer.check_output("TrendSurfer", short_output)
        assert not any(v.violation_type == "output_too_long" for v in violations)

    def test_unknown_agent_no_violations(self):
        from shared.reliability.role_enforcer import RoleEnforcer
        enforcer = RoleEnforcer()
        violations = enforcer.check_output("UnknownAgent", "I recommend everything.")
        assert len(violations) == 0

    def test_dict_output_checked(self):
        from shared.reliability.role_enforcer import RoleEnforcer
        enforcer = RoleEnforcer()
        violations = enforcer.check_output(
            "TrendSurfer",
            {"text": "I recommend this topic"}
        )
        assert len(violations) >= 1


class TestRoleEnforcerCustomConstraints:
    """Custom constraint configuration."""

    def test_custom_constraints(self):
        from shared.reliability.role_enforcer import RoleEnforcer
        enforcer = RoleEnforcer({
            "MyAgent": {
                "forbidden_patterns": [r"(?i)\bFORBIDDEN\b"],
                "max_output_length": 100,
            }
        })
        violations = enforcer.check_output("MyAgent", "This is FORBIDDEN text")
        assert len(violations) == 1

    def test_add_constraint(self):
        from shared.reliability.role_enforcer import RoleEnforcer
        enforcer = RoleEnforcer({"ExistingAgent": {"forbidden_patterns": [], "max_output_length": 1000}})
        enforcer.add_constraint("AddedAgent", {
            "forbidden_patterns": [r"(?i)\bno\b"],
            "max_output_length": 50,
        })
        violations = enforcer.check_output("AddedAgent", "No way this is short enough to pass")
        assert len(violations) >= 1

    def test_get_constraint(self):
        from shared.reliability.role_enforcer import RoleEnforcer
        enforcer = RoleEnforcer()
        constraint = enforcer.get_constraint("TrendSurfer")
        assert constraint is not None
        assert "forbidden_patterns" in constraint

    def test_get_constraint_missing(self):
        from shared.reliability.role_enforcer import RoleEnforcer
        enforcer = RoleEnforcer()
        assert enforcer.get_constraint("Nonexistent") is None


class TestRoleViolation:
    """RoleViolation data class."""

    def test_violation_attributes(self):
        from shared.reliability.role_enforcer import RoleViolation
        v = RoleViolation(
            agent="TestAgent",
            violation_type="forbidden_pattern",
            message="Found pattern",
            severity="error",
        )
        assert v.agent == "TestAgent"
        assert v.violation_type == "forbidden_pattern"
        assert v.message == "Found pattern"
        assert v.severity == "error"

    def test_violation_repr(self):
        from shared.reliability.role_enforcer import RoleViolation
        v = RoleViolation("A", "t", "m")
        assert "RoleViolation" in repr(v)

    def test_default_severity_warning(self):
        from shared.reliability.role_enforcer import RoleViolation
        v = RoleViolation("A", "t", "m")
        assert v.severity == "warning"


class TestDefaultRoleConstraints:
    """Verify default constraints are properly defined."""

    def test_default_constraints_exist(self):
        from shared.reliability.role_enforcer import DEFAULT_ROLE_CONSTRAINTS
        assert "TrendSurfer" in DEFAULT_ROLE_CONSTRAINTS
        assert "CorpusScout" in DEFAULT_ROLE_CONSTRAINTS
        assert "InsightSummarizer" in DEFAULT_ROLE_CONSTRAINTS
        assert "DataCollector" in DEFAULT_ROLE_CONSTRAINTS

    def test_each_constraint_has_required_keys(self):
        from shared.reliability.role_enforcer import DEFAULT_ROLE_CONSTRAINTS
        for agent, constraint in DEFAULT_ROLE_CONSTRAINTS.items():
            assert "allowed_output_types" in constraint, f"{agent} missing allowed_output_types"
            assert "forbidden_patterns" in constraint, f"{agent} missing forbidden_patterns"
            assert "max_output_length" in constraint, f"{agent} missing max_output_length"

    def test_trendsurfer_forbids_recommend(self):
        from shared.reliability.role_enforcer import RoleEnforcer
        enforcer = RoleEnforcer()
        violations = enforcer.check_output("TrendSurfer", "I recommend investing in AI.")
        assert len(violations) >= 1

    def test_datacollector_forbids_conclusions(self):
        from shared.reliability.role_enforcer import RoleEnforcer
        enforcer = RoleEnforcer()
        violations = enforcer.check_output("DataCollector", "We can conclude that GDP grew.")
        assert len(violations) >= 1


# ── CheckpointManager ────────────────────────────────────────────────

class TestCheckpointManagerCore:
    """Core checkpoint save/load/resume."""

    def test_save_creates_file(self):
        from shared.reliability.checkpoint import CheckpointManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(checkpoint_dir=tmpdir)
            path = mgr.save("Stage1", {"result": "ok"})
            assert os.path.exists(path)

    def test_save_file_is_valid_json(self):
        from shared.reliability.checkpoint import CheckpointManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(checkpoint_dir=tmpdir)
            path = mgr.save("Stage1", {"result": "ok"})
            with open(path, "r") as f:
                data = json.load(f)
            assert data["stage_name"] == "Stage1"
            assert data["data"]["result"] == "ok"

    def test_load_latest(self):
        from shared.reliability.checkpoint import CheckpointManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(checkpoint_dir=tmpdir)
            mgr.save("Stage1", {"v": 1})
            time.sleep(0.01)
            mgr.save("Stage2", {"v": 2})
            result = mgr.load_latest()
            assert result is not None
            stage_name, data, metadata = result
            assert stage_name == "Stage2"
            assert data["v"] == 2

    def test_load_latest_empty(self):
        from shared.reliability.checkpoint import CheckpointManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(checkpoint_dir=tmpdir)
            assert mgr.load_latest() is None

    def test_can_resume_true(self):
        from shared.reliability.checkpoint import CheckpointManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(checkpoint_dir=tmpdir)
            mgr.save("Stage1", {"v": 1})
            assert mgr.can_resume() is True

    def test_can_resume_false(self):
        from shared.reliability.checkpoint import CheckpointManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(checkpoint_dir=tmpdir)
            assert mgr.can_resume() is False


class TestCheckpointManagerListing:
    """Listing and tracking checkpoints."""

    def test_list_checkpoints_ordered(self):
        from shared.reliability.checkpoint import CheckpointManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(checkpoint_dir=tmpdir)
            mgr.save("S1", {"v": 1})
            time.sleep(0.01)
            mgr.save("S2", {"v": 2})
            time.sleep(0.01)
            mgr.save("S3", {"v": 3})
            cps = mgr.list_checkpoints()
            assert len(cps) == 3
            assert cps[0].stage_name == "S1"
            assert cps[2].stage_name == "S3"

    def test_get_completed_stages(self):
        from shared.reliability.checkpoint import CheckpointManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(checkpoint_dir=tmpdir)
            mgr.save("StageA", {"v": 1})
            mgr.save("StageB", {"v": 2})
            stages = mgr.get_completed_stages()
            assert "StageA" in stages
            assert "StageB" in stages


class TestCheckpointManagerCleanup:
    """Checkpoint cleanup operations."""

    def test_clean_all(self):
        from shared.reliability.checkpoint import CheckpointManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(checkpoint_dir=tmpdir)
            mgr.save("S1", {"v": 1})
            mgr.save("S2", {"v": 2})
            mgr.clean(keep_latest=0)
            assert not mgr.can_resume()

    def test_clean_keep_latest(self):
        from shared.reliability.checkpoint import CheckpointManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(checkpoint_dir=tmpdir)
            mgr.save("S1", {"v": 1})
            mgr.save("S2", {"v": 2})
            mgr.save("S3", {"v": 3})
            mgr.clean(keep_latest=1)
            cps = mgr.list_checkpoints()
            assert len(cps) == 1


class TestCheckpointManagerMetadata:
    """Metadata and pipeline_run_id support."""

    def test_save_with_metadata(self):
        from shared.reliability.checkpoint import CheckpointManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(checkpoint_dir=tmpdir)
            path = mgr.save("S1", {"v": 1}, metadata={"duration": 120, "items": 25})
            with open(path, "r") as f:
                data = json.load(f)
            assert data["metadata"]["duration"] == 120
            assert data["metadata"]["items"] == 25

    def test_pipeline_run_id(self):
        from shared.reliability.checkpoint import CheckpointManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(checkpoint_dir=tmpdir, pipeline_run_id="run-123")
            path = mgr.save("S1", {"v": 1})
            with open(path, "r") as f:
                data = json.load(f)
            assert data["pipeline_run_id"] == "run-123"

    def test_checkpoint_repr(self):
        from shared.reliability.checkpoint import Checkpoint
        cp = Checkpoint("TestStage", {"v": 1}, {}, 1000.0)
        assert "TestStage" in repr(cp)

    def test_checkpoint_file_path_attribute(self):
        from shared.reliability.checkpoint import CheckpointManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(checkpoint_dir=tmpdir)
            mgr.save("S1", {"v": 1})
            cps = mgr.list_checkpoints()
            assert cps[0].file_path != ""

    def test_sequential_file_naming(self):
        from shared.reliability.checkpoint import CheckpointManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(checkpoint_dir=tmpdir)
            p1 = mgr.save("S1", {"v": 1})
            p2 = mgr.save("S2", {"v": 2})
            assert "checkpoint_000" in p1
            assert "checkpoint_001" in p2

    def test_corrupted_checkpoint_skipped(self):
        from shared.reliability.checkpoint import CheckpointManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(checkpoint_dir=tmpdir)
            mgr.save("S1", {"v": 1})
            # Write a corrupted file
            corrupt_path = os.path.join(tmpdir, "checkpoint_999_bad.json")
            with open(corrupt_path, "w") as f:
                f.write("{invalid json")
            cps = mgr.list_checkpoints()
            assert len(cps) == 1  # only valid one loaded


class TestCheckpointManagerDirCreation:
    """Directory creation behavior."""

    def test_creates_dir_if_not_exists(self):
        from shared.reliability.checkpoint import CheckpointManager
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "nested", "checkpoints")
            mgr = CheckpointManager(checkpoint_dir=subdir)
            assert os.path.isdir(subdir)


# ── Imports via __init__ ──────────────────────────────────────────────

class TestReliabilityImports:
    """Verify reliability module exports."""

    def test_import_state_guard(self):
        from shared.reliability import StateGuard
        assert StateGuard is not None

    def test_import_checkpoint_manager(self):
        from shared.reliability import CheckpointManager
        assert CheckpointManager is not None

    def test_import_role_enforcer(self):
        from shared.reliability import RoleEnforcer
        assert RoleEnforcer is not None

    def test_all_exports(self):
        import shared.reliability
        assert "StateGuard" in shared.reliability.__all__
        assert "CheckpointManager" in shared.reliability.__all__
        assert "RoleEnforcer" in shared.reliability.__all__


# ── Cross-component: StateGuard + CheckpointManager ──────────────────

class TestReliabilityCrossComponent:
    """Cross-component interactions."""

    def test_state_guard_with_checkpoint_data(self):
        """Checkpoint data can be verified with StateGuard."""
        from shared.reliability.state_guard import StateGuard
        from shared.reliability.checkpoint import CheckpointManager
        with tempfile.TemporaryDirectory() as tmpdir:
            guard = StateGuard()
            mgr = CheckpointManager(checkpoint_dir=tmpdir)

            data = {"research_questions": ["Q1", "Q2"]}
            h = guard.sign_output(data, stage_name="IdeationTeam:SourcingStage")
            mgr.save("IdeationTeam:SourcingStage", data, metadata={"hash": h})

            # Load and verify
            result = mgr.load_latest()
            assert result is not None
            stage_name, loaded_data, metadata = result
            assert guard.verify_input(loaded_data, metadata["hash"])

    def test_role_enforcer_on_checkpoint_output(self):
        """Role enforcer can validate data loaded from checkpoint."""
        from shared.reliability.role_enforcer import RoleEnforcer
        from shared.reliability.checkpoint import CheckpointManager
        with tempfile.TemporaryDirectory() as tmpdir:
            enforcer = RoleEnforcer()
            mgr = CheckpointManager(checkpoint_dir=tmpdir)

            clean_output = "Found 5 trending topics in macroeconomics."
            mgr.save("TrendSurfer", {"text": clean_output})

            result = mgr.load_latest()
            _, loaded_data, _ = result
            violations = enforcer.check_output("TrendSurfer", loaded_data["text"])
            assert len(violations) == 0


# ── BudgetController Loop Limits ──────────────────────────────────────

class TestBudgetControllerLoopLimits:
    """Tests for coordination loop limits in BudgetController."""

    def test_record_retry_increments_count(self):
        from shared.guardrails.budget_controller import BudgetController
        bc = BudgetController(max_retries_per_stage=5, max_total_retries=20)
        bc.record_retry("Stage1")
        counts = bc.get_retry_counts()
        assert counts["stage_retries"]["Stage1"] == 1
        assert counts["total_retries"] == 1

    def test_record_retry_multiple_stages(self):
        from shared.guardrails.budget_controller import BudgetController
        bc = BudgetController(max_retries_per_stage=5, max_total_retries=20)
        bc.record_retry("Stage1")
        bc.record_retry("Stage1")
        bc.record_retry("Stage2")
        counts = bc.get_retry_counts()
        assert counts["stage_retries"]["Stage1"] == 2
        assert counts["stage_retries"]["Stage2"] == 1
        assert counts["total_retries"] == 3

    def test_per_stage_limit_exceeded(self):
        from shared.guardrails.budget_controller import BudgetController, BudgetExceededError
        bc = BudgetController(max_retries_per_stage=2, max_total_retries=100)
        bc.record_retry("Stage1")
        bc.record_retry("Stage1")
        with pytest.raises(BudgetExceededError) as exc_info:
            bc.record_retry("Stage1")
        assert exc_info.value.budget_type == "retry_per_stage"

    def test_total_retry_limit_exceeded(self):
        from shared.guardrails.budget_controller import BudgetController, BudgetExceededError
        bc = BudgetController(max_retries_per_stage=5, max_total_retries=3)
        bc.record_retry("S1")
        bc.record_retry("S2")
        bc.record_retry("S3")
        with pytest.raises(BudgetExceededError) as exc_info:
            bc.record_retry("S4")
        assert exc_info.value.budget_type == "retry_total"

    def test_default_loop_limits(self):
        from shared.guardrails.budget_controller import BudgetController
        bc = BudgetController()
        limits = bc.get_limits()
        assert limits["max_retries_per_stage"] == 3
        assert limits["max_total_retries"] == 10

    def test_custom_loop_limits(self):
        from shared.guardrails.budget_controller import BudgetController
        bc = BudgetController(max_retries_per_stage=10, max_total_retries=50)
        limits = bc.get_limits()
        assert limits["max_retries_per_stage"] == 10
        assert limits["max_total_retries"] == 50

    def test_get_retry_counts_initial(self):
        from shared.guardrails.budget_controller import BudgetController
        bc = BudgetController()
        counts = bc.get_retry_counts()
        assert counts["total_retries"] == 0
        assert counts["stage_retries"] == {}

    def test_within_limits_no_error(self):
        from shared.guardrails.budget_controller import BudgetController
        bc = BudgetController(max_retries_per_stage=3, max_total_retries=10)
        for i in range(3):
            bc.record_retry(f"Stage{i}")
        # Should not raise — 3 stages × 1 retry each = 3 total, under all limits


# ── Pipeline StateGuard Integration ──────────────────────────────────

class TestPipelineStateGuard:
    """Tests for StateGuard wiring in research_pipeline.py."""

    def test_pipeline_has_state_guard(self):
        from pipeline.pipeline_config import PipelineConfig, StageConfig, BudgetConfig
        from pipeline.research_pipeline import ResearchPipelineOrchestrator
        config = PipelineConfig(
            name="test", mode="ModeNoWcNoHITL",
            stages=[StageConfig(team="IdeationTeam", mode="ModeNoWcNoHITL",
                                inputs=["research_topic"], outputs=["rq"], depends_on=[])],
            budget=BudgetConfig(max_total_cost_usd=1.0, max_per_team_cost_usd=1.0, max_total_tokens=100000),
            output_dir=os.path.join(tempfile.gettempdir(), "test_pipeline_sg"),
        )
        pipeline = ResearchPipelineOrchestrator(config)
        assert hasattr(pipeline, "state_guard")
        from shared.reliability.state_guard import StateGuard
        assert isinstance(pipeline.state_guard, StateGuard)

    def test_state_guard_import_in_pipeline(self):
        """Verify StateGuard is importable from pipeline module."""
        import pipeline.research_pipeline as rp
        assert hasattr(rp, "StateGuard")


# ── Checkpoint in run_ael_pipeline.py ─────────────────────────────────────

class TestRunPipelineCheckpoint:
    """Tests for CheckpointManager wiring in run_ael_pipeline.py."""

    def test_checkpoint_manager_import(self):
        """Verify CheckpointManager is imported in run_ael_pipeline."""
        import run_ael_pipeline
        assert hasattr(run_ael_pipeline, "CheckpointManager")

    def test_build_default_config(self):
        """Verify build_default_config still works."""
        from run_ael_pipeline import build_default_config
        config = build_default_config()
        assert config.name == "full_research_pipeline"
        assert len(config.stages) == 7


# ── StateGuard in MasterOrchestrators ─────────────────────────────────

class TestMasterOrchestratorStateGuard:
    """Verify StateGuard is imported in MasterOrchestrator files."""

    TEAMS_AND_MODES = [
        ("IdeationTeam", "ModeNoWcNoHITL"),
        ("IdeationTeam", "ModeNoWcWithHITL"),
        ("IdeationTeam", "ModeWithWcNoHITL"),
        ("IdeationTeam", "ModeWithWcWithHITL"),
        ("LiteratureTeam", "ModeNoWcNoHITL"),
        ("LiteratureTeam", "ModeNoWcWithHITL"),
        ("LiteratureTeam", "ModeWithWcNoHITL"),
        ("LiteratureTeam", "ModeWithWcWithHITL"),
        ("ModelTeam", "ModeNoWcNoHITL"),
        ("ModelTeam", "ModeNoWcWithHITL"),
        ("ModelTeam", "ModeWithWcNoHITL"),
        ("ModelTeam", "ModeWithWcWithHITL"),
        ("DataTeam", "ModeOpenSourceAPI"),
        ("DataTeam", "ModePremiumSubscribed"),
        ("DataTeam", "ModeUserUploaded"),
    ]

    @pytest.mark.parametrize("team,mode", TEAMS_AND_MODES)
    def test_state_guard_import(self, team, mode):
        """Each MasterOrchestrator imports StateGuard."""
        agents_dir = os.path.join(os.path.dirname(__file__), "..", "..")
        filepath = os.path.join(agents_dir, team, "ael", mode, "0-MasterOrchestrator.py")
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        assert "StateGuard" in content, f"{team}/{mode} missing StateGuard import"

    @pytest.mark.parametrize("team,mode", TEAMS_AND_MODES)
    def test_state_guard_instantiation(self, team, mode):
        """Each MasterOrchestrator instantiates StateGuard."""
        agents_dir = os.path.join(os.path.dirname(__file__), "..", "..")
        filepath = os.path.join(agents_dir, team, "ael", mode, "0-MasterOrchestrator.py")
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        assert "state_guard = StateGuard()" in content or "StateGuard()" in content, \
            f"{team}/{mode} missing StateGuard instantiation"
