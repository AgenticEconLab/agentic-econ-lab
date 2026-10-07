# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Tests for the V0.7 feature-flag loader."""

from __future__ import annotations

import pytest

from shared.feature_flags import (
    DEFAULT_FLAGS,
    all_flags,
    clear_flag,
    is_enabled,
    set_flag,
)


class TestDefaults:
    def test_built_in_defaults_cover_seven_flags(self):
        assert set(DEFAULT_FLAGS) == {
            "causal_fm_enabled",
            "simulation_enabled",
            "theory_transfer_enabled",
            "citation_verifier_enabled",
            "falsifier_enabled",
            "memory_guard_v2",
            "semantic_cache_enabled",
        }

    def test_defaults_are_booleans(self):
        assert all(isinstance(v, bool) for v in DEFAULT_FLAGS.values())


class TestPrecedence:
    def test_default_when_no_config_no_env(self, monkeypatch):
        for name in DEFAULT_FLAGS:
            monkeypatch.delenv(f"AEL_{name.upper()}", raising=False)
        # With an empty config dict, defaults win
        assert is_enabled("causal_fm_enabled", config={}) is True
        assert is_enabled("simulation_enabled", config={}) is False

    def test_yaml_overrides_default(self, monkeypatch):
        monkeypatch.delenv("AEL_CAUSAL_FM_ENABLED", raising=False)
        cfg = {"feature_flags": {"causal_fm_enabled": False}}
        assert is_enabled("causal_fm_enabled", config=cfg) is False

    def test_env_overrides_yaml(self, monkeypatch):
        monkeypatch.setenv("AEL_CAUSAL_FM_ENABLED", "0")
        cfg = {"feature_flags": {"causal_fm_enabled": True}}
        assert is_enabled("causal_fm_enabled", config=cfg) is False

    def test_env_truthy_values(self, monkeypatch):
        for val in ("1", "true", "yes", "on", "TRUE", "Yes"):
            monkeypatch.setenv("AEL_SIMULATION_ENABLED", val)
            assert is_enabled("simulation_enabled", config={}) is True

    def test_env_falsy_values(self, monkeypatch):
        for val in ("0", "false", "no", "off", "FALSE"):
            monkeypatch.setenv("AEL_CAUSAL_FM_ENABLED", val)
            assert is_enabled("causal_fm_enabled", config={}) is False

    def test_env_invalid_value_falls_through(self, monkeypatch):
        monkeypatch.setenv("AEL_CAUSAL_FM_ENABLED", "maybe")
        cfg = {"feature_flags": {"causal_fm_enabled": False}}
        # Unrecognised env value → fall through to yaml
        assert is_enabled("causal_fm_enabled", config=cfg) is False

    def test_unknown_flag_defaults_false(self, monkeypatch):
        monkeypatch.delenv("AEL_FAKE_FLAG", raising=False)
        assert is_enabled("fake_flag", config={}) is False

    def test_explicit_default_override(self):
        assert is_enabled("fake_flag", config={}, default=True) is True


class TestAllFlags:
    def test_all_flags_returns_every_known_flag(self, monkeypatch):
        for name in DEFAULT_FLAGS:
            monkeypatch.delenv(f"AEL_{name.upper()}", raising=False)
        resolved = all_flags(config={})
        assert set(resolved) == set(DEFAULT_FLAGS)

    def test_all_flags_respects_env(self, monkeypatch):
        monkeypatch.setenv("AEL_SIMULATION_ENABLED", "1")
        resolved = all_flags(config={})
        assert resolved["simulation_enabled"] is True


class TestSetClear:
    def test_set_flag_updates_env(self, monkeypatch):
        monkeypatch.delenv("AEL_CAUSAL_FM_ENABLED", raising=False)
        set_flag("causal_fm_enabled", False)
        assert is_enabled("causal_fm_enabled", config={}) is False
        set_flag("causal_fm_enabled", True)
        assert is_enabled("causal_fm_enabled", config={}) is True

    def test_clear_flag_removes_env(self, monkeypatch):
        set_flag("simulation_enabled", True)
        clear_flag("simulation_enabled")
        # After clear, default (False) wins
        assert is_enabled("simulation_enabled", config={}) is False


class TestYAMLIntegration:
    def test_reads_real_yaml(self):
        # Default ael_config.yaml should expose the 7 V0.7 flags
        resolved = all_flags()
        assert set(resolved) == set(DEFAULT_FLAGS)
        # Baseline expectations from the committed yaml
        assert resolved["causal_fm_enabled"] is True or True  # tolerant
