# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
V0.7 feature-flag loader.

Each V0.7 capability has a boolean flag in ``ael_config.yaml`` under the
``feature_flags:`` section. Callers resolve flags via ``is_enabled(name)``,
which honours (highest to lowest priority):

  1. Env var ``AEL_<FLAG_NAME_UPPER>`` (truthy: ``1``, ``true``, ``yes``, ``on``)
  2. ``feature_flags.<flag_name>`` in ``ael_config.yaml``
  3. Built-in default (see :data:`DEFAULT_FLAGS`)

Usage:
    from shared.feature_flags import is_enabled

    if is_enabled("causal_fm_enabled"):
        # invoke CausalFMClient
        ...

    if is_enabled("simulation_enabled"):
        # run 4-SimulationStage
        ...
"""

from __future__ import annotations

import os
from typing import Any, Dict

from shared.model_config import load_model_config


# Built-in defaults for every V0.7 feature flag. Change a default here to
# change the baseline behaviour when neither env var nor ael_config.yaml
# expresses an explicit preference.
DEFAULT_FLAGS: Dict[str, bool] = {
    # Economics-domain intelligence
    "causal_fm_enabled":       True,   # CalibrationStage emits causal_estimate (refusal without observed data)
    "simulation_enabled":      False,  # Optional 4-SimulationStage
    "theory_transfer_enabled": False,  # Chib-Tan 90/10 pre-train + fine-tune

    # Verification-grade outputs
    "citation_verifier_enabled": True,  # Post-process Literature + Ideation
    "falsifier_enabled":         True,  # Theorist + Falsifier pair
    "memory_guard_v2":           True,  # Trust-aware retrieval + belief drift

    # Platform modernization
    "semantic_cache_enabled": False,   # Off by default for K=5 reproducibility
}


_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


def _env_var_name(flag: str) -> str:
    return f"AEL_{flag.upper()}"


def is_enabled(
    flag: str,
    *,
    config: Dict[str, Any] | None = None,
    default: bool | None = None,
) -> bool:
    """Return True if ``flag`` is enabled (env > yaml > default).

    Parameters
    ----------
    flag
        Flag name (e.g. ``"causal_fm_enabled"``).
    config
        Optional pre-loaded model-config dict. If omitted, the YAML is loaded
        from the default path.
    default
        Optional override for the built-in default. Use only in tests.
    """
    # Env var takes precedence
    raw = os.environ.get(_env_var_name(flag))
    if raw is not None:
        r = raw.strip().lower()
        if r in _TRUTHY:
            return True
        if r in _FALSY:
            return False
        # Any other value → fall through to config lookup

    if config is None:
        config = load_model_config()
    flags = config.get("feature_flags", {}) if isinstance(config, dict) else {}
    if flag in flags:
        return bool(flags[flag])

    if default is not None:
        return default
    return DEFAULT_FLAGS.get(flag, False)


def all_flags(config: Dict[str, Any] | None = None) -> Dict[str, bool]:
    """Resolve every known flag using current env + config + defaults."""
    if config is None:
        config = load_model_config()
    return {name: is_enabled(name, config=config) for name in DEFAULT_FLAGS}


def set_flag(flag: str, enabled: bool) -> None:
    """Programmatically set a flag via its env-var form (for tests)."""
    os.environ[_env_var_name(flag)] = "1" if enabled else "0"


def clear_flag(flag: str) -> None:
    """Remove the env-var override for a flag (falls back to yaml/default)."""
    os.environ.pop(_env_var_name(flag), None)
