# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Per-team, per-stage model configuration loader.

Reads ael_config.yaml to determine which LLM model to use for each
pipeline stage. This is AEL's key differentiator: users can assign
different models to different stages without editing any code.

Usage in MasterOrchestrators:
    from shared.model_config import load_model_config, stage_model

    config = load_model_config()

    with stage_model(config, "IdeationTeam", "SourcingStage"):
        orchestrator = stage1.MultiAgentOrchestrator(...)
        results = orchestrator.run(...)

The context manager sets AEL_MODEL env var so all LLMClient instances
created within the block use the configured model.
"""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional

# Default model when no config file exists: the open-weight model of the reported runs,
# served by a local vLLM server (VLLM_BASE_URL, default http://localhost:11434).
_DEFAULT_MODEL = "vllm/qwen3.6-27b-fp8"


def load_model_config(config_path: Optional[str] = None) -> Dict:
    """
    Load model configuration from YAML file.

    Search order:
    1. Explicit config_path argument
    2. AEL_MODEL_CONFIG env var
    3. ael_config.yaml in the repository root (auto-discovered)

    Returns dict with 'default_model' and per-team/stage overrides.
    """
    if config_path is None:
        config_path = os.environ.get("AEL_MODEL_CONFIG")

    if config_path is None:
        # Auto-discover: walk up from shared/ to find ael_config.yaml
        candidates = [
            Path(__file__).resolve().parent.parent / "ael_config.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                config_path = str(candidate)
                break

    if config_path is None or not Path(config_path).exists():
        return {"default_model": _DEFAULT_MODEL}

    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if "default_model" not in data:
            data["default_model"] = _DEFAULT_MODEL
        return data
    except Exception:
        return {"default_model": _DEFAULT_MODEL}


def get_stage_model(config: Dict, team: str, stage: str) -> str:
    """
    Resolve the model for a specific team and stage.

    Priority (highest to lowest):
    1. AEL_MODEL env var (global override, always wins)
    2. Team/stage-specific entry in config (e.g., IdeationTeam.SourcingStage)
    3. default_model in config
    4. Built-in fallback: vllm/qwen3.6-27b-fp8
    """
    # Global env override takes precedence
    env_override = os.environ.get("AEL_MODEL")
    if env_override:
        return env_override

    # Team/stage-specific config
    team_config = config.get(team, {})
    if isinstance(team_config, dict) and stage in team_config:
        return team_config[stage]

    # Default model from config
    return config.get("default_model", _DEFAULT_MODEL)


def default_model() -> str:
    """The model an LLMClient uses when neither the caller nor AEL_MODEL names one:
    ``default_model`` in ael_config.yaml, else the built-in local default."""
    return load_model_config().get("default_model") or _DEFAULT_MODEL


@contextmanager
def stage_model(config: Dict, team: str, stage: str):
    """
    Context manager that sets AEL_MODEL for the duration of a stage.

    All LLMClient instances created within this block will use the
    model configured for this team/stage in ael_config.yaml.

    Usage:
        with stage_model(config, "IdeationTeam", "SourcingStage"):
            orchestrator = MultiAgentOrchestrator(...)
            results = orchestrator.run(...)
    """
    model = get_stage_model(config, team, stage)
    prev = os.environ.get("AEL_MODEL")

    os.environ["AEL_MODEL"] = model
    try:
        yield model
    finally:
        if prev is None:
            os.environ.pop("AEL_MODEL", None)
        else:
            os.environ["AEL_MODEL"] = prev
