# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""The default model is local, and it has one source: ael_config.yaml ``default_model``.

Without AEL_MODEL, every agent must reach the local vLLM default rather than a hosted model, and
no agent may require OPENAI_API_KEY to start: the client checks the key of the provider it
actually calls."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TEAMS = ["IdeationTeam", "LiteratureTeam", "ModelTeam", "DataTeam", "CodeTeam",
         "EstimationTeam", "ReportingTeam"]


@pytest.fixture(autouse=True)
def _no_env_model(monkeypatch):
    monkeypatch.delenv("AEL_MODEL", raising=False)
    monkeypatch.delenv("AEL_MODEL_CONFIG", raising=False)


def test_shipped_default_is_local_vllm():
    from shared.model_config import default_model, _DEFAULT_MODEL
    assert default_model() == "vllm/qwen3.6-27b-fp8"
    assert _DEFAULT_MODEL == "vllm/qwen3.6-27b-fp8"


def test_llmclient_without_model_uses_default(monkeypatch):
    from shared.llm import LLMClient
    client = LLMClient(agent_name="t")
    assert client.model == "vllm/qwen3.6-27b-fp8"
    assert client.provider == "vllm"
    monkeypatch.setenv("AEL_MODEL", "ollama/qwen3-32b")
    assert LLMClient(agent_name="t").model == "ollama/qwen3-32b"


def test_shipped_config_has_no_active_stage_override():
    from shared.model_config import load_model_config, get_stage_model
    config = load_model_config()
    for team in TEAMS:
        assert not isinstance(config.get(team), dict), f"{team} overrides the default model"
    assert get_stage_model(config, "ModelTeam", "TheoryStage") == "vllm/qwen3.6-27b-fp8"


def test_no_stage_names_a_model_or_requires_openai_key():
    literal = re.compile(r'\bmodel\s*=\s*"(gpt-|claude-|gemini-|o\d-)')
    gate = "OpenAI API key required"
    offenders = []
    for team in TEAMS:
        for path in (ROOT / team).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if literal.search(text) or gate in text:
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
