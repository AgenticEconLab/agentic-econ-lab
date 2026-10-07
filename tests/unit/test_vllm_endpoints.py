# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Per-model vLLM endpoint routing (AEL_VLLM_ENDPOINTS) — lets a diverse committee
address several vLLM servers at once."""

import shared.llm as llm
from shared.llm import _vllm_endpoint, LLMClient


def test_endpoint_lookup(monkeypatch):
    monkeypatch.setenv(
        "AEL_VLLM_ENDPOINTS",
        "qwen3.6-27b-fp8=http://localhost:11434/v1/chat/completions,"
        "mistral-small-3.2-24b-fp8=http://localhost:11435/v1/chat/completions",
    )
    assert _vllm_endpoint("qwen3.6-27b-fp8") == "http://localhost:11434/v1/chat/completions"
    assert _vllm_endpoint("mistral-small-3.2-24b-fp8") == "http://localhost:11435/v1/chat/completions"
    assert _vllm_endpoint("gemma-3-27b-it-fp8") is None  # unmapped -> fallback


def test_endpoint_unset(monkeypatch):
    monkeypatch.delenv("AEL_VLLM_ENDPOINTS", raising=False)
    assert _vllm_endpoint("qwen3.6-27b-fp8") is None


class _Resp:
    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": "hi"}}], "usage": {}}


def test_client_routes_per_model(monkeypatch):
    monkeypatch.delenv("AEL_MODEL", raising=False)
    monkeypatch.setenv("AEL_VLLM_ENDPOINTS",
                       "mistral-small-3.2-24b-fp8=http://localhost:11435/v1/chat/completions")
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        return _Resp()

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    LLMClient(model="vllm/mistral-small-3.2-24b-fp8").invoke([{"role": "user", "content": "hi"}])
    assert captured["url"] == "http://localhost:11435/v1/chat/completions"


def test_allow_env_override_pins_committee_member_model(monkeypatch):
    # AEL_MODEL (the agents' model) must NOT collapse a distinct committee member onto it.
    monkeypatch.setenv("AEL_MODEL", "vllm/qwen3.6-27b-fp8")
    pinned = LLMClient(model="vllm/mistral-small-3.2-24b-fp8", allow_env_override=False)
    assert pinned.model == "vllm/mistral-small-3.2-24b-fp8"
    # default (override allowed) -> AEL_MODEL still wins for ordinary agents
    agent = LLMClient(model="vllm/mistral-small-3.2-24b-fp8")
    assert agent.model == "vllm/qwen3.6-27b-fp8"


def test_committee_factory_pins_member_model(monkeypatch):
    monkeypatch.setenv("AEL_MODEL", "vllm/qwen3.6-27b-fp8")
    from shared.llm_economist import LLMEconomistCommittee

    c = LLMEconomistCommittee(members=["vllm/qwen3.6-27b-fp8", "vllm/mistral-small-3.2-24b-fp8",
                                       "vllm/gemma-3-27b-it-fp8"], log_path="")
    for m in c.members:
        assert c._default_client_factory(m).model == m  # each member keeps its own model


def test_invoke_max_tokens_override_and_clamp(monkeypatch):
    """A per-call max_tokens bounds a verbose call (e.g. equation formulation); it overrides
    the AEL_MAX_TOKENS default but stays clamped to the remaining context window."""
    monkeypatch.delenv("AEL_MODEL", raising=False)
    monkeypatch.delenv("AEL_VLLM_ENDPOINTS", raising=False)
    monkeypatch.delenv("AEL_MAX_TOKENS", raising=False)
    monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:11434/v1/chat/completions")
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["body"] = json
        return _Resp()

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    c = LLMClient(model="vllm/qwen3.6-27b-fp8")

    c.invoke([{"role": "user", "content": "alpha"}], max_tokens=6000)
    assert captured["body"]["max_tokens"] == 6000                 # per-call cap honored

    c.invoke([{"role": "user", "content": "beta"}])
    assert captured["body"]["max_tokens"] == 8192                 # default when no override

    c.invoke([{"role": "user", "content": "gamma"}], max_tokens=999999)
    assert captured["body"]["max_tokens"] < 16384                 # clamped to remaining context


def test_client_falls_back_to_base_url(monkeypatch):
    monkeypatch.delenv("AEL_MODEL", raising=False)
    monkeypatch.delenv("AEL_VLLM_ENDPOINTS", raising=False)
    monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:11434/v1/chat/completions")
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        return _Resp()

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    LLMClient(model="vllm/qwen3.6-27b-fp8").invoke([{"role": "user", "content": "hi"}])
    assert captured["url"] == "http://localhost:11434/v1/chat/completions"
