# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Regression tests for _thinking_kwargs (shared/llm.py).

A prior bug gated disable-thinking on provider=='ollama' only, so the vLLM HPC runs (provider
'vllm') never sent enable_thinking:false -> Qwen reasoned to max_tokens every call -> timeouts.
"""
from shared.llm import _thinking_kwargs


def test_disabled_for_local_qwen_when_env_set(monkeypatch):
    monkeypatch.setenv("AEL_DISABLE_THINKING", "1")
    assert _thinking_kwargs("vllm", "vllm/qwen3.6-27b-fp8") == {"enable_thinking": False}
    assert _thinking_kwargs("ollama", "qwen3") == {"enable_thinking": False}


def test_noop_for_non_qwen_models(monkeypatch):
    # Mistral/Gemma tokenizers 400 on this kwarg -> must be a no-op even if env is set
    monkeypatch.setenv("AEL_DISABLE_THINKING", "1")
    assert _thinking_kwargs("vllm", "mistral-small-3.2-24b") is None
    assert _thinking_kwargs("vllm", "gemma-3-27b-it") is None


def test_noop_for_cloud_providers(monkeypatch):
    monkeypatch.setenv("AEL_DISABLE_THINKING", "1")
    assert _thinking_kwargs("openai", "gpt-4o") is None
    assert _thinking_kwargs("anthropic", "claude-opus-4-8") is None


def test_noop_when_env_unset(monkeypatch):
    monkeypatch.delenv("AEL_DISABLE_THINKING", raising=False)
    assert _thinking_kwargs("vllm", "qwen3.6-27b-fp8") is None
