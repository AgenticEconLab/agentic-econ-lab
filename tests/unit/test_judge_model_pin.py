# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Judge/evaluator models must be PINNED — AEL_MODEL (the subject/agents' model) must
not collapse a multi-model consensus (Mistral+Gemma) onto one model (or the subject =
self-evaluation). Mirrors the committee's allow_env_override behaviour."""

import pytest


def test_llm_evaluator_pins_judge_model(monkeypatch):
    monkeypatch.setenv("AEL_MODEL", "vllm/qwen3.6-27b-fp8")  # subject/agents' model
    from evaluation.scoring.llm_evaluator import LLMEvaluator

    ev = LLMEvaluator(model="vllm/mistral-small-3.2-24b-fp8")
    assert ev._llm_client is not None
    # judge stays Mistral, NOT collapsed onto AEL_MODEL (Qwen)
    assert ev._llm_client.model == "vllm/mistral-small-3.2-24b-fp8"


def test_two_judges_stay_distinct_under_ael_model(monkeypatch):
    # the dual-judge / consensus case: both judges must keep their own models
    monkeypatch.setenv("AEL_MODEL", "vllm/qwen3.6-27b-fp8")
    from evaluation.scoring.llm_evaluator import LLMEvaluator

    m = LLMEvaluator(model="vllm/mistral-small-3.2-24b-fp8")
    g = LLMEvaluator(model="vllm/gemma-3-27b-it-fp8")
    assert m._llm_client.model == "vllm/mistral-small-3.2-24b-fp8"
    assert g._llm_client.model == "vllm/gemma-3-27b-it-fp8"
    assert m._llm_client.model != g._llm_client.model  # not collapsed onto one


def test_baseline_generator_pins_model(monkeypatch):
    monkeypatch.setenv("AEL_MODEL", "vllm/qwen3.6-27b-fp8")
    from evaluation.baselines.baseline_generator import BaselineGenerator

    bg = BaselineGenerator(model="vllm/mistral-small-3.2-24b-fp8")
    assert bg._llm_client.model == "vllm/mistral-small-3.2-24b-fp8"
