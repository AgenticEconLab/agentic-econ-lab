# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Tier-2 LLM-as-reviewer scoring engine.

- `llm_evaluator.py` — `LLMEvaluator`, the rubric-driven judge that produces
  per-dimension content scores (open-weight vLLM judge by default).
- `rubrics.py` — the dimension rubric definitions the judge scores against.

Import fully-qualified, e.g. `from evaluation.scoring.llm_evaluator import LLMEvaluator`.
"""
