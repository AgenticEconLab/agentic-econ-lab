# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Constrained decoding wrapper (V0.7).

Reference: native constrained decoding shipped in Claude 4.x (Nov 2025 GA) and
OpenAI GPT-5.x; Instructor is the Pydantic wrapper. This module provides a
lightweight client-side enforcement layer that converts JSON mode + retry
loops into a single-call contract: hand it a Pydantic schema and a prompt,
get back either a validated instance or a ``ConstrainedValidationError``.

Production callers should pass an ``llm_client`` that supports provider-native
grammar-guided decoding; this wrapper falls back to JSON-mode + json_repair +
Pydantic validation when the provider lacks native support.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError


M = TypeVar("M", bound=BaseModel)


class ConstrainedValidationError(Exception):
    """Raised when the wrapper cannot produce a schema-valid response."""


class ConstrainedLLM:
    """Pydantic-schema-enforced LLM calls.

    Parameters
    ----------
    llm_client
        Object with an ``invoke(prompt, *, schema=None, json_mode=True) -> str``
        method (or compatible). ``json_mode=True`` is the fallback path;
        ``schema`` is passed through when the provider supports native
        grammar-guided decoding.
    repair_fn
        Optional JSON repair function. Defaults to ``shared.json_repair.repair_json``.
    """

    def __init__(
        self,
        llm_client: Any,
        *,
        repair_fn: Optional[Callable[[str], str]] = None,
        collector: Any = None,
    ) -> None:
        self.llm_client = llm_client
        self.repair_fn = repair_fn or _default_repair
        self.collector = collector

    def invoke(
        self,
        prompt: str,
        schema: Type[M],
        *,
        max_retries: int = 1,
    ) -> M:
        """Run a single-shot constrained decoding call.

        Total LLM calls = ``max_retries + 1``. On each validation failure the
        previous error message is appended to the prompt so the provider has
        a chance to self-correct (important for semantic violations like
        ``ge=0`` that a pure JSON-repair pass cannot fix).
        """
        attempts = max_retries + 1
        last_err: Optional[Exception] = None
        augmented_prompt = prompt

        for attempt in range(attempts):
            raw = self._call(augmented_prompt, schema=schema, json_mode=True)
            # 1. Direct validation
            try:
                return self._validate(raw, schema)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_err = exc
            # 2. JSON repair pass (fixes formatting, cannot fix semantic violations)
            try:
                repaired = self.repair_fn(raw)
                return self._validate(repaired, schema)
            except Exception as exc2:
                last_err = exc2

            if attempt == attempts - 1:
                break
            # Augment prompt with the error so the next LLM call can self-correct
            augmented_prompt = (
                f"{prompt}\n\nPrevious attempt failed validation with: {last_err}\n"
                "Return strictly valid JSON satisfying the schema."
            )

        raise ConstrainedValidationError(
            f"failed to produce schema-valid output after {attempts} attempts: {last_err}"
        )

    # ------------------------------------------------------------------

    def _call(self, prompt: str, *, schema: Type[BaseModel], json_mode: bool) -> str:
        """Dispatch to the LLM client, preferring provider-native schema support."""
        try:
            # Provider-native path (e.g. Anthropic native grammar)
            return self.llm_client.invoke(prompt, schema=schema, json_mode=json_mode)
        except TypeError:
            # Client doesn't accept schema/json_mode kwargs; fall back to plain call
            return self.llm_client.invoke(prompt)

    @staticmethod
    def _validate(raw: str, schema: Type[M]) -> M:
        data = json.loads(raw) if isinstance(raw, str) else raw
        return schema.model_validate(data)


def _default_repair(text: str) -> str:
    try:
        from shared.json_repair import repair_json
        return repair_json(text)
    except Exception:
        return text
