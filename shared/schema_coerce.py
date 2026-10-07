# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Shared LLM-output robustness: coerce structurally-richer LLM JSON into a model's declared
scalar/list types BEFORE Pydantic type-checks them.

LLM agents routinely return a dict where a model declares ``str``, a list-of-dicts where it
declares ``List[str]``, a non-numeric string where it declares ``int``/``float``, a single
object where a ``List[Model]`` is expected, or a stray scalar mixed into an object list. With a
strict schema each of these raises a ValidationError that crashes the whole stage (confirmed
across IdeationTeam ConceptList/QuestionList AND ModelTeam CalibratedParameter/ModelParameter).

``LLMCoercedModel`` is a base class with one type-driven ``model_validator(mode='before')`` that
coerces these shapes to the declared type for every current and future field — ending the crash
class with zero per-field boilerplate. Freeform ``Any`` / ``List[Any]`` / ``Dict`` fields and
nested ``BaseModel`` fields are left to validate naturally (nested LLMCoercedModel subclasses run
their own before-validator recursively). Apply ONLY to LLM-parsed models — keep code-set models
(API/HITL payloads) strict so they keep surfacing real upstream bugs.
"""

from __future__ import annotations

import json
import re
from typing import Any, List, Union, get_args, get_origin

from pydantic import BaseModel, model_validator

# Primary identifier fields: a missing one means the object is genuinely broken, so it is
# NOT backfilled by the missing-required-field guard (it should still fail validation).
_PRIMARY_FIELDS = frozenset({"question", "concept_title", "title", "name", "statement", "moment_key"})

_STR_KEYS = ("query", "text", "value", "name", "title", "keyword", "term",
             "method", "methodology", "step", "theme", "area", "focus",
             "alternative", "assumption", "gap", "concept", "rationale",
             "reason", "framework", "contribution", "impact", "relevance",
             "description", "summary", "content", "label", "question",
             "statement", "definition", "rank", "score", "justification")


def coerce_to_str(value):
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, dict):
        for k in _STR_KEYS:
            v = value.get(k)
            if isinstance(v, str) and v.strip():
                return v
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    if isinstance(value, list):
        return ", ".join(coerce_to_str(v) for v in value if v is not None)
    return str(value)


def coerce_to_str_list(value):
    if value is None:
        return value
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [coerce_to_str(value)]
    if isinstance(value, list):
        return [coerce_to_str(v) for v in value if v is not None]
    return [coerce_to_str(value)]


def coerce_to_number(value):
    if value is None or isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        s = value.strip()
        try:
            return int(s) if re.fullmatch(r"-?\d+", s) else float(s)
        except Exception:
            return None  # non-numeric string -> None (Optional default absorbs it)
    return None  # dict / list -> None


def _unwrap_optional(ann):
    if get_origin(ann) is Union:
        non_none = [a for a in get_args(ann) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return ann


class LLMCoercedModel(BaseModel):
    """Base for models built from LLM JSON: coerce richer shapes to the declared type."""

    @model_validator(mode="before")
    @classmethod
    def _coerce_llm_scalars(cls, data):
        if not isinstance(data, dict):
            return data
        out = dict(data)
        for name, field in cls.model_fields.items():
            if name not in out or out[name] is None:
                # Backfill a MISSING required scalar/list so a partial LLM object (e.g. a
                # question the model returned without its required 'rationale') degrades
                # gracefully instead of raising "Field required [missing]". Only str /
                # List[str]; Optional fields and richer types keep their natural default.
                # PRIMARY identifier fields are NOT backfilled — a missing 'question'/'title'
                # means the object is genuinely broken and should still fail validation.
                if name not in out and field.is_required() and name not in _PRIMARY_FIELDS:
                    _ann = _unwrap_optional(field.annotation)
                    _args = get_args(_ann)
                    if _ann is str:
                        out[name] = ""
                    elif get_origin(_ann) in (list, List) and _args and _args[0] is str:
                        out[name] = []
                continue
            ann = _unwrap_optional(field.annotation)
            origin = get_origin(ann)
            args = get_args(ann)
            if ann is str:
                out[name] = coerce_to_str(out[name])
            elif origin in (list, List):
                if args and args[0] is str:
                    out[name] = coerce_to_str_list(out[name])
                elif args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
                    if isinstance(out[name], dict):           # single object -> 1-element list
                        out[name] = [out[name]]
                    elif isinstance(out[name], list):         # drop stray non-object items
                        out[name] = [x for x in out[name] if isinstance(x, (dict, args[0]))]
            elif ann in (int, float):
                out[name] = coerce_to_number(out[name])
            elif isinstance(ann, type) and issubclass(ann, BaseModel):
                v = out[name]                                  # nested model field
                if isinstance(v, list):
                    out[name] = next((x for x in v if isinstance(x, dict)), None)
                elif not isinstance(v, dict):
                    out[name] = None
        return out
