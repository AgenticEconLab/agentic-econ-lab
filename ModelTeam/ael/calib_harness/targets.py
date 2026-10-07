# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""External empirical calibration targets for the deterministic calibration harness.

Loads the shipped ``macro_targets.yaml`` into ``TargetRow`` objects (the LOCKED contract in
``calib_harness.types``). These targets are the harness's INDEPENDENT yardstick: a model-implied
moment is scored only by joining to a target here via ``moment_key`` (exact match). A moment with
no row is dropped upstream as ``no_external_target_match`` — targets are NEVER LLM-invented and
NEVER back-filled from a model's own parameters.

Public API:
    load_targets(path=None) -> Dict[str, TargetRow]   # default loads the shipped YAML
    get_target(moment_key, path=None) -> Optional[TargetRow]

Notes
-----
* The YAML may carry an optional ``fred_series_id`` per row (documentation for an OPTIONAL FRED
  recompute that flows only into the target side). The locked ``TargetRow`` dataclass has no such
  field, so unknown YAML keys are silently dropped when constructing rows. This keeps the loader
  forward-compatible without mutating the contract.
* Every row is validated on load: non-empty ``moment_key`` and ``source_citation``, numeric finite
  ``value``, and ``std_error > 0``. A malformed row raises ``ValueError`` rather than being silently
  scored against garbage.
"""

from __future__ import annotations

import math
import os
from dataclasses import fields as _dataclass_fields
from typing import Dict, Optional

import yaml

from .types import TargetRow

# Default shipped target file lives beside this module.
_DEFAULT_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "macro_targets.yaml")

# Field names the locked TargetRow accepts; any other YAML key (e.g. fred_series_id) is dropped.
_TARGETROW_FIELDS = {f.name for f in _dataclass_fields(TargetRow)}

# Cache keyed by absolute path so repeated default lookups don't re-read/re-parse the file.
_CACHE: Dict[str, Dict[str, TargetRow]] = {}


def _coerce_row(raw: dict, index: int) -> TargetRow:
    """Validate one raw YAML mapping and build a TargetRow (dropping unknown keys)."""
    if not isinstance(raw, dict):
        raise ValueError(f"target row #{index} is not a mapping: {raw!r}")

    moment_key = raw.get("moment_key")
    if not moment_key or not isinstance(moment_key, str) or not moment_key.strip():
        raise ValueError(f"target row #{index} has a missing/empty 'moment_key'")

    value = raw.get("value")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"target '{moment_key}' has a non-numeric/non-finite 'value': {value!r}")

    std_error = raw.get("std_error")
    if not isinstance(std_error, (int, float)) or isinstance(std_error, bool) or not (float(std_error) > 0.0):
        raise ValueError(f"target '{moment_key}' must have numeric 'std_error' > 0, got {std_error!r}")

    citation = raw.get("source_citation")
    if not citation or not isinstance(citation, str) or not citation.strip():
        raise ValueError(f"target '{moment_key}' has a missing/empty 'source_citation'")

    # Keep only fields the locked dataclass knows about (drops fred_series_id and any future extras).
    kwargs = {k: v for k, v in raw.items() if k in _TARGETROW_FIELDS}
    kwargs["value"] = float(value)
    kwargs["std_error"] = float(std_error)
    return TargetRow(**kwargs)


def load_targets(path: Optional[str] = None) -> Dict[str, TargetRow]:
    """Load calibration targets keyed by ``moment_key``.

    Parameters
    ----------
    path:
        YAML file to load. ``None`` (default) loads the shipped ``macro_targets.yaml``.

    Returns
    -------
    Dict[str, TargetRow]
        Mapping ``moment_key -> TargetRow``. Duplicate keys are an error (ambiguous targets).
    """
    resolved = os.path.abspath(path or _DEFAULT_YAML)
    if resolved in _CACHE:
        return _CACHE[resolved]

    if not os.path.exists(resolved):
        raise FileNotFoundError(f"calibration targets file not found: {resolved}")

    with open(resolved, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)

    # Accept either a top-level mapping with a 'targets:' list or a bare list.
    if isinstance(doc, dict):
        rows = doc.get("targets")
    else:
        rows = doc
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"targets file {resolved} has no non-empty 'targets' list")

    out: Dict[str, TargetRow] = {}
    for i, raw in enumerate(rows):
        row = _coerce_row(raw, i)
        if row.moment_key in out:
            raise ValueError(f"duplicate moment_key '{row.moment_key}' in {resolved}")
        out[row.moment_key] = row

    _CACHE[resolved] = out
    return out


def get_target(moment_key: str, path: Optional[str] = None) -> Optional[TargetRow]:
    """Return the TargetRow for ``moment_key`` (default target set), or ``None`` if absent."""
    return load_targets(path).get(moment_key)
