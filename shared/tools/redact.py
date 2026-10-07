# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Secret redaction for tool error messages and logged URLs.

requests' HTTPError text embeds the full request URL, query string included — the FRED API
key was printed in SLURM job logs ("400 Client Error ... ?series_id=X&api_key=...").
Every tool error that can reach a log passes through :func:`redact_secrets`."""

from __future__ import annotations

import re

# query/form parameters whose values are credentials
_SECRET_PARAM = re.compile(
    r"(?i)(\b(?:api[_-]?key|apikey|key|token|access[_-]?token|auth[_-]?token|insttoken|"
    r"secret|client[_-]?secret|password|passwd|subscription[_-]?key)=)([^&\s'\"<>]+)")
# header-style "Authorization: Bearer xyz" / "X-API-KEY: xyz" fragments
_SECRET_HEADER = re.compile(
    r"(?i)((?:authorization|x-api-key|x-subscription-token|x-els-apikey|x-els-insttoken)"
    r"['\"]?\s*[:=]\s*['\"]?(?:bearer\s+)?)([A-Za-z0-9._\-]{6,})")


def redact_secrets(text) -> str:
    """Replace credential values in ``text`` with ``REDACTED`` (idempotent; never raises)."""
    try:
        s = str(text) if text is not None else ""
    except Exception:
        return ""
    s = _SECRET_PARAM.sub(r"\1REDACTED", s)
    return _SECRET_HEADER.sub(r"\1REDACTED", s)
