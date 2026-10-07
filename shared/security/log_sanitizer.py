# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Log sanitizer — defends against Log-To-Leak prompt-injection via MCP logging tools.

Reference: OpenReview UVgbFuXPaO (2026) — demonstrates covert data exfil through
logging tools on GPT-4o/5 and Claude Sonnet 4.

This module:
  1. Strips template-injection tokens (``{{..}}``, ``<?..?>``, ``${..}``)
  2. Redacts high-entropy secrets (API keys, JWTs, AWS keys)
  3. Optionally drops sensitive values by key name (auth headers, etc.)

Used by shared/mcp/server.py logging hooks and evaluation observability.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List


TEMPLATE_INJECTION_RE = re.compile(
    r"\{\{[\s\S]*?\}\}"       # Jinja / Handlebars (multi-line aware)
    r"|<\?[\s\S]*?\?>"        # PHP / XML processing instructions
    r"|\$\{[\s\S]*?\}"        # Shell / ES6 (multi-line aware)
    r"|<%[\s\S]*?%>",         # ERB / ASP
)

# High-entropy token patterns (order matters — match longer first).
SECRET_PATTERNS = [
    # OpenAI keys (current and legacy)
    re.compile(r"sk-proj-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{32,}"),
    # Anthropic
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    # Generic Bearer tokens (case-insensitive)
    re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]{20,}", re.IGNORECASE),
    # AWS access keys
    re.compile(r"AKIA[0-9A-Z]{16}"),
    # JWTs
    re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),
    # GitHub tokens
    re.compile(r"ghp_[A-Za-z0-9]{30,}"),
    re.compile(r"gho_[A-Za-z0-9]{30,}"),
    # Google API keys
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),
]

SENSITIVE_KEYS = {
    "authorization", "x-api-key", "api_key", "api-key", "apikey",
    "password", "passwd", "secret", "client_secret", "private_key",
    "cookie", "set-cookie", "session", "access_token", "refresh_token",
    "x-auth-token", "x-anthropic-api-key", "x-goog-api-key",
}


class LogSanitizer:
    """Sanitize log payloads before they are emitted or persisted."""

    REDACTED = "[REDACTED]"
    INJECTION_PLACEHOLDER = "[TEMPLATE_REDACTED]"

    def __init__(
        self,
        *,
        sensitive_keys: Iterable[str] | None = None,
        extra_patterns: Iterable[re.Pattern[str]] | None = None,
    ) -> None:
        self._sensitive_keys = {k.lower() for k in (sensitive_keys or SENSITIVE_KEYS)}
        self._patterns: List[re.Pattern[str]] = list(SECRET_PATTERNS)
        if extra_patterns:
            self._patterns.extend(extra_patterns)

    def sanitize(self, payload: Any) -> Any:
        """Recursively sanitize a log payload (dict / list / string)."""
        if isinstance(payload, dict):
            return {k: self._sanitize_value(k, v) for k, v in payload.items()}
        if isinstance(payload, list):
            return [self.sanitize(v) for v in payload]
        if isinstance(payload, str):
            return self._sanitize_string(payload)
        return payload

    def sanitize_string(self, text: str) -> str:
        """Public alias for string-only sanitization."""
        return self._sanitize_string(text)

    def contains_template_injection(self, text: str) -> bool:
        return bool(TEMPLATE_INJECTION_RE.search(text))

    def contains_secret(self, text: str) -> bool:
        return any(p.search(text) for p in self._patterns)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _sanitize_value(self, key: str, value: Any) -> Any:
        if key.lower() in self._sensitive_keys:
            return self.REDACTED
        return self.sanitize(value)

    def _sanitize_string(self, text: str) -> str:
        # 1. Strip template-injection tokens
        text = TEMPLATE_INJECTION_RE.sub(self.INJECTION_PLACEHOLDER, text)
        # 2. Redact high-entropy secrets
        for pat in self._patterns:
            text = pat.sub(self.REDACTED, text)
        return text


_default_sanitizer = LogSanitizer()


def sanitize(payload: Any) -> Any:
    """Convenience wrapper using the default sanitizer."""
    return _default_sanitizer.sanitize(payload)
