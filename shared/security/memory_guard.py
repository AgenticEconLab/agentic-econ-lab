# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
MemoryGuard — Defense against memory poisoning attacks (V0.6 Phase 4, OWASP ASI06).

Provides input sanitization, session isolation, entry validation, and
integrity checking for AEL's memory and cache systems.

Usage:
    from shared.security.memory_guard import MemoryGuard

    guard = MemoryGuard()
    clean = guard.sanitize_input("user provided text")
    result = guard.validate_memory_entry({"key": "value", "session": "abc"})
"""

import hashlib
import re
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MemoryGuardConfig(BaseModel):
    """Configuration for MemoryGuard."""

    max_entry_size: int = 10000  # chars
    max_entries_per_session: int = 1000
    expiry_hours: int = 720  # 30 days
    sanitize_urls: bool = True
    block_encoded_payloads: bool = True
    session_isolation: bool = True


class ValidationResult(BaseModel):
    """Result of a memory entry validation."""

    valid: bool = True
    issues: List[str] = Field(default_factory=list)
    sanitized_content: Optional[str] = None


class IntegrityReport(BaseModel):
    """Report from integrity audit of memory store."""

    total_entries: int = 0
    expired_entries: int = 0
    oversized_entries: int = 0
    suspicious_entries: int = 0
    issues: List[str] = Field(default_factory=list)
    clean: bool = True


# Patterns commonly used in injection attacks
_INJECTION_PATTERNS = [
    re.compile(r"<script[^>]*>", re.IGNORECASE),
    re.compile(r"javascript:", re.IGNORECASE),
    re.compile(r"data:text/html", re.IGNORECASE),
    re.compile(r"on\w+\s*=", re.IGNORECASE),  # onload=, onerror=, etc.
    re.compile(r"\{\{.*\}\}"),  # Template injection
    re.compile(r"__import__\s*\("),  # Python code injection
    re.compile(r"eval\s*\("),
    re.compile(r"exec\s*\("),
]

# Base64 encoded payload patterns
_ENCODED_PATTERNS = [
    re.compile(r"base64[,;]", re.IGNORECASE),
    re.compile(r"[A-Za-z0-9+/]{50,}={0,2}"),  # Long base64 strings
]


class MemoryGuard:
    """Defense against memory poisoning attacks (OWASP ASI06).

    Args:
        config: Guard configuration. Defaults to MemoryGuardConfig().
    """

    def __init__(self, config: Optional[MemoryGuardConfig] = None):
        self.config = config or MemoryGuardConfig()
        self._session_counts: Dict[str, int] = {}

    def sanitize_input(self, content: str) -> str:
        """Remove injection patterns, encoded payloads, malicious URIs.

        Args:
            content: Raw input text.

        Returns:
            Sanitized text.
        """
        result = content

        # Remove injection patterns
        for pattern in _INJECTION_PATTERNS:
            result = pattern.sub("[REMOVED]", result)

        # Remove encoded payloads
        if self.config.block_encoded_payloads:
            for pattern in _ENCODED_PATTERNS:
                result = pattern.sub("[ENCODED_REMOVED]", result)

        # Sanitize URLs (remove potentially malicious ones)
        if self.config.sanitize_urls:
            result = re.sub(
                r"(javascript|data|vbscript):[^\s\"']+",
                "[URL_REMOVED]",
                result,
                flags=re.IGNORECASE,
            )

        # Truncate if too long
        if len(result) > self.config.max_entry_size:
            result = result[: self.config.max_entry_size]

        return result

    def validate_memory_entry(self, entry: Dict[str, Any]) -> ValidationResult:
        """Check entry against poisoning indicators before persistence.

        Args:
            entry: Memory entry dict to validate.

        Returns:
            ValidationResult with valid flag and issues list.
        """
        issues = []
        content = str(entry)

        # Size check
        if len(content) > self.config.max_entry_size:
            issues.append(f"Entry exceeds max size ({len(content)} > {self.config.max_entry_size})")

        # Injection pattern check
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(content):
                issues.append(f"Potential injection pattern detected: {pattern.pattern}")

        # Check for suspiciously structured data
        if isinstance(entry, dict):
            for key in entry:
                if "__" in str(key):
                    issues.append(f"Suspicious key with double underscores: {key}")

        sanitized = self.sanitize_input(content) if issues else None

        return ValidationResult(
            valid=len(issues) == 0,
            issues=issues,
            sanitized_content=sanitized,
        )

    def enforce_session_isolation(self, session_id: str, entry: Dict[str, Any]) -> bool:
        """Ensure entries cannot cross session boundaries without validation.

        Args:
            session_id: Current session ID.
            entry: Entry to check.

        Returns:
            True if entry is allowed, False if blocked.
        """
        if not self.config.session_isolation:
            return True

        # Check per-session entry count
        count = self._session_counts.get(session_id, 0)
        if count >= self.config.max_entries_per_session:
            return False

        # Check if entry references a different session
        entry_session = entry.get("session_id", session_id)
        if entry_session != session_id:
            return False

        self._session_counts[session_id] = count + 1
        return True

    def check_integrity(self, memory_store: Dict[str, Any]) -> IntegrityReport:
        """Periodic integrity audit of stored memories.

        Args:
            memory_store: Dict of memory entries to audit.

        Returns:
            IntegrityReport with findings.
        """
        report = IntegrityReport(total_entries=len(memory_store))
        now = time.time()
        expiry_seconds = self.config.expiry_hours * 3600

        for key, entry in memory_store.items():
            entry_str = str(entry)

            # Check size
            if len(entry_str) > self.config.max_entry_size:
                report.oversized_entries += 1
                report.issues.append(f"Oversized entry: {key}")

            # Check expiry
            if isinstance(entry, dict):
                ts = entry.get("timestamp", 0)
                if ts and (now - ts) > expiry_seconds:
                    report.expired_entries += 1
                    report.issues.append(f"Expired entry: {key}")

            # Check for injection patterns
            for pattern in _INJECTION_PATTERNS:
                if pattern.search(entry_str):
                    report.suspicious_entries += 1
                    report.issues.append(f"Suspicious entry: {key}")
                    break

        report.clean = (
            report.oversized_entries == 0
            and report.suspicious_entries == 0
        )
        return report
