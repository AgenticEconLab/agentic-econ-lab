# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
PII Detector — Scans text for personally identifiable information.

Regex-based detection for common PII patterns in DataTeam outputs and
any agent-generated text. Designed as a guardrail to prevent leaking
sensitive information in pipeline outputs.

Supported PII types:
- Email addresses
- US Social Security Numbers (SSN)
- US phone numbers (various formats)
- Credit card numbers (Visa, MC, Amex, Discover)
- IP addresses (IPv4)

Usage:
    from shared.guardrails.pii_detector import PIIDetector, PIIFinding

    detector = PIIDetector()
    report = detector.scan("Contact john@example.com or call 555-123-4567")
    if report.has_pii:
        for finding in report.findings:
            print(f"{finding.pii_type}: {finding.redacted}")

    # Redact PII from text
    clean = detector.redact("Email me at john@example.com")
    # -> "Email me at [EMAIL_REDACTED]"
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class PIIFinding:
    """A single PII detection result."""

    pii_type: str  # e.g., "email", "ssn", "phone", "credit_card", "ip_address"
    matched_text: str  # The raw matched text
    redacted: str  # Redacted version (e.g., "[EMAIL_REDACTED]")
    start: int = 0  # Character offset in original text
    end: int = 0  # End offset

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pii_type": self.pii_type,
            "matched_text": self.matched_text,
            "redacted": self.redacted,
            "start": self.start,
            "end": self.end,
        }


@dataclass
class PIIScanReport:
    """Result of scanning text for PII."""

    has_pii: bool
    findings: List[PIIFinding] = field(default_factory=list)
    scanned_length: int = 0
    pii_types_found: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_pii": self.has_pii,
            "findings_count": len(self.findings),
            "pii_types_found": sorted(self.pii_types_found),
            "scanned_length": self.scanned_length,
            "findings": [f.to_dict() for f in self.findings],
        }


# PII detection patterns
# Each tuple: (pii_type, compiled_regex, redaction_label)
_PII_PATTERNS = [
    (
        "email",
        re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        ),
        "[EMAIL_REDACTED]",
    ),
    (
        "ssn",
        re.compile(
            r"\b\d{3}-\d{2}-\d{4}\b"
        ),
        "[SSN_REDACTED]",
    ),
    (
        "phone",
        re.compile(
            r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
        ),
        "[PHONE_REDACTED]",
    ),
    (
        "credit_card",
        re.compile(
            r"\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6(?:011|5\d{2}))[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"
        ),
        "[CREDIT_CARD_REDACTED]",
    ),
    (
        "ip_address",
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        ),
        "[IP_REDACTED]",
    ),
]


class PIIDetector:
    """
    Scans text for personally identifiable information (PII).

    Provides both detection (scan) and remediation (redact) capabilities.
    Can be configured to check only specific PII types.
    """

    def __init__(
        self,
        enabled_types: Optional[Set[str]] = None,
    ):
        """
        Args:
            enabled_types: Set of PII types to detect. If None, all types
                          are enabled. Valid types: email, ssn, phone,
                          credit_card, ip_address.
        """
        all_types = {p[0] for p in _PII_PATTERNS}
        if enabled_types is not None:
            invalid = enabled_types - all_types
            if invalid:
                raise ValueError(f"Unknown PII types: {invalid}. Valid: {all_types}")
            self.enabled_types = enabled_types
        else:
            self.enabled_types = all_types

    def scan(self, text: str) -> PIIScanReport:
        """
        Scan text for PII patterns.

        Args:
            text: Input text to scan.

        Returns:
            PIIScanReport with all findings.
        """
        if not text:
            return PIIScanReport(has_pii=False, scanned_length=0)

        findings: List[PIIFinding] = []
        pii_types: Set[str] = set()

        for pii_type, pattern, redaction in _PII_PATTERNS:
            if pii_type not in self.enabled_types:
                continue
            for match in pattern.finditer(text):
                findings.append(PIIFinding(
                    pii_type=pii_type,
                    matched_text=match.group(),
                    redacted=redaction,
                    start=match.start(),
                    end=match.end(),
                ))
                pii_types.add(pii_type)

        # Sort by position
        findings.sort(key=lambda f: f.start)

        return PIIScanReport(
            has_pii=len(findings) > 0,
            findings=findings,
            scanned_length=len(text),
            pii_types_found=pii_types,
        )

    def redact(self, text: str) -> str:
        """
        Replace all detected PII with redaction labels.

        Args:
            text: Input text to redact.

        Returns:
            Text with PII replaced by type-specific labels.
        """
        if not text:
            return text

        report = self.scan(text)
        if not report.has_pii:
            return text

        # Replace from end to start to preserve offsets
        result = text
        for finding in reversed(report.findings):
            result = result[:finding.start] + finding.redacted + result[finding.end:]

        return result

    def scan_dict(
        self,
        data: Dict[str, Any],
        max_depth: int = 5,
    ) -> PIIScanReport:
        """
        Recursively scan all string values in a dict for PII.

        Useful for scanning entire stage output payloads.

        Args:
            data: Dictionary to scan.
            max_depth: Maximum recursion depth.

        Returns:
            Combined PIIScanReport for all string values.
        """
        all_text = self._extract_strings(data, max_depth)
        combined = "\n".join(all_text)
        return self.scan(combined)

    def _extract_strings(
        self, obj: Any, max_depth: int, depth: int = 0
    ) -> List[str]:
        """Recursively extract string values from nested data."""
        if depth >= max_depth:
            return []

        strings: List[str] = []
        if isinstance(obj, str):
            strings.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                strings.extend(self._extract_strings(v, max_depth, depth + 1))
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                strings.extend(self._extract_strings(item, max_depth, depth + 1))

        return strings

    @staticmethod
    def available_types() -> Set[str]:
        """Return all supported PII types."""
        return {p[0] for p in _PII_PATTERNS}
