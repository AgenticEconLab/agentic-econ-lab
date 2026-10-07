# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Tests for PIIDetector — PII scanning and redaction."""

import pytest
from shared.guardrails.pii_detector import (
    PIIDetector,
    PIIFinding,
    PIIScanReport,
)


# ---------------------------------------------------------------------------
# Email detection tests
# ---------------------------------------------------------------------------

class TestEmailDetection:
    def test_simple_email(self):
        d = PIIDetector()
        report = d.scan("Contact john@example.com for details")
        assert report.has_pii is True
        assert "email" in report.pii_types_found
        assert report.findings[0].matched_text == "john@example.com"

    def test_email_with_dots(self):
        d = PIIDetector()
        report = d.scan("Email: first.last@company.co.uk")
        assert report.has_pii is True
        assert report.findings[0].pii_type == "email"

    def test_email_with_plus(self):
        d = PIIDetector()
        report = d.scan("user+tag@gmail.com")
        assert report.has_pii is True

    def test_no_email(self):
        d = PIIDetector()
        report = d.scan("No email addresses here")
        assert "email" not in report.pii_types_found


# ---------------------------------------------------------------------------
# SSN detection tests
# ---------------------------------------------------------------------------

class TestSSNDetection:
    def test_ssn_standard(self):
        d = PIIDetector()
        report = d.scan("SSN: 123-45-6789")
        assert report.has_pii is True
        assert "ssn" in report.pii_types_found
        assert report.findings[0].matched_text == "123-45-6789"

    def test_not_ssn_format(self):
        d = PIIDetector()
        report = d.scan("Number: 12345-6789")
        assert "ssn" not in report.pii_types_found


# ---------------------------------------------------------------------------
# Phone detection tests
# ---------------------------------------------------------------------------

class TestPhoneDetection:
    def test_phone_dashes(self):
        d = PIIDetector()
        report = d.scan("Call 555-123-4567")
        assert report.has_pii is True
        assert "phone" in report.pii_types_found

    def test_phone_dots(self):
        d = PIIDetector()
        report = d.scan("Phone: 555.123.4567")
        assert report.has_pii is True

    def test_phone_parens(self):
        d = PIIDetector()
        report = d.scan("Call (555) 123-4567")
        assert report.has_pii is True

    def test_phone_with_country_code(self):
        d = PIIDetector()
        report = d.scan("+1-555-123-4567")
        assert report.has_pii is True


# ---------------------------------------------------------------------------
# Credit card detection tests
# ---------------------------------------------------------------------------

class TestCreditCardDetection:
    def test_visa(self):
        d = PIIDetector()
        report = d.scan("Card: 4111-1111-1111-1111")
        assert report.has_pii is True
        assert "credit_card" in report.pii_types_found

    def test_mastercard(self):
        d = PIIDetector()
        report = d.scan("MC: 5111 1111 1111 1111")
        assert report.has_pii is True

    def test_not_credit_card(self):
        d = PIIDetector()
        report = d.scan("ID: 1234-5678-9012")
        assert "credit_card" not in report.pii_types_found


# ---------------------------------------------------------------------------
# IP address detection tests
# ---------------------------------------------------------------------------

class TestIPAddressDetection:
    def test_ipv4(self):
        d = PIIDetector()
        report = d.scan("Server at 192.168.1.100")
        assert report.has_pii is True
        assert "ip_address" in report.pii_types_found

    def test_localhost(self):
        d = PIIDetector()
        report = d.scan("Listening on 127.0.0.1")
        assert report.has_pii is True

    def test_invalid_ip_rejected(self):
        d = PIIDetector()
        report = d.scan("Version 999.999.999.999")
        # 999 > 255, should not match our pattern
        assert "ip_address" not in report.pii_types_found


# ---------------------------------------------------------------------------
# Redaction tests
# ---------------------------------------------------------------------------

class TestRedaction:
    def test_redact_email(self):
        d = PIIDetector()
        result = d.redact("Contact john@example.com please")
        assert "john@example.com" not in result
        assert "[EMAIL_REDACTED]" in result

    def test_redact_ssn(self):
        d = PIIDetector()
        result = d.redact("SSN is 123-45-6789")
        assert "123-45-6789" not in result
        assert "[SSN_REDACTED]" in result

    def test_redact_multiple(self):
        d = PIIDetector()
        text = "Email john@example.com, SSN 123-45-6789"
        result = d.redact(text)
        assert "[EMAIL_REDACTED]" in result
        assert "[SSN_REDACTED]" in result

    def test_redact_no_pii(self):
        d = PIIDetector()
        text = "No PII here, just normal text."
        result = d.redact(text)
        assert result == text

    def test_redact_empty(self):
        d = PIIDetector()
        assert d.redact("") == ""


# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------

class TestPIIDetectorConfig:
    def test_enabled_types_filter(self):
        d = PIIDetector(enabled_types={"email"})
        report = d.scan("john@example.com and 123-45-6789")
        assert "email" in report.pii_types_found
        assert "ssn" not in report.pii_types_found

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Unknown PII types"):
            PIIDetector(enabled_types={"invalid_type"})

    def test_available_types(self):
        types = PIIDetector.available_types()
        assert "email" in types
        assert "ssn" in types
        assert "phone" in types
        assert "credit_card" in types
        assert "ip_address" in types
        assert len(types) == 5


# ---------------------------------------------------------------------------
# scan_dict tests
# ---------------------------------------------------------------------------

class TestScanDict:
    def test_scan_flat_dict(self):
        d = PIIDetector()
        data = {"name": "John", "email": "john@example.com"}
        report = d.scan_dict(data)
        assert report.has_pii is True

    def test_scan_nested_dict(self):
        d = PIIDetector()
        data = {"person": {"contact": {"email": "test@example.com"}}}
        report = d.scan_dict(data)
        assert report.has_pii is True

    def test_scan_list_in_dict(self):
        d = PIIDetector()
        data = {"emails": ["a@b.com", "c@d.com"]}
        report = d.scan_dict(data)
        assert report.has_pii is True

    def test_scan_clean_dict(self):
        d = PIIDetector()
        data = {"gdp": 21.43, "year": 2024, "country": "USA"}
        report = d.scan_dict(data)
        assert report.has_pii is False

    def test_max_depth_respected(self):
        d = PIIDetector()
        # Deeply nested — beyond max_depth
        deep = {"a": {"b": {"c": {"d": {"e": {"f": "john@example.com"}}}}}}
        report = d.scan_dict(deep, max_depth=3)
        # Depth 3: a->b->c, stops before d, so email not found
        assert report.has_pii is False


# ---------------------------------------------------------------------------
# PIIScanReport tests
# ---------------------------------------------------------------------------

class TestPIIScanReport:
    def test_report_to_dict(self):
        report = PIIScanReport(
            has_pii=True,
            findings=[PIIFinding(
                pii_type="email", matched_text="a@b.com",
                redacted="[EMAIL_REDACTED]", start=0, end=7,
            )],
            scanned_length=100,
            pii_types_found={"email"},
        )
        d = report.to_dict()
        assert d["has_pii"] is True
        assert d["findings_count"] == 1
        assert d["pii_types_found"] == ["email"]

    def test_empty_report(self):
        d = PIIDetector()
        report = d.scan("")
        assert report.has_pii is False
        assert report.scanned_length == 0

    def test_scanned_length_tracked(self):
        d = PIIDetector()
        text = "Hello world, no PII here."
        report = d.scan(text)
        assert report.scanned_length == len(text)
