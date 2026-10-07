# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Tests for shared.json_repair module."""

import json

from shared.json_repair import repair_json, safe_json_loads


# ── repair_json ──────────────────────────────────────────────────────────

class TestRepairJson:
    """Tests for repair_json()."""

    def test_none_returns_empty_object(self):
        assert repair_json(None) == "{}"

    def test_empty_string_returns_empty_object(self):
        assert repair_json("") == "{}"

    def test_whitespace_only_returns_empty_object(self):
        assert repair_json("   \n  ") == "{}"

    def test_valid_json_unchanged(self):
        original = '{"key": "value", "num": 42}'
        result = repair_json(original)
        assert json.loads(result) == {"key": "value", "num": 42}

    def test_latex_beta_parseable(self):
        """\\beta starts with valid JSON escape \\b, so it parses as backspace+eta.
        The key thing is that json.loads() succeeds without error."""
        text = r'{"param": "\beta coefficient"}'
        result = repair_json(text)
        parsed = json.loads(result)
        assert "param" in parsed

    def test_latex_leq_escaped(self):
        """\\leq has \\l which is NOT a valid JSON escape, so it gets double-escaped."""
        text = r'{"constraint": "0 \leq x \leq 1"}'
        result = repair_json(text)
        parsed = json.loads(result)
        assert "leq" in parsed["constraint"]

    def test_latex_alpha_gamma_escaped(self):
        """\\alpha (\\a is invalid) and \\gamma (\\g is invalid) get escaped."""
        text = r'{"formula": "\alpha + \gamma = 1"}'
        result = repair_json(text)
        parsed = json.loads(result)
        assert "alpha" in parsed["formula"]
        assert "gamma" in parsed["formula"]

    def test_valid_escapes_preserved_newline(self):
        text = '{"text": "line1\\nline2"}'
        result = repair_json(text)
        parsed = json.loads(result)
        assert parsed["text"] == "line1\nline2"

    def test_valid_escapes_preserved_tab(self):
        text = '{"text": "col1\\tcol2"}'
        result = repair_json(text)
        parsed = json.loads(result)
        assert parsed["text"] == "col1\tcol2"

    def test_valid_escapes_preserved_unicode(self):
        text = '{"char": "\\u0041"}'
        result = repair_json(text)
        parsed = json.loads(result)
        assert parsed["char"] == "A"

    def test_valid_escapes_preserved_backslash(self):
        text = '{"path": "C:\\\\Users\\\\test"}'
        result = repair_json(text)
        parsed = json.loads(result)
        assert "Users" in parsed["path"]

    def test_code_fence_json_stripped(self):
        text = '```json\n{"key": "value"}\n```'
        result = repair_json(text)
        parsed = json.loads(result)
        assert parsed == {"key": "value"}

    def test_code_fence_plain_stripped(self):
        text = '```\n{"key": "value"}\n```'
        result = repair_json(text)
        parsed = json.loads(result)
        assert parsed == {"key": "value"}

    def test_multiple_latex_in_one_string(self):
        """Mix of valid (\\b, \\f) and invalid (\\a, \\g, \\d) JSON escapes in LaTeX."""
        text = r'{"eq": "\alpha \gamma \delta"}'
        result = repair_json(text)
        parsed = json.loads(result)
        assert "alpha" in parsed["eq"]
        assert "gamma" in parsed["eq"]
        assert "delta" in parsed["eq"]


# ── safe_json_loads ──────────────────────────────────────────────────────

class TestSafeJsonLoads:
    """Tests for safe_json_loads()."""

    def test_valid_json_parses(self):
        assert safe_json_loads('{"a": 1}') == {"a": 1}

    def test_valid_array_parses(self):
        assert safe_json_loads('[1, 2, 3]') == [1, 2, 3]

    def test_empty_returns_default_empty_dict(self):
        assert safe_json_loads("") == {}

    def test_empty_returns_custom_fallback(self):
        assert safe_json_loads("", fallback=[]) == []

    def test_none_returns_default_empty_dict(self):
        assert safe_json_loads(None) == {}

    def test_latex_backslashes_repaired(self):
        """\\alpha has invalid \\a escape — repair makes it parseable."""
        text = r'{"param": "\alpha value"}'
        result = safe_json_loads(text)
        assert result is not None
        assert "param" in result
        assert "alpha" in result["param"]

    def test_prose_around_json_extracted(self):
        text = 'Here is the result:\n{"key": "value"}\nDone.'
        result = safe_json_loads(text)
        assert result == {"key": "value"}

    def test_prose_around_array_extracted(self):
        text = 'The queries are:\n["query1", "query2"]\nEnd.'
        result = safe_json_loads(text)
        assert result == ["query1", "query2"]

    def test_total_garbage_returns_fallback(self):
        assert safe_json_loads("This is not JSON at all", fallback={"default": True}) == {"default": True}

    def test_total_garbage_returns_empty_dict_default(self):
        assert safe_json_loads("no json here") == {}

    def test_code_fences_with_latex(self):
        text = '```json\n{"eq": "\\\\beta = 0.95"}\n```'
        result = safe_json_loads(text)
        assert result is not None
        assert "eq" in result

    def test_nested_json_object(self):
        text = '{"outer": {"inner": "value"}, "list": [1, 2]}'
        result = safe_json_loads(text)
        assert result["outer"]["inner"] == "value"
        assert result["list"] == [1, 2]

    def test_fallback_none_explicitly(self):
        result = safe_json_loads("garbage", fallback=None)
        assert result is None


class TestHardening:
    """Missing-comma + truncation repair — attempted ONLY when the text doesn't parse."""

    def test_parseable_text_returned_byte_identical(self):
        # the heuristics must never touch valid JSON (incl. weird-but-valid whitespace)
        good = '{"a": "x"\n , "b": [1, 2]}'
        assert repair_json(good) == good

    def test_missing_comma_between_string_members(self):
        # the observed "Expecting ',' delimiter" class (Theory synthesis)
        bad = '{"components": ["alpha"\n "beta"\n "gamma"]}'
        out = json.loads(repair_json(bad))
        assert out["components"] == ["alpha", "beta", "gamma"]

    def test_missing_comma_between_objects_in_array(self):
        bad = '{"items": [{"id": 1}\n {"id": 2}]}'
        out = json.loads(repair_json(bad))
        assert [i["id"] for i in out["items"]] == [1, 2]

    def test_truncated_mid_string_drops_partial_member(self):
        # generation cut by max_tokens mid-value: partial member dropped, structure closed
        bad = '{"questions": [{"q": "How does AI affect"'
        out = json.loads(repair_json(bad))
        assert "questions" in out

    def test_truncated_after_key_colon(self):
        bad = '{"a": "done", "b":'
        out = json.loads(repair_json(bad))
        assert out["a"] == "done" and "b" not in out

    def test_truncated_nested_brackets_closed(self):
        bad = '{"outer": {"inner": [1, 2'
        out = json.loads(repair_json(bad))
        assert out["outer"]["inner"] == [1, 2]

    def test_unrepairable_returns_original_behavior(self):
        # if nothing yields parseable JSON, the original text comes back (caller errors fire)
        garbage = "not json ]["
        assert repair_json(garbage) == garbage
