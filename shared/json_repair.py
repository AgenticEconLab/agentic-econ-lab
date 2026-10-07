# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
JSON repair utilities for handling malformed LLM output.

Handles common issues:
- Unescaped LaTeX backslashes (\\beta -> invalid \\b escape)
- Empty or whitespace-only responses
- Markdown code fences around JSON
- Prose text surrounding JSON objects/arrays

Usage:
    from shared.json_repair import repair_json, safe_json_loads

    # Fix backslash issues before parsing
    text = repair_json(raw_llm_output)
    data = json.loads(text)

    # Or use safe_json_loads for full fallback chain
    data = safe_json_loads(raw_llm_output, fallback={})
"""

import json
import re
from typing import Any, Optional


def repair_json(text: str) -> str:
    """
    Repair common JSON issues in LLM output.

    Steps:
    1. Return "{}" for empty/None input
    2. Strip markdown code fences (```json ... ```)
    3. Escape unrecognized backslash sequences (e.g. \\beta -> \\\\beta)
    4. ONLY if still unparseable (never touches parseable text):
       a. insert missing commas between adjacent siblings ("...\"\\n\"...", "}\\n{", "]\\n[")
       b. close a truncated tail (generation cut mid-string / mid-object): drop a dangling
          partial member, close the open string and every open bracket

    Valid JSON escapes (\\", \\\\, \\/, \\b, \\f, \\n, \\r, \\t, \\uXXXX)
    are preserved. If no repair yields parseable JSON the step-3 text is returned
    unchanged (previous behavior), so callers' own error paths still fire.

    Args:
        text: Raw LLM output string.

    Returns:
        Repaired string suitable for json.loads().
    """
    if not text or not text.strip():
        return "{}"

    text = text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            last_fence = text.rfind("```")
            if last_fence > first_nl:
                text = text[first_nl + 1:last_fence].strip()

    # Escape unrecognized backslash sequences.
    # Valid JSON escapes: " \ / b f n r t u
    # Anything else (e.g. \leq, \alpha, \gamma) needs double-escaping.
    # Use negative lookbehind to skip already-escaped backslashes (\\).
    text = re.sub(r'(?<!\\)\\([^"\\/bfnrtu])', r'\\\\\1', text)

    # The fixes below are heuristic, so they are attempted ONLY when the text does not
    # already parse, and only a variant that PARSES is returned — parseable input is
    # returned unchanged.
    if _parses(text):
        return text

    # (a) missing commas between adjacent string members / objects / arrays — the observed
    # "Expecting ',' delimiter" class (e.g. ModelTeam Theory synthesis).
    fixed = re.sub(r'"\s*\n(\s*")', '",\n\\1', text)
    fixed = re.sub(r'\}\s*\n(\s*\{)', '},\n\\1', fixed)
    fixed = re.sub(r'\]\s*\n(\s*\[)', '],\n\\1', fixed)
    if _parses(fixed):
        return fixed

    # (b) truncated tail (max_tokens cut): close the open string/brackets.
    for candidate in (_close_truncated(fixed), _close_truncated(text)):
        if candidate and _parses(candidate):
            return candidate

    return text


def _parses(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def _close_truncated(text: str) -> Optional[str]:
    """Close a JSON string cut off mid-generation.

    Walks the text tracking string/escape state and the open-bracket stack; then drops a
    dangling partial member (an unterminated string, or a trailing ``,`` / ``"key":``) and
    appends the missing closers. Returns None when there is nothing to close (already
    balanced) or the structure never opened."""
    stack = []
    in_string = False
    escape = False
    last_safe = -1          # index AFTER the last completed value/close at depth
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
                last_safe = i + 1
        else:
            if ch == '"':
                in_string = True
            elif ch in "{[":
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]":
                if stack:
                    stack.pop()
                last_safe = i + 1
            elif ch in ",0123456789.true false null e+-":
                # values/numbers complete implicitly; track nothing extra
                pass
    if not stack and not in_string:
        return None                          # balanced — nothing to repair here

    body = text
    if in_string:
        # drop the unterminated string (a partial member) back to the last safe point
        body = text[:last_safe] if last_safe > 0 else text + '"'
    # drop a dangling separator or an orphan key ("key":  /  trailing comma)
    body = re.sub(r'[,\s]+$', "", body)
    body = re.sub(r',?\s*"(?:[^"\\]|\\.)*"\s*:\s*$', "", body)
    body = re.sub(r'[,\s]+$', "", body)

    # recompute the stack for the (possibly shortened) body
    stack = []
    in_string = False
    escape = False
    for ch in body:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch in "{[":
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]":
                if stack:
                    stack.pop()
    if in_string:
        body += '"'
    return body + "".join(reversed(stack))


_SENTINEL = object()


def safe_json_loads(text: str, fallback: Any = _SENTINEL) -> Any:
    """
    Parse JSON from LLM output with multiple fallback strategies.

    Strategy chain:
    1. Direct json.loads()
    2. repair_json() then json.loads()
    3. Extract first JSON object/array from text
    4. Return fallback value (default: empty dict {})

    Args:
        text: Raw LLM output string.
        fallback: Value to return if all parsing fails. Default is {}.
                  Pass None explicitly to get None on failure.

    Returns:
        Parsed JSON value, or fallback on failure.
    """
    _fallback = {} if fallback is _SENTINEL else fallback

    if not text or not text.strip():
        return _fallback

    text = text.strip()

    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: Repair then parse
    try:
        repaired = repair_json(text)
        return json.loads(repaired)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 3: Extract JSON object/array from surrounding text
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start_idx = text.find(start_char)
        if start_idx == -1:
            continue
        # Find matching closing bracket
        depth = 0
        for i in range(start_idx, len(text)):
            if text[i] == start_char:
                depth += 1
            elif text[i] == end_char:
                depth -= 1
                if depth == 0:
                    candidate = text[start_idx:i + 1]
                    try:
                        return json.loads(candidate)
                    except (json.JSONDecodeError, ValueError):
                        # Try with repair
                        try:
                            return json.loads(repair_json(candidate))
                        except (json.JSONDecodeError, ValueError):
                            pass
                    break

    # Strategy 4: Fallback
    return _fallback
