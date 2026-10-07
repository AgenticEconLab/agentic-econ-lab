# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Deterministic number consistency: every number in the final draft must trace to a
source artifact (§4.7 Proofreader's numeric core).

The checker builds a pool of the AUTHORITATIVE numeric values in the source artifacts, then
extracts every number from the draft and verifies each against the pool — exactly, or as a
correct ROUNDING of a pool value (a report that prints 0.13 for an artifact's 0.128473 is
consistent; one that prints 0.31 is not). LLM narrative blocks are where hallucinated
numbers enter; this is the machine that catches them.

A pool that took numbers out of ANY string, including LLM-written literature prose and
specification rationales, would let a fabricated 73.42 in literature prose verify "The
estimated effect is 73.42"; exempting every integer up to 12 would make "Our effect is
9 percent" consistent with no artifacts at all. Therefore:

- the pool holds JSON numeric fields (int/float), numeric-only strings ("0.128", "12.5%"),
  date-only strings ("1960-01-01" contributes 1960, 1 and 1), numeric-only dict keys, the
  recorded derivations (the 'derived' source) and the titles returned by the retrieval APIs
  (``_SOURCE_TITLE_PATHS``: API-returned titles such as "S&P 500", "Covid-19") — never
  numbers inside other free text (literature prose, rationales, notes);
- digits inside a URL or DOI in the draft are an identifier, not a reported number;
- a small integer (|n| <= floor) is exempt only in an enumeration, ordinal or count
  context recognised by syntax: after an ordinal word ("Section 3", "round 2", "#4"), as a
  list marker or table index at the start of a line ("1. ", "| 2 |"), in parentheses
  ("(1)", "AR(1)"), or before a counted noun ("3 models", "2 of 4 hypotheses"); a conventional
  confidence level (90/95/99) is exempt only next to "CI"/"confidence"/"interval". A number
  printed with decimals or followed by %/percent is never exempt.

Scope: this verifies that each printed value OCCURS in an upstream artifact (exactly or as a
presentation rounding); it does not verify that the sentence around it is supported by that
artifact. When artifacts are passed by name, each verified number records the artifacts its
matched value occurs in and the kind of match."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

from .types import ConsistencyReport, UnverifiedNumber, VerifiedNumber

# A pattern that matched inside DOIs and hashes ("...2020.4e903", "439e62711") would parse
# them as inf, and inf would "verify" many numbers. A number must therefore stand alone: not preceded by a word character, '.' or '/' (so DOI segments, URL paths
# and identifiers like x1 are not numbers), not followed by a word character or '/', and not
# continued by ".<digit>" (version strings). A '-' is a minus sign only when it does not follow
# a word character ("0.2-0.5" is the range 0.2 to 0.5). Exponents have at most three digits.
_NUM_RE = re.compile(r"(?<![\w./])-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d{1,3})?"
                     r"(?![\w/])(?!\.\d)")
# Percent rescaling (stored 0.128 printed as 12.8) is accepted only for numbers printed as
# percentages: followed by '%' or the word percent / per cent / percentage point(s).
_PCT_RE = re.compile(r"\s?(?:%|per\s?cent|percentage\s+points?)", re.IGNORECASE)

# LLM-WRITTEN text fields must NEVER back-verify the draft: the narrative that hallucinated a
# number is itself stored in the stage artifacts, and pooling it would let every
# hallucination verify against its own source. Keys here are skipped when their value is
# free text (str, or a dict/list of str as in DraftingResult.narratives).
_LLM_TEXT_KEYS = frozenset({"narrative", "narratives", "proofreader_notes",
                            "interpretation", "report_markdown"})


# Titles returned by the retrieval APIs (OpenAlex/Crossref items, FRED/World Bank series):
# source metadata the report reprints in the References and the data table, never written
# by an agent. LLM-written titles (model_title, section_title, ...) are NOT in this set.
# Titles returned by retrieval APIs, by (parent list key, field). A literature review's own
# 'title' and an LLM-chosen 'series_name' are generated text and stay out.
_SOURCE_TITLE_PATHS = frozenset({("literature_items", "title"), ("citations", "paper_title"),
                                 ("papers", "title"), ("retrieved_data", "source_title")})

# Spans of the draft that are identifiers, not numbers: URLs and DOIs.
_URL_RE = re.compile(r"(?:https?://|www\.|doi:\s*)\S+|\b10\.\d{4,9}/\S+", re.IGNORECASE)


def _is_free_text(v: Any) -> bool:
    if isinstance(v, str):
        return True
    if isinstance(v, dict):
        return all(isinstance(x, str) for x in v.values())
    if isinstance(v, (list, tuple)):
        return all(isinstance(x, str) for x in v)
    return False


# A string is an authoritative value only when it IS a number (optionally a percentage) or a
# date; any other string is free text and contributes nothing to the pool.
_NUMERIC_STR_RE = re.compile(r"\s*[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d{1,3})?\s*%?\s*")
_DATE_STR_RE = re.compile(r"\s*(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?"
                          r"(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?\s*")


# Structured fields that hold a declared numeric range as a string ('[1.2, 1.8]'); a list
# inside any other string field (a rationale, a note) is never a source.
_RANGE_KEYS = frozenset({"typical_range", "range", "bounds", "prior_range", "admissible_range"})

_NUM_LIST_RE = re.compile(
    r"\s*[\[(]?\s*-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?:\s*[,;]\s*-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)+"
    r"\s*[\])]?\s*")


def _string_values(s: str) -> List[float]:
    """The numeric value(s) a structured string carries: a numeric-only string -> its
    number; a date-only string -> its year, month and day. Free text -> []. Numeric lists
    are read only at declared range fields (see _RANGE_KEYS)."""
    if _NUMERIC_STR_RE.fullmatch(s):
        try:
            return [float(s.strip().rstrip("%").strip().replace(",", ""))]
        except ValueError:
            return []
    m = _DATE_STR_RE.fullmatch(s)
    if m:
        return [float(g) for g in m.groups() if g]
    return []


def _walk_numbers(obj: Any, pool: Set[float], label: str = "",
                  prov: Optional[Dict[float, Set[str]]] = None, parent: str = "") -> None:
    def _add(x: float) -> None:
        if not math.isfinite(x):          # inf/nan never enter the pool
            return
        pool.add(x)
        if prov is not None and label:
            prov.setdefault(x, set()).add(label)

    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        if math.isfinite(float(obj)):
            _add(float(obj))
        return
    if isinstance(obj, str):
        for x in _string_values(obj):     # numeric-only / date-only strings; never free text
            _add(x)
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            _walk_numbers(str(k), pool, label, prov)   # numeric keys (e.g. {"2020": ...})
            if str(k) in _LLM_TEXT_KEYS and _is_free_text(v):
                continue                          # never pool LLM narration
            if (str(k) in _RANGE_KEYS and isinstance(v, str) and _NUM_LIST_RE.fullmatch(v)
                    and not _NUMERIC_STR_RE.fullmatch(v)):     # '1,234' is one number
                for t in re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", v):
                    try:
                        _add(float(t))
                    except ValueError:
                        pass
                continue
            if (parent, str(k)) in _SOURCE_TITLE_PATHS and isinstance(v, str):
                # a title returned by a retrieval API is an identifier the report reprints
                # verbatim ("S&P 500", "Covid-19"): its numbers are names, not estimates
                for t in _NUM_RE.findall(v):
                    try:
                        _add(float(t.replace(",", "")))
                    except ValueError:
                        pass
                continue
            _walk_numbers(v, pool, label, prov, parent=str(k))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _walk_numbers(v, pool, label, prov, parent=parent)


def artifact_number_pool(artifacts: Iterable[Any]) -> Set[float]:
    """The authoritative numbers in the artifacts — numeric leaves, numeric-only and
    date-only strings, numeric dict keys — never numbers inside free text."""
    pool: Set[float] = set()
    for art in artifacts:
        _walk_numbers(art, pool)
    return pool


def _extract(text: str) -> List[tuple]:
    """(value, context, printed_as_percent, match) for every number in the draft."""
    out = []
    text = text or ""
    url_spans = [u.span() for u in _URL_RE.finditer(text)]
    for m in _NUM_RE.finditer(text):
        if any(a <= m.start() < b for a, b in url_spans):
            continue                      # digits of a URL or DOI are not a reported number
        try:
            v = float(m.group(0).replace(",", ""))
        except ValueError:
            continue
        # a Unicode minus or en dash written as a sign ("−0.075", "–0.057"), not as a range
        # ("1960–2024"): the character before it is not a digit
        st = m.start()
        if v > 0 and st >= 1 and text[st - 1] in "\u2212\u2013":
            before = text[:st - 1]
            back = before.rstrip()
            # joined to a word or number ('A–5', '1960–2024'): a connector; after a space,
            # a range only when a number precedes ('1960 –2024'); otherwise a sign
            # ('coefficient –0.057', '(−0.075)')
            joined = bool(before) and before[-1].isalnum()
            spaced_range = bool(back) and back != before and back[-1].isdigit()
            if text[st - 1] == "\u2212":
                # a true minus sign is never a range separator ('0.12 −0.057')
                if not joined:
                    v = -v
            elif not joined and not spaced_range:
                v = -v
        lo, hi = max(0, m.start() - 35), min(len(text), m.end() + 35)
        is_pct = _PCT_RE.match(text, m.end()) is not None
        out.append((v, text[lo:hi].replace("\n", " "), is_pct, m))
    return out


def extract_numbers(text: str) -> List[tuple]:
    """(value, context) for every number in the draft."""
    return [(v, ctx) for v, ctx, _pct, _m in _extract(text)]


# Words after which a small integer is an ordinal/label ("Section 3", "round 2", "lag 1").
_ORDINAL_BEFORE_RE = re.compile(
    r"(?:\b(?:sections?|§|tables?|figures?|figs?\.|stages?|steps?|rounds?|models?|equations?|"
    r"eqs?\.|appendix|chapters?|columns?|rows?|panels?|hypothes[ie]s|lags?|orders?|teams?|"
    r"attempts?|candidates?|specifications?|specs?|waves?|parts?|items?|questions?|rq|"
    r"no\.|nos\.|versions?|v|level|tier|phase|layer|pillar|check|mode)[\s-]*|#\s*)$",
    re.IGNORECASE)
# Words after a small integer that make it a magnitude rather than a count.
_NOT_COUNT_WORDS = frozenset({
    "percent", "per", "percentage", "pct", "pp", "bp", "bps", "basis", "point", "points",
    "times", "fold", "unit", "units", "and", "or", "but", "to", "in", "on", "at", "for",
    "is", "are", "was", "were", "with", "than", "the", "a", "an", "as", "by", "from",
    "dollars", "usd", "euros", "eur", "cents", "standard", "sd", "sds", "log", "x"})
_COUNT_AFTER_RE = re.compile(r"[ \t]+([A-Za-z][A-Za-z-]*)")
# Structural counts of the pipeline's own objects; a count of findings, effects, estimates,
# observations or hypotheses is a claim and needs provenance.
_STRUCT_COUNT_WORDS = frozenset({
    "model", "models", "equation", "equations", "stage", "stages", "team", "teams", "round",
    "rounds", "cycle", "cycles", "section", "sections", "table", "tables", "figure", "figures",
    "question", "questions", "member", "members", "judge", "judges", "lens", "lenses",
    "checkpoint", "checkpoints", "step", "steps", "agent", "agents", "candidate", "candidates",
    "column", "columns", "panel", "panels", "appendix", "chapter", "chapters"})
_CONF_RE = re.compile(r"\b(?:CI|CIs|confidence|interval)\b", re.IGNORECASE)


def _exempt_small_int(text: str, m: "re.Match", v: float, is_pct: bool,
                      floor: int) -> bool:
    """Whether a draft number is an enumeration/ordinal/count (or a confidence level) by
    SYNTAX. Decimals, percentages and negative numbers are never exempt."""
    tok = m.group(0)
    start, end = m.start(), m.end()
    if v in _ALLOWED_CONSTANTS:
        window = text[max(0, start - 25):min(len(text), end + 25)]
        if _CONF_RE.search(window):
            return True
    if is_pct or v < 0 or any(c in tok for c in ".eE-") or not float(v).is_integer() \
            or abs(v) > floor:
        return False
    before, after = text[max(0, start - 40):start], text[end:end + 40]
    if _ORDINAL_BEFORE_RE.search(before):
        return True
    line_prefix = text[text.rfind("\n", 0, start) + 1:start]
    if (re.fullmatch(r"[ \t]*(?:[-*+>][ \t]*|#+[ \t]*|\|[ \t]*)?", line_prefix)
            and re.match(r"(?:[.)][ \t]|[ \t]*\|)", after)):
        return True                                   # list marker / table index
    if before.endswith("(") and after.startswith(")"):
        pre = before[:-1]
        if pre and (pre[-1].isalnum() or pre[-1] == "_"):
            return True                               # AR(1), MA(2): an identifier's order
        if re.fullmatch(r"[ \t]*", line_prefix[:-1] if line_prefix.endswith("(") else line_prefix):
            return True                               # "(1)" opening a list item
    w = _COUNT_AFTER_RE.match(after)
    if w and w.group(1).lower() in _STRUCT_COUNT_WORDS:
        return True                                   # 3 models, 2 rounds
    return False


def _decimals(token) -> int:
    """Decimal places as PRINTED: '122' -> 0 (float 122.0 would give 1), '0.075' -> 3."""
    s = token if isinstance(token, str) else f"{token}"
    s = re.split(r"[eE]", s.replace(",", ""))[0]
    if isinstance(token, str):
        return len(s.split(".")[1]) if "." in s else 0
    return len(s.split(".")[1]) if "." in s and not s.endswith(".0") else 0


def _sigfigs(v: float) -> int:
    """Significant digits of the value as it would print (e.g. 2.2e-05 -> 2)."""
    mant = f"{abs(v):e}".split("e")[0].rstrip("0").rstrip(".")
    return max(1, len(mant.replace(".", "")))


def _sig_round(x: float, sig: int) -> float:
    if x == 0 or not math.isfinite(x):
        return x
    return round(x, -int(math.floor(math.log10(abs(x)))) + (sig - 1))


def _match(v: float, pool: Set[float],
           allow_percent: bool = True, printed: Optional[str] = None) -> Optional[Tuple[str, float]]:
    """How ``v`` matches the pool: ``(kind, matched_pool_value)``, preferring an exact match
    over a rounding, significant-figure, or percent-scaling match; None if it does not.
    Percent scaling is tried only when ``allow_percent`` (the number was printed as a
    percentage); non-finite values never match."""
    if not math.isfinite(v):
        return None
    best: Optional[Tuple[str, float]] = None
    rank = {"exact": 0, "rounding": 1, "significant_figures": 2, "percent_scaling": 3}
    for p in pool:
        if not math.isfinite(p):
            continue
        kind = None
        if p == v or abs(p - v) <= 1e-9 * max(1.0, abs(p)):
            kind = "exact"
        else:
            # v is a correct rounding of p (presentation rounding is legitimate)
            try:
                if round(p, _decimals(printed if printed is not None else v)) == v:
                    kind = "rounding"
            except (OverflowError, ValueError):
                continue
            # Tiny magnitudes print in scientific notation, where
            # decimal-place rounding is meaningless ('2.2e-05' vs stored 2.223e-05) —
            # compare at the draft value's significant figures instead.
            if kind is None and 0.0 < abs(v) < 1e-3 and math.isfinite(p) and p != 0.0:
                if _sig_round(p, _sigfigs(v)) == v:
                    kind = "significant_figures"
            # percentages: draft prints 12.8% for a stored share 0.128 (and vice versa)
            if kind is None and allow_percent:
                for scaled in (p * 100.0, p / 100.0):
                    if math.isfinite(scaled) and round(scaled, _decimals(printed if printed is not None else v)) == v:
                        kind = "percent_scaling"
                        break
        if kind is not None and (best is None or rank[kind] < rank[best[0]]):
            best = (kind, p)
            if kind == "exact":
                break
    return best


def _matches(v: float, pool: Set[float]) -> bool:
    return _match(v, pool) is not None


# Template constants that are legitimate without artifact backing: conventional confidence
# levels printed next to "CI"/"confidence"/"interval" ("95% CI").
_ALLOWED_CONSTANTS = frozenset({90.0, 95.0, 99.0})


def check_consistency(report_text: str, artifacts: "Mapping[str, Any] | Iterable[Any]",
                      small_int_floor: int = 12) -> ConsistencyReport:
    """``artifacts``: a mapping ``{artifact_name: artifact}`` (preferred: verified numbers then
    record the artifacts their value occurs in) or a plain iterable (labelled by position)."""
    named = (list(artifacts.items()) if isinstance(artifacts, Mapping)
             else [(f"artifact[{i}]", a) for i, a in enumerate(artifacts)])
    pool: Set[float] = set()
    prov: Dict[float, Set[str]] = {}
    for label, art in named:
        _walk_numbers(art, pool, label, prov)
    numbers = _extract(report_text)
    if not numbers:
        return ConsistencyReport(verdict="empty")
    report = ConsistencyReport(total_numbers=len(numbers))
    for v, ctx, is_pct, m in numbers:
        if math.isfinite(v) and _exempt_small_int(report_text, m, v, is_pct,
                                                   small_int_floor):
            report.skipped_small_ints += 1
            continue
        hit = _match(v, pool, allow_percent=is_pct, printed=m.group(0))
        if hit is not None:
            kind, p = hit
            srcs = sorted(prov.get(p, set()))
            report.verified += 1
            report.verified_numbers.append(VerifiedNumber(
                value=v, context=ctx, match=kind, matched_value=p, sources=srcs))
            report.match_kinds[kind] = report.match_kinds.get(kind, 0) + 1
            for src in srcs:
                report.verified_by_source[src] = report.verified_by_source.get(src, 0) + 1
        else:
            report.unverified.append(UnverifiedNumber(value=v, context=ctx))
    report.verdict = "has_unverified" if report.unverified else "consistent"
    return report
