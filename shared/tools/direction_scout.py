# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Fair research-direction discovery (no hardcoded seed, no field favoritism).

Why: every validation run had seeded the pipeline with the same hardcoded macro topic, so
every model choice inherited a direction no evidence ever chose. This module replaces the
seed with a two-step, fully disclosed mechanism:

1. **Field-balanced slate** (``discover_directions``): one search per JEL top-level field —
   the discipline's own neutral taxonomy — then ONE grounded LLM extraction that turns each
   field's live signals into a concrete candidate direction. Fields with no signals are
   DROPPED with a note, never invented.
2. **Committee choice** (``choose_direction``): each LLM-economist committee member (with its
   theory/data/policy lens) independently ranks the slate; Borda count picks the winner.
   Every ballot — model, lens, ranking, rationale — is returned as provenance, so the run
   records WHY this direction was chosen and by whom.

The human ``--topic`` override remains first-class (a researcher choosing their own topic is
legitimate HITL); what this removes is the FRAMEWORK imposing one.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

# JEL top-level classification — the neutral, discipline-canonical field taxonomy.
JEL_FIELDS: List[tuple] = [
    ("C", "econometrics and quantitative methods"),
    ("D", "microeconomics and behavioral economics"),
    ("E", "macroeconomics and monetary economics"),
    ("F", "international economics and trade"),
    ("G", "financial economics"),
    ("H", "public economics and taxation"),
    ("I", "health, education, and welfare economics"),
    ("J", "labor and demographic economics"),
    ("K", "law and economics"),
    ("L", "industrial organization"),
    ("N", "economic history"),
    ("O", "development, innovation, and growth"),
    ("Q", "agricultural, environmental, and energy economics"),
    ("R", "urban, regional, and transportation economics"),
]

_SNIPPET_CAP = 220          # chars per search result fed to the extractor
_RESULTS_PER_FIELD = 4


def _parse_json_block(raw: str):
    text = str(raw).strip()
    m = re.search(r"\[.*\]|\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    from shared.json_repair import repair_json
    return json.loads(repair_json(text))


def discover_directions(fields: Optional[List[tuple]] = None,
                        collector=None) -> Dict[str, Any]:
    """Build the field-balanced candidate slate from live signals.

    Returns {"candidates": [{field_code, field, direction, evidence}], "notes": [...],
    "grounded": bool}. When NO field yields search signals (offline node), falls back to an
    ungrounded LLM slate across the same fields — still field-balanced, and DISCLOSED."""
    fields = fields or JEL_FIELDS
    notes: List[str] = []
    signals: Dict[str, List[Dict[str, str]]] = {}

    try:
        from shared.tools.news_search import search_news
    except Exception:                                       # pragma: no cover - defensive
        search_news = None

    if search_news is not None:
        for code, name in fields:
            try:
                hits = search_news(f"{name} research", max_results=_RESULTS_PER_FIELD)
            except Exception:
                hits = []
            if hits:
                signals[code] = hits
            else:
                notes.append(f"field {code} ({name}): no live signals — dropped, not invented")

    from shared.llm import LLMClient
    llm = LLMClient(temperature=0.3, collector=collector, agent_name="DirectionScout")

    if signals:
        blocks = []
        for code, name in fields:
            if code not in signals:
                continue
            lines = [f"[{code}] {name}:"]
            for h in signals[code]:
                lines.append(f"  - {h.get('title', '')[:120]} :: "
                             f"{str(h.get('content', ''))[:_SNIPPET_CAP]} ({h.get('url', '')})")
            blocks.append("\n".join(lines))
        prompt = (
            "For EACH field below, extract ONE concrete, researchable economics question "
            "grounded in that field's signals (cite which signal). Do not favor any field; "
            "do not merge fields; skip a field only if its signals contain nothing "
            "researchable.\n\n" + "\n\n".join(blocks) +
            "\n\nReturn STRICT JSON: [{\"field_code\": \"..\", \"direction\": \"one-sentence "
            "research question\", \"evidence\": \"url or title used\"}]")
        grounded = True
    else:
        notes.append("NO live signals in any field — falling back to an UNGROUNDED "
                     "LLM slate (field-balanced, but not evidence-based); disclosed")
        prompt = (
            "For EACH of these economics fields, state ONE concrete open research question "
            "currently debated in that field. Do not favor any field.\n"
            + "\n".join(f"[{c}] {n}" for c, n in fields) +
            "\nReturn STRICT JSON: [{\"field_code\": \"..\", \"direction\": \"...\", "
            "\"evidence\": \"none (ungrounded)\"}]")
        grounded = False

    field_names = dict(fields)
    candidates: List[Dict[str, Any]] = []
    try:
        raw = llm.invoke([
            {"role": "system", "content":
                "You are a field-neutral research scout for economics. STRICT JSON only."},
            {"role": "user", "content": prompt},
        ], max_tokens=2200)
        for item in _parse_json_block(raw):
            if not isinstance(item, dict):
                continue
            code = str(item.get("field_code", "")).strip().upper()
            direction = str(item.get("direction", "")).strip()
            if code in field_names and direction:
                candidates.append({
                    "field_code": code,
                    "field": field_names[code],
                    "direction": direction,
                    "evidence": str(item.get("evidence", ""))[:300],
                })
    except Exception as e:
        notes.append(f"slate extraction failed: {e}")

    return {"candidates": candidates, "notes": notes, "grounded": grounded}


def choose_direction(slate: Dict[str, Any], collector=None) -> Dict[str, Any]:
    """Committee-ranked choice over the slate; full ballot provenance returned.

    Each committee member (its own model + lens) ranks its top 3 candidates; Borda count
    aggregates. Without a committee (non-llm_economist runs), a single AEL_MODEL ranking is
    used and labeled as such. Returns {"chosen", "method", "ballots", "scores"}."""
    candidates = list(slate.get("candidates") or [])
    if not candidates:
        return {"chosen": None, "method": "none", "ballots": [],
                "error": "empty slate — pass --topic explicitly"}

    from shared.llm import LLMClient
    listing = "\n".join(f"{i}: [{c['field_code']}] {c['direction'][:180]}"
                        for i, c in enumerate(candidates))
    base_prompt = (
        "Candidate research directions (index: [JEL field] question):\n" + listing +
        "\n\nRank the THREE most promising by scientific merit (novelty, tractability, "
        "data prospects) — judge the QUESTION, not the field; do not favor macroeconomics "
        "or any other field.\nReturn STRICT JSON: {\"ranking\": [best_idx, second_idx, "
        "third_idx], \"rationale\": \"<one sentence>\"}")

    members: List[tuple] = []
    if os.environ.get("AEL_HITL_MODE", "").strip().lower() == "llm_economist":
        try:
            from shared.llm_economist import get_committee_models, _lens_for
            models = get_committee_models()
            members = [(m, _lens_for(i)) for i, m in enumerate(models)]
        except Exception:
            members = []
    if not members:
        from shared.model_config import default_model
        members = [(os.environ.get("AEL_MODEL") or default_model(), ("single", ""))]

    ballots: List[Dict[str, Any]] = []
    scores = [0.0] * len(candidates)
    for model, (lens_name, lens_directive) in members:
        try:
            llm = LLMClient(model=model, temperature=0.2, collector=collector,
                            agent_name=f"DirectionJudge:{lens_name}")
            system = ("You are one member of an economics review committee choosing a "
                      "research direction. " + (lens_directive or ""))
            raw = llm.invoke([{"role": "system", "content": system},
                              {"role": "user", "content": base_prompt}], max_tokens=400)
            parsed = _parse_json_block(raw)
            ranking = [int(i) for i in parsed.get("ranking", [])
                       if isinstance(i, (int, float)) and 0 <= int(i) < len(candidates)][:3]
            for pos, idx in enumerate(ranking):
                scores[idx] += 3 - pos                       # Borda: 3/2/1
            ballots.append({"model": model, "lens": lens_name, "ranking": ranking,
                            "rationale": str(parsed.get("rationale", ""))[:300]})
        except Exception as e:
            ballots.append({"model": model, "lens": lens_name, "ranking": [],
                            "error": str(e)[:200]})

    if not any(scores):
        return {"chosen": None, "method": "committee_failed", "ballots": ballots,
                "error": "no valid rankings — pass --topic explicitly"}
    winner = max(range(len(candidates)), key=lambda i: scores[i])
    return {
        "chosen": candidates[winner],
        "method": "committee_borda" if len(members) > 1 else "single_llm",
        "ballots": ballots,
        "scores": {str(i): s for i, s in enumerate(scores) if s},
    }
