#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Build the rater packets (one per item_id in item_manifest.csv) for the
human-expert validation study.

Reuses LLMEvaluator._load_outputs directly rather than re-deriving the file
list, so a rater sees EXACTLY the files the Tier-2 LLM judges scored for that
(team, run) -- no separate summarization step that could introduce a mismatch
between what was judged and what a human rates.

Usage (from the repository root):
    python evaluation/human_validation/build_rater_packets.py \
        --manifest <round_dir>/item_manifest.csv \
        --scratch-root <runs_root> \
        --out-dir <round_dir>/rater_packets
"""
import argparse
import os
import csv
import json
from pathlib import Path

from evaluation.scoring.llm_evaluator import LLMEvaluator

def run_dir_name(run_id: str, scratch_root: Path = None) -> str:
    """Scratch run directory for a job id: the original campaign's
    ``full_pipeline_committee_smoke_<id>``, or a labelled run ``<prefix>_<label>_<id>``
    (e.g. ``v072_rep07_<job>``) found under ``scratch_root``."""
    legacy = f"full_pipeline_committee_smoke_{run_id}"
    if scratch_root is not None and not (Path(scratch_root) / legacy).exists():
        hits = sorted(Path(scratch_root).glob(f"*_{run_id}"))
        if hits:
            return hits[0].name
    return legacy


OUTPUT_TYPE_LABEL = {
    "research_questions": "Ideation -- prioritized research questions",
    "literature_review": "Literature -- review, gap analysis, synthesis",
    "model_specification": "Model -- theory, formal design, calibration",
    "data_report": "Data -- retrieval, cleaning, quality assurance",
    "code_modules": "Code -- generated modules, validation, experimentation",
    "estimation_results": "Estimation -- specification, verdict, diagnostics, robustness",
    "final_report": "Reporting -- terminal research report",
}


_TABLE_KEY_CANDIDATES = ["title", "name", "question", "citation", "id"]
_TABLE_EXTRA_CANDIDATES = ["authors", "year", "publication_year", "rank", "priority_rank"]
_LONG_LIST_THRESHOLD = 15  # a list this long of similar records -> table, not raw JSON


def _compact_table(items: list) -> str:
    """Render a long list of similar dict records as a compact Markdown table
    (title/name + a couple of identifying fields), not full pretty-printed JSON.
    Keeps the SAME facts (nothing dropped from the underlying artifact -- this
    is a rendering choice for readability, the file itself is unmodified and
    named right above this table), just not every field repeated per row."""
    key_col = next((k for k in _TABLE_KEY_CANDIDATES if k in items[0]), None)
    extra_cols = [k for k in _TABLE_EXTRA_CANDIDATES if k in items[0]][:2]
    if key_col is None:
        return None  # doesn't fit the pattern; caller falls back to full JSON
    header = ["#", key_col] + extra_cols
    rows = [header, ["---"] * len(header)]
    for i, it in enumerate(items, 1):
        val = str(it.get(key_col, ""))[:100].replace("|", "/").replace("\n", " ")
        extras = [str(it.get(c, ""))[:40].replace("|", "/") for c in extra_cols]
        rows.append([str(i), val, *extras])
    lines = [f"*({len(items)} items; compact table -- full record for each, including "
             "abstracts/full text, is in the named JSON file itself)*", ""]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _render_json_value(content) -> list:
    """Render one file's parsed JSON. A long top-level list of dict records
    (e.g. literature_batch.json's 44-paper `literature_items`) becomes a
    compact table under its own key heading; everything else stays as
    pretty-printed JSON. One level deep only -- the fields that actually
    grow long in this project's real artifacts (see build script's own
    inspection of literature_batch.json) are all top-level keys, and going
    deeper risks splicing JSON fences incorrectly for no real benefit here."""
    out = []
    if isinstance(content, dict):
        compact_keys = {
            k: v for k, v in content.items()
            if isinstance(v, list) and len(v) >= _LONG_LIST_THRESHOLD
            and v and isinstance(v[0], dict)
        }
        rest = {k: v for k, v in content.items() if k not in compact_keys}
        if rest:
            out.append("```json")
            out.append(json.dumps(rest, indent=2, ensure_ascii=False))
            out.append("```")
        for k, v in compact_keys.items():
            out.append("")
            out.append(f"**`{k}`** ({len(v)} items)")
            out.append("")
            table = _compact_table(v)
            out.append(table if table else "```json\n" + json.dumps(v, indent=2, ensure_ascii=False) + "\n```")
    else:
        out.append("```json")
        out.append(json.dumps(content, indent=2, ensure_ascii=False))
        out.append("```")
    return out


def render_packet(item_id: str, team: str, output_type: str, run_id: str,
                   outputs: dict) -> str:
    lines = [
        f"# Rating item {item_id}",
        "",
        f"**Stage:** {OUTPUT_TYPE_LABEL.get(output_type, output_type)}",
        f"**Source run:** {run_id} (job {run_id})",
        "",
        "*This is exactly the set of files the Tier-2 LLM judges (Mistral-Small-3.2-24B, "
        "Gemma-3-27B) were given for this stage -- nothing added, nothing summarized. Long "
        "lists of similar records (e.g. the literature corpus) are shown as a compact table "
        "for readability; the full record for each, unabridged, is in the named JSON file "
        "in the repository.*",
        "",
        "---",
        "",
    ]
    if not outputs:
        lines.append("*(no output files found -- flag this item, do not score it)*")
        return "\n".join(lines)

    for filename, content in outputs.items():
        lines.append(f"## `{filename}`")
        lines.append("")
        if isinstance(content, str):
            body = content.splitlines()
        else:
            body = _render_json_value(content)
        # _render_json_value's list elements can each be a whole multi-line json.dumps
        # blob -- split so the line cap below actually counts rendered lines, not blobs.
        body = [ln for chunk in body for ln in chunk.splitlines()] if body else body
        lines.extend(_cap_lines(body, filename))
        lines.append("")
    return "\n".join(lines)


_MAX_LINES_PER_FILE = 350  # hard backstop: no single file balloons a packet regardless
                            # of its JSON shape (nested lists the table heuristic misses,
                            # e.g. knowledge_graph.json's 223 nodes/416 edges). What's
                            # shown is always a genuine PREFIX of the real file -- never
                            # reordered or cherry-picked -- so a rater who wants more reads
                            # the named file itself in the repo.


def _cap_lines(body_lines: list, filename: str) -> list:
    if len(body_lines) <= _MAX_LINES_PER_FILE:
        return body_lines
    fence_open = body_lines[0].startswith("```")
    kept = body_lines[:_MAX_LINES_PER_FILE]
    if fence_open and not kept[-1].startswith("```"):
        kept.append("```")
    kept.append("")
    kept.append(f"*[truncated: {len(body_lines) - _MAX_LINES_PER_FILE} more lines in "
                f"`{filename}` -- this is the file's own genuine prefix, nothing "
                f"reordered; the full file is in the repository]*")
    return kept


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="the round's item_manifest.csv")
    ap.add_argument("--scratch-root", default=os.environ.get("AEL_RUNS_DIR"),
                     required=not os.environ.get("AEL_RUNS_DIR"),
                     help="directory holding the run folders (default: $AEL_RUNS_DIR)")
    ap.add_argument("--out-dir", required=True, help="the round's rater_packets/ folder")
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_root = Path(a.scratch_root)
    ev = LLMEvaluator(model="vllm/dummy")  # only ._load_outputs() is used -- no LLM call

    rows = list(csv.DictReader(open(a.manifest)))
    missing = []
    for row in rows:
        item_id, team, output_type = row["item_id"], row["team"], row["output_type"]
        run_id = item_id.split("-")[-1]
        team_dir = scratch_root / run_dir_name(run_id, scratch_root) / team
        outputs = ev._load_outputs(team, mode="n/a", base_path=scratch_root, output_dir=team_dir)
        if not outputs:
            missing.append((item_id, f"no files loaded from {team_dir}"))
        packet = render_packet(item_id, team, output_type, run_id, outputs)
        (out_dir / f"{item_id}.md").write_text(packet, encoding="utf-8")
        print(f"  {item_id:16s} <- {len(outputs)} file(s) from {team_dir}")

    print(f"\nWrote {len(rows) - len(missing)}/{len(rows)} packets to {out_dir}")
    if missing:
        print("MISSING / EMPTY:")
        for item_id, reason in missing:
            print(f"  {item_id}: {reason}")


if __name__ == "__main__":
    main()
