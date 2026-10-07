#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Export the Tier-2 ensemble scores of a run campaign to llm_scores.csv, the input of
select_sample.py (one row per item x dimension; item_id = <STAGE>-<job id>).

Usage (from the repository root):
    python evaluation/human_validation/export_llm_scores.py \\
        --runs-root <runs_root> --prefix v072 \\
        --out <round_dir>/llm_scores.csv
"""
import argparse
import csv
import glob
import json
import os

STAGES = {"IdeationTeam": ("IDE", "research_questions"),
          "LiteratureTeam": ("LIT", "literature_review"),
          "ModelTeam": ("MOD", "model_specification"),
          "DataTeam": ("DAT", "data_report"),
          "CodeTeam": ("COD", "code_modules"),
          "EstimationTeam": ("EST", "estimation_results"),
          "ReportingTeam": ("REP", "final_report")}
DIMS = ["correctness", "soundness", "innovation_potential", "transparency",
        "decision_quality", "economic_rigor"]
JUDGES = {"mistral": "mistral-small-3.2-24b-fp8", "gemma": "gemma-3-27b-it-fp8"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--prefix", default="v072")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows, incomplete = [], []
    for run in sorted(glob.glob(os.path.join(a.runs_root, a.prefix + "_*"))):
        job = os.path.basename(run).rsplit("_", 1)[-1]
        scores = {}
        for jk, slug in JUDGES.items():
            p = os.path.join(run, "_eval_view", f"tier2_multirun_{slug}.json")
            scores[jk] = json.load(open(p)).get("configs", {}) if os.path.exists(p) else {}
        for team, (code, otype) in STAGES.items():
            vals = {}
            for jk in JUDGES:
                k = next((c for c in scores[jk] if c.startswith(team + "_")), None)
                vals[jk] = {d: ((scores[jk].get(k) or {}).get(d) or {}).get("mean") for d in DIMS} if k else {}
            for d in DIMS:
                m, g = vals["mistral"].get(d), vals["gemma"].get(d)
                if m is None or g is None:
                    incomplete.append(f"{code}-{job}:{d}")
                    continue
                rows.append({"item_id": f"{code}-{job}", "team": team, "output_type": otype,
                             "dimension": d, "mistral": m, "gemma": g})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "team", "output_type", "dimension", "mistral", "gemma"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {a.out}; incomplete cells: {len(incomplete)}")
    if incomplete:
        print("  e.g.", incomplete[:10])


if __name__ == "__main__":
    main()
