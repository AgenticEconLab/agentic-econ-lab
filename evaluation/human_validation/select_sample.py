#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Select a spread-stratified sample of workflow outputs for the human-expert
validation study (the human-validation table of the paper).

For each team we pick items spanning the LLM score range (min / lower-mid /
upper-mid / max) rather than a random draw, because the Tier-2 scores cluster
narrowly and a narrow cluster gives an unstable human-vs-LLM correlation.

Usage:
    python select_sample.py --llm-scores llm_scores.csv --n-per-team 4 --out-dir .

Inputs
    llm_scores.csv : item_id, team, output_type, dimension, mistral, gemma
Outputs
    item_manifest.csv        : chosen items + their LLM mean score
    rating_sheet_template.csv : blank sheet for the economists (one row per item)
"""
import argparse
import sys

import pandas as pd

DIMS = ["correctness", "soundness", "innovation_potential", "transparency",
        "decision_quality", "economic_rigor"]


def pick_spread(df_team, n):
    """Return n items evenly spaced across the team's ranked LLM-mean range."""
    d = df_team.sort_values("llm_mean").reset_index(drop=True)
    m = len(d)
    if m <= n:
        idx = list(range(m))
    else:
        idx = sorted({round(i * (m - 1) / (n - 1)) for i in range(n)})
    return d.iloc[idx]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm-scores", required=True)
    ap.add_argument("--n-per-team", type=int, default=4)
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()

    df = pd.read_csv(a.llm_scores)
    need = {"item_id", "team", "output_type", "dimension", "mistral", "gemma"}
    missing = need - set(df.columns)
    if missing:
        sys.exit(f"llm_scores.csv missing columns: {sorted(missing)}")

    df["ensemble"] = (df["mistral"] + df["gemma"]) / 2.0
    per_item = (df.groupby(["item_id", "team", "output_type"])["ensemble"]
                  .mean().reset_index().rename(columns={"ensemble": "llm_mean"}))

    sample = pd.concat([pick_spread(g, a.n_per_team)
                        for _, g in per_item.groupby("team")], ignore_index=True)

    manifest = sample[["item_id", "team", "output_type", "llm_mean"]] \
        .sort_values(["team", "llm_mean"])
    manifest.to_csv(f"{a.out_dir}/item_manifest.csv", index=False)

    sheet = sample[["item_id", "team", "output_type"]].copy()
    sheet.insert(0, "rater_id", "")
    for d in DIMS:
        sheet[d] = ""
    sheet["notes"] = ""
    sheet.sort_values(["team", "item_id"]).to_csv(
        f"{a.out_dir}/rating_sheet_template.csv", index=False)

    print(f"Selected {len(sample)} items across {sample['team'].nunique()} teams")
    print(f"  wrote {a.out_dir}/item_manifest.csv")
    print(f"  wrote {a.out_dir}/rating_sheet_template.csv  (blank; give one copy per rater)")
    print("\nPer-team spread of LLM mean score (aim for a wide range):")
    for team, g in manifest.groupby("team"):
        vals = ", ".join(f"{v:.2f}" for v in g["llm_mean"])
        print(f"  {team:16s} n={len(g)}  LLM means: [{vals}]")


if __name__ == "__main__":
    main()
