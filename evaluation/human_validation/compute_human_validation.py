#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Compare economist ratings with the LLM-as-reviewer ensemble and emit the
numbers for the human-validation table of the paper.

Merges human ratings (1--5 Likert, one score per content dimension) with the
two-judge LLM ensemble (0--1) on the same (item x dimension) pairs, then reports
per team and pooled: expert mean, LLM mean, signed gap (LLM - expert), and the
Spearman rank correlation. Also reports inter-rater reliability. Prints LaTeX
rows ready to paste into the manuscript.

Usage:
    python compute_human_validation.py \
        --human rater1.csv rater2.csv [rater3.csv ...] \
        --llm-scores llm_scores.csv --out results.csv

Inputs
    human sheets : rater_id, item_id, team, [correctness, soundness,
                   innovation, transparency]  (1--5; blank cells are ignored)
    llm_scores   : item_id, team, dimension, mistral, gemma  (0--1)
"""
import argparse
import itertools
import sys

import pandas as pd
from scipy.stats import spearmanr

DIMS = ["correctness", "soundness", "innovation_potential", "transparency",
        "decision_quality", "economic_rigor"]


def to01(x):
    """Likert 1..5 -> 0..1."""
    return (x - 1.0) / 4.0


def rho(a, b):
    """Spearman correlation, NaN when undefined (too few / constant values)."""
    a, b = pd.Series(list(a)), pd.Series(list(b))
    if len(a) < 3 or a.nunique() < 2 or b.nunique() < 2:
        return float("nan")
    return float(spearmanr(a, b).correlation)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--human", required=True, nargs="+",
                    help="one or more filled rating-sheet CSVs")
    ap.add_argument("--llm-scores", required=True)
    ap.add_argument("--out", default="results.csv")
    a = ap.parse_args()

    # --- human: wide -> long -> mean over raters ---
    h = pd.concat([pd.read_csv(f) for f in a.human], ignore_index=True)
    for c in ["rater_id", "item_id", "team", *DIMS]:
        if c not in h.columns:
            sys.exit(f"human sheet missing column: {c}")
    hl = h.melt(id_vars=["rater_id", "item_id", "team"], value_vars=DIMS,
                var_name="dimension", value_name="likert")
    hl["likert"] = pd.to_numeric(hl["likert"], errors="coerce")
    hl = hl.dropna(subset=["likert"])
    hl["expert01"] = to01(hl["likert"])
    expert = hl.groupby(["item_id", "team", "dimension"])["expert01"].mean().reset_index()

    # --- llm ensemble ---
    L = pd.read_csv(a.llm_scores)
    L["llm01"] = (L["mistral"] + L["gemma"]) / 2.0
    llm = L[["item_id", "team", "dimension", "llm01"]]

    m = expert.merge(llm, on=["item_id", "team", "dimension"], how="inner")
    if m.empty:
        sys.exit("no overlapping (item, dimension) pairs between human and LLM inputs")

    rows = []
    for team, g in m.groupby("team"):
        rows.append(dict(team=team, n=len(g), expert=g.expert01.mean(),
                         llm=g.llm01.mean(), gap=g.llm01.mean() - g.expert01.mean(),
                         rho=rho(g.expert01, g.llm01)))
    rows.append(dict(team="Pooled", n=len(m), expert=m.expert01.mean(),
                     llm=m.llm01.mean(), gap=m.llm01.mean() - m.expert01.mean(),
                     rho=rho(m.expert01, m.llm01)))
    res = pd.DataFrame(rows)
    res.to_csv(a.out, index=False)

    # --- inter-rater reliability: mean pairwise Spearman across raters ---
    raters = sorted(h["rater_id"].dropna().unique())
    irr = float("nan")
    if len(raters) >= 2:
        piv = hl.pivot_table(index=["item_id", "dimension"], columns="rater_id",
                             values="expert01")
        pair_rhos = []
        for r1, r2 in itertools.combinations(raters, 2):
            if r1 in piv.columns and r2 in piv.columns:
                sub = piv[[r1, r2]].dropna()
                pair_rhos.append(rho(sub[r1], sub[r2]))
        pair_rhos = [x for x in pair_rhos if x == x]  # drop NaN
        if pair_rhos:
            irr = sum(pair_rhos) / len(pair_rhos)

    # --- report ---
    print(f"\nHuman-expert validation  "
          f"(raters={len(raters)}, items={m['item_id'].nunique()}, pairs={len(m)})")
    print(f"Inter-rater reliability (mean pairwise Spearman): "
          f"{irr:.3f}" if irr == irr else "Inter-rater reliability: n/a (<2 raters)")
    print()
    show = res.copy()
    for c in ["expert", "llm", "gap", "rho"]:
        show[c] = show[c].map(lambda v: f"{v:.3f}" if v == v else "--")
    print(show.to_string(index=False))

    def fmt(v):
        return f"{v:.3f}" if v == v else "--"

    tex_name = {"IdeationTeam": "Ideation", "LiteratureTeam": "Literature",
                "ModelTeam": "Model", "DataTeam": "Data", "CodeTeam": "Code",
                "EstimationTeam": "Estimation", "ReportingTeam": "Reporting",
                "Pooled": "\\textbf{Pooled}"}
    print("\nLaTeX rows for tab:human-validation:")
    for _, r in res.iterrows():
        name = tex_name.get(r.team, r.team)
        print(f"{name:16s} & {fmt(r.expert)} & {fmt(r.llm)} & "
              f"{fmt(r.gap)} & {fmt(r.rho)} \\\\")


if __name__ == "__main__":
    main()
