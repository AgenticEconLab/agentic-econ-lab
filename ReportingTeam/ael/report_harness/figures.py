# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Publication figures from the estimation artifact (§4.7 VisualDesigner's numeric core).

Deterministic matplotlib (Agg backend — headless HPC safe); every plotted point comes from the
stored analysis panel or the coefficient table. Returns the list of files written; failures
degrade to an empty list with a note, never crash the report."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def make_figures(estimation_results: Dict, out_dir: str) -> Tuple[List[str], List[str]]:
    """Write (a) normalized time-series of all analysis variables and (b) a coefficient/CI
    plot. Returns (files, notes)."""
    files: List[str] = []
    notes: List[str] = []
    outcome = (estimation_results or {}).get("outcome") or {}
    if outcome.get("verdict") == "inestimable" or not outcome.get("analysis_data"):
        return files, ["no figures: upstream estimation not estimable"]
    os.makedirs(out_dir, exist_ok=True)

    try:
        frame = pd.DataFrame(outcome["analysis_data"])
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.set_index("date").astype(float)
        norm = (frame - frame.mean()) / frame.std()
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for col in norm.columns:
            ax.plot(norm.index, norm[col], label=col, linewidth=1.0)
        ax.set_title("Analysis variables (standardized)")
        ax.legend(loc="best", fontsize=8)
        ax.set_ylabel("standard deviations from mean")
        path = os.path.join(out_dir, "analysis_series.png")
        fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
        files.append(path)
    except Exception as e:                                    # pragma: no cover - defensive
        notes.append(f"series figure skipped: {e}")

    try:
        coefs = [c for c in outcome.get("coefficients") or [] if c.get("name") != "const"]
        if coefs:
            fig, ax = plt.subplots(figsize=(6, 0.8 + 0.6 * len(coefs)))
            ys = range(len(coefs))
            for y, c in zip(ys, coefs):
                ax.plot([c["ci_low"], c["ci_high"]], [y, y], color="tab:blue", linewidth=2)
                ax.plot(c["estimate"], y, "o", color="tab:blue")
            ax.axvline(0.0, color="gray", linestyle="--", linewidth=1)
            ax.set_yticks(list(ys))
            ax.set_yticklabels([c["name"] for c in coefs])
            ax.set_title(f"Coefficient estimates with 95% CI ({outcome.get('cov_type', '')})")
            path = os.path.join(out_dir, "coefficients_ci.png")
            fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
            files.append(path)
    except Exception as e:                                    # pragma: no cover - defensive
        notes.append(f"coefficient figure skipped: {e}")

    return files, notes
