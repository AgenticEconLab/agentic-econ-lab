#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Re-run the report's number-consistency check on a finished pipeline run, offline.

The check is deterministic (no LLM call): it rebuilds the source pool exactly as the
reporting runner does (team_runners.run_reporting_team -> QualityStage) from the run's own
artifacts and the stored draft, and writes the result next to the original
(``ReportingTeam/consistency_recheck.json``) without touching ``quality_output.json``.

Usage:
    python3 pipeline/recheck_consistency.py <run_dir> [<run_dir> ...]
"""

import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from ReportingTeam.ael.report_harness.consistency import check_consistency  # noqa: E402


def _load(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _artifact(run_dir, name):
    d = _load(os.path.join(run_dir, "artifacts", f"{name}.json"), {}) or {}
    return d.get("data", d) if isinstance(d, dict) else d


def source_pool(run_dir):
    """Same keys and objects as run_reporting_team passes to the Quality stage."""
    pool = {
        "research_questions": _artifact(run_dir, "research_questions"),
        "literature_review": _artifact(run_dir, "literature_review"),
        "model_specification": _artifact(run_dir, "model_specification"),
        "data_source": _artifact(run_dir, "data_source"),
        "estimation_results": _artifact(run_dir, "estimation_results"),
        "interpretation": _load(os.path.join(run_dir, "ReportingTeam",
                                             "interpretation_output.json"), {}) or {},
        "literature_batch": _load(os.path.join(run_dir, "LiteratureTeam",
                                               "literature_batch.json"), {}) or {},
        "code_generation": _load(os.path.join(run_dir, "CodeTeam",
                                              "generation_output.json"), {}) or {},
        "code_validation": _load(os.path.join(run_dir, "CodeTeam",
                                              "validation_output.json"), {}) or {},
    }
    return pool


def recheck(run_dir):
    drafting = _load(os.path.join(run_dir, "ReportingTeam", "drafting_output.json"), {}) or {}
    markdown = drafting.get("report_markdown", "") or ""
    pool = source_pool(run_dir)
    derived = ((drafting.get("drafting") or {}).get("metadata") or {}).get("derived_numbers") or {}
    if derived:
        pool["derived"] = derived
    rep = check_consistency(markdown, pool)
    out = rep.model_dump() if hasattr(rep, "model_dump") else dict(rep.__dict__)
    try:
        commit = subprocess.run(["git", "-C", _HERE, "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        commit = ""
    out["recheck"] = {"checker_commit": commit, "draft": "ReportingTeam/drafting_output.json"}
    return out


def main(argv):
    for run_dir in argv:
        out = recheck(run_dir)
        path = os.path.join(run_dir, "ReportingTeam", "consistency_recheck.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, default=str)
        print(f"{os.path.basename(run_dir.rstrip('/'))}: {out.get('verdict')} "
              f"{out.get('verified')}/{out.get('total_numbers')} "
              f"unverified={len(out.get('unverified') or [])}")


if __name__ == "__main__":
    main(sys.argv[1:])
