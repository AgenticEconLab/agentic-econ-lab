#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Build the ``_eval_view`` adapter tree for a chained pipeline run, so
``evaluation/run_ael_evaluation.py``'s ``--multirun-dir`` machinery
(``Team/Mode/run_NNN/``) can Tier-2-score a chained run the same way it
scores a standalone multirun.

Validated by regenerating the ``_eval_view/`` of a reference chained run into a
scratch copy and diffing every file against the original.

Per team: symlinks (absolute target, so run_001/ still resolves if the view
is moved) every file in that team's real scratch output directory, then
synthesizes execution_log.json from that team's own entry in the
pipeline-level execution_log.json's ``stages`` list -- provenance/freshness
only; run_ael_evaluation.py's staleness guard reads ``started_at`` from it.

Usage:
    python3 pipeline/build_eval_view.py <scratch_run_dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# team -> its real per-team execution mode, matching run_ael_pipeline.py's
# _DEFAULT_PIPELINE_MODES (the mode name becomes part of the adapter path).
TEAM_MODES = {
    "IdeationTeam": "ModeWithWcWithHITL",
    "LiteratureTeam": "ModeWithWcWithHITL",
    "ModelTeam": "ModeWithWcWithHITL",
    "DataTeam": "ModeOpenSourceAPI",
    "CodeTeam": "ModeNoWcWithHITL",
    "EstimationTeam": "ModeNoWcWithHITL",
    "ReportingTeam": "ModeNoWcWithHITL",
}


def build(run_dir: Path) -> None:
    pipeline_log = json.loads((run_dir / "execution_log.json").read_text())
    stages_by_team = {s["name"]: s for s in pipeline_log.get("stages", [])}

    eval_view = run_dir / "_eval_view"
    for team, mode in TEAM_MODES.items():
        team_dir = run_dir / team
        if not team_dir.is_dir():
            print(f"  {team}: no output directory -- skipped (team not run this run)")
            continue
        stage = stages_by_team.get(team)
        if stage is None:
            print(f"  {team}: no matching stage in execution_log.json -- skipped")
            continue

        run_001 = eval_view / team / mode / "run_001"
        run_001.mkdir(parents=True, exist_ok=True)

        n = 0
        for f in sorted(team_dir.iterdir()):
            if f.is_file():
                link = run_001 / f.name
                if link.exists() or link.is_symlink():
                    link.unlink()
                link.symlink_to(f.resolve())
                n += 1

        synth = {
            "execution_id": pipeline_log.get("execution_id", ""),
            "team": team,
            "mode": mode,
            "framework": "ael",
            "started_at": stage["started_at"],
            "completed_at": stage["completed_at"],
            "total_duration_seconds": stage["duration_seconds"],
            "stages": [stage],
            "errors": [],
            "success": stage.get("status") == "success",
            "metadata": {"source": f"chained pipeline run {run_dir.name} (adapter view)"},
        }
        (run_001 / "execution_log.json").write_text(json.dumps(synth, indent=1))
        print(f"  {team}: {n} file(s) linked -> {run_001}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="scratch chained-run directory (not the published output/ view)")
    a = ap.parse_args()
    build(Path(a.run_dir))


if __name__ == "__main__":
    main()
