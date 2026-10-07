# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Publish a pipeline run as a curated, browsable folder under output/ in the repository root.

Copy variant: small text deliverables are COPIED so review
artifacts survive scratch purges; only bulky internals stay symlinks (_run, checkpoints).

    output/<YYYYMMDD>_fullpipe_7team_<job>/
        REPORT.md                <- the deliverable, at top level
        SUMMARY.md               <- deterministic one-pager (generated here, no LLM)
        figures/
        1_question/   final_questions.{json,txt}
        2_literature/ review.txt  gaps.json  knowledge_graph.json  corpus.json
        3_model/      theory.json  design.json  calibration.json
        4_data/       retrieved.json  cleaning.json  quality.json
        5_code/       generation.json  validation.json  experiments.json  generated_models/
        6_estimation/ results.json  diagnostics.json  inference.json
        _internals/   manifest.json  committee_ballots.jsonl  feasibility_report.json
                      execution_log.json  checkpoints -> ...  _run -> raw scratch dir

``output/latest`` points at the newest published run. stdlib-only on purpose — must run
under system python even when the venv is unreachable.

Usage:
    python3 pipeline/link_outputs.py <run_dir> [--output-root output]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# (destination inside the view, run-relative source, "copy"|"copytree"|"symlink")
_PUBLISH_MAP: List[Tuple[str, str, str]] = [
    ("REPORT.md", "ReportingTeam/research_report.md", "copy"),
    ("figures", "ReportingTeam/figures", "copytree"),
    ("1_question/final_questions.json", "IdeationTeam/finalized_research_questions.json", "copy"),
    ("1_question/final_questions.txt", "IdeationTeam/finalized_research_questions.txt", "copy"),
    ("2_literature/review.txt", "LiteratureTeam/literature_review.txt", "copy"),
    ("2_literature/gaps.json", "LiteratureTeam/gap_analysis_results.json", "copy"),
    ("2_literature/knowledge_graph.json", "LiteratureTeam/knowledge_graph.json", "copy"),
    ("2_literature/corpus.json", "LiteratureTeam/literature_batch.json", "copy"),
    ("3_model/theory.json", "ModelTeam/theory_output.json", "copy"),
    ("3_model/design.json", "ModelTeam/model_design_output.json", "copy"),
    ("3_model/calibration.json", "ModelTeam/calibration_output.json", "copy"),
    ("4_data/retrieved.json", "DataTeam/api_source_output.json", "copy"),
    ("4_data/cleaning.json", "DataTeam/api_cleaning_output.json", "copy"),
    ("4_data/quality.json", "DataTeam/api_qa_output.json", "copy"),
    ("5_code/generation.json", "CodeTeam/generation_output.json", "copy"),
    ("5_code/validation.json", "CodeTeam/validation_output.json", "copy"),
    ("5_code/experiments.json", "CodeTeam/experimentation_output.json", "copy"),
    ("5_code/generated_models", "CodeTeam/generated_models", "copytree"),
    ("6_estimation/results.json", "EstimationTeam/estimation_output.json", "copy"),
    ("6_estimation/diagnostics.json", "EstimationTeam/validation_output.json", "copy"),
    ("6_estimation/inference.json", "EstimationTeam/inference_output.json", "copy"),
    ("_internals/manifest.json", "pipeline_manifest.json", "copy"),
    ("_internals/committee_ballots.jsonl", "hitl_committee_ballots.jsonl", "copy"),
    ("_internals/feasibility_report.json", "feasibility_report.json", "copy"),
    ("_internals/execution_log.json", "execution_log.json", "copy"),
    ("_internals/checkpoints", "checkpoints", "symlink"),
    ("_internals/_run", ".", "symlink"),
]

_TEAM_DIRS = ("IdeationTeam", "LiteratureTeam", "ModelTeam", "DataTeam",
              "CodeTeam", "EstimationTeam", "ReportingTeam")

_LABEL_RE = re.compile(r"^\d{8}_fullpipe_\d+team_")


def _load(path: Path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def _run_label(run_dir: Path) -> str:
    """``YYYYMMDD_fullpipe_{N}team_{jobid}`` — the user's viewing convention (local date)."""
    m = re.search(r"(\d{5,})$", run_dir.name)
    job = m.group(1) if m else ""
    date = None
    manifest = _load(run_dir / "pipeline_manifest.json")
    ts = manifest.get("timestamp", "")
    if ts:
        try:
            date = datetime.fromisoformat(ts).astimezone().strftime("%Y%m%d")
        except ValueError:
            pass
    if date is None:
        date = datetime.fromtimestamp(run_dir.stat().st_mtime).strftime("%Y%m%d")
    # single readdir, not per-entry stat — per-entry is_dir() flakes on degraded Lustre
    try:
        entries = set(os.listdir(run_dir))
    except OSError:
        entries = set()
    n_teams = len(entries & set(_TEAM_DIRS))
    return f"{date}_fullpipe_{n_teams}team_{job or run_dir.name}"


# ---------------------------------------------------------------------------------------
# SUMMARY.md — deterministic one-pager (no LLM): status table + headlines + limitations.
# ---------------------------------------------------------------------------------------

def _fmt(x, nd: int = 3) -> str:
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    return str(x) if x is not None else "—"


def build_summary(run: Path, label: str) -> str:
    manifest = _load(run / "pipeline_manifest.json")
    lines: List[str] = []
    add = lines.append
    add(f"# Run Summary — {label}")
    add(f"\n**Topic:** {manifest.get('research_topic', '?')}")
    add(f"**Pipeline:** `{manifest.get('pipeline_run_id', '?')}` | mode "
        f"{manifest.get('mode', '?')} | success: **{manifest.get('success', '?')}** | "
        f"last session {round(float(manifest.get('total_duration_sec', 0)) / 60)} min")

    completed = list(manifest.get("teams_completed") or [])
    failed = list(manifest.get("teams_failed") or [])

    # per-team headline facts
    facts = {}
    fq = _load(run / "IdeationTeam" / "finalized_research_questions.json")
    qs = fq.get("final_questions", []) if isinstance(fq, dict) else []
    if qs:
        facts["IdeationTeam"] = f"{len(qs)} final questions"
    batch = _load(run / "LiteratureTeam" / "literature_batch.json")
    items = batch.get("literature_items", []) if isinstance(batch, dict) else []
    if items:
        cov = ((batch.get("metadata") or {}).get("abstract_coverage") or {}).get("summary", "")
        facts["LiteratureTeam"] = f"{len(items)} sources" + (f"; {cov}" if cov else "")
    calib = _load(run / "ModelTeam" / "calibration_output.json")

    def _calib_status(m):
        status = (m.get("metadata") or {}).get("calibration_status")
        if not status:  # metadata not uniformly populated — parse the harness assessment
            fa = str((m.get("fit_metrics") or {}).get("fit_assessment", ""))
            hit = re.search(r"status=([a-z_]+)", fa)
            status = hit.group(1) if hit else "?"
        return str(status)

    verdicts = [_calib_status(m)
                for m in calib.get("calibrated_models", []) if isinstance(m, dict)]
    if verdicts:
        facts["ModelTeam"] = f"{len(verdicts)} models: " + ", ".join(verdicts)
    src = _load(run / "DataTeam" / "api_source_output.json")
    rd = src.get("retrieved_data", []) if isinstance(src, dict) else []
    if rd:
        real = sum(1 for x in rd if not x.get("data_simulated"))
        repairs = sum(1 for x in rd if "repaired" in str(x.get("quality_notes", "")))
        facts["DataTeam"] = (f"{real}/{len(rd)} real series"
                             + (f", {repairs} disclosed repair(s)" if repairs else ""))
    gen = _load(run / "CodeTeam" / "generation_output.json")
    gv = [r.get("verdict", "?") for r in gen.get("results", []) if isinstance(r, dict)]
    if gv:
        facts["CodeTeam"] = "generation: " + ", ".join(gv)
    inf = _load(run / "EstimationTeam" / "inference_output.json")
    outcome = inf.get("outcome") or {}
    if outcome:
        hyps = (inf.get("inference") or {}).get("hypotheses") or []
        supported = sum(1 for h in hyps if h.get("supported"))
        facts["EstimationTeam"] = (f"`{outcome.get('verdict', '?')}` "
                                   f"({outcome.get('dependent_name', '?')}, "
                                   f"n={outcome.get('n_obs', '?')}, "
                                   f"R²={_fmt(outcome.get('r_squared'))}); "
                                   f"hypotheses {supported}/{len(hyps)} supported")
    qual = _load(run / "ReportingTeam" / "quality_output.json")
    cons = ((qual.get("quality") or {}).get("consistency")) or {}
    if cons:
        facts["ReportingTeam"] = (f"consistency `{cons.get('verdict', '?')}` "
                                  f"({cons.get('verified', '?')}/"
                                  f"{cons.get('total_numbers', '?')} verified)")

    add("\n## Teams")
    add("| # | Team | Status | Headline |")
    add("|---|------|--------|----------|")
    for i, team in enumerate(_TEAM_DIRS, 1):
        if team in completed:
            status = "✅"
        elif team in failed:
            status = "❌ failed"
        elif (run / team).name in set(os.listdir(run) if run.exists() else []):
            status = "◻︎ ran (not in manifest)"
        else:
            continue
        add(f"| {i} | {team} | {status} | {facts.get(team, '—')} |")

    feas = _load(run / "feasibility_report.json")
    if feas:
        add(f"\n**Model–data feasibility:** `{feas.get('status', '?')}` after "
            f"{feas.get('cycles', '?')} cycle(s).")

    # limitations — lifted verbatim from the report (heading "Limitations"; older reports used "Honest Limitations")
    try:
        report = (run / "ReportingTeam" / "research_report.md").read_text(encoding="utf-8")
        _m = re.search(r"## 7\. (?:Honest )?Limitations", report)
        section = report[_m.end():] if _m else report.split("Limitations", 1)[1]
        section = section.split("\n## ", 1)[0]        # stop at the next header (References!)
        bullets = [ln for ln in section.splitlines() if ln.strip().startswith("- ")]
        if bullets:
            add("\n## Limitations (from the report)")
            lines.extend(bullets[:8])
    except (OSError, IndexError):
        pass

    add("\n---")
    add(f"*Generated deterministically by `pipeline/link_outputs.py` from the run artifacts; "
        f"raw run: `{run}`*")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------
# publisher
# ---------------------------------------------------------------------------------------

def _force_symlink(target: Path, link: Path) -> None:
    if link.is_symlink() or link.exists():
        if link.is_dir() and not link.is_symlink():
            shutil.rmtree(link)
        else:
            link.unlink()
    link.symlink_to(target)


def publish_run(run_dir: str, output_root: Optional[str] = None) -> Optional[str]:
    """Create/refresh the curated copy-view for one run. Returns the view path or None."""
    run = Path(run_dir).resolve()
    if not run.is_dir():
        print(f"[link_outputs] run dir not found: {run}")
        return None
    root = Path(output_root) if output_root else Path(__file__).resolve().parent.parent / "output"
    root.mkdir(parents=True, exist_ok=True)

    label = _run_label(run)
    view = root / label
    # rebuild from scratch — the view is wholly publisher-owned (guard: only our labels)
    if (view.is_symlink() or view.exists()) and _LABEL_RE.match(view.name):
        if view.is_symlink():
            view.unlink()
        else:
            shutil.rmtree(view)
    view.mkdir(parents=True, exist_ok=True)

    published = 0
    for dest_rel, src_rel, kind in _PUBLISH_MAP:
        src = run if src_rel == "." else run / src_rel
        if not src.exists():
            continue
        dest = view / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            if kind == "copy":
                shutil.copy2(src, dest)
            elif kind == "copytree":
                # Never publish bytecode caches or execution scratch — the code
                # folder is a deliverable, not a working directory.
                shutil.copytree(src, dest, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("__pycache__", "_exec_scratch",
                                                              "*.pyc"))
            else:
                _force_symlink(src.resolve() if src_rel != "." else run, dest)
            published += 1
        except OSError as e:
            print(f"[link_outputs] skip {dest_rel}: {e}")

    try:
        (view / "SUMMARY.md").write_text(build_summary(run, label), encoding="utf-8")
        published += 1
    except Exception as e:                                    # pragma: no cover - defensive
        print(f"[link_outputs] SUMMARY.md failed: {e}")

    # Every published run carries its final human-review report (regenerate it once
    # dual-judge evaluation scores exist).
    try:
        # In script mode (python3 pipeline/link_outputs.py) sys.path[0]
        # is pipeline/ itself, so the package import fails — fall back to the sibling module.
        try:
            from pipeline.run_audit_report import build_audit_report
        except ImportError:
            from run_audit_report import build_audit_report
        audit_text = build_audit_report(str(run))
        (view / "RUN_AUDIT.md").write_text(audit_text, encoding="utf-8")
        (run / "RUN_AUDIT.md").write_text(audit_text, encoding="utf-8")
        published += 1
    except Exception as e:                                    # pragma: no cover - defensive
        print(f"[link_outputs] RUN_AUDIT.md failed: {e}")

    _force_symlink(view, root / "latest")
    print(f"[link_outputs] {label}: {published} items -> {view}")
    return str(view)


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish a run's outputs as a curated view.")
    parser.add_argument("run_dir", help="Pipeline run directory (on scratch)")
    parser.add_argument("--output-root", default=None,
                        help="View root (default: output/ in the repository root)")
    args = parser.parse_args()
    return 0 if publish_run(args.run_dir, args.output_root) else 1


if __name__ == "__main__":
    sys.exit(main())
