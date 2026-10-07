# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Generate RUN_AUDIT.md — the per-run final report for HUMAN review.

Every published pipeline run gets one (link_outputs calls this after publishing; rerun
it once dual-judge evaluation scores exist). It is a
deterministic collection of the run's own machine verdicts — nothing here is authored
by a language model — followed by a review checklist for the human auditor.

Stdlib-only, absence-tolerant: every section renders from whatever artifacts exist and
says plainly when one is missing.

CLI:
    python3 pipeline/run_audit_report.py <run_dir> [--out <path>]
"""

from __future__ import annotations

import glob
import json
import os
import statistics
import sys
from typing import Any, Dict, List, Optional


def _load(path: str) -> Optional[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _data(obj: Optional[Dict]) -> Dict:
    """Unwrap the {data: ...} envelope some stages use."""
    if not isinstance(obj, dict):
        return {}
    inner = obj.get("data")
    return inner if isinstance(inner, dict) else obj


def _fmt(v: Any, nd: int = 3) -> str:
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v) if v is not None else "—"


def _manifest_section(run: str, add) -> None:
    m = _load(os.path.join(run, "pipeline_manifest.json")) or {}
    add("## 1. Run")
    add(f"- Run id: `{m.get('pipeline_run_id', os.path.basename(run))}`")
    add(f"- Topic: {m.get('research_topic', '—')}")
    add(f"- Mode: `{m.get('mode', '—')}` | success: **{m.get('success', '—')}**")
    dur = m.get("total_duration_sec")
    add(f"- Duration: {dur / 3600:.2f} h" if isinstance(dur, (int, float)) else "- Duration: —")
    done, failed = m.get("teams_completed") or [], m.get("teams_failed") or []
    add(f"- Teams completed: {len(done)}/{len(done) + len(failed)} "
        f"({', '.join(done) if done else 'none'})")
    if failed:
        add(f"- **Teams FAILED: {', '.join(failed)}**")


def _verdict_rows(run: str) -> List[str]:
    rows: List[str] = []

    calib = _data(_load(os.path.join(run, "ModelTeam", "calibration_output.json")))
    if calib:
        vs = []
        for mm in calib.get("calibrated_models") or []:
            sv = mm.get("sandbox_validation") or {}
            vs.append(sv.get("verdict") or "?")
        rows.append(f"| ModelTeam calibration | {' / '.join(vs) if vs else '—'} |")
    else:
        rows.append("| ModelTeam calibration | artifact missing |")

    qa = _data(_load(os.path.join(run, "DataTeam", "api_qa_output.json")))
    ds = (qa.get("documented_datasets") or [{}])[0] if qa else {}
    if ds:
        rows.append(f"| DataTeam QA | {_fmt(ds.get('quality_score'), 1)}/100, "
                    f"{ds.get('certification_level', '—')}"
                    f"{' (contains simulated series)' if ds.get('data_simulated') else ''} |")
    else:
        rows.append("| DataTeam QA | artifact missing |")

    gen = _data(_load(os.path.join(run, "CodeTeam", "generation_output.json")))
    val = _data(_load(os.path.join(run, "CodeTeam", "validation_output.json")))
    if gen:
        gv = [g.get("verdict", "?") for g in gen.get("results") or []]
        vv = [v.get("verdict", "?") for v in (val.get("results") or [])] if val else []
        rows.append(f"| CodeTeam | generation: {' / '.join(gv) or '—'}; "
                    f"validation: {' / '.join(vv) or '—'} |")
    else:
        rows.append("| CodeTeam | not run (optional node) |")

    est = _data(_load(os.path.join(run, "EstimationTeam", "estimation_output.json")))
    o = est.get("outcome") or {}
    if o:
        detail = f"`{o.get('verdict', '?')}`"
        if o.get("reason"):
            detail += f" ({o['reason']})"
        if o.get("verdict") in ("estimated", "fragile"):
            detail += (f", n={o.get('n_obs')}, R²={_fmt(o.get('r_squared'))}")
        inf = _data(_load(os.path.join(run, "EstimationTeam", "inference_output.json")))
        inner = inf.get("inference") or {}
        hyps = inner.get("hypotheses") or []
        if hyps:
            supported = sum(1 for h in hyps if h.get("supported"))
            detail += f"; hypotheses {supported}/{len(hyps)} supported"
        if inner.get("stability_score") is not None:
            detail += f"; sign-stability {_fmt(inner['stability_score'], 2)}"
        rows.append(f"| EstimationTeam | {detail} |")
    else:
        rows.append("| EstimationTeam | artifact missing |")

    q = _data(_load(os.path.join(run, "ReportingTeam", "quality_output.json")))
    cons = (q.get("quality") or {}).get("consistency") or q.get("consistency") or {}
    if cons:
        rows.append(f"| Reporting consistency | `{cons.get('verdict', '?')}` "
                    f"({cons.get('verified', '?')}/{cons.get('total_numbers', '?')} numbers) |")
    else:
        rows.append("| Reporting consistency | artifact missing |")
    return rows


def _feasibility_section(run: str, add) -> None:
    fr = _data(_load(os.path.join(run, "feasibility_report.json")))
    add("\n## 3. Model–Data Feasibility")
    if not fr:
        add("- feasibility report missing (loop disabled or linear-only run)")
        return
    rep = fr.get("final_data_availability_report") or {}
    add(f"- Status: `{fr.get('status', '—')}` after {fr.get('cycles', '—')} cycle(s); "
        f"{len(fr.get('model_revision_requests') or [])} model revision request(s)")
    add(f"- Requirements: {rep.get('n_feasible', '—')} feasible / "
        f"{rep.get('n_proxy', '—')} proxy-only / {rep.get('n_unavailable', '—')} unavailable")
    supp = [f for f in rep.get("findings") or [] if f.get("fetchable")]
    if supp:
        add(f"- Scout-resolved (fetchable) requirements: {len(supp)} — " +
            ", ".join(f"{f['variable_name']}→{f['matched_series']}" for f in supp[:6]) +
            (" …" if len(supp) > 6 else ""))


def _ballots_section(run: str, add) -> None:
    path = os.path.join(run, "hitl_committee_ballots.jsonl")
    add("\n## 4. Committee Checkpoints")
    if not os.path.exists(path):
        add("- no committee ballots recorded (auto/interactive mode)")
        return
    kinds: Dict[str, int] = {}
    degenerate_discarded = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                b = json.loads(line)
                kinds[b.get("kind", "?")] = kinds.get(b.get("kind", "?"), 0) + 1
                if "degenerate" in (b.get("note") or ""):
                    degenerate_discarded += 1
    except Exception:
        add("- ballot log unreadable")
        return
    total = sum(kinds.values())
    add(f"- {total} ballots: " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    if degenerate_discarded:
        add(f"- degenerate feedback drafts discarded (disclosed): {degenerate_discarded}")


def _evaluation_section(run: str, add) -> None:
    add("\n## 5. Evaluation")
    t1 = _load(os.path.join(run, "tier1_pipeline.json"))
    if t1:
        m = t1.get("metrics") or {}
        add(f"- Tier-1 (pipeline-level): stage completion {_fmt(m.get('stage_completion_rate'), 2)}, "
            f"ordering {_fmt(m.get('stage_ordering'), 2)}, error rate {_fmt(m.get('error_rate'), 2)}")
    else:
        add("- Tier-1 pipeline metrics: not yet computed")
    score_files = sorted(glob.glob(os.path.join(run, "_eval_view", "tier2_multirun_*.json")))
    if not score_files:
        add("- Tier-2 dual-judge scores: **not yet run** — regenerate this report after "
            "the Tier-2 evaluation step")
        return
    per_team: Dict[str, Dict[str, float]] = {}
    judges: List[str] = []
    for f in score_files:
        d = _load(f) or {}
        judge = str(d.get("judge_model", "?")).split("/")[-1]
        judges.append(judge)
        for cfg, dims in (d.get("configs") or {}).items():
            team = cfg.split("_")[0]
            vals = [v.get("mean") for v in dims.values() if isinstance(v, dict)]
            if vals:
                per_team.setdefault(team, {})[judge] = statistics.mean(vals)
    add(f"- Tier-2 judges: {', '.join(judges)} (team-specific rubrics; mean over six "
        "content dimensions)")
    add("\n| Team | " + " | ".join(judges) + " |")
    add("|---" * (len(judges) + 1) + "|")
    order = ["IdeationTeam", "LiteratureTeam", "ModelTeam", "DataTeam",
             "CodeTeam", "EstimationTeam", "ReportingTeam"]
    for team in order:
        if team in per_team:
            add(f"| {team} | " +
                " | ".join(_fmt(per_team[team].get(j)) for j in judges) + " |")


def _cycle_state() -> Optional[Dict]:
    """Optional state file of a maintainer's audit cycle (a JSON of improvements and open
    items), read from $AEL_AUDIT_STATE. Absent unless that variable is set."""
    path = os.environ.get("AEL_AUDIT_STATE")
    return _load(path) if path else None


def _improvements_section(add) -> bool:
    """Sections 6-7, written only when $AEL_AUDIT_STATE is set; returns whether they were."""
    if not os.environ.get("AEL_AUDIT_STATE"):
        return False
    state = _cycle_state()
    add("\n## 6. Infrastructure Improved This Cycle — and Its Measured Effect")
    if not state:
        add("- audit-cycle state file missing")
        add("\n## 7. Potential Future Improvements (open backlog)")
        add("- audit-cycle state file missing")
        return True
    add(f"*(audit-cycle round {state.get('round', '?')}; the diff history of the "
        "state file is the cycle-over-cycle changelog)*\n")
    for it in state.get("improvements") or []:
        add(f"- **{it.get('id')}** — {it.get('title')}")
        if it.get("effect"):
            add(f"  - *Effect:* {it['effect']}")
    add("\n## 7. Potential Future Improvements (open backlog)")
    for it in state.get("open") or []:
        add(f"- **{it.get('id')}** ({it.get('priority')}): {it.get('title')}")
    prev = state.get("previous_report")
    if prev:
        add(f"\n*Previous cycle report (the next audit cycle reads it first): `{prev}`*")
    return True


_CHECKLIST = """
## {n}. Human Review Checklist

- [ ] Research questions: coherent, on-topic, genuinely distinct?
- [ ] Model verdicts: does each calibration verdict match the artifact it describes?
- [ ] Data: any silently drifted series (wrong geography / wrong component / wrong concept)?
- [ ] Estimation: is the specification economically sensible; are refusals/negatives justified?
- [ ] Code: do the modules read as this model's economics; are refusals correctly scoped?
- [ ] Report: spot-check three numbers against their artifacts; read the limitations section.
- [ ] Committee: sample three ballots — is the dissent substantive?
- [ ] Anything the machine verdicts MISSED (this checklist exists because they can):

*Reviewer notes:*

"""


def build_audit_report(run_dir: str) -> str:
    run = os.path.abspath(run_dir)
    lines: List[str] = []
    add = lines.append
    add(f"# Run Audit Report — {os.path.basename(run)}")
    add("")
    add("*Deterministically generated from the run's own artifacts for human review; "
        "machine verdicts are reproduced verbatim, never re-graded. "
        "Regenerate with `python3 pipeline/run_audit_report.py <run_dir>` "
        "(evaluation scores appear once the dual-judge step has run).*")
    add("")
    _manifest_section(run, add)
    add("\n## 2. Stage Verdicts (machine-generated, verbatim)")
    add("\n| Stage | Verdict |")
    add("|---|---|")
    for row in _verdict_rows(run):
        add(row)
    _feasibility_section(run, add)
    _ballots_section(run, add)
    _evaluation_section(run, add)
    with_log = _improvements_section(add)
    add(_CHECKLIST.replace("{n}", "8" if with_log else "6"))
    return "\n".join(lines)


def write_audit_report(run_dir: str, out_path: Optional[str] = None) -> str:
    text = build_audit_report(run_dir)
    out = out_path or os.path.join(run_dir, "RUN_AUDIT.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    _out = None
    if "--out" in sys.argv:
        _out = sys.argv[sys.argv.index("--out") + 1]
    print("wrote", write_audit_report(sys.argv[1], _out))
