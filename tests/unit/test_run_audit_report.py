# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""RUN_AUDIT.md — every published run's final report for human review: deterministic,
verbatim machine verdicts, absence-tolerant, checklist included."""

import json

from pipeline.run_audit_report import build_audit_report, write_audit_report


def _seed_run(tmp_path):
    (tmp_path / "pipeline_manifest.json").write_text(json.dumps({
        "pipeline_run_id": "pipeline-test", "research_topic": "T", "mode": "M",
        "success": True, "total_duration_sec": 7200.0,
        "teams_completed": ["IdeationTeam", "ModelTeam"], "teams_failed": []}))
    (tmp_path / "ModelTeam").mkdir()
    (tmp_path / "ModelTeam" / "calibration_output.json").write_text(json.dumps({
        "data": {"calibrated_models": [
            {"sandbox_validation": {"verdict": "point_calibrated"}},
            {"sandbox_validation": {"verdict": "uncalibratable"}}]}}))
    (tmp_path / "ReportingTeam").mkdir()
    (tmp_path / "ReportingTeam" / "quality_output.json").write_text(json.dumps({
        "data": {"quality": {"consistency": {
            "total_numbers": 100, "verified": 90, "verdict": "consistent"}}}}))
    return tmp_path


def test_report_reproduces_verdicts_verbatim(tmp_path):
    run = _seed_run(tmp_path)
    text = build_audit_report(str(run))
    assert "point_calibrated / uncalibratable" in text
    assert "`consistent` (90/100 numbers)" in text
    assert "success: **True**" in text and "Duration: 2.00 h" in text


def test_absence_is_stated_not_hidden(tmp_path):
    run = _seed_run(tmp_path)
    text = build_audit_report(str(run))
    assert "not run (optional node)" in text            # CodeTeam absent
    assert "artifact missing" in text                   # DataTeam QA / Estimation absent
    assert "not yet run" in text                        # Tier-2 scores absent
    assert "Human Review Checklist" in text


def test_evaluation_table_renders_when_scores_exist(tmp_path):
    run = _seed_run(tmp_path)
    ev = run / "_eval_view"
    ev.mkdir()
    (ev / "tier2_multirun_mistral.json").write_text(json.dumps({
        "judge_model": "vllm/mistral", "configs": {
            "IdeationTeam_ModeX": {"correctness": {"mean": 0.8}, "soundness": {"mean": 0.6}}}}))
    text = build_audit_report(str(run))
    assert "| IdeationTeam | 0.700 |" in text           # mean of dims per judge


def test_write_creates_file(tmp_path):
    run = _seed_run(tmp_path)
    out = write_audit_report(str(run))
    assert out.endswith("RUN_AUDIT.md")
    assert (run / "RUN_AUDIT.md").exists()


def test_improvements_section_renders_from_state(tmp_path, monkeypatch):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({
        "round": 21, "previous_report": "output/prev/RUN_AUDIT.md",
        "improvements": [{"id": "C8", "title": "Compound LHS", "effect": "0 pseudo-vars"}],
        "open": [{"id": "E17", "priority": "P3", "title": "identification caveats"}]}))
    monkeypatch.setenv("AEL_AUDIT_STATE", str(state))
    text = build_audit_report(str(_seed_run(tmp_path)))
    assert "Infrastructure Improved This Cycle" in text
    assert "**C8** — Compound LHS" in text and "*Effect:* 0 pseudo-vars" in text
    assert "**E17** (P3): identification caveats" in text
    assert "`output/prev/RUN_AUDIT.md`" in text          # chain pointer


def test_no_state_variable_omits_the_improvement_sections(tmp_path, monkeypatch):
    monkeypatch.delenv("AEL_AUDIT_STATE", raising=False)
    text = build_audit_report(str(_seed_run(tmp_path)))
    assert "Infrastructure Improved" not in text and "state file missing" not in text
    assert "## 6. Human Review Checklist" in text


def test_missing_state_is_stated(tmp_path, monkeypatch):
    monkeypatch.setenv("AEL_AUDIT_STATE", str(tmp_path / "nope.json"))
    text = build_audit_report(str(_seed_run(tmp_path)))
    assert "state file missing" in text
    heads = [l.split(".")[0] for l in text.splitlines() if l.startswith("## ")]
    assert heads == [f"## {i}" for i in range(1, 9)]


def test_publisher_script_mode_writes_run_audit(tmp_path):
    """In script mode sys.path[0] is pipeline/ itself, so the package
    import failed and RUN_AUDIT.md was silently skipped. Locked via a real subprocess."""
    import subprocess, sys
    from pathlib import Path
    run = tmp_path / "full_pipeline_committee_smoke_99999999"
    run.mkdir()
    (run / "pipeline_manifest.json").write_text(json.dumps({
        "pipeline_run_id": "p-test", "research_topic": "T", "mode": "M", "success": True,
        "total_duration_sec": 60.0, "teams_completed": ["IdeationTeam"], "teams_failed": []}))
    out_root = tmp_path / "out"
    script = Path(__file__).resolve().parent.parent.parent / "pipeline" / "link_outputs.py"
    r = subprocess.run([sys.executable, str(script), str(run), "--output-root", str(out_root)],
                       capture_output=True, text=True, timeout=120)
    views = list(out_root.glob("*_99999999"))
    assert views, r.stdout + r.stderr
    assert (views[0] / "RUN_AUDIT.md").exists(), r.stdout + r.stderr
