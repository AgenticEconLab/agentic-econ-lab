# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Curated output view, copy variant (user-approved): REPORT/SUMMARY at top, one folder
per stage with COPIED deliverables (survive scratch purges), internals hidden."""

import json

from pipeline.link_outputs import build_summary, publish_run


def _fake_run(tmp_path, jobid="1234567"):
    run = tmp_path / f"full_pipeline_committee_smoke_{jobid}"
    (run / "ReportingTeam" / "figures").mkdir(parents=True)
    (run / "ReportingTeam" / "research_report.md").write_text(
        "# Report\n## 7. Honest Limitations\n- Series `SIM` is DISCLOSED SIMULATION.\n")
    (run / "ReportingTeam" / "figures" / "a.png").write_bytes(b"png")
    (run / "ReportingTeam" / "quality_output.json").write_text(json.dumps(
        {"quality": {"consistency": {"verdict": "consistent", "verified": 10,
                                     "total_numbers": 12}}}))
    (run / "EstimationTeam").mkdir()
    (run / "EstimationTeam" / "inference_output.json").write_text(json.dumps(
        {"outcome": {"verdict": "estimated", "dependent_name": "y", "n_obs": 61,
                     "r_squared": 0.319},
         "inference": {"hypotheses": [{"supported": False}, {"supported": True}]}}))
    (run / "IdeationTeam").mkdir()
    (run / "IdeationTeam" / "finalized_research_questions.json").write_text(json.dumps(
        {"final_questions": [{"question": "Q1?"}, {"question": "Q2?"}]}))
    (run / "checkpoints").mkdir()
    (run / "pipeline_manifest.json").write_text(json.dumps(
        {"timestamp": "2026-07-11T01:34:00+00:00", "research_topic": "HANK things",
         "pipeline_run_id": "pipeline-x", "mode": "M", "success": True,
         "total_duration_sec": 900.0,
         "teams_completed": ["IdeationTeam", "EstimationTeam", "ReportingTeam"],
         "teams_failed": []}))
    return run


def test_publish_copies_deliverables_and_builds_summary(tmp_path):
    run = _fake_run(tmp_path)
    root = tmp_path / "output"
    view_path = publish_run(str(run), output_root=str(root))
    v = root / "20260711_fullpipe_3team_1234567"
    assert view_path and v.is_dir()

    # deliverables are REAL COPIES (survive scratch purge), not symlinks
    assert (v / "REPORT.md").is_file() and not (v / "REPORT.md").is_symlink()
    assert (v / "figures" / "a.png").is_file() and not (v / "figures").is_symlink()
    assert (v / "1_question" / "final_questions.json").is_file()
    assert (v / "6_estimation" / "inference.json").is_file()
    # internals hidden; bulky things stay symlinks
    assert (v / "_internals" / "manifest.json").is_file()
    assert (v / "_internals" / "_run").is_symlink()
    assert (v / "_internals" / "checkpoints").is_symlink()
    assert (root / "latest").resolve() == v.resolve()
    # absent stages simply don't appear
    assert not (v / "5_code").exists()

    summary = (v / "SUMMARY.md").read_text()
    assert "HANK things" in summary
    assert "| 6 | EstimationTeam | ✅ |" in summary
    assert "hypotheses 1/2 supported" in summary
    assert "consistency `consistent` (10/12 verified)" in summary
    assert "DISCLOSED SIMULATION" in summary          # limitations lifted from the report


def test_republish_is_idempotent_and_survives_source_deletion(tmp_path):
    run = _fake_run(tmp_path)
    root = tmp_path / "output"
    publish_run(str(run), output_root=str(root))
    publish_run(str(run), output_root=str(root))      # rebuild in place, no error
    v = root / "20260711_fullpipe_3team_1234567"
    (run / "ReportingTeam" / "research_report.md").unlink()   # simulate scratch purge
    assert "# Report" in (v / "REPORT.md").read_text()        # the COPY survives


def test_summary_alone_never_crashes_on_sparse_run(tmp_path):
    run = tmp_path / "run_99"
    run.mkdir()
    text = build_summary(run, "label_99")
    assert "label_99" in text
