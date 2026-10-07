# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Unit tests for the Provenance & serving-integrity auditor (deterministic layer)
in evaluation/audits/hallucination_audit.py — checks A1/A3/A4/A5/B1/B2/B3/B4/J3/G3.

The module is loaded directly by file path so these tests do NOT import
evaluation/__init__.py (which pulls in pydantic): the provenance layer is
stdlib-only by design, and so is this test.
"""
import importlib.util
import json
import sys
from pathlib import Path

_MODPATH = Path(__file__).resolve().parents[2] / "evaluation" / "audits" / "hallucination_audit.py"
_spec = importlib.util.spec_from_file_location("hallucination_audit_under_test", _MODPATH)
ha = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ha          # let @dataclass resolve cls.__module__
_spec.loader.exec_module(ha)

REPO = "Qwen/Qwen3.6-27B-FP8"
REV = "e89b16ebf1988b3d6befa7de50abc2d76f26eb09"
SERVED = "qwen3.6-27b-fp8"
COMMIT = "abc123def456"
VLLM_VER = "0.22.1"

CONFIG_YAML = f"""\
models:
  primary:
    served_name: "{SERVED}"          # AEL_MODEL = ollama/{SERVED}
    hf_repo: "{REPO}"
    hf_revision: "{REV}"
run:
  repo_commit_sha: "{COMMIT}"
  temperature_distribution:   # counts across tracked AEL stages
    "0.1": 8
    "0.3": 48
    "0.7": 4
  notes: >
    anything after the block should be ignored by the parser.
serving:
  vllm_version: "{VLLM_VER}"
"""

PROVENANCE = f"""\
model_repo={REPO}
served_name={SERVED}
commit_sha={COMMIT}
vllm_version={VLLM_VER}
k_runs=3
"""


def _vllm_log(repo_dashed=REPO.replace("/", "--"), rev=REV, served=SERVED,
              ver=VLLM_VER, inits=1, oom=False) -> str:
    snap = f"/scratch/hf-cache/hub/models--{repo_dashed}/snapshots/{rev}"
    lines = [
        f"INFO non-default args: {{'model': '{REPO}', 'served_model_name': ['{served}']}}",
        f"INFO HF_HUB_OFFLINE is True, replace model_id to model_path [{snap}]",
    ]
    for _ in range(inits):
        lines.append(f"INFO Initializing a V1 LLM engine (v{ver}) with config: "
                     f"model='{snap}', served_model_name={served}, seed=0,")
    lines.append(f"INFO Starting to load model {snap}...")
    if oom:
        lines.append("ERROR torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate ...")
    return "\n".join(lines) + "\n"


def _exec_log(cost=0.0, tokens=100, temps=None) -> str:
    llm = {"total_cost_usd": cost, "total_tokens": tokens, "total_calls": 5}
    if temps is not None:
        llm["temperatures"] = temps
    return json.dumps({"team": "ModelTeam", "success": True, "observability": {"llm": llm}})


def _manifest(base: str, n_runs=3) -> str:
    return json.dumps({
        "team": "ModelTeam", "mode": "ModeNoWcNoHITL", "n_runs": n_runs,
        "runs": [{"run_number": i, "output_dir": f"{base}/run_{i:03d}", "success": True}
                 for i in range(1, n_runs + 1)],
    })


def _build_tree(tmp_path: Path, *, cost=0.0, tokens=100, stage_model="ollama/" + SERVED,
                manifest_n=3, think_leak=False, extra_provenance_commit=None, temps=None) -> Path:
    """Construct a minimal multirun tree under tmp_path/runs and return its root."""
    root = tmp_path / "runs"
    mode_dir = root / "ModelTeam" / "ModeNoWcNoHITL"
    run_dir = mode_dir / "run_001"
    run_dir.mkdir(parents=True)
    (run_dir / "execution_log.json").write_text(_exec_log(cost, tokens, temps), encoding="utf-8")
    (run_dir / "run_001.log").write_text(f"  [i] Stage model: {stage_model}\n", encoding="utf-8")
    if think_leak:
        (run_dir / "theory_output.json").write_text('{"x": "<think>secret</think> done"}', encoding="utf-8")
    (mode_dir / "multirun_manifest.json").write_text(_manifest(str(mode_dir), manifest_n), encoding="utf-8")
    prov_dir = root / "_provenance"; prov_dir.mkdir(exist_ok=True)
    (prov_dir / "provenance-1.txt").write_text(PROVENANCE, encoding="utf-8")  # current subfolder layout
    if extra_provenance_commit:
        (prov_dir / "provenance-2.txt").write_text(
            PROVENANCE.replace(COMMIT, extra_provenance_commit), encoding="utf-8")
    return root


def _statuses(prov):
    return {c.id: c.status for c in prov}


# --------------------------------------------------------------------------- #
# parsers
# --------------------------------------------------------------------------- #

def test_read_config_fields(tmp_path):
    cfg = tmp_path / "v.yaml"; cfg.write_text(CONFIG_YAML, encoding="utf-8")
    f = ha._read_config_fields(cfg)
    assert f["hf_repo"] == REPO
    assert f["hf_revision"] == REV
    assert f["served_name"] == SERVED          # inline comment stripped
    assert f["repo_commit_sha"] == COMMIT
    assert f["vllm_version"] == VLLM_VER


def test_read_config_fields_missing_file_returns_none(tmp_path):
    f = ha._read_config_fields(tmp_path / "nope.yaml")
    assert all(v is None for v in f.values())


def test_read_provenance(tmp_path):
    (tmp_path / "_provenance-1.txt").write_text(PROVENANCE, encoding="utf-8")
    prov = ha._read_provenance(tmp_path)
    assert len(prov) == 1
    assert prov[0]["model_repo"] == REPO
    assert prov[0]["k_runs"] == "3"


def test_read_vllm_logs(tmp_path):
    log = tmp_path / "vllm-1.log"; log.write_text(_vllm_log(), encoding="utf-8")
    info = ha._read_vllm_logs([log])
    assert info["repos"] == {REPO.replace("/", "--")}
    assert info["snapshots"] == {REV}
    assert SERVED in info["served"]
    assert info["init_count"] == 1
    assert info["vllm_versions"] == {VLLM_VER}
    assert info["oom_lines"] == []


def test_read_vllm_logs_none_when_empty():
    assert ha._read_vllm_logs([]) is None


# --------------------------------------------------------------------------- #
# all-pass path
# --------------------------------------------------------------------------- #

def test_all_checks_pass(tmp_path):
    root = _build_tree(tmp_path)
    cfg = tmp_path / "v.yaml"; cfg.write_text(CONFIG_YAML, encoding="utf-8")
    log = tmp_path / "vllm-1.log"; log.write_text(_vllm_log(), encoding="utf-8")
    st = _statuses(ha.audit_provenance(root, [log], cfg, None))
    assert st["A1"] == "PASS"
    assert st["A2"] == "SKIP"           # config has a distribution but this tree logs no temps
    assert st["A3"] == "PASS"
    assert st["A4"] == "PASS"
    assert st["A5"] == "PASS"
    assert st["B1"] == "PASS"
    assert st["B2"] == "PASS"
    assert st["B3"] == "PASS"
    assert st["B4"] == "PASS"
    assert st["J3"] == "PASS"
    assert st["G3"] == "SKIP"          # no --tier1-passes supplied


def test_log_and_config_optional_mark_skip_not_fail(tmp_path):
    """Without --config / --vllm-log, log/config-dependent checks SKIP (never FAIL)."""
    root = _build_tree(tmp_path)
    st = _statuses(ha.audit_provenance(root, [], None, None))
    assert st["A1"] == "SKIP"           # no config = no pinned claim to compare against
    assert st["B2"] == "SKIP"           # needs vllm log
    assert st["J3"] == "SKIP"           # needs vllm log
    assert st["B1"] == "PASS"           # execution-log only
    assert st["A3"] == "PASS"           # provenance internally consistent


# --------------------------------------------------------------------------- #
# fail paths (one mutation each)
# --------------------------------------------------------------------------- #

def test_A1_fail_on_snapshot_mismatch(tmp_path):
    root = _build_tree(tmp_path)
    cfg = tmp_path / "v.yaml"; cfg.write_text(CONFIG_YAML, encoding="utf-8")
    log = tmp_path / "vllm-1.log"
    log.write_text(_vllm_log(rev="0000000000000000000000000000000000000000"), encoding="utf-8")
    st = _statuses(ha.audit_provenance(root, [log], cfg, None))
    assert st["A1"] == "FAIL"


def test_A3_fail_on_mixed_commits(tmp_path):
    root = _build_tree(tmp_path, extra_provenance_commit="ffffffffffff")
    cfg = tmp_path / "v.yaml"; cfg.write_text(CONFIG_YAML, encoding="utf-8")
    st = _statuses(ha.audit_provenance(root, [], cfg, None))
    assert st["A3"] == "FAIL"           # provenance files disagree on commit_sha


def test_A4_fail_on_foreign_model(tmp_path):
    root = _build_tree(tmp_path, stage_model="ollama/gpt-4o-mini")
    cfg = tmp_path / "v.yaml"; cfg.write_text(CONFIG_YAML, encoding="utf-8")
    st = _statuses(ha.audit_provenance(root, [], cfg, None))
    assert st["A4"] == "FAIL"


def test_A5_fail_on_wrong_run_count(tmp_path):
    root = _build_tree(tmp_path, manifest_n=2)   # K=3 from provenance, manifest says 2
    st = _statuses(ha.audit_provenance(root, [], None, None))
    assert st["A5"] == "FAIL"


def test_B1_fail_on_nonzero_cost(tmp_path):
    root = _build_tree(tmp_path, cost=0.012)
    st = _statuses(ha.audit_provenance(root, [], None, None))
    assert st["B1"] == "FAIL"


def test_B3_fail_on_think_leak(tmp_path):
    root = _build_tree(tmp_path, think_leak=True)
    st = _statuses(ha.audit_provenance(root, [], None, None))
    assert st["B3"] == "FAIL"


def test_B4_fail_on_zero_tokens(tmp_path):
    root = _build_tree(tmp_path, tokens=0)
    st = _statuses(ha.audit_provenance(root, [], None, None))
    assert st["B4"] == "FAIL"


def test_J3_fail_on_oom(tmp_path):
    root = _build_tree(tmp_path)
    cfg = tmp_path / "v.yaml"; cfg.write_text(CONFIG_YAML, encoding="utf-8")
    log = tmp_path / "vllm-1.log"; log.write_text(_vllm_log(oom=True), encoding="utf-8")
    st = _statuses(ha.audit_provenance(root, [log], cfg, None))
    assert st["J3"] == "FAIL"


def test_G3_diff_pass_and_fail(tmp_path):
    root = _build_tree(tmp_path)
    a = tmp_path / "passA.txt"; b = tmp_path / "passB.txt"
    a.write_text("score 1.0\n", encoding="utf-8"); b.write_text("score 1.0\n", encoding="utf-8")
    st = _statuses(ha.audit_provenance(root, [], None, [a, b]))
    assert st["G3"] == "PASS"
    b.write_text("score 0.9\n", encoding="utf-8")
    st = _statuses(ha.audit_provenance(root, [], None, [a, b]))
    assert st["G3"] == "FAIL"


def test_B2_fail_on_multiple_engine_inits(tmp_path):
    root = _build_tree(tmp_path)
    cfg = tmp_path / "v.yaml"; cfg.write_text(CONFIG_YAML, encoding="utf-8")
    log = tmp_path / "vllm-1.log"; log.write_text(_vllm_log(inits=2), encoding="utf-8")
    st = _statuses(ha.audit_provenance(root, [log], cfg, None))
    assert st["B2"] == "FAIL"           # 2 inits in 1 log = a reload/restart


# --------------------------------------------------------------------------- #
# A2 — temperature claim vs temperatures actually run
# --------------------------------------------------------------------------- #

def test_read_config_temp_keys(tmp_path):
    cfg = tmp_path / "v.yaml"; cfg.write_text(CONFIG_YAML, encoding="utf-8")
    assert ha._read_config_temp_keys(cfg) == [0.1, 0.3, 0.7]   # parsed, "notes:" excluded


def test_A2_pass_when_ran_temps_are_a_subset_of_claimed(tmp_path):
    root = _build_tree(tmp_path, temps=[0.1, 0.3])             # ⊆ claimed {0.1,0.3,0.7}
    cfg = tmp_path / "v.yaml"; cfg.write_text(CONFIG_YAML, encoding="utf-8")
    checks = {c.id: c for c in ha.audit_provenance(root, [], cfg, None)}
    assert checks["A2"].status == "PASS"
    assert "claimed-but-not-seen" in checks["A2"].evidence  # 0.7 noted, not failed


def test_A2_fail_on_unexpected_temperature(tmp_path):
    root = _build_tree(tmp_path, temps=[0.0, 0.3])             # 0.0 was never claimed
    cfg = tmp_path / "v.yaml"; cfg.write_text(CONFIG_YAML, encoding="utf-8")
    st = _statuses(ha.audit_provenance(root, [], cfg, None))
    assert st["A2"] == "FAIL"           # this is the classic "claimed temp=0 but it sampled" catch


def test_A2_skip_when_no_temps_logged(tmp_path):
    root = _build_tree(tmp_path)                               # pre-instrumentation: no temperatures
    cfg = tmp_path / "v.yaml"; cfg.write_text(CONFIG_YAML, encoding="utf-8")
    st = _statuses(ha.audit_provenance(root, [], cfg, None))
    assert st["A2"] == "SKIP"


def test_A2_skip_when_config_has_no_distribution(tmp_path):
    root = _build_tree(tmp_path, temps=[0.3])
    cfg = tmp_path / "v.yaml"
    cfg.write_text("models:\n  primary:\n    hf_repo: \"x/y\"\n", encoding="utf-8")
    st = _statuses(ha.audit_provenance(root, [], cfg, None))
    assert st["A2"] == "SKIP"


# --------------------------------------------------------------------------- #
# zero-config auto-discovery of vLLM logs (so a bare invocation works)
# --------------------------------------------------------------------------- #

def test_discover_vllm_logs_from_job_ids(tmp_path, monkeypatch):
    sl = tmp_path / "slurm"; sl.mkdir()
    (sl / "vllm-111.log").write_text("x", encoding="utf-8")
    (sl / "vllm-222.log").write_text("y", encoding="utf-8")
    monkeypatch.setenv("AGENTIC_SLURM_LOGS", str(sl))
    got = ha._discover_vllm_logs([{"job": "111"}, {"job": "222"}, {"job": "999"}])  # 999 has no log
    assert sorted(p.name for p in got) == ["vllm-111.log", "vllm-222.log"]


def test_discover_vllm_logs_prefers_explicit_path(tmp_path):
    f = tmp_path / "v.log"; f.write_text("x", encoding="utf-8")
    got = ha._discover_vllm_logs([{"vllm_log": str(f)}, {"vllm_log": str(tmp_path / "missing.log")}])
    assert [p.name for p in got] == ["v.log"]   # missing path dropped


# --------------------------------------------------------------------------- #
# citation matcher: identifier detection (offline — no network for a non-id URL)
# --------------------------------------------------------------------------- #

def test_verify_by_identifier_no_id_is_inconclusive_offline():
    # a plain web URL carries no S2/DOI/arXiv id -> returns "" without any network call
    assert ha._verify_by_identifier(ha.Citation(title="x", url="https://example.com/blog/post")) == ""
    assert ha._verify_by_identifier(ha.Citation(title="x", url="")) == ""


def test_collect_citations_reads_bibliography_references_with_url(tmp_path):
    """WS2b: the audit must read bibliography['references'] (with the threaded URL),
    not fall back to the URL-less bibliography.txt."""
    rd = tmp_path / "LiteratureTeam" / "ModeNoWcNoHITL" / "run_001"
    rd.mkdir(parents=True)
    syn = {"bibliography": {"references": [
        {"citation_key": "Smith2024", "formatted_citation": "A Real Paper. (2024). Venue.",
         "reference_type": "journal", "importance": "High",
         "title": "A Real Paper", "url": "https://www.semanticscholar.org/paper/abc123"}]}}
    (rd / "synthesis_results.json").write_text(json.dumps(syn), encoding="utf-8")
    (rd / "bibliography.txt").write_text("[Smith2024] A Real Paper. (2024). Research Paper.\n", encoding="utf-8")
    cites = ha.collect_citations("LiteratureTeam/ModeNoWcNoHITL/run_001", rd)
    assert len(cites) == 1
    assert cites[0].title == "A Real Paper"
    assert "semanticscholar.org/paper/abc123" in cites[0].url      # URL carried through
    assert cites[0].origin.endswith("synthesis_results.json")      # not the txt fallback
