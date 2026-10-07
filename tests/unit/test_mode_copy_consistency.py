# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Guard the per-mode stage-file copies.

Ideation/Literature/Model duplicate each stage file across the 4 Mode* directories. Two
invariants keep the cross-team pipeline working in EVERY mode, and both have so far been
enforced only by hand (repeated md5 checks during the hardening window):

1. **Byte-identity where designed**: ModelTeam stages 1-4 and Literature SynthesisStage carry no
   mode-specific code — all 4 copies must stay byte-identical. A fix applied to one copy but not
   the others silently forks behavior by mode.
2. **Interface compatibility everywhere**: the pipeline (`pipeline/team_runners.py`) drives the
   stage modules via ``import_module("<stage>").<Orchestrator>(...)`` with keyword arguments.
   Every mode copy must expose those classes/methods and accept those keywords — a signature
   drift in one mode raises TypeError only when that mode is live on GPU.

The CONTRACT below mirrors team_runners' actual call sites; update it when team_runners changes.
Pure AST checks — no imports of the stage modules, no LLM, no network."""

import ast
import hashlib
from pathlib import Path

import pytest

AGENTS = Path(__file__).resolve().parents[2]
MODES = ["ModeNoWcNoHITL", "ModeNoWcWithHITL", "ModeWithWcNoHITL", "ModeWithWcWithHITL"]

# --- invariant 1: designed byte-identical copies -----------------------------------------
BYTE_IDENTICAL = [
    ("ModelTeam", "1-TheoryStage.py"),
    ("ModelTeam", "2-ModelDesignStage.py"),
    ("ModelTeam", "3-CalibrationStage.py"),
    ("ModelTeam", "4-SimulationStage.py"),
    ("LiteratureTeam", "3-SynthesisStage.py"),
]

# --- invariant 2: the pipeline's call contract (team_runners.py call sites) --------------
# (team, stage-file, class, {method: [required kwargs]}).
# A method key may be prefixed "NoHITL:"/"WithHITL:" when team_runners only calls it in that
# mode subset (e.g. run_automated_* exists only in NoHITL copies; the HITL driver uses
# run_*_round and only ever imports WithHITL directories).
CONTRACT = [
    ("IdeationTeam", "1-SourcingStage.py", "MultiAgentOrchestrator", {
        "__init__": ["quiet", "collector"],
        "WithHITL:run_search_round": ["research_topic", "round_number"],
        "NoHITL:run_automated_search": [],
        "save_results": [],
    }),
    ("IdeationTeam", "2-RefinementStage.py", "RefinementOrchestrator", {
        "__init__": ["quiet", "collector"],
        "WithHITL:run_refinement_round": ["literature_df", "round_number"],
        "NoHITL:run_automated_refinement": [],
        "save_results": [],
    }),
    ("IdeationTeam", "3-IntegrationStage.py", "IntegrationOrchestrator", {
        "__init__": ["quiet", "collector"],
        "WithHITL:run_integration_round": ["questions", "round_number"],
        "NoHITL:run_automated_integration": [],
        "save_final_questions": [],
    }),
    ("LiteratureTeam", "1-LiteratureGatheringStage.py", "LiteratureGatheringOrchestrator", {
        "__init__": ["collector"],
        "run_gathering_pipeline": ["research_questions", "max_papers_per_question"],
        "save_literature_batch": [],
    }),
    ("LiteratureTeam", "2-GapDetectionStage.py", "GapDetectionOrchestrator", {
        "__init__": ["collector"],
        "run_gap_detection_pipeline": [],
        "save_gap_analysis": [],
        "save_graph_json": [],
    }),
    ("LiteratureTeam", "3-SynthesisStage.py", "SynthesisOrchestrator", {
        "__init__": ["collector"],
        "run_synthesis_pipeline": [],
        "save_literature_review": [],
        "save_synthesis_result": [],
    }),
    ("ModelTeam", "1-TheoryStage.py", "TheoryStageOrchestrator", {
        "__init__": ["collector"],
        "run_theory_pipeline": ["research_questions", "literature_batch"],
        "save_theory_output": [],
    }),
    ("ModelTeam", "2-ModelDesignStage.py", "ModelDesignOrchestrator", {
        "__init__": ["collector"],
        "run_design_pipeline": ["theory_output", "revision_request"],
        "save_design_output": [],
    }),
    ("ModelTeam", "3-CalibrationStage.py", "CalibrationOrchestrator", {
        "__init__": ["collector"],
        "run_calibration_pipeline": ["model_design_output", "targets_only"],
        "save_calibration_output": [],
    }),
]


# EstimationTeam: 2 modes only — no Wc axis. Stage files are byte-identical
# across the two mode dirs; HITL behavior lives entirely in the per-mode MasterOrchestrator.
ESTIMATION_MODES = ["ModeNoWcNoHITL", "ModeNoWcWithHITL"]
ESTIMATION_BYTE_IDENTICAL = [
    "1-EstimationStage.py",
    "2-ValidationDiagnosticsStage.py",
    "3-InferenceRobustnessStage.py",
]
ESTIMATION_CONTRACT = [
    ("1-EstimationStage.py", "EstimationOrchestrator", {
        "__init__": ["collector"],
        "run_estimation_pipeline": ["model_spec_data", "data_artifact", "research_question",
                                    "feedback"],
        "collect_human_feedback": ["round_number", "context"],
        "save_estimation_output": [],
    }),
    ("2-ValidationDiagnosticsStage.py", "ValidationOrchestrator", {
        "__init__": ["collector"],
        "run_validation_pipeline": ["estimation_output_data"],
        "save_validation_output": [],
    }),
    ("3-InferenceRobustnessStage.py", "InferenceOrchestrator", {
        "__init__": ["collector"],
        "run_inference_pipeline": ["validation_output_data", "research_question"],
        "save_inference_output": [],
    }),
]


# ReportingTeam: 2 modes only — no Wc axis. Stage files are
# byte-identical; HITL lives in the master (gated on MODE) and in the pipeline runner.
REPORTING_MODES = ["ModeNoWcNoHITL", "ModeNoWcWithHITL"]
REPORTING_BYTE_IDENTICAL = [
    "1-InterpretationStage.py",
    "2-DraftingStage.py",
    "3-QualityStage.py",
]
REPORTING_CONTRACT = [
    ("1-InterpretationStage.py", "InterpretationOrchestrator", {
        "__init__": ["collector"],
        "run_interpretation_pipeline": ["estimation_results", "research_question",
                                        "figures_dir"],
        "save_interpretation_output": [],
    }),
    ("2-DraftingStage.py", "DraftingOrchestrator", {
        "__init__": ["collector"],
        "run_drafting_pipeline": ["research_questions", "literature_review",
                                  "model_specification", "data_source",
                                  "estimation_results", "interpretation_output",
                                  "feasibility_report", "report_file", "feedback",
                                  "literature_batch"],
        "collect_human_feedback": ["round_number", "context"],
        "save_drafting_output": [],
    }),
    ("3-QualityStage.py", "QualityOrchestrator", {
        "__init__": ["collector"],
        "run_quality_pipeline": ["drafting_output", "source_artifacts"],
        "save_quality_output": [],
    }),
]


# CodeTeam: 2 modes only — no Wc axis; derived-code stages are
# byte-identical, HITL is a record-only review in the MODE-gated master.
CODE_MODES = ["ModeNoWcNoHITL", "ModeNoWcWithHITL"]
CODE_BYTE_IDENTICAL = [
    "1-CodeGenerationStage.py",
    "2-ValidationStage.py",
    "3-ExperimentationStage.py",
]
CODE_CONTRACT = [
    ("1-CodeGenerationStage.py", "CodeGenerationOrchestrator", {
        "__init__": ["collector"],
        "run_generation_pipeline": ["model_design_output", "calibration_output", "modules_dir"],
        "save_generation_output": [],
    }),
    ("2-ValidationStage.py", "CodeValidationOrchestrator", {
        "__init__": ["collector"],
        "run_validation_pipeline": ["generation_output", "workdir"],
        "save_validation_output": [],
    }),
    ("3-ExperimentationStage.py", "ExperimentationOrchestrator", {
        "__init__": ["collector"],
        "run_experimentation_pipeline": ["generation_output", "validation_output", "workdir"],
        "save_experimentation_output": [],
    }),
]


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def _class_methods(path: Path, class_name: str):
    """{method_name: (param_names, has_var_keyword)} for one class in one file (AST only)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            out = {}
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    a = item.args
                    names = [p.arg for p in (a.posonlyargs + a.args + a.kwonlyargs)]
                    out[item.name] = (names, a.kwarg is not None)
            return out
    return None


# DataTeam: 3 workflow dirs share class/method NAMES but have workflow-specific stage-1
# signatures (run_data_team filters kwargs by signature) — so the contract here is
# presence + the kwargs run_data_team ALWAYS relies on.
DATA_WORKFLOWS = ["ModeOpenSourceAPI", "ModePremiumSubscribed", "ModeUserUploaded"]
DATA_CONTRACT = [
    ("1-DataSourceStage.py", "DataSourceOrchestrator",
     {"__init__": ["collector"], "run_source_pipeline": ["research_question", "enable_hitl"],
      "save_source_output": []}),
    ("2-DataCleaningStage.py", "DataCleaningOrchestrator",
     {"__init__": ["collector"],
      "run_cleaning_pipeline": ["data_source_output", "research_question", "enable_hitl"],
      "save_cleaning_output": []}),
    ("3-QualityAssuranceStage.py", "QualityAssuranceOrchestrator",
     {"__init__": ["collector"],
      "run_qa_pipeline": ["data_cleaning_output", "research_question"],
      "save_qa_output": []}),
]


@pytest.mark.parametrize("team,fname", BYTE_IDENTICAL, ids=lambda x: str(x))
def test_designed_identical_copies_stay_in_sync(team, fname):
    files = [AGENTS / team / "ael" / m / fname for m in MODES]
    present = [f for f in files if f.exists()]
    assert len(present) == 4, f"{team}/{fname}: expected 4 mode copies, found {len(present)}"
    sums = {_md5(f) for f in present}
    assert len(sums) == 1, (
        f"{team}/{fname}: mode copies have DIVERGED ({len(sums)} distinct contents). "
        f"A fix was applied to some modes but not all — re-sync the copies "
        f"(cp from the edited one) so behavior cannot fork by mode.")


@pytest.mark.parametrize("team,fname,cls,methods", CONTRACT,
                         ids=[f"{t}/{f}::{c}" for t, f, c, _ in CONTRACT])
def test_every_mode_copy_honors_the_pipeline_contract(team, fname, cls, methods):
    for mode in MODES:
        path = AGENTS / team / "ael" / mode / fname
        assert path.exists(), f"{team}/{mode}/{fname} missing"
        found = _class_methods(path, cls)
        assert found is not None, f"{team}/{mode}/{fname}: class {cls} not found"
        for key, required_kwargs in methods.items():
            subset, _, meth = key.rpartition(":")
            if subset == "NoHITL" and "WithHITL" in mode:
                continue
            if subset == "WithHITL" and "WithHITL" not in mode:
                continue
            assert meth in found, f"{team}/{mode}/{fname}: {cls}.{meth} missing"
            params, has_var_kw = found[meth]
            for kw in required_kwargs:
                assert kw in params or has_var_kw, (
                    f"{team}/{mode}/{fname}: {cls}.{meth} does not accept "
                    f"keyword '{kw}' used by pipeline/team_runners.py")


@pytest.mark.parametrize("fname", ESTIMATION_BYTE_IDENTICAL, ids=str)
def test_estimation_mode_copies_stay_in_sync(fname):
    files = [AGENTS / "EstimationTeam" / "ael" / m / fname for m in ESTIMATION_MODES]
    present = [f for f in files if f.exists()]
    assert len(present) == len(ESTIMATION_MODES), (
        f"EstimationTeam/{fname}: expected {len(ESTIMATION_MODES)} mode copies, "
        f"found {len(present)}")
    sums = {_md5(f) for f in present}
    assert len(sums) == 1, (
        f"EstimationTeam/{fname}: mode copies have DIVERGED — re-sync with /bin/cp so "
        f"behavior cannot fork by mode (HITL differences belong in 0-MasterOrchestrator.py)")


@pytest.mark.parametrize("fname,cls,methods", ESTIMATION_CONTRACT,
                         ids=[f"{f}::{c}" for f, c, _ in ESTIMATION_CONTRACT])
def test_every_estimation_mode_honors_the_stage_contract(fname, cls, methods):
    for mode in ESTIMATION_MODES:
        path = AGENTS / "EstimationTeam" / "ael" / mode / fname
        assert path.exists(), f"EstimationTeam/{mode}/{fname} missing"
        found = _class_methods(path, cls)
        assert found is not None, f"EstimationTeam/{mode}/{fname}: class {cls} not found"
        for meth, required_kwargs in methods.items():
            assert meth in found, f"EstimationTeam/{mode}/{fname}: {cls}.{meth} missing"
            params, has_var_kw = found[meth]
            for kw in required_kwargs:
                assert kw in params or has_var_kw, (
                    f"EstimationTeam/{mode}/{fname}: {cls}.{meth} does not accept "
                    f"keyword '{kw}' fixed by the stage contract")


@pytest.mark.parametrize("fname", REPORTING_BYTE_IDENTICAL, ids=str)
def test_reporting_mode_copies_stay_in_sync(fname):
    files = [AGENTS / "ReportingTeam" / "ael" / m / fname for m in REPORTING_MODES]
    present = [f for f in files if f.exists()]
    assert len(present) == len(REPORTING_MODES), (
        f"ReportingTeam/{fname}: expected {len(REPORTING_MODES)} mode copies, "
        f"found {len(present)}")
    sums = {_md5(f) for f in present}
    assert len(sums) == 1, (
        f"ReportingTeam/{fname}: mode copies have DIVERGED — re-sync with /bin/cp")


@pytest.mark.parametrize("fname,cls,methods", REPORTING_CONTRACT,
                         ids=[f"{f}::{c}" for f, c, _ in REPORTING_CONTRACT])
def test_every_reporting_mode_honors_the_stage_contract(fname, cls, methods):
    for mode in REPORTING_MODES:
        path = AGENTS / "ReportingTeam" / "ael" / mode / fname
        assert path.exists(), f"ReportingTeam/{mode}/{fname} missing"
        found = _class_methods(path, cls)
        assert found is not None, f"ReportingTeam/{mode}/{fname}: class {cls} not found"
        for meth, required_kwargs in methods.items():
            assert meth in found, f"ReportingTeam/{mode}/{fname}: {cls}.{meth} missing"
            params, has_var_kw = found[meth]
            for kw in required_kwargs:
                assert kw in params or has_var_kw, (
                    f"ReportingTeam/{mode}/{fname}: {cls}.{meth} does not accept "
                    f"keyword '{kw}' fixed by the stage contract")


@pytest.mark.parametrize("fname", CODE_BYTE_IDENTICAL, ids=str)
def test_code_mode_copies_stay_in_sync(fname):
    files = [AGENTS / "CodeTeam" / "ael" / m / fname for m in CODE_MODES]
    present = [f for f in files if f.exists()]
    assert len(present) == len(CODE_MODES), (
        f"CodeTeam/{fname}: expected {len(CODE_MODES)} mode copies, found {len(present)}")
    sums = {_md5(f) for f in present}
    assert len(sums) == 1, (
        f"CodeTeam/{fname}: mode copies have DIVERGED — re-sync with /bin/cp")


@pytest.mark.parametrize("fname,cls,methods", CODE_CONTRACT,
                         ids=[f"{f}::{c}" for f, c, _ in CODE_CONTRACT])
def test_every_code_mode_honors_the_stage_contract(fname, cls, methods):
    for mode in CODE_MODES:
        path = AGENTS / "CodeTeam" / "ael" / mode / fname
        assert path.exists(), f"CodeTeam/{mode}/{fname} missing"
        found = _class_methods(path, cls)
        assert found is not None, f"CodeTeam/{mode}/{fname}: class {cls} not found"
        for meth, required_kwargs in methods.items():
            assert meth in found, f"CodeTeam/{mode}/{fname}: {cls}.{meth} missing"
            params, has_var_kw = found[meth]
            for kw in required_kwargs:
                assert kw in params or has_var_kw, (
                    f"CodeTeam/{mode}/{fname}: {cls}.{meth} does not accept "
                    f"keyword '{kw}' fixed by the stage contract")


@pytest.mark.parametrize("fname,cls,methods", DATA_CONTRACT,
                         ids=[f"{f}::{c}" for f, c, _ in DATA_CONTRACT])
def test_every_data_workflow_honors_the_pipeline_contract(fname, cls, methods):
    for wf in DATA_WORKFLOWS:
        path = AGENTS / "DataTeam" / "ael" / wf / fname
        assert path.exists(), f"DataTeam/{wf}/{fname} missing"
        found = _class_methods(path, cls)
        assert found is not None, f"DataTeam/{wf}/{fname}: class {cls} not found"
        for meth, required_kwargs in methods.items():
            assert meth in found, f"DataTeam/{wf}/{fname}: {cls}.{meth} missing"
            params, has_var_kw = found[meth]
            for kw in required_kwargs:
                assert kw in params or has_var_kw, (
                    f"DataTeam/{wf}/{fname}: {cls}.{meth} does not accept keyword "
                    f"'{kw}' relied on by pipeline/team_runners.py::run_data_team")
