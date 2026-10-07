# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""ModelTeam designs/calibrates its 2-3 models CONCURRENTLY (vLLM batches the cross-model LLM
calls). Lock the behavior of the parallelized loops: every model is kept, input order is preserved,
and a single per-model failure drops only that model (the batch survives)."""

import importlib.util
import sys
import typing
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("httpx")
pydantic = pytest.importorskip("pydantic")

_AGENTS = Path(__file__).resolve().parents[2]
if str(_AGENTS) not in sys.path:
    sys.path.insert(0, str(_AGENTS))


def _load(stage_file, mod_name):
    path = _AGENTS / "ModelTeam" / "ael" / "ModeNoWcNoHITL" / stage_file
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


MD = _load("2-ModelDesignStage.py", "design_stage_under_test")
CAL = _load("3-CalibrationStage.py", "calibration_stage_under_test_parallel")


def _val(ann):
    """A schema-valid dummy value for a required field of type ``ann``."""
    if typing.get_origin(ann) in (list, typing.List):
        return []
    if isinstance(ann, type) and issubclass(ann, pydantic.BaseModel):
        return _minimal(ann)
    return {str: "x", float: 0.0, int: 0, bool: False}.get(ann, "x")


def _minimal(cls):
    """Construct a schema-valid instance of a pydantic model by filling its required fields."""
    return cls(**{n: _val(f.annotation) for n, f in cls.model_fields.items() if f.is_required()})


def test_design_pipeline_parallel_keeps_all_in_order_and_survives_one_failure():
    orch = MD.ModelDesignOrchestrator.__new__(MD.ModelDesignOrchestrator)
    template = _minimal(MD.FormalMathematicalModel)

    def fake_design(fw, revision_request=None):
        title = getattr(fw, "framework_title", "") or ""
        if title == "BOOM":
            raise RuntimeError("boom")
        return template.model_copy(update={"model_title": f"M::{title}"})

    orch.model_designer = SimpleNamespace(design_formal_model=fake_design)
    fw = _minimal(MD.TheoreticalFramework).model_dump()
    theory = {"theoretical_frameworks": [
        {**fw, "framework_title": t} for t in ("A", "BOOM", "C")]}

    out = orch.run_design_pipeline(theory_output=theory)
    titles = [m.model_title for m in out.formal_models]
    assert titles == ["M::A", "M::C"]                 # order preserved; the failing model dropped
    assert out.metadata["num_models"] == 2


def test_calibration_pipeline_parallel_keeps_all_in_order_and_survives_one_failure():
    orch = CAL.CalibrationOrchestrator.__new__(CAL.CalibrationOrchestrator)
    bare = CAL.Calibrator.__new__(CAL.Calibrator)
    bare.agent_name = "Calibrator"

    def fake_calibrate(fm, targets_only=False):
        title = getattr(fm, "model_title", "") or ""
        if title == "BOOM":
            raise RuntimeError("boom")
        return bare._targets_only_model(fm, []).model_copy(update={"model_title": f"C::{title}"})

    orch.calibrator = SimpleNamespace(calibrate_model=fake_calibrate)
    fm = _minimal(CAL.FormalMathematicalModel).model_dump()
    design = {"formal_models": [{**fm, "model_title": t} for t in ("A", "BOOM", "C")]}

    out = orch.run_calibration_pipeline(model_design_output=design)
    titles = [m.model_title for m in out.calibrated_models]
    assert titles == ["C::A", "C::C"]                 # order preserved; the failing model dropped
    assert out.metadata["num_calibrated"] == 2
