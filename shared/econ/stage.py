# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Optional 4-SimulationStage helper — shared implementation for all 4 ModelTeam
AEL modes (NoFcNoHITL / NoFcWithHITL / WithFcNoHITL / WithFcWithHITL).

Each ModelTeam mode ships a thin ``4-SimulationStage.py`` that calls
:func:`run_simulation_stage` with its own paths. The shared helper keeps the
per-mode wrappers < 20 lines of boilerplate each.

Feature flag: ``simulation_enabled`` (default False) preserves V0.6 behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from shared.econ.simulation import (
    SCENARIOS,
    SimulationResult,
    SimulationSpec,
    run_simulation,
)


def run_simulation_stage(
    calibration_output_path: str | Path,
    output_path: str | Path,
    *,
    simulation_enabled: bool = False,
    backend: Optional[str] = None,
    scenario: Optional[str] = None,
    n_agents: int = 100,
    n_periods: int = 60,
    random_seed: int = 42,
    extra_calibration: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run (or skip) the optional 4th ModelTeam stage.

    Parameters
    ----------
    calibration_output_path
        Input JSON produced by 3-CalibrationStage (``calibration_output.json``).
    output_path
        Where to write ``simulation_output.json``.
    simulation_enabled
        Hard flag. Default False preserves V0.6 pipeline behaviour — the stage
        becomes a no-op and writes a ``{simulation_skipped: true}`` sentinel.
    backend / scenario
        Override the automatic scenario selection (which reads ``recommended_scenario``
        from the calibration output, falling back to ``"dsge_monetary"``).
    """
    if not simulation_enabled:
        skipped = {"simulation_skipped": True}
        Path(output_path).write_text(json.dumps(skipped, indent=2))
        return skipped

    calibration = _load_json(calibration_output_path)
    scen_name = scenario or calibration.get("recommended_scenario") or "dsge_monetary"
    if scen_name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scen_name!r}")
    scen = SCENARIOS[scen_name]
    backend_name = backend or scen.default_backend  # type: ignore[assignment]

    spec = SimulationSpec(
        backend=backend_name,  # type: ignore[arg-type]
        scenario=scen_name,
        n_agents=n_agents,
        n_periods=n_periods,
        random_seed=random_seed,
    )
    calib = dict(calibration.get("parameters") or {})
    if extra_calibration:
        calib.update(extra_calibration)

    result = run_simulation(spec, calib)
    payload = {
        "simulation_result": result.to_dict(),
        "metrics": result.metrics,
        "time_series": result.time_series,
        "scenario_reference": scen.reference,
    }
    Path(output_path).write_text(json.dumps(payload, indent=2))
    return payload


def _load_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)
