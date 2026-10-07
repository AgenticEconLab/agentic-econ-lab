# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Simulation Stage (optional) - ModeNoWcNoHITL (added in V0.7)

Takes calibrated model parameters and runs an agent-based economic simulation
(ABIDES-Economist / LLM-Economist / MALLES depending on scenario).

Disabled by default — set ``simulation_enabled=True`` in ael_config.yaml
(or pass via environment) to activate.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_agents_dir = Path(__file__).resolve().parent.parent.parent.parent
if str(_agents_dir) not in sys.path:
    sys.path.insert(0, str(_agents_dir))

from shared.econ import run_simulation_stage


def main() -> None:
    here = Path(__file__).resolve().parent
    enabled = os.getenv("AEL_SIMULATION_ENABLED", "0").lower() in ("1", "true", "yes")
    run_simulation_stage(
        calibration_output_path=here / "calibration_output.json",
        output_path=here / "simulation_output.json",
        simulation_enabled=enabled,
    )


if __name__ == "__main__":
    main()
