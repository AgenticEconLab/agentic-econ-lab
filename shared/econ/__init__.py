# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Economics-domain intelligence modules (V0.7).

Subpackages:
  - causal_fm : Zero-shot causal inference via PFN in-context learning.
  - simulation : Agent-based economic simulation (ABIDES, LLM-Economist, MALLES).
  - dsge_generator : Synthetic DSGE trajectory generator.
  - theory_transfer : Chib-Tan 90/10 theory-guided transfer learning.
"""

from shared.econ.causal_fm import (
    CausalAdjustment,
    CausalEstimate,
    CausalFMClient,
)
from shared.econ.dsge_generator import (
    DSGEGenerator,
    DSGESpec,
    flatten_to_records,
)
from shared.econ.theory_transfer import (
    TheoryTransferConfig,
    TheoryTransferModel,
    TheoryTransferTrainer,
    drl_dsge_not_implemented,
)
from shared.econ.stage import run_simulation_stage
from shared.econ.model_team_hooks import (
    attach_causal_estimate,
    attach_falsifier_reports,
    maybe_run_simulation_stage,
    run_causal_fm_pass,
    run_falsifier_pass,
)
from shared.econ.simulation import (
    ABIDESEconomistBackend,
    BACKENDS,
    LLMEconomistBackend,
    MALLESBackend,
    SCENARIOS,
    Scenario,
    SimulationBackend,
    SimulationResult,
    SimulationSpec,
    run_simulation,
)

__all__ = [
    "CausalAdjustment",
    "CausalEstimate",
    "CausalFMClient",
    "ABIDESEconomistBackend",
    "BACKENDS",
    "LLMEconomistBackend",
    "MALLESBackend",
    "SCENARIOS",
    "Scenario",
    "SimulationBackend",
    "SimulationResult",
    "SimulationSpec",
    "run_simulation",
    "run_simulation_stage",
    "DSGEGenerator",
    "DSGESpec",
    "flatten_to_records",
    "TheoryTransferConfig",
    "TheoryTransferModel",
    "TheoryTransferTrainer",
    "drl_dsge_not_implemented",
    "run_falsifier_pass",
    "run_causal_fm_pass",
    "maybe_run_simulation_stage",
    "attach_falsifier_reports",
    "attach_causal_estimate",
]
