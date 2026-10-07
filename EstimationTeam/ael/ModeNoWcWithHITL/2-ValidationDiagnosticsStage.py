# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Validation & Diagnostics Stage — the deterministic battery, then LLM narration.

Pipeline:
1. Re-fit the stored analysis panel deterministically (no network, no re-proposal)
2. Diagnostics battery: Breusch-Pagan, Durbin-Watson, Ljung-Box, Jarque-Bera,
   residual-ADF spurious-regression guard, RESET, VIF, sample-size check
3. Verdict update: severe failures downgrade 'estimated' -> 'fragile' (harness's decision)
4. Validator (LLM) narrates the COMPUTED statistics — narration can never change a verdict

Input:  estimation_output.json (Stage 1)
Output: validation_output.json (ValidationStageOutput)

This file is byte-identical across EstimationTeam mode directories (mode-copy invariant).
"""

import json
import os
import sys
import time
from pathlib import Path as _Path
from typing import Dict, Optional

from dotenv import load_dotenv

_agents_dir = _Path(__file__).resolve().parent.parent.parent.parent
if str(_agents_dir) not in sys.path:
    sys.path.insert(0, str(_agents_dir))

from shared.auto_input import auto_input
from shared.llm import LLMClient
from shared.observability import MetricsCollector

from EstimationTeam.ael.estim_harness import (
    VERDICT_INESTIMABLE,
    apply_diagnostics_verdict,
    run_diagnostics,
)
from EstimationTeam.ael.schemas.stage_outputs import EstimationStageOutput, ValidationStageOutput

load_dotenv()


# The specification record (every fitted spec, the committee's review rounds
# and any objections) travels with the outcome to the final artifact the report reads.
_CARRIED_METADATA = ("fitted_specifications", "spec_review", "committee_objections",
                     "executable_model_use")


def _carried(metadata):
    return {k: metadata[k] for k in _CARRIED_METADATA if k in (metadata or {})}


# ========== AGENT ==========

class Validator:
    """Narrates the computed diagnostics; has no authority over verdicts."""

    def __init__(self, openai_api_key: str = "", collector: Optional[MetricsCollector] = None):
        self.agent_name = "Validator"
        self.llm = LLMClient(
            temperature=0.3,
            api_key=openai_api_key or os.environ.get("OPENAI_API_KEY", ""),
            collector=collector,
            agent_name=self.agent_name,
        )

    def interpret(self, diagnostics_dump: str, outcome_summary: str) -> str:
        raw = self.llm.invoke([
            {"role": "system", "content":
                "You are an econometrician reviewing computed diagnostic statistics. "
                "Interpret ONLY the numbers given; do not invent statistics or change verdicts."},
            {"role": "user", "content":
                f"Estimation summary:\n{outcome_summary[:1200]}\n\n"
                f"Computed diagnostics (statsmodels):\n{diagnostics_dump[:2400]}\n\n"
                "In <=150 words: what do these diagnostics imply for the reliability of the "
                "estimates, and which caveats must accompany any interpretation?"},
        ], max_tokens=400)
        return str(raw).strip()


# ========== ORCHESTRATOR ==========

class ValidationOrchestrator:
    """Runs the deterministic battery and applies the verdict update."""

    def __init__(self, openai_api_key: str = "", collector: Optional[MetricsCollector] = None,
                 quiet: bool = False):
        self.collector = collector
        self.quiet = quiet
        self.validator = Validator(openai_api_key, collector=collector)
        self.output: Optional[ValidationStageOutput] = None

    def _print(self, msg: str):
        if not self.quiet:
            print(msg)

    def run_validation_pipeline(self, estimation_output_data: Dict) -> ValidationStageOutput:
        stage1 = EstimationStageOutput(**estimation_output_data)
        outcome = stage1.outcome
        inherited_timing = dict(stage1.timing_sec)

        if outcome.verdict == VERDICT_INESTIMABLE:
            self._print("[ValidationStage] Upstream verdict is inestimable — nothing to diagnose")
            from EstimationTeam.ael.estim_harness.types import DiagnosticsReport
            report = DiagnosticsReport(results=[], overall="fragile",
                                       severe_failures=["upstream inestimable"])
            self.output = ValidationStageOutput(diagnostics=report, outcome=outcome,
                                                timing_sec=inherited_timing,
                                                metadata=_carried(stage1.metadata))
            return self.output

        self._print("[ValidationStage] Running deterministic diagnostics battery...")
        t0 = time.perf_counter()
        report = run_diagnostics(outcome)
        updated = apply_diagnostics_verdict(outcome, report)
        diagnostics_sec = time.perf_counter() - t0
        self._print(f"[ValidationStage] Battery: {report.overall} "
                    f"({len(report.results)} tests, severe={report.severe_failures or 'none'}) "
                    f"-> verdict {updated.verdict}")

        # LLM narration of the computed stats (optional — the stage succeeds without it)
        t0 = time.perf_counter()
        try:
            diag_dump = json.dumps([r.model_dump() for r in report.results], default=str)
            coef_summary = ", ".join(
                f"{c.name}={c.estimate:.4g} (p={c.p_value:.3f})" for c in updated.coefficients)
            outcome_summary = (f"{updated.dependent_name} regression, n={updated.n_obs}, "
                               f"R2={updated.r_squared}, cov={updated.cov_type}; {coef_summary}")
            report = report.model_copy(update={
                "interpretation": self.validator.interpret(diag_dump, outcome_summary)})
        except Exception as e:
            self._print(f"[ValidationStage] Narration skipped (non-critical): {e}")
        narration_sec = time.perf_counter() - t0

        timing_sec = {**inherited_timing,
                     "stage2_diagnostics_det_sec": diagnostics_sec,
                     "stage2_narration_llm_sec": narration_sec}
        self.output = ValidationStageOutput(diagnostics=report, outcome=updated,
                                            timing_sec=timing_sec,
                                            metadata=_carried(stage1.metadata))
        return self.output

    def save_validation_output(self, filename: str = "validation_output.json") -> str:
        if self.output is None:
            raise RuntimeError("run_validation_pipeline first")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.output.model_dump(), f, indent=2, default=str)
        self._print(f"[ValidationStage] Output saved to {filename}")
        return filename


if __name__ == "__main__":
    est_path = auto_input("Path to estimation_output.json: ",
                          default="estimation_output.json").strip()
    with open(est_path, "r", encoding="utf-8") as f:
        estimation_output_data = json.load(f)
    orch = ValidationOrchestrator()
    orch.run_validation_pipeline(estimation_output_data)
    orch.save_validation_output()
