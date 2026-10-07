# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Inference & Robustness Stage — theory-derived hypothesis tests and stability sweeps.

Pipeline:
1. Hypothesis tests: each sign/zero restriction carried in the EstimationSpec is tested
   deterministically (one-sided t for sign restrictions, two-sided for zero)
2. Robustness sweeps: drop-one-regressor, split-sample halves, covariance-estimator swap —
   all re-estimated on the STORED analysis panel (no network, fully reproducible)
3. HypothesisTester (LLM) narrates the computed results — narration only
4. Optimizer (§4.6): aggregates the real wall-clock timing recorded across all 3 stages,
   split into deterministic-compute vs. LLM time — deterministic, no LLM, profiles only

Economic-magnitude interpretation (elasticities/marginal effects, narrated into economic
significance) deliberately lives one stage downstream in ReportingTeam's ResultsInterpreter
(§4.7), which already computes it more completely (one-SD effects, standardized betas,
elasticity-at-means from the real data descriptives) from this stage's own output — adding a
second, less complete ResultsInterpreter here would just be unread, duplicated computation.

Input:  validation_output.json (Stage 2)
Output: inference_output.json (InferenceStageOutput)

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

from EstimationTeam.ael.estim_harness import VERDICT_INESTIMABLE, run_inference
from EstimationTeam.ael.estim_harness.types import PerformanceProfile
from EstimationTeam.ael.schemas.stage_outputs import InferenceStageOutput, ValidationStageOutput

load_dotenv()


# The specification record (every fitted spec, the committee's review rounds
# and any objections) travels with the outcome to the final artifact the report reads.
_CARRIED_METADATA = ("fitted_specifications", "spec_review", "committee_objections",
                     "executable_model_use")


def _carried(metadata):
    return {k: metadata[k] for k in _CARRIED_METADATA if k in (metadata or {})}


# ========== AGENTS ==========

class HypothesisTester:
    """Narrates computed hypothesis/robustness results; no authority over the numbers."""

    def __init__(self, openai_api_key: str = "", collector: Optional[MetricsCollector] = None):
        self.agent_name = "HypothesisTester"
        self.llm = LLMClient(
            temperature=0.3,
            api_key=openai_api_key or os.environ.get("OPENAI_API_KEY", ""),
            collector=collector,
            agent_name=self.agent_name,
        )

    def interpret(self, inference_dump: str, question: str) -> str:
        raw = self.llm.invoke([
            {"role": "system", "content":
                "You are an econometrician summarizing computed hypothesis tests and "
                "robustness checks. Interpret ONLY the numbers given."},
            {"role": "user", "content":
                f"Research question: {question[:300]}\n\n"
                f"Computed inference results:\n{inference_dump[:2600]}\n\n"
                "In <=150 words: which theoretical predictions does the evidence support, "
                "how stable are the findings, and what must be qualified?"},
        ], max_tokens=400)
        return str(raw).strip()


class Optimizer:
    """Performance profiler over the 3-stage estimation workflow. Deterministic, no LLM:
    aggregates the real wall-clock timing each stage already measured, split into
    deterministic-compute vs. LLM proposal/narration time, so a researcher can see where
    the econometric bottleneck sits (e.g. repeated re-estimation in the robustness sweeps
    vs. LLM latency). It profiles the harness; it does not rewrite it — an Optimizer that
    edited estimation code would forfeit the harness's 'the verdict is always the
    harness's' guarantee."""

    def __init__(self, collector: Optional[MetricsCollector] = None):
        self.agent_name = "Optimizer"
        self.collector = collector

    def profile(self, timing_sec: Dict[str, float]) -> PerformanceProfile:
        print(f"\n[{self.agent_name}] Profiling {len(timing_sec)} timed step(s) across "
              "3 stages...")
        if not timing_sec:
            return PerformanceProfile(notes=["no timing recorded"])
        det_sec = sum(v for k, v in timing_sec.items() if k.endswith("_det_sec"))
        llm_sec = sum(v for k, v in timing_sec.items() if k.endswith("_llm_sec"))
        total = det_sec + llm_sec
        bottleneck = max(timing_sec, key=timing_sec.get)
        notes = []
        if total and llm_sec / total > 0.6:
            notes.append("LLM proposal/narration dominates wall time; deterministic "
                         "econometric computation is comparatively cheap at this sample size.")
        elif total and det_sec / total > 0.6:
            notes.append("Deterministic computation (estimation/diagnostics/robustness "
                         "sweeps) dominates wall time; a larger panel or more robustness "
                         "variants would benefit most from optimization here.")
        profile = PerformanceProfile(
            stage_timing_sec={k: round(v, 4) for k, v in timing_sec.items()},
            deterministic_sec=round(det_sec, 4), llm_sec=round(llm_sec, 4),
            total_sec=round(total, 4), bottleneck_step=bottleneck, notes=notes,
        )
        print(f"  Total: {profile.total_sec:.2f}s (deterministic {profile.deterministic_sec:.2f}s "
              f"/ LLM {profile.llm_sec:.2f}s); slowest step: {bottleneck}")
        return profile


# ========== ORCHESTRATOR ==========

class InferenceOrchestrator:
    """Runs deterministic inference on the stage-2 outcome."""

    def __init__(self, openai_api_key: str = "", collector: Optional[MetricsCollector] = None,
                 quiet: bool = False):
        self.collector = collector
        self.quiet = quiet
        self.tester = HypothesisTester(openai_api_key, collector=collector)
        self.optimizer = Optimizer(collector=collector)
        self.output: Optional[InferenceStageOutput] = None

    def _print(self, msg: str):
        if not self.quiet:
            print(msg)

    def run_inference_pipeline(self, validation_output_data: Dict,
                               research_question: str = "") -> InferenceStageOutput:
        stage2 = ValidationStageOutput(**validation_output_data)
        outcome = stage2.outcome
        inherited_timing = dict(stage2.timing_sec)

        if outcome.verdict == VERDICT_INESTIMABLE:
            self._print("[InferenceStage] Upstream verdict is inestimable — no inference to run")
            from EstimationTeam.ael.estim_harness.types import InferenceReport
            profile = self.optimizer.profile(inherited_timing)
            self.output = InferenceStageOutput(
                inference=InferenceReport(), outcome=outcome,
                summary="No inference: upstream estimation was honestly declined "
                        f"({outcome.reason}).",
                performance_profile=profile, metadata=_carried(stage2.metadata))
            return self.output

        self._print("[InferenceStage] Running hypothesis tests + robustness sweeps...")
        t0 = time.perf_counter()
        report = run_inference(outcome)
        inference_sec = time.perf_counter() - t0
        n_supported = sum(1 for h in report.hypotheses if h.supported)
        _counts = {}
        for h in report.hypotheses:
            _counts[h.outcome or "absent"] = _counts.get(h.outcome or "absent", 0) + 1
        _breakdown = ", ".join(f"{k} {v}" for k, v in sorted(_counts.items())) or "none"
        # Only variants that could be estimated are robustness checks;
        # the non-estimable ones are counted separately, never folded into the total.
        n_est = sum(1 for c in report.robustness if c.variant_estimate is not None)
        n_not = len(report.robustness) - n_est
        _rob = (f"{n_est} estimable robustness variant(s)"
                + (f" ({n_not} not estimable)" if n_not else ""))
        self._print(f"[InferenceStage] Hypotheses: {n_supported}/{len(report.hypotheses)} "
                    f"supported ({_breakdown}); {_rob}; "
                    f"sign-stability: {report.stability_score}")

        summary = (f"{n_supported}/{len(report.hypotheses)} theory-derived hypotheses supported "
                   f"({_breakdown}); {_rob}, "
                   f"sign-stability {report.stability_score}; final verdict: {outcome.verdict}")

        t0 = time.perf_counter()
        try:
            dump = json.dumps(report.model_dump(), default=str)
            report = report.model_copy(update={
                "interpretation": self.tester.interpret(dump, research_question)})
        except Exception as e:
            self._print(f"[InferenceStage] Narration skipped (non-critical): {e}")
        hypothesis_narration_sec = time.perf_counter() - t0

        timing_sec = {**inherited_timing,
                     "stage3_inference_det_sec": inference_sec,
                     "stage3_hypothesis_narration_llm_sec": hypothesis_narration_sec}
        profile = self.optimizer.profile(timing_sec)

        self.output = InferenceStageOutput(inference=report, outcome=outcome, summary=summary,
                                           performance_profile=profile,
                                           metadata=_carried(stage2.metadata))
        return self.output

    def save_inference_output(self, filename: str = "inference_output.json") -> str:
        if self.output is None:
            raise RuntimeError("run_inference_pipeline first")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.output.model_dump(), f, indent=2, default=str)
        self._print(f"[InferenceStage] Output saved to {filename}")
        return filename


if __name__ == "__main__":
    val_path = auto_input("Path to validation_output.json: ",
                          default="validation_output.json").strip()
    with open(val_path, "r", encoding="utf-8") as f:
        validation_output_data = json.load(f)
    orch = InferenceOrchestrator()
    orch.run_inference_pipeline(validation_output_data)
    orch.save_inference_output()
