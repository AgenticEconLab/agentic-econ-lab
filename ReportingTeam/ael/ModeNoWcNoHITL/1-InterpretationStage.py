# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Interpretation Stage — deterministic economic magnitudes + figures, LLM narration only.

Pipeline:
1. Compute effect sizes (one-SD effects, standardized betas, elasticities at means) and
   descriptives from the estimation artifact's stored analysis panel — pure pandas
2. Render figures (standardized series, coefficient/CI plot) via matplotlib Agg
3. ResultsInterpreter (LLM) narrates the COMPUTED magnitudes — narration only; an upstream
   `inestimable` passes through honestly with no invented numbers

Input:  estimation_results (EstimationTeam inference output)
Output: interpretation_output.json (+ figures/)

This file is byte-identical across ReportingTeam mode directories (mode-copy invariant).
"""

import json
import os
import sys
from pathlib import Path as _Path
from typing import Dict, Optional

from dotenv import load_dotenv

_agents_dir = _Path(__file__).resolve().parent.parent.parent.parent
if str(_agents_dir) not in sys.path:
    sys.path.insert(0, str(_agents_dir))

from shared.auto_input import auto_input
from shared.llm import LLMClient
from shared.observability import MetricsCollector

from ReportingTeam.ael.report_harness import interpret_estimation, make_figures
from ReportingTeam.ael.schemas.stage_outputs import InterpretationStageOutput

load_dotenv()


# ========== AGENT ==========

class ResultsInterpreter:
    """Narrates computed magnitudes; has no authority over any number."""

    def __init__(self, openai_api_key: str = "", collector: Optional[MetricsCollector] = None):
        self.agent_name = "ResultsInterpreter"
        self.llm = LLMClient(
            temperature=0.3,
            api_key=openai_api_key or os.environ.get("OPENAI_API_KEY", ""),
            collector=collector,
            agent_name=self.agent_name,
        )

    def interpret(self, magnitudes_dump: str, research_question: str) -> str:
        raw = self.llm.invoke([
            {"role": "system", "content":
                "You are an economist translating computed statistical magnitudes into "
                "economic meaning. Use ONLY the numbers given — repeat them verbatim, never "
                "compute or invent new ones."},
            {"role": "user", "content":
                f"Research question: {research_question[:300]}\n\n"
                f"Computed magnitudes (deterministic harness):\n{magnitudes_dump[:2400]}\n\n"
                "In <=180 words: what do these effect sizes mean economically, and how large "
                "are they in practical terms?"},
        ], max_tokens=450)
        return str(raw).strip()


# ========== ORCHESTRATOR ==========

class InterpretationOrchestrator:
    """Runs the interpretation stage: magnitudes -> figures -> narration."""

    def __init__(self, openai_api_key: str = "", collector: Optional[MetricsCollector] = None,
                 quiet: bool = False):
        self.collector = collector
        self.quiet = quiet
        self.interpreter = ResultsInterpreter(openai_api_key, collector=collector)
        self.output: Optional[InterpretationStageOutput] = None

    def _print(self, msg: str):
        if not self.quiet:
            print(msg)

    def run_interpretation_pipeline(self, estimation_results: Dict,
                                    research_question: str = "",
                                    figures_dir: str = "figures") -> InterpretationStageOutput:
        self._print("[InterpretationStage] Computing economic magnitudes (deterministic)...")
        interpretation = interpret_estimation(estimation_results)
        files, fig_notes = make_figures(estimation_results, figures_dir)
        interpretation.figure_files = files
        interpretation.notes.extend(fig_notes)
        self._print(f"[InterpretationStage] {len(interpretation.effects)} effect sizes, "
                    f"{len(files)} figure(s); verdict passthrough: {interpretation.verdict}")

        if interpretation.effects:
            try:
                dump = json.dumps([e.model_dump() for e in interpretation.effects], default=str)
                interpretation.narrative = self.interpreter.interpret(dump, research_question)
            except Exception as e:
                self._print(f"[InterpretationStage] Narration skipped (non-critical): {e}")

        self.output = InterpretationStageOutput(interpretation=interpretation)
        return self.output

    def save_interpretation_output(self, filename: str = "interpretation_output.json") -> str:
        if self.output is None:
            raise RuntimeError("run_interpretation_pipeline first")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.output.model_dump(), f, indent=2, default=str)
        self._print(f"[InterpretationStage] Output saved to {filename}")
        return filename


if __name__ == "__main__":
    est_path = auto_input("Path to estimation inference_output.json: ",
                          default="../../EstimationTeam/ael/ModeNoWcNoHITL/inference_output.json").strip()
    with open(est_path, "r", encoding="utf-8") as f:
        estimation_results = json.load(f)
    orch = InterpretationOrchestrator()
    orch.run_interpretation_pipeline(estimation_results)
    orch.save_interpretation_output()
